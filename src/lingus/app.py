"""Entrypoint: wire adapters into the perception-cognition loop.

The offline loop can replay chat, transcript and pre-captioned scene state,
build a compact context snapshot, decide whether to speak, and post a short
deterministic reply. Live adapters and hosted generation land in later phases.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import time

from .adapters.base import ChatAdapter, StreamCaptureAdapter
from .adapters.file_replay import (
    FileReplayCaptureAdapter,
    FileReplayChatAdapter,
    paced_rows,
    read_scene,
    read_transcript,
)
from .arbiter import ArbiterDecision, SimpleArbiter
from .chat_trends import ChatTrend, ChatTrendDetector
from .config import Settings
from .context import ContextSnapshot, build_context_snapshot
from .control import ControlState
from .generator import ReplyGenerator, TemplateReplyGenerator
from .logging import get_logger, setup_logging
from .memory import (
    EpisodicSummarizer,
    ExtractiveSummarizer,
    FactExtractor,
    HeuristicFactExtractor,
    RepetitionGuard,
    SemanticStore,
)
from .models.base import ASRBackend, LLMBackend, ModerationBackend
from .monitor import Monitor, NullMonitor, TickReport
from .output_governor import OutputGovernor
from .persona.loader import load_persona
from .persona.schema import PersonaSpec
from .world_state import Event, SceneState, WorldState

log = get_logger("lingus.app")


def build_adapters(
    settings: Settings, segment_override: str | None, video_override: str | None, speed: float
) -> tuple[StreamCaptureAdapter, ChatAdapter, str | None]:
    """Construct capture + chat adapters for the configured platform."""
    if settings.platform == "file_replay":
        segment = segment_override or settings.file_replay.segment_path
        if not segment:
            raise SystemExit(
                "file_replay platform needs --segment or config.file_replay.segment_path"
            )
        return (
            FileReplayCaptureAdapter(segment, speed=speed),
            FileReplayChatAdapter(segment, speed=speed),
            segment,
        )
    if settings.platform == "youtube":
        from .adapters.youtube import ObserveChatAdapter, YouTubeCaptureAdapter

        video = video_override or settings.youtube.video_id
        if not video:
            raise SystemExit("youtube platform needs --video or config.youtube.video_id")
        # Observe mode: a live capture + ASR spine, but replies are logged not posted.
        return YouTubeCaptureAdapter(video), ObserveChatAdapter(), None
    # Twitch adapter arrives in the final phase.
    raise SystemExit(f"platform '{settings.platform}' not implemented yet")


class BotLoop:
    def __init__(
        self,
        settings: Settings,
        persona: PersonaSpec,
        capture: StreamCaptureAdapter,
        chat: ChatAdapter,
        segment: str | None,
        replay_speed: float = 1.0,
        reply_generator: ReplyGenerator | None = None,
        monitor: Monitor | None = None,
        asr: ASRBackend | None = None,
        controls: ControlState | None = None,
        summarizer: EpisodicSummarizer | None = None,
        fact_extractor: FactExtractor | None = None,
        safety: ModerationBackend | None = None,
    ) -> None:
        self.settings = settings
        self.persona = persona
        self.capture = capture
        self.chat = chat
        self.segment = segment
        self.replay_speed = replay_speed
        self.monitor = monitor or NullMonitor()
        self.asr = asr
        self.safety = safety
        self.controls = controls
        self.arbiter = SimpleArbiter(
            fire_threshold=settings.arbiter.fire_threshold,
            cooldown_seconds=settings.arbiter.cooldown_seconds,
            min_seconds_between_posts=settings.output.min_seconds_between_posts,
            weights=settings.arbiter.weights,
            cooldown_bump=settings.arbiter.cooldown_bump,
            lull_after_seconds=settings.arbiter.lull_after_seconds,
            mood_threshold_gain=settings.arbiter.mood_threshold_gain,
        )
        self.reply_generator = reply_generator or TemplateReplyGenerator()
        ct = settings.chat_trends
        self.trends = (
            ChatTrendDetector(
                window_seconds=ct.window_seconds,
                min_senders=ct.min_senders,
                min_fraction=ct.min_fraction,
                max_token_chars=ct.max_token_chars,
                follow_probability=ct.follow_probability,
                fatigue_seconds=ct.fatigue_seconds,
                cooldown_seconds=ct.cooldown_seconds,
            )
            if ct.enabled
            else None
        )
        self.governor = OutputGovernor(
            max_chars=settings.output.max_chars,
            min_seconds_between_posts=settings.output.min_seconds_between_posts,
            burst=settings.output.burst,
            posts_per_minute=settings.output.posts_per_minute,
            typing_cps=settings.output.typing_cps,
            typing_base_seconds=settings.output.typing_base_seconds,
            typing_min_seconds=settings.output.typing_min_seconds,
            typing_max_seconds=settings.output.typing_max_seconds,
        )
        self.repetition = RepetitionGuard(
            similarity_threshold=settings.memory.similarity_threshold,
            fatigue_seconds=settings.memory.fatigue_seconds,
        )
        self.summarizer = summarizer or ExtractiveSummarizer(
            max_chars=settings.memory.episodic_max_chars
        )
        self.semantic = (
            SemanticStore(max_facts=settings.memory.semantic_max_facts)
            if settings.memory.semantic_enabled
            else None
        )
        self.fact_extractor = fact_extractor or HeuristicFactExtractor()
        self.world = WorldState()
        self._stop = asyncio.Event()
        self._task_error: BaseException | None = None
        self._last_tick = time.monotonic()  # for mood decay between ticks

    async def run(self) -> None:
        tasks: list[asyncio.Task[None]] = []
        capture_started = False
        chat_started = False
        monitor_started = False
        try:
            self.monitor.start()
            monitor_started = True
            await self.capture.start()
            capture_started = True
            await self.chat.start()
            chat_started = True
            log.info("loop started as persona '%s'", self.persona.name)

            # Long-term memory survives restarts: load facts from prior streams,
            # and seed the context with the most-established ones up front.
            if self.semantic is not None:
                self.semantic.load_file(self.settings.memory.semantic_path)
                self.world.semantic_facts = [
                    f.text for f in self.semantic.retrieve("", self.settings.memory.semantic_top_k)
                ]
                log.info("semantic memory: %d durable facts loaded", len(self.semantic))

            tasks = [
                asyncio.create_task(self._ingest_chat(), name="ingest_chat"),
                asyncio.create_task(self._cognition_loop(), name="cognition"),
            ]
            mem = self.settings.memory
            if mem.episodic_enabled or mem.semantic_enabled:
                tasks.append(asyncio.create_task(self._consolidate_loop(), name="consolidate"))
            if self.asr is not None:
                # Live: capture audio -> ASR -> world state.
                tasks.append(asyncio.create_task(self._ingest_audio_asr(), name="ingest_asr"))
            else:
                # Offline convenience: replay pre-transcribed speech + pre-captioned scenes.
                tasks.append(
                    asyncio.create_task(self._ingest_transcript(), name="ingest_transcript")
                )
                tasks.append(asyncio.create_task(self._ingest_scene(), name="ingest_scene"))
            for task in tasks:
                task.add_done_callback(self._remember_task_failure)
            await self._stop.wait()
            # Fold any still-pending evicted lines into memory before teardown.
            with contextlib.suppress(Exception):
                await self._consolidate(force=True)
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                try:
                    await t
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    if self._task_error is None:
                        self._task_error = exc
            if chat_started:
                await self.chat.stop()
            if capture_started:
                await self.capture.stop()
            if monitor_started:
                self.monitor.stop()
            # Persist long-term memory so the next stream starts knowing the channel.
            if self.semantic is not None:
                with contextlib.suppress(OSError):
                    self.semantic.save_file(self.settings.memory.semantic_path)
            log.info("loop stopped (%d events seen)", len(self.world.events))
        if self._task_error is not None:
            raise self._task_error

    def request_stop(self) -> None:
        self._stop.set()

    def _remember_task_failure(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        if self._task_error is None:
            self._task_error = exc
            log.error(
                "task %s failed",
                task.get_name(),
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        self.request_stop()

    # --- perception -> world state ---
    async def _ingest_chat(self) -> None:
        async for msg in self.chat.incoming():
            self.world.add_event(
                Event(
                    source="chat",
                    kind="message",
                    payload={"author": msg.author, "text": msg.text},
                )
            )
            if self.trends is not None:
                # Feed the pile-on detector. The bot's own echoes never reach here
                # (they go through record_own_message, not the chat ingest), so a
                # mirror can't re-trigger itself. A platform that loops the bot's
                # posts back in would need to filter them by author at the adapter.
                self.trends.observe(msg.author, msg.text, time.monotonic())
            log.debug("chat %s: %s", msg.author, msg.text)

    async def _ingest_audio_asr(self) -> None:
        """Live: pull captured audio through ASR into the world state."""
        if self.asr is None:
            return
        async for tr in self.asr.transcribe_stream(self.capture.audio_frames()):
            text = tr.text.strip()
            if not text:
                continue
            self.world.add_event(Event(source="speech", kind="transcript", payload={"text": text}))
            log.info("ASR: %s", text)

    async def _ingest_transcript(self) -> None:
        """Phase-0 convenience: replay pre-transcribed speech (stands in for ASR)."""
        if not self.segment:
            return
        async for row in paced_rows(read_transcript(self.segment), self.replay_speed):
            self.world.add_event(
                Event(source="speech", kind="transcript", payload={"text": row.get("text", "")})
            )

    async def _ingest_scene(self) -> None:
        """Phase-0 convenience: replay pre-captioned scene state (stands in for VLM)."""
        if not self.segment:
            return
        async for row in paced_rows(read_scene(self.segment), self.replay_speed):
            scene = _scene_from_row(row)
            self.world.update_scene(scene)
            self.world.add_event(Event(source="scene", kind="scene_change", payload=dict(row)))

    # --- cognition ---
    async def _cognition_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(1.0)
            await self._cognition_tick()
            # Phase-0 stop condition: if replay is exhausted, end after a quiet beat.
            if self.settings.platform == "file_replay" and self.world.seconds_since_own_message():
                last = self.world.last_event()
                if last and last.age() > 2.0:
                    self.request_stop()

    async def _consolidate_loop(self) -> None:
        """Move aged-out transcript into the slower memory layers.

        Runs off the hot path on a slow cadence and only fires once a batch of
        lines has fallen out of the working window, so the (LLM) summarizer and
        fact extractor are called rarely — not every tick.
        """
        while not self._stop.is_set():
            await asyncio.sleep(5.0)
            await self._consolidate()

    async def _consolidate(self, *, force: bool = False) -> None:
        mem = self.settings.memory
        if not force and self.world.pending_summary_count() < mem.episodic_batch_lines:
            return
        lines = self.world.drain_pending_summary()
        if force:
            # Teardown: also sweep the still-live working window, so a short stream
            # (nothing ever evicted) still persists the facts it heard.
            lines += list(self.world.transcript)
        if not lines:
            return
        if mem.episodic_enabled:
            summary = await self.summarizer.summarize(self.world.episodic_summary, lines)
            self.world.set_episodic_summary(summary)
            log.debug("episodic summary updated (%d lines folded)", len(lines))
        if self.semantic is not None:
            for ef in await self.fact_extractor.extract(lines):
                self.semantic.add(ef.text, ef.subject)
            # Refresh the facts surfaced into context against the current moment.
            self.world.semantic_facts = [
                f.text
                for f in self.semantic.retrieve(self.world.recent_transcript(), mem.semantic_top_k)
            ]

    async def _cognition_tick(self) -> None:
        # Live tuning: push the latest web-UI knob values onto the arbiter,
        # governor and generator before we score this tick.
        if self.controls is not None:
            self.controls.apply(self.arbiter, self.governor, self.reply_generator)
        self._decay_mood()
        # Trend mirror runs *before* the arbiter/generator path: a cresting pile-on
        # is its own decision, joined verbatim. If we follow one this tick, we've
        # spoken — skip the normal "should I speak / what do I say" path.
        if await self._maybe_follow_trend():
            return
        snapshot = build_context_snapshot(self.world)
        seconds_since_own = self.world.seconds_since_own_message()
        decision = self.arbiter.decide(
            snapshot,
            persona_name=self.persona.name,
            seconds_since_own_message=seconds_since_own,
            mood=self.persona.mood.value,
        )
        log.debug(
            "tick: %d events, score=%.2f/thr=%.2f mood=%.2f reasons=%s transcript=%r",
            len(self.world.events),
            decision.score,
            decision.threshold,
            self.persona.mood.value,
            ",".join(decision.reasons) or "-",
            self.world.recent_transcript(3),
        )
        self._nudge_mood(decision)
        posted, dropped = await self._maybe_reply(snapshot, decision)
        self._emit_tick(decision, snapshot, posted=posted, dropped=dropped)

    async def _maybe_reply(
        self,
        snapshot: ContextSnapshot,
        decision: ArbiterDecision,
    ) -> tuple[str | None, str | None]:
        """Generate, temporize, run the Gate-B staleness check, and post.

        Order matters: we compose the reply, "type" it for a human-like beat, and
        only *then* re-check staleness and the hard output gate. The world keeps
        moving while the bot types, so the staleness re-check after the delay
        doubles as a cheap barge-in: if something newer took over, we drop.

        Returns (posted, dropped).
        """
        if not decision.should_reply:
            return None, None

        # Master switch: when chatting is disabled we still perceive and score
        # (so the dashboard keeps moving), but we never generate or post.
        if self.controls is not None and not self.controls.chat_enabled:
            return None, "🔇 muted (would have spoken)"

        reply = await self.reply_generator.generate(
            snapshot, decision, self.persona, max_chars=self.settings.output.max_chars
        )
        # One tighter regeneration if the reply blew past the cap, so the governor's
        # truncation is a fallback rather than the common case.
        out_cfg = self.settings.output
        if reply and out_cfg.regenerate_on_overflow and self.governor.over_length(reply):
            reply = await self.reply_generator.generate(
                snapshot, decision, self.persona, max_chars=int(out_cfg.max_chars * 0.8)
            )
        if not reply:
            return None, None
        # Safety pre-check with one regeneration: unlike the trend mirror (a
        # verbatim copy that can only be dropped), a generated reply that trips
        # moderation gets a second draft before we give up. The authoritative
        # gate still lives in _post_message; this just spares a salvageable reply.
        if self.safety is not None and not (await self.safety.check(reply)).allowed:
            log.info("reply failed safety; regenerating once: %s", reply)
            reply = await self.reply_generator.generate(
                snapshot, decision, self.persona, max_chars=self.settings.output.max_chars
            )
            if not reply:
                return None, None
        # Self-memory dedup + bit-fatigue: drop a reply that repeats (verbatim or
        # reworded) something the bot said recently, or that leans on a spent
        # catchphrase. Repetition is the #1 immersion-killer (CLAUDE.md §5).
        if self.repetition.is_repetitive(reply, self.world.own_messages, self.persona):
            log.info("dropping repetitive reply: %s", reply)
            return None, reply

        # Temporizer: emulate the time a human takes to type this message, so a
        # sentence can't land instantly on the heels of the prior one. Offline we
        # compress it by the replay speed, same as the rest of the replay clock.
        delay = self.governor.typing_delay(reply) / max(self.replay_speed, 1e-9)
        await asyncio.sleep(delay)

        # Gate B — staleness: re-check that the moment we decided to speak into
        # still wants a reply and that nothing newer became the thing worth
        # reacting to. If it moved on (incl. during typing), drop and let the next
        # tick re-evaluate rather than posting a stale line.
        recheck = self.arbiter.decide(
            build_context_snapshot(self.world),
            persona_name=self.persona.name,
            seconds_since_own_message=self.world.seconds_since_own_message(),
            mood=self.persona.mood.value,
        )
        if not recheck.should_reply or recheck.trigger_event is not decision.trigger_event:
            log.info("dropping stale reply (context moved on): %s", reply)
            return None, reply

        posted, dropped = await self._post_message(reply, drop_context="reply")
        if posted is not None:
            log.info("bot replied: %s", posted)
        return posted, dropped

    async def _maybe_follow_trend(self) -> str | None:
        """Trend mirror: when chat converges on an emote/phrase, pile on with the
        same line — bypassing the generator entirely.

        This is a *copy*, not a "what do I say", so it skips the LLM and the
        typing temporizer: a pile-on that lands seconds late reads worse than not
        joining at all. The deterministic guardrails still apply — the chat-enabled
        master switch and the output governor (rate limit + length cap) — and a
        successful follow feeds back into self-memory like any other post.

        Returns the posted text, or ``None`` if nothing was followed.
        """
        if self.trends is None:
            return None
        # Feature toggle (web UI): trend-following can be switched off independently
        # of the chat master switch. Off → don't even poll (poll has decision state).
        if self.controls is not None and not self.controls.trends_enabled:
            return None
        now = time.monotonic()
        trend = self.trends.poll(now)
        if trend is None:
            return None

        # Master switch: when muted we still detect (so the dashboard shows the
        # wave) but post nothing — and leave the trend un-followed so it can fire
        # the moment chatting is re-enabled.
        if self.controls is not None and not self.controls.chat_enabled:
            self._report_trend_tick(trend, posted=None, dropped="🔇 muted (would have followed)")
            return None

        self.world.chat.trend = trend  # observability
        posted, dropped = await self._post_message(trend.message, drop_context="trend mirror")
        if posted is None:
            self._report_trend_tick(trend, posted=None, dropped=dropped)
            return None

        self.trends.mark_followed(trend, now)
        log.info(
            "bot followed chat trend (%dx, %d senders, %s): %s",
            trend.count,
            trend.senders,
            trend.phase,
            posted,
        )
        self._report_trend_tick(trend, posted=posted, dropped=None)
        return posted

    async def _post_message(
        self,
        text: str,
        *,
        drop_context: str,
    ) -> tuple[str | None, str | None]:
        """Run the final deterministic post path and update self-memory on success.

        Order — safety first: an unsafe line must never post even if the governor
        would rate-limit it away this tick anyway. This is the last, authoritative
        moderation gate (CLAUDE.md §9); the generated-reply path may pre-check and
        regenerate upstream, but everything (incl. verbatim trend mirrors) is
        re-checked here so nothing slips through a bypass.
        """
        if self.safety is not None:
            verdict = await self.safety.check(text)
            if not verdict.allowed:
                log.warning("safety dropped %s (%s): %s", drop_context, verdict.reason, text)
                return None, f"⚠ unsafe ({verdict.reason})"
        outcome = self.governor.admit(text)
        if outcome.action == "drop":
            log.info("governor dropped %s (%s): %s", drop_context, outcome.reason, text)
            return None, text
        if outcome.truncated:
            log.info("governor truncated %s to %d chars", drop_context, len(outcome.text))

        await self.chat.post(outcome.text)
        self.world.record_own_message(outcome.text)
        self.repetition.note_post(outcome.text, self.persona)
        return outcome.text, None

    def _report_trend_tick(
        self, trend: ChatTrend, *, posted: str | None, dropped: str | None
    ) -> None:
        """Emit a monitor tick for a trend-follow, so the dashboard shows it like
        any other decision (the trend path doesn't run the arbiter, so we synth a
        decision describing what fired)."""
        decision = ArbiterDecision(
            should_reply=posted is not None,
            score=float(trend.senders),
            reasons=["chat_trend", trend.phase],
            threshold=0.0,
        )
        snapshot = build_context_snapshot(self.world)
        self._emit_tick(decision, snapshot, posted=posted, dropped=dropped)

    def _emit_tick(
        self,
        decision: ArbiterDecision,
        snapshot: ContextSnapshot,
        *,
        posted: str | None,
        dropped: str | None,
    ) -> None:
        self.monitor.on_tick(
            self._build_tick_report(
                decision,
                snapshot,
                posted=posted,
                dropped=dropped,
            )
        )

    def _build_tick_report(
        self,
        decision: ArbiterDecision,
        snapshot: ContextSnapshot,
        *,
        posted: str | None,
        dropped: str | None,
    ) -> TickReport:
        return TickReport(
            t=time.monotonic(),
            decision=decision,
            mood=self.persona.mood.value,
            n_events=len(self.world.events),
            transcript_tail=self.world.recent_transcript(3),
            recent_chat=list(snapshot.recent_chat),
            scene_summary=snapshot.scene_summary(),
            posted=posted,
            dropped=dropped,
        )

    def _decay_mood(self) -> None:
        now = time.monotonic()
        elapsed_minutes = (now - self._last_tick) / 60.0
        self._last_tick = now
        self.persona.mood.decay(elapsed_minutes)

    def _nudge_mood(self, decision: ArbiterDecision) -> None:
        """Energy rises with excitement; it decays back to baseline on its own."""
        if "hype" in decision.reasons:
            self.persona.mood.nudge(0.15)
        if "scene_event" in decision.reasons or "streamer_mishap" in decision.reasons:
            self.persona.mood.nudge(0.1)


def _scene_from_row(row: dict) -> SceneState:
    salient_objects = row.get("salient_objects", [])
    if not isinstance(salient_objects, list):
        salient_objects = []
    return SceneState(
        activity=str(row.get("activity", "")),
        setting=str(row.get("setting", "")),
        on_screen_text=str(row.get("on_screen_text", "")),
        salient_objects=[str(item) for item in salient_objects],
        last_event=str(row.get("last_event", "")),
    )


def _build_monitor(
    args: argparse.Namespace,
    persona: PersonaSpec,
    platform: str,
    controls: ControlState | None,
) -> Monitor:
    if args.web:
        try:
            import aiohttp  # noqa: F401  (presence check; imported for real in webui)
        except ImportError as exc:
            raise SystemExit("--web needs the 'web' extra: pip install -e '.[web]'") from exc
        from .webui import WebMonitor

        assert controls is not None  # created alongside the --web flag in _amain
        return WebMonitor(controls, persona.name, platform, port=args.web_port)
    if not args.dashboard:
        return NullMonitor()
    try:
        from .dashboard import RichDashboard
    except ImportError as exc:  # rich not installed
        raise SystemExit(
            "--dashboard needs the 'dashboard' extra: pip install -e '.[dashboard]'"
        ) from exc
    return RichDashboard(persona.name, platform)


def _build_safety(settings: Settings) -> ModerationBackend | None:
    """The moderation gate. Off only when config says so — and then loudly,
    because observe-mode tuning is the only legitimate reason to run without it."""
    from .safety import build_moderation

    safety = build_moderation(settings)
    if safety is None:
        log.warning(
            "moderation DISABLED (moderation.backend=%s) — safe only for offline tuning",
            settings.models.moderation.backend,
        )
    else:
        log.info("moderation: %s", settings.models.moderation.backend)
    return safety


def _build_asr(settings: Settings) -> ASRBackend | None:
    """Live audio needs ASR; file replay ships its own pre-transcribed text."""
    if settings.platform == "file_replay":
        return None
    cfg = settings.models.asr
    if cfg.backend != "faster_whisper":
        raise SystemExit(f"asr backend '{cfg.backend}' not implemented")
    from .models.faster_whisper_asr import FasterWhisperASR

    return FasterWhisperASR(
        model_size=cfg.model_size,
        device=cfg.device,
        language=cfg.language,
        window_seconds=cfg.window_seconds,
    )


def _build_llm_backend(settings: Settings) -> LLMBackend | None:
    """Shared hosted-LLM backend for the generator and the episodic summarizer.

    Returns None when no key is configured, so callers fall back to their
    deterministic offline path (template generator / extractive summarizer).
    """
    cfg = settings.models.llm
    if cfg.backend == "openai_compat" and settings.secrets.openai_api_key:
        from .models.openai_llm import OpenAICompatLLM

        return OpenAICompatLLM(
            api_key=settings.secrets.openai_api_key,
            base_url=settings.secrets.openai_base_url,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    return None


def _build_generator(settings: Settings, backend: LLMBackend | None) -> ReplyGenerator | None:
    """Hosted LLM when a key is present; else the deterministic template."""
    if backend is not None:
        from .models.openai_llm import LLMReplyGenerator

        log.info("generator: hosted LLM (%s)", settings.models.llm.model)
        return LLMReplyGenerator(backend)
    log.info("generator: template (no API key set)")
    return None  # BotLoop falls back to TemplateReplyGenerator


def _build_summarizer(
    settings: Settings,
    backend: LLMBackend | None,
) -> EpisodicSummarizer | None:
    """Episodic memory: LLM summarizer when a key is present; else extractive."""
    if not settings.memory.episodic_enabled:
        return None
    if backend is not None:
        from .memory import LLMSummarizer

        return LLMSummarizer(backend, max_chars=settings.memory.episodic_max_chars)
    return ExtractiveSummarizer(max_chars=settings.memory.episodic_max_chars)


def _build_fact_extractor(
    settings: Settings,
    backend: LLMBackend | None,
) -> FactExtractor | None:
    """Semantic memory: LLM extraction when a key is present; else heuristic."""
    if not settings.memory.semantic_enabled:
        return None
    if backend is not None:
        from .memory import LLMFactExtractor

        return LLMFactExtractor(backend)
    return HeuristicFactExtractor()


async def _seed_research(
    args: argparse.Namespace,
    settings: Settings,
    llm_backend: LLMBackend | None,
) -> None:
    """Cold-start: profile the channel and seed durable memory before the loop.

    Runs once per channel (cached; re-researched after `research.refresh_days`),
    writes the resulting facts into the semantic store with source="research", and
    persists it — so `BotLoop.run()` then loads them like any other durable fact
    and surfaces them into the generator's context. Entirely best-effort: a
    research failure logs and returns, it never blocks the stream from starting.
    """
    cfg = settings.research
    if args.no_research or not cfg.enabled or not settings.memory.semantic_enabled:
        return
    from .research import research_channel, resolve_identity

    video = args.video or settings.youtube.video_id
    identity = await resolve_identity(
        settings.platform, video=video, channel_name=cfg.channel
    )
    if identity is None:
        log.info("research: no channel to research (set research.channel for this platform)")
        return
    log.info("research: profiling channel '%s' (%s)", identity.name, identity.platform)
    profile = await research_channel(
        identity,
        web_backend=cfg.web_search.backend,
        cache_dir=cfg.cache_dir,
        refresh_days=cfg.refresh_days,
        max_facts=cfg.max_facts,
        max_queries=cfg.web_search.max_queries,
        max_results=cfg.web_search.max_results,
        llm=llm_backend,
        force=args.research,
    )
    if profile is None or not profile.facts:
        log.info("research: no facts produced for '%s'", identity.name)
        return
    store = SemanticStore(max_facts=settings.memory.semantic_max_facts)
    store.load_file(settings.memory.semantic_path)
    before = len(store)
    for fact in profile.facts:
        store.add(fact, subject="channel", source="research")
    store.save_file(settings.memory.semantic_path)
    log.info(
        "research: seeded %d/%d durable facts about '%s' into memory",
        len(store) - before,
        len(profile.facts),
        identity.name,
    )


async def _run_eval(args: argparse.Namespace) -> None:
    """Phase 6: replay a recorded segment offline and score the bot's outputs.

    Uses the real generator (hosted LLM when a key is set, else the template) and
    the LLM-as-judge when a backend is available, falling back to the deterministic
    heuristic judge otherwise — so `--eval` produces a report with or without keys.
    """
    from .eval import HeuristicJudge, LLMJudge, evaluate_segment

    settings = Settings.load(args.config)
    persona = load_persona(args.persona or settings.persona.path)
    segment = args.segment or settings.file_replay.segment_path
    if not segment:
        raise SystemExit("--eval needs --segment (a recorded segment directory)")
    llm_backend = _build_llm_backend(settings)
    reply_generator = _build_generator(settings, llm_backend)
    judge = LLMJudge(llm_backend) if llm_backend is not None else HeuristicJudge()
    log.info("eval judge: %s", type(judge).__name__)
    report = await evaluate_segment(
        settings,
        persona,
        segment,
        judge=judge,
        reply_generator=reply_generator,
        speed=args.speed,
    )
    print("\n".join(report.summary_lines()))
    if args.eval_json:
        import json

        with open(args.eval_json, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        log.info("eval report written to %s", args.eval_json)


async def _amain(args: argparse.Namespace) -> None:
    settings = Settings.load(args.config)
    if args.platform:
        settings.platform = args.platform
    # The dashboard owns the terminal, so send logs to a file while it runs.
    setup_logging(
        settings.logging.level,
        settings.logging.as_json,
        log_file=args.log_file or ("lingus.log" if args.dashboard else None),
    )
    persona = load_persona(args.persona or settings.persona.path)
    # Per-run ASR overrides. Streams are usually single-language, so pinning the
    # language skips Whisper's per-window detection (no more flapping to the wrong
    # language on low-confidence chunks) and trims a little latency. `auto`/`none`
    # forces re-detection back on even if config pins a language.
    if args.language is not None:
        settings.models.asr.language = (
            None if args.language.lower() in ("auto", "none", "") else args.language
        )
    if args.asr_model is not None:
        settings.models.asr.model_size = args.asr_model
    if args.asr_window is not None:
        settings.models.asr.window_seconds = args.asr_window
    # Replay speed only makes sense offline; live capture runs in real time.
    speed = args.speed if settings.platform == "file_replay" else 1.0
    capture, chat, segment = build_adapters(settings, args.segment, args.video, speed)
    controls = ControlState(settings) if args.web else None
    monitor = _build_monitor(args, persona, settings.platform, controls)
    asr = _build_asr(settings)
    safety = _build_safety(settings)
    llm_backend = _build_llm_backend(settings)
    reply_generator = _build_generator(settings, llm_backend)
    summarizer = _build_summarizer(settings, llm_backend)
    fact_extractor = _build_fact_extractor(settings, llm_backend)
    loop = BotLoop(
        settings,
        persona,
        capture,
        chat,
        segment,
        replay_speed=speed,
        monitor=monitor,
        asr=asr,
        reply_generator=reply_generator,
        controls=controls,
        summarizer=summarizer,
        fact_extractor=fact_extractor,
        safety=safety,
    )

    # Cold-start: research the channel and seed durable memory before the loop
    # loads it. Best-effort — never let it stop the stream from starting.
    try:
        await _seed_research(args, settings, llm_backend)
    except Exception:  # noqa: BLE001 - research must not block the run
        log.exception("research: cold-start seeding failed; continuing without it")

    running = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            running.add_signal_handler(sig, loop.request_stop)

    await loop.run()


def main() -> None:
    parser = argparse.ArgumentParser(prog="lingus")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--segment", default=None, help="recorded segment dir (file_replay)")
    parser.add_argument("--video", default=None, help="youtube video id or URL (youtube platform)")
    parser.add_argument(
        "--platform",
        default=None,
        choices=["file_replay", "youtube", "twitch"],
        help="override config.yaml platform for this run",
    )
    parser.add_argument("--persona", default=None, help="path to a persona YAML")
    parser.add_argument(
        "--research",
        action="store_true",
        help="force cold-start channel research, ignoring any cached profile",
    )
    parser.add_argument(
        "--no-research",
        action="store_true",
        help="skip cold-start channel research for this run",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="pin ASR language for this run (e.g. 'it', 'en'); 'auto' to autodetect per window",
    )
    parser.add_argument(
        "--asr-model",
        default=None,
        help="override ASR model size for this run (tiny/base/small/medium/large-v3)",
    )
    parser.add_argument(
        "--asr-window",
        type=float,
        default=None,
        help="ASR window seconds for this run (bigger = lower RTF + more context, +latency)",
    )
    parser.add_argument("--speed", type=float, default=10.0, help="replay speed multiplier")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Phase-6 eval: replay --segment offline and score the bot's outputs, then exit",
    )
    parser.add_argument(
        "--eval-json", default=None, help="also write the eval report as JSON to this path"
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="live terminal dashboard (needs [dashboard] extra)"
    )
    parser.add_argument("--web", action="store_true", help="live web tuner UI (needs [web] extra)")
    parser.add_argument("--web-port", type=int, default=8080, help="port for --web (default 8080)")
    parser.add_argument(
        "--log-file", default=None, help="write logs here (default: lingus.log under --dashboard)"
    )
    args = parser.parse_args()
    if args.eval:
        # Report goes to stdout; keep logs on stderr and out of the way.
        setup_logging("WARNING", False, log_file=args.log_file)
        asyncio.run(_run_eval(args))
        return
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
