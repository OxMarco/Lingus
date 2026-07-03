# Live-Stream Interaction Bot — Project Spec & Implementation Plan

A real-time bot that watches a live stream (video + audio + chat), understands what
is happening, and posts characterful replies into the stream's text chat. Optimized
for **personality** over raw coverage.

---

## 1. Project idea

The bot perceives a live stream through three channels — video, audio, and chat —
fuses them into a coherent picture of "what's happening right now," and decides when
it's worth saying something. When it is, it generates a short, in-character message
and posts it to chat. It remembers the stream so far (and prior streams) so it can
make callbacks, sustain running jokes, and feel like a familiar presence rather than
a stateless responder.

**Output channel:** text chat (no voice/TTS for v1).
**Primary design priority:** personality — being recognizable, well-timed, and
non-annoying — over comprehensiveness.

---

## 2. Core design principles

These are the non-obvious decisions that shape the whole system. They matter more
than any single library choice.

1. **It's a perception–cognition loop on a clock, not a request/response system.**
   The hardest problems are about *timing*, not about any single perception module.

2. **Perception writes to a shared world-state; cognition reads state, never raw streams.**
   Video, audio, and chat arrive at totally different rates. A shared, timestamped
   "blackboard" decouples them, lets each module fail independently, and gives the
   LLM one coherent, time-aligned context.

3. **"Should I speak?" is a separate decision from "What do I say?"**
   A cheap, always-on **arbiter** decides *whether/when* to react; the expensive
   **generator** only runs when the arbiter fires. This is the main lever for both
   cost and not-being-spammy.

4. **Personality is distributed across three subsystems — not just the prompt.**
   - **Arbiter** → *timing = temperament.* What a character reacts to defines it as
     much as what it says.
   - **Generator** → *voice.* Phrasing, lexicon, opinions.
   - **Memory** → *relationships & callbacks.* A personality with no memory of its
     own jokes has no character.

5. **Speech is the spine.** For most streams, the streamer's voice carries the
   semantic load and is cheap to process as text. Video provides grounding/event
   detection; invest in good ASR before investing in video.

6. **Video produces a running scene-state, not per-frame captions.** Gate local
   analysis behind cheap frame-diffing; only update scene state on meaningful
   change; report *what changed* relative to the prior state.

7. **Chat is its own perception problem.** Don't feed the raw firehose to the LLM.
   Aggregate into a "chat state": questions to the bot, hype/sentiment spikes,
   emergent topics, raids/spam.

8. **Staleness is the real latency risk.** The world moves while the bot thinks.
   Timestamp the triggering context, re-check relevance before posting, and support
   aborting an in-flight reply if something more important happens.

9. **Review output** A regex filter pass runs on every generated
   message before it is posted preventing spam or offensive language (but jokes, satire, colloquial and fun language is allowed).

10. **You can't tune personality blind.** A replay/eval harness is a first-class
    component, not an afterthought.

---

## 3. Architecture

```
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │  VIDEO   │   │  AUDIO   │   │   CHAT   │     PERCEPTION
        │ scene    │   │ speech + │   │ questions│
        │ events   │   │ sound    │   │ + hype   │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             └──────────────┼──────────────┘
                            ▼
              ┌───────────────────────────┐
              │     SHARED WORLD STATE     │          (timestamped event timeline)
              └─────────────┬─────────────┘
                            ▼
                 ┌────────────────────┐
                 │ ARBITER            │                "should I speak?"
                 │ salience + cooldown│                (cheap, always-on)
                 └─────────┬──────────┘
                           │ on trigger
                           ▼
   ┌──────────┐   ┌────────────────────┐
   │  MEMORY   │──▶│ GENERATOR          │               LLM + persona + context
   │ short +  │   │ generate reply     │
   │ long-term│   └─────────┬──────────┘
   └──────────┘             ▼
                  ┌────────────────────┐
                  │ SAFETY FILTER      │                moderation pass
                  └─────────┬──────────┘
                            ▼
                  ┌────────────────────┐
                  │ OUTPUT → post to   │
                  │ stream chat        │
                  └────────────────────┘
```

The bot's own posted messages re-enter the world state (so it doesn't repeat itself
and can follow its own thread).

---

## 4. Component breakdown

### Perception
- **Video → scene state:** sample frames; gate with perceptual hash / SSIM diffing
  so local analysis only runs on real change. Maintain a structured object
  (`activity`, `setting`, `on_screen_text`, `salient_objects`, `last_event`) and
  record *changes*, not static per-frame captions.
- **Audio → speech + events:** streaming ASR for the streamer's voice (the richest,
  cheapest signal); optional lightweight tagging for non-speech sounds.
- **Chat → chat state:** aggregate the firehose into questions-to-bot, sentiment/hype
  spikes, emergent topics, spam/raid detection.

### World state
- A single in-memory object updated by each perception module, with timestamps.
- Rolling window of recent events; older events get compressed (see Memory).

### Arbiter ("should I speak?")
- Computes a **salience score** from signals: direct address, unanswered question,
  dramatic on-screen event, chat hype spike, conversational lull, time-since-last-bot-message.
- Fires when score crosses a threshold; the threshold **rises after the bot speaks**
  (cooldown) so it can't dominate.
- **This is where temperament lives** — signal weights and thresholds *are* the
  character. Mostly heuristics; optionally a small classifier.

### Generator ("what do I say?")
- Runs only when the arbiter fires.
- Assembles: persona spec + relevant memory + current world state + bot's own recent
  messages → short in-character reply.
- Enforces brevity, no "assistant voice," no trailing "how can I help?".
- Driven by an **exemplar bank** (concrete sample reactions), not adjective lists.

### Safety filter
- Moderation pass on the generated text before posting. Drop/regenerate on fail.

### Output governor
- The **deterministic last line** before a message is posted. Everything upstream is
  probabilistic — the arbiter *decides* whether to react, the generator (an LLM) *decides*
  what to say and will sometimes ignore a length instruction. Spam rate and message length
  are **invariants**, so they're enforced here in code, where no model decision can bypass
  them. The arbiter's matching rate pre-filter only exists to avoid generating a reply we'd
  drop anyway; the governor is the authority.
- **Rate:** a token bucket (sustained `posts_per_minute` + `burst` ceiling) plus a hard
  `min_seconds_between_posts` floor. Drops, never queues — a queued reply posts stale.
- **Length:** a hard `max_chars` cap with **sentence-aware truncation** (cut at a sentence,
  else a word boundary + ellipsis — never mid-word, never over the cap). The generator no
  longer truncates; the cap lives here so it's a single source of truth, not dead code.
- **Optional typing delay:** a flavor knob only, disabled by default for live
  responsiveness. The hot path should not wait on theatrical pacing.
- Lives between the safety filter and the post; tunable via `output:` in `config.yaml`.

### Output
- Post to stream chat via the platform library. Re-inject into world state.

---

## 5. Personality design (priority feature)

**Persona spec** is a structured artifact, not a prose blurb:
- voice/tone, lexicon (words it uses/avoids, catchphrases)
- interests, opinions/stances (it should have takes and be allowed to be wrong in-character)
- relationships (to the streamer, to regulars), boundaries (what it won't engage)
- **exemplar bank**: concrete sample lines reacting to specific situations
- optional bounded, decaying **mood/energy** variable that nudges arbiter thresholds
  and generator phrasing

**Anti-patterns to engineer against**
- Assistant-voice leak (the #1 immersion-killer): hedging, over-explaining, balanced
  both-sides answers, constant questions.
- Repetition: track "bit fatigue" (recent uses of each catchphrase/joke) + dedup
  against the bot's own recent outputs.
- Drift: keep mood bounded and decaying.

**Reference design:** SillyTavern character cards + lorebooks are a battle-tested
instantiation of "structured persona spec + running-bits memory." Borrow the patterns.

---

## 6. Memory architecture

Four layers, built differently:

| Layer | What | How to build |
|---|---|---|
| **Working** | last few minutes of transcript, scene states, chat highlights | rolling buffer, fed directly into context |
| **Episodic** | "the stream so far" narrative + prior stream summaries | periodic summarization of old working memory; persist compact per-stream summaries; evict raw |
| **Semantic / long-term** | durable facts: streamer, regulars, running bits across streams | vector store + extraction (hand-rolled, or Mem0) |
| **Self-memory** | the bot's own recent messages | feed back in + dedup (commonly forgotten — don't) |

> RAG / vector DB only touches the **semantic** layer. At one-streamer scale it is a
> minor component, not the centerpiece. Build working + episodic yourself; keep
> the durable episodic archive JSON-backed until evals show retrieval misses.

---

## 7. Tech stack

**Language decision: Python core.**
We keep perception local for cost and control, and that ecosystem
(faster-whisper, NeMo/Parakeet, torch, OpenCV, PyAV) is Python-first in practice.
The pipeline framework (Pipecat), the Twitch library, and the audio/video stack
are all Python-first too. Node's concurrency strength solves a problem we don't
have (our bottleneck is inference latency, not socket count), and the GIL is a
non-issue here (inference releases it via C extensions; asyncio orchestrates I/O;
perception workers run as separate processes). Hosted LLM calls are reserved for
text summarization and bot replies.
**Optional TypeScript/React** only for a web overlay or dashboard, as a sidecar over a
websocket — not the core.

| Component | Primary pick | Alternative / backup | Notes |
|---|---|---|---|
| Stream A/V capture | Streamlink + FFmpeg (PyAV) | yt-dlp | Pulls live feed, demuxes audio + frames |
| Pipeline spine | Pipecat | thin asyncio loop | Processors per stage; arbiter & safety are custom processors |
| Chat ingestion | TwitchIO (Twitch) | pytchat / YouTube Data API | Async, EventSub over WebSocket |
| Speech → text (ASR) | faster-whisper `turbo` | Parakeet on NVIDIA | M4 Max/64 GB can target large-v3-turbo locally; Silero VAD is bundled |
| Video frame gating | local RGB diff + OpenCV/PySceneDetect later | — | Cheap diff; analyze only on real change |
| Scene state | MLX-VLM + Qwen2.5-VL 3B 4-bit | `local_cv` fallback / 7B offline override | Local Apple Silicon VLM; no hosted vision API |
| Speech tagging | sherpa-onnx embeddings | spectral/HF audio gate | Next local audio semantics slice |
| Audio events (optional) | lightweight audio tagger | skip for v1 | Non-speech sounds; low priority |
| Generator LLM | hosted API *or* self-host Qwen3 / Mistral / GLM | — | Text output gives latency headroom |
| LLM serving (if self-host) | vLLM (prod) | Ollama (dev) | vLLM for sub-200ms TTFT + batching |
| Arbiter | custom (heuristic scoring) + optional small classifier | — | The temperament — you build it |
| Working/episodic memory | custom (rolling buffer + summarization) | — | Don't reach for a framework here |
| Long-term/semantic memory | Mem0 (optional) | hand-rolled extraction | Adopt when manual version creaks |
| Vector DB | Chroma (proto) → pgvector or Qdrant | — | Near-non-decision at this scale |
| Output moderation | `RegexModeration` first | platform/guard API only if needed | Cheap pre-post safety pass; keep model-based moderation out of the hot path unless evals prove the regex gate is insufficient |
| Eval / replay harness | custom recorded segments + heuristic judge | optional local/explicit judge later | Lets you tune personality without hidden hosted calls |
| Persona patterns (reference) | SillyTavern character cards + lorebooks | — | Design reference, not a runtime dep |

> **Skipped on purpose:** LangChain/LangGraph for the hot loop. As of v1.0 they're
> toolkit + stateful runtime used together; their value (checkpointing, HITL, branching)
> is marginal for a simple latency-sensitive loop, and the ecosystem churns. Use the
> model SDK directly. Pipecat is the better-fit spine.

> **The "custom" rows are not gaps in tooling.** The arbiter, working/episodic memory,
> and the replay harness are the parts that make this bot good rather than generic —
> which is exactly why no library ships them. Spend your real time here.

---

## 8. Implementation plan (build order)

Build in **vertical slices** so you always have something live to test. The latency
loop is the thing to validate early.

### Phase 0 — Skeleton
- Repo, config, env, logging. Define the **world-state** object and the **persona spec**
  schema up front (everything reads/writes these).

### Phase 1 — Speech → reply → chat (the vertical slice that already feels alive)
- Capture stream audio (Streamlink/FFmpeg).
- Streaming ASR → world state.
- Trivial arbiter (e.g. fire on direct mention / question).
- Generator with a first-draft persona + exemplar bank.
- **Output governor** (lands here, not Phase 5): hard length cap + sentence-aware
  truncation, a basic rate floor + token bucket, and optional typing delay disabled by default. ~one module,
  and a bot that spams the test chat or posts paragraphs is painful from the first run.
- Post to chat via TwitchIO.
- **Goal:** validate the end-to-end latency loop with the cheapest possible path.

### Phase 2 — Chat perception
- Ingest full chat; build the chat-state aggregator (questions, hype, spam).
- Expand arbiter signals to include chat (hype spikes, lulls).

### Phase 3 — Memory
- Working buffer + self-memory + dedup (kills repetition).
- Episodic summarization (survives long streams).
- Semantic store for durable facts / running bits (hand-rolled first; Mem0 if it creaks).

### Phase 3.5 — Memory hardening
- Persist compact per-stream episodic summaries so prior stream context can be
  recalled without raw transcript retention.
- Keep short-term working memory in `WorldState`; keep retrieval cheap
  (token-overlap + recency/popularity). Do not add a vector/RAG database until
  the corpus is large enough or evals show obvious callback misses.

### Phase 4 — Video
- Frame capture + diff-gating (OpenCV/PySceneDetect).
- Local scene-state analysis with change-reporting.
- Add on-screen events to arbiter signals.
- *(Last because it's the most expensive and lowest early marginal value.)*

### Phase 5 — Lean hardening
- Keep only cheap deterministic guardrails in the hot path: pre-ASR audio gating,
  output moderation, output rate/length caps, and repetition/bit-fatigue.
- Use the existing cheap trigger-age/staleness checks, but do not build mid-flight
  abort/barge-in machinery or per-signal cooldown systems unless evals show a real
  problem. Reply latency beats cleverness here.

### Phase 6 — Eval loop (start informally in Phase 1, formalize here)
- Record real stream segments (transcript + chat + scene states).
- Replay the bot offline; score outputs (in-character / not-generic / not-repetitive)
  with a deterministic heuristic judge by default.
- Use it to tune arbiter weights and the exemplar bank.

---

## 9. Failure modes to watch

- **Spammy / talks too much** → arbiter thresholds + cooldowns.
- **Stale/awkward** → cheap trigger-age/staleness checks; avoid blocking abort
  machinery unless evals prove it is needed.
- **Repetitive** → self-memory dedup + bit-fatigue tracking.
- **Hallucinating on-screen content** → keep local analysis conservative; do not
  invent objects/text the local analyzer cannot verify.
- **Assistant-voice leak** → brevity caps, no trailing questions, opinions allowed.
- **Context loss over long streams** → episodic summarization.
- **Unsafe output** → moderation pass before posting (never skip).

---

## 10. Open decisions (deferred)

- Hosted API vs self-hosted for the **generator** (text output leaves room for either).
- Which exact ASR (Parakeet vs faster-whisper vs streaming API) — depends on GPU access.
- When to adopt Mem0 vs keep the semantic layer hand-rolled.
- Multi-platform (YouTube alongside Twitch) — additive via a second chat adapter.

---

## 11. Build progress (as of 2026-07-01)

State of the implementation against the Phase plan in §8. 214 tests pass, ruff clean.

### Done
- **Phase 0 / 0.5 — skeleton & offline loop.** Config, timestamped world-state,
  persona spec (schema + loader, `default.yaml`), adapter/model ABCs, context
  snapshot, heuristic arbiter, **output governor** (rate bucket + sentence-aware
  length cap + optional typing delay disabled by default), dashboard/web UI, monitor. Offline replay loop
  (`--segment`) runs end-to-end.
- **Phase 1 — speech → reply → chat.** Validated **live** on a real YouTube stream
  in observe mode (logs, never posts):
  - Capture: yt-dlp → ffmpeg → 16 kHz PCM (`adapters/youtube.py`).
  - ASR: faster-whisper, **target `turbo` / `en` / 10s window** on this M4 Max
    64 GB machine; `medium` remains a safe fallback. **VAD-based
    utterance segmentation** (Silero, bundled) with amortization so each Whisper
    call clears the ~4s break-even; cuts on pauses, not mid-word. Hallucination
    filter (confidence + denylist) for the "Sottotitoli a cura di…" family.
    Pre-ASR audio gate (`models.audio_gate`) drops silence/music-like windows
    before Whisper, replacing them with short silence so music/lyrics don't
    poison transcript context; schema/default config is lightweight `spectral`.
    The locally cached `hf_ast` Hugging Face AudioSet gate is optional when the
    audio ML dependencies are installed.
    CLI overrides: `--language`, `--asr-model`, `--asr-window`.
  - Arbiter → generator → governor → post path: **fired live** on a real question.
  - Generator: hosted LLM wired (`openai_compat`), falls back to deterministic
    template when no API key.
- **Phase 2 — chat perception (partial).** `ChatTrendDetector` (pile-on / hype
  with fatigue + cooldown) built and wired into the arbiter. Chat-state has
  hype_level / questions_to_bot / trend.
- **Phase 3 — memory (COMPLETE).** `src/lingus/memory/`:
  - Working buffer + self-memory (rolling deques on `WorldState`).
  - **Dedup + bit-fatigue** (`RepetitionGuard`) — token-Jaccard near-dup +
    catchphrase fatigue; deterministic gate in the post path.
  - **Episodic summarization** — evicted transcript lines folded into a "stream
    so far" narrative; `LLMSummarizer` with `ExtractiveSummarizer` fallback.
  - **Durable episodic archive** — per-stream summaries persisted to
    `.lingus/episodes.json` via `EpisodicArchive`, then surfaced as "Past stream
    memories" in generator context. This is Phase 3.5's lightweight alternative
    to introducing a vector/RAG database too early.
  - **Semantic / long-term** — `SemanticStore`, durable facts persisted to
    `.lingus/semantic.json` and **loaded across streams**; hand-rolled
    token-overlap retrieval; runtime extraction uses `HeuristicFactExtractor`
    (regex) so hosted LLM calls stay limited to summarization and replies.
  - Episodic + semantic both fed from one eviction stream in `_consolidate_loop`.
  - Memory facts injected into generator context ("Known facts:" / "Stream so far:").
- **Phase 5 — lean safety/hardening gates (DONE).** Pre-ASR speech/music gate
  (`src/lingus/models/audio_gate.py`) keeps music/lyrics out of transcript
  context. `src/lingus/safety.py`: `RegexModeration`
  (implements the `ModerationBackend` ABC) — a deterministic denylist for hate
  slurs (leetspeak-tolerant), threats of violence (human-target-shaped, so
  "kill this boss" passes), CSAE, doxxing, and (toggleable) spam shapes. Jokes,
  satire, swearing and slang are deliberately allowed (CLAUDE.md §9). Wired as
  the authoritative last gate in `_post_message` (covers generated replies AND
  verbatim trend mirrors); the generated path also pre-checks and **regenerates
  once** before dropping. `build_moderation()` from `moderation:` config;
  `backend: none` disables it (logs a warning — offline tuning only). The output
  governor enforces rate and length caps; the optional typing delay is disabled
  by default so live replies are not held. Config: `check_spam`, `extra_patterns`.

- **Cold-start channel research (DONE).** `src/lingus/research/`: before the loop
  starts, profile the streamer and seed the durable (semantic) memory so the bot
  walks in already knowing the channel. `resolve_identity` reuses yt-dlp
  (`--dump-single-json`) for the YouTube channel name/description/tags (config
  `research.channel` for other platforms); `ChannelResearcher` gathers live web
  snippets (`WebSearchBackend` seam → keyless `DuckDuckGoSearch`, `[research]`
  extra, degrades to `NullSearch`) and distills identity + snippets into durable
  facts with the deterministic metadata/snippet fallback in app runtime.
  Per-channel JSON cache (`.lingus/research/`, `refresh_days` TTL) → researched
  once, not every boot. `_seed_research` in `app.py` runs pre-loop, adds facts to
  `SemanticStore` with `source="research"`, and persists — `BotLoop.run()` then
  loads them like any other durable fact and surfaces them into generator context.
  Entirely best-effort (never blocks the run). CLI: `--research` (force refresh),
  `--no-research` (skip). Config: `research:` block. 26 tests in `tests/research/`.
- **Phase 4 — video (STARTED).** YouTube frame capture yields raw RGB frames from
  `video_frames()`, and `FrameGate` deterministically filters frames by RGB diff
  + minimum interval before local analysis runs. `MLXVLMSceneAnalyzer` targets
  local MLX-VLM with `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` for live
  latency on this M4 Max / 64 GB machine, with `LocalFrameAnalyzer` as a
  no-network fallback. The 7B 4-bit model fits but was too slow in the first
  live stream test, so keep it as an explicit offline/high-detail override.
  Unchanged frames do not create scene events.
- **Phase 6 — eval/replay harness (DONE).** `src/lingus/eval.py`: replay a
  recorded segment through the *real* `BotLoop` (via a `CollectingMonitor` that
  captures every posted line + its triggering context), then score each line on
  three axes — `in_character`, `not_generic`, `not_repetitive` — behind one
  `Judge` seam. App runtime uses `HeuristicJudge` so eval judging is local
  (assistant-voice tells, filler fraction, context grounding, self-similarity)
  and needs no key.
  `evaluate_segment()` disables semantic memory for reproducibility. CLI:
  `lingus --eval --segment <dir> [--eval-json out.json]`. Tune arbiter weights +
  the exemplar bank from the report.

### Remaining
- **Phase 1 finish:** validate the **real LLM generator** live (needs an API key
  in `.env`); the template fallback is a placeholder, not the personality.
- **Phase 2 follow-up:** validate keyless YouTube **live-chat ingestion** against
  more live streams as YouTube's InnerTube payloads drift.
- **Phase 4 — video:** local MLX-VLM scene-state backend is wired. Still open:
  install/download validation (`pip install -e ".[video-mlx]"`, model cache) and
  richer OpenCV/PySceneDetect gating.
- **Phase 6 — eval/replay harness (DONE, see above).** Still open: a curated
  library of recorded segments to regression-test personas against, and wiring the
  report back into an automated weight-tuning sweep.
- **Final — Twitch adapter.** Not started.

### Notable environment notes
- Use `./.venv312/bin/python` for the local model stack; the old `venv` is Python
  3.14 and cannot satisfy the project pin (`>=3.12,<3.14`).
- `mlx-vlm` is installed in `.venv312`; in the Codex sandbox it reports no visible
  Metal device and falls back to `local_cv`. A normal user-run terminal may expose
  Metal and load the configured Qwen2.5-VL model.
- `turbo` ASR and the MLX-VLM model may need first-run downloads if not cached.
- `.lingus/semantic.json` is the cross-stream memory file (gitignored).
- `.lingus/episodes.json` is the durable per-stream summary archive (gitignored).
