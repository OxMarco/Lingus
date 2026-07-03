import asyncio
import time

import pytest

from lingus.adapters.base import AudioChunk, ChatAdapter, ChatMessage, Frame, StreamCaptureAdapter
from lingus.app import BotLoop, _build_fact_extractor, _build_vlm
from lingus.arbiter import ArbiterDecision
from lingus.config import Settings
from lingus.context import build_context_snapshot
from lingus.models.base import Transcript
from lingus.persona.schema import PersonaSpec
from lingus.video import FrameGate
from lingus.world_state import Event, SceneState


class EmptyCaptureAdapter(StreamCaptureAdapter):
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def audio_frames(self):
        if False:
            yield AudioChunk(pcm=b"", sample_rate=16_000, ts=0.0)

    async def video_frames(self):
        if False:
            yield Frame(data=b"", width=1, height=1, ts=0.0)


class VideoCaptureAdapter(EmptyCaptureAdapter):
    def __init__(self, frames: list[Frame]) -> None:
        self.frames = frames

    async def video_frames(self):
        for frame in self.frames:
            yield frame


class AudioCaptureAdapter(EmptyCaptureAdapter):
    def __init__(self, chunks: list[AudioChunk]) -> None:
        self.chunks = chunks

    async def audio_frames(self):
        for chunk in self.chunks:
            yield chunk


class BrokenChatAdapter(ChatAdapter):
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def incoming(self):
        raise RuntimeError("chat stream failed")
        yield ChatMessage(author="viewer", text="hello", ts=0.0)

    async def post(self, text: str) -> None:
        pass


class CollectingChatAdapter(ChatAdapter):
    def __init__(self) -> None:
        self.posts: list[str] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def incoming(self):
        if False:
            yield ChatMessage(author="viewer", text="hello", ts=0.0)

    async def post(self, text: str) -> None:
        self.posts.append(text)


class FailingSummarizer:
    async def summarize(self, prior: str, new_lines: list[str]) -> str:
        raise RuntimeError("summary model fell over")


class FailingFactExtractor:
    async def extract(self, lines: list[str]):
        raise RuntimeError("fact model fell over")


class FakeVLM:
    def __init__(self) -> None:
        self.frames: list[Frame] = []

    async def describe_change(self, frame: Frame, prev: SceneState) -> SceneState:
        self.frames.append(frame)
        return SceneState(
            activity=f"frame {len(self.frames)}",
            last_event=f"video changed at {frame.ts:.1f}s",
        )


class DuplicateEventVLM:
    def __init__(self) -> None:
        self.frames: list[Frame] = []

    async def describe_change(self, frame: Frame, prev: SceneState) -> SceneState:
        self.frames.append(frame)
        return SceneState(
            activity=f"state refresh {len(self.frames)}",
            salient_objects=[f"object {len(self.frames)}"],
            last_event="same visible event",
        )


class RecordingASR:
    def __init__(self) -> None:
        self.chunks: list[AudioChunk] = []

    async def transcribe_stream(self, chunks):
        self.chunks = [chunk async for chunk in chunks]
        if self.chunks:
            yield Transcript(text="hello from speech", ts=self.chunks[0].ts)


class DropAllAudioGate:
    async def gate_stream(self, chunks):
        async for _chunk in chunks:
            continue
        if False:
            yield AudioChunk(pcm=b"", sample_rate=16_000, ts=0.0)


class StaticReplyGenerator:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def generate(self, snapshot, decision, persona, *, max_chars: int) -> str:
        self.calls += 1
        return self.text


@pytest.mark.asyncio
async def test_bot_loop_propagates_background_task_failures():
    settings = Settings.model_validate({"platform": "file_replay"})
    persona = PersonaSpec(name="test", voice="brief")
    loop = BotLoop(
        settings=settings,
        persona=persona,
        capture=EmptyCaptureAdapter(),
        chat=BrokenChatAdapter(),
        segment=None,
    )

    with pytest.raises(RuntimeError, match="chat stream failed"):
        await asyncio.wait_for(loop.run(), timeout=1.0)


@pytest.mark.asyncio
async def test_bot_loop_replies_from_replayed_context():
    settings = Settings.model_validate(
        {
            "platform": "file_replay",
            "arbiter": {"weights": {"streamer_mishap": 1.1}},
        }
    )
    persona = PersonaSpec(name="Gremlin", voice="brief")
    chat = CollectingChatAdapter()
    loop = BotLoop(
        settings=settings,
        persona=persona,
        capture=EmptyCaptureAdapter(),
        chat=chat,
        segment="tests/samples/cake",
        replay_speed=100.0,
    )

    await loop._ingest_scene()
    await loop._ingest_transcript()
    await loop._cognition_tick()

    assert chat.posts == ["be careful next time, chocolate stains are hard to remove"]
    assert list(loop.world.own_messages) == chat.posts


@pytest.mark.asyncio
async def test_file_replay_stop_waits_for_replay_tasks_to_finish():
    settings = Settings.model_validate({"platform": "file_replay"})
    loop = BotLoop(
        settings=settings,
        persona=PersonaSpec(name="test", voice="brief"),
        capture=EmptyCaptureAdapter(),
        chat=CollectingChatAdapter(),
        segment=None,
    )
    pending = asyncio.create_task(asyncio.sleep(3600))
    loop._replay_tasks = [pending]
    loop.world.add_event(
        Event(
            source="speech",
            kind="transcript",
            payload={"text": "first sparse event"},
            ts=time.monotonic() - 10.0,
        )
    )

    try:
        assert not loop._replay_should_stop()
    finally:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending


@pytest.mark.asyncio
async def test_file_replay_stop_allows_empty_finished_replay():
    settings = Settings.model_validate({"platform": "file_replay"})
    loop = BotLoop(
        settings=settings,
        persona=PersonaSpec(name="test", voice="brief"),
        capture=EmptyCaptureAdapter(),
        chat=CollectingChatAdapter(),
        segment=None,
    )
    done = asyncio.create_task(asyncio.sleep(0))
    await done
    loop._replay_tasks = [done]

    assert loop._replay_should_stop()


@pytest.mark.asyncio
async def test_memory_consolidation_model_failures_do_not_stop_loop():
    settings = Settings.model_validate({"platform": "file_replay"})
    loop = BotLoop(
        settings=settings,
        persona=PersonaSpec(name="test", voice="brief"),
        capture=EmptyCaptureAdapter(),
        chat=CollectingChatAdapter(),
        segment=None,
        summarizer=FailingSummarizer(),
        fact_extractor=FailingFactExtractor(),
    )
    loop.world.add_event(
        Event(source="speech", kind="transcript", payload={"text": "My name is Marco"})
    )

    await loop._consolidate(force=True)

    assert loop.world.episodic_summary == ""
    assert loop.semantic is not None and len(loop.semantic) == 0


@pytest.mark.asyncio
async def test_memory_consolidation_persists_episodic_summary(tmp_path):
    settings = Settings.model_validate(
        {
            "platform": "file_replay",
            "memory": {
                "episodic_path": str(tmp_path / "episodes.json"),
                "semantic_enabled": False,
            },
        }
    )
    loop = BotLoop(
        settings=settings,
        persona=PersonaSpec(name="test", voice="brief"),
        capture=EmptyCaptureAdapter(),
        chat=CollectingChatAdapter(),
        segment="tests/samples/cake",
    )
    loop.world.add_event(
        Event(source="speech", kind="transcript", payload={"text": "Tuchel named the squad today"})
    )

    await loop._consolidate(force=True)

    assert "Tuchel named the squad today" in loop.world.episodic_summary
    assert loop.episodes is not None
    assert loop.episodes.summaries() == [loop.world.episodic_summary]
    assert (tmp_path / "episodes.json").exists()


def test_episodic_history_refresh_uses_prior_streams_not_current():
    settings = Settings.model_validate(
        {"platform": "file_replay", "memory": {"episodic_top_k": 2}}
    )
    loop = BotLoop(
        settings=settings,
        persona=PersonaSpec(name="test", voice="brief"),
        capture=EmptyCaptureAdapter(),
        chat=CollectingChatAdapter(),
        segment="current",
    )
    assert loop.episodes is not None
    loop.episodes.add(
        "prior stream ended with a chocolate cake stain", stream_id="file:old", now=1.0
    )
    loop.episodes.add("current stream summary", stream_id="file:current", now=2.0)
    loop.world.add_event(Event(source="speech", kind="transcript", payload={"text": "cake stain"}))

    loop._refresh_episodic_history()

    assert loop.world.episodic_history == ["prior stream ended with a chocolate cake stain"]


@pytest.mark.asyncio
async def test_video_vlm_ingest_gates_frames_into_scene_events():
    frames = [
        Frame(data=b"\x00" * 12, width=2, height=2, ts=0.0),
        Frame(data=b"\x00" * 12, width=2, height=2, ts=1.0),
        Frame(data=b"\xff" * 12, width=2, height=2, ts=2.0),
    ]
    vlm = FakeVLM()
    loop = BotLoop(
        settings=Settings.model_validate({"platform": "youtube"}),
        persona=PersonaSpec(name="test", voice="brief"),
        capture=VideoCaptureAdapter(frames),
        chat=CollectingChatAdapter(),
        segment=None,
        vlm=vlm,
        frame_gate=FrameGate(diff_threshold=0.5, min_interval_seconds=0.0),
    )

    await loop._ingest_video_vlm()

    assert [frame.ts for frame in vlm.frames] == [0.0, 2.0]
    assert loop.world.scene.activity == "frame 2"
    assert [e.source for e in loop.world.events] == ["scene", "scene"]


@pytest.mark.asyncio
async def test_video_vlm_refreshes_scene_without_duplicate_event_text():
    frames = [
        Frame(data=b"\x00" * 12, width=2, height=2, ts=0.0),
        Frame(data=b"\xff" * 12, width=2, height=2, ts=2.0),
    ]
    loop = BotLoop(
        settings=Settings.model_validate({"platform": "youtube"}),
        persona=PersonaSpec(name="test", voice="brief"),
        capture=VideoCaptureAdapter(frames),
        chat=CollectingChatAdapter(),
        segment=None,
        vlm=DuplicateEventVLM(),
        frame_gate=FrameGate(diff_threshold=0.5, min_interval_seconds=0.0),
    )

    await loop._ingest_video_vlm()

    assert loop.world.scene.activity == "state refresh 2"
    assert [e.payload["last_event"] for e in loop.world.events] == ["same visible event"]


@pytest.mark.asyncio
async def test_audio_gate_blocks_chunks_before_asr_context():
    asr = RecordingASR()
    loop = BotLoop(
        settings=Settings.model_validate({"platform": "youtube"}),
        persona=PersonaSpec(name="test", voice="brief"),
        capture=AudioCaptureAdapter(
            [AudioChunk(pcm=b"\1" * 3200, sample_rate=16_000, ts=3.0)]
        ),
        chat=CollectingChatAdapter(),
        segment=None,
        asr=asr,
        audio_gate=DropAllAudioGate(),
    )

    await loop._ingest_audio_asr()

    assert asr.chunks == []
    assert list(loop.world.events) == []


@pytest.mark.asyncio
async def test_dropped_repetitive_reply_sets_short_backoff():
    generator = StaticReplyGenerator("same joke")
    loop = BotLoop(
        settings=Settings.model_validate(
            {
                "platform": "youtube",
                "output": {"min_seconds_between_posts": 8.0},
                "memory": {"similarity_threshold": 0.6},
            }
        ),
        persona=PersonaSpec(name="test", voice="brief"),
        capture=EmptyCaptureAdapter(),
        chat=CollectingChatAdapter(),
        segment=None,
        reply_generator=generator,
    )
    event = Event(source="speech", kind="transcript", payload={"text": "what just happened?"})
    loop.world.add_event(event)
    loop.world.own_messages.append("same joke")
    decision = ArbiterDecision(
        should_reply=True,
        score=2.0,
        reasons=["unanswered_question"],
        trigger_event=event,
        threshold=1.0,
    )
    snapshot = build_context_snapshot(loop.world)

    posted, dropped = await loop._maybe_reply(snapshot, decision)
    posted_again, dropped_again = await loop._maybe_reply(snapshot, decision)

    assert posted is None
    assert dropped == "same joke"
    assert posted_again is None
    assert dropped_again is None
    assert generator.calls == 1


def test_build_vlm_disabled_by_default():
    assert _build_vlm(Settings.model_validate({})) is None


def test_build_vlm_uses_local_analyzer_for_live_video():
    from lingus.models.local_vision import LocalFrameAnalyzer, MLXVLMSceneAnalyzer

    settings = Settings.model_validate({"platform": "youtube"})
    vlm = _build_vlm(settings)

    assert isinstance(vlm, MLXVLMSceneAnalyzer)
    assert isinstance(vlm.fallback, LocalFrameAnalyzer)
    assert vlm.model_name == "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"


def test_build_vlm_rejects_remote_video_backend():
    settings = Settings.model_validate(
        {"platform": "youtube", "models": {"vlm": {"backend": "openai_compat"}}}
    )

    with pytest.raises(SystemExit, match="openai_compat"):
        _build_vlm(settings)


def test_fact_extractor_stays_local_even_when_llm_backend_exists():
    from lingus.memory import HeuristicFactExtractor

    settings = Settings.model_validate({"memory": {"semantic_enabled": True}})

    assert isinstance(_build_fact_extractor(settings, object()), HeuristicFactExtractor)


@pytest.mark.asyncio
async def test_post_message_drops_unsafe_output():
    from lingus.safety import RegexModeration

    settings = Settings.model_validate({"platform": "file_replay"})
    persona = PersonaSpec(name="test", voice="brief")
    chat = CollectingChatAdapter()
    loop = BotLoop(
        settings=settings,
        persona=persona,
        capture=EmptyCaptureAdapter(),
        chat=chat,
        segment=None,
        safety=RegexModeration(),
    )

    posted, dropped = await loop._post_message("kys loser", drop_context="reply")
    assert posted is None
    assert dropped is not None and "unsafe" in dropped
    assert chat.posts == []
    assert list(loop.world.own_messages) == []


@pytest.mark.asyncio
async def test_post_message_admits_safe_output():
    from lingus.safety import RegexModeration

    settings = Settings.model_validate({"platform": "file_replay"})
    persona = PersonaSpec(name="test", voice="brief")
    chat = CollectingChatAdapter()
    loop = BotLoop(
        settings=settings,
        persona=persona,
        capture=EmptyCaptureAdapter(),
        chat=chat,
        segment=None,
        safety=RegexModeration(),
    )

    posted, dropped = await loop._post_message(
        "that was clean, honestly unreal", drop_context="reply"
    )
    assert posted == "that was clean, honestly unreal"
    assert dropped is None
    assert chat.posts == [posted]
