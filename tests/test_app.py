import asyncio

import pytest

from lingus.adapters.base import AudioChunk, ChatAdapter, ChatMessage, Frame, StreamCaptureAdapter
from lingus.app import BotLoop
from lingus.config import Settings
from lingus.persona.schema import PersonaSpec


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
