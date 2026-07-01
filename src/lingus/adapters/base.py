"""Platform-agnostic adapter contracts.

The core loop reads/writes ONLY through these interfaces, so swapping YouTube for
Twitch (or a recorded segment for dev) never touches cognition. Capture and chat
are separate adapters because a stream's A/V feed and its chat feed are different
services with different lifecycles.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass(slots=True)
class AudioChunk:
    """A slice of mono PCM audio."""

    pcm: bytes  # 16-bit little-endian mono samples
    sample_rate: int
    ts: float  # monotonic capture time of the chunk start


@dataclass(slots=True)
class Frame:
    """A single decoded video frame (RGB)."""

    data: bytes  # raw RGB bytes
    width: int
    height: int
    ts: float


@dataclass(slots=True)
class ChatMessage:
    author: str
    text: str
    ts: float
    is_moderator: bool = False
    is_owner: bool = False
    raw: dict = field(default_factory=dict)


class StreamCaptureAdapter(abc.ABC):
    """Pulls the live A/V feed and yields audio chunks and/or video frames."""

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    def audio_frames(self) -> AsyncIterator[AudioChunk]:
        """Yield audio chunks until the stream ends or stop() is called."""

    @abc.abstractmethod
    def video_frames(self) -> AsyncIterator[Frame]:
        """Yield video frames. May yield nothing if video capture is disabled."""


class ChatAdapter(abc.ABC):
    """Reads incoming chat and posts the bot's replies."""

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    def incoming(self) -> AsyncIterator[ChatMessage]:
        """Yield chat messages as they arrive."""

    @abc.abstractmethod
    async def post(self, text: str) -> None:
        """Post a message to the stream chat."""
