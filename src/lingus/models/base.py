"""Backend-agnostic model contracts.

Small perception models (ASR, VLM) run locally; the generator LLM is hosted.
Both sit behind these interfaces so backends are swappable by config and faked
in tests.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ..adapters.base import AudioChunk, Frame
from ..world_state import SceneState


@dataclass(slots=True)
class Transcript:
    text: str
    ts: float
    is_final: bool = True
    confidence: float | None = None


@dataclass(slots=True)
class ChatTurn:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(slots=True)
class ModerationResult:
    allowed: bool
    reason: str = ""
    categories: list[str] = field(default_factory=list)


class ASRBackend(abc.ABC):
    @abc.abstractmethod
    def transcribe_stream(
        self, chunks: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[Transcript]:
        """Consume audio chunks, yield transcripts as speech is recognized."""


class VLMBackend(abc.ABC):
    @abc.abstractmethod
    async def describe_change(self, frame: Frame, prev: SceneState) -> SceneState:
        """Report what CHANGED relative to the prior scene state."""


class LLMBackend(abc.ABC):
    @abc.abstractmethod
    async def generate(self, messages: list[ChatTurn], **opts) -> str:
        """Generate a single completion from a chat-style message list."""


class ModerationBackend(abc.ABC):
    @abc.abstractmethod
    async def check(self, text: str) -> ModerationResult:
        """Decide whether a generated message is safe to post."""
