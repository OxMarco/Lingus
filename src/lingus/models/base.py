"""Backend-agnostic model contracts.

Small perception models (ASR, VLM) run locally; the generator LLM is hosted.
Both sit behind these interfaces so backends are swappable by config and faked
in tests.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from ..adapters.base import AudioChunk, Frame
from ..world_state import SceneState


@dataclass(slots=True)
class Transcript:
    text: str
    ts: float
    is_final: bool = True
    confidence: float | None = None


@dataclass(slots=True)
class AudioGateDecision:
    """Speech/music gate decision for an audio window before ASR."""

    allow_asr: bool
    mode: Literal["speech", "music", "mixed", "silence", "unknown"]
    speech_score: float = 0.0
    music_score: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class ChatTurn:
    role: str  # "system" | "user" | "assistant"
    content: str


class ASRBackend(abc.ABC):
    @abc.abstractmethod
    def transcribe_stream(
        self, chunks: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[Transcript]:
        """Consume audio chunks, yield transcripts as speech is recognized."""


class AudioGateBackend(abc.ABC):
    @abc.abstractmethod
    def gate_stream(self, chunks: AsyncIterator[AudioChunk]) -> AsyncIterator[AudioChunk]:
        """Yield only audio that should reach ASR, replacing dropped spans with silence."""


class VLMBackend(abc.ABC):
    @abc.abstractmethod
    async def describe_change(self, frame: Frame, prev: SceneState) -> SceneState:
        """Report what CHANGED relative to the prior scene state.

        The historical name is VLMBackend, but implementations may be local
        deterministic analyzers; live video should not require a hosted vision API.
        """


class LLMBackend(abc.ABC):
    @abc.abstractmethod
    async def generate(self, messages: list[ChatTurn], **opts) -> str:
        """Generate a single completion from a chat-style message list."""
