"""Offline replay adapters — drive the whole loop from a recorded segment.

Runnable with zero heavy deps (audio via stdlib `wave`), so the pipeline is
testable from day one and this doubles as the spine of the Phase 6 eval harness.

Segment directory layout (all files optional):
  <segment>/chat.jsonl       one JSON object per line: {"t": 0.0, "author": "...", "text": "..."}
  <segment>/audio.wav        mono PCM WAV of the stream audio
  <segment>/transcript.jsonl one per line: {"t": 0.0, "text": "..."}  (pre-ASR convenience)
  <segment>/scene.jsonl      one per line: {"t": 0.0, "activity": "..."} (pre-VLM convenience)
"""

from __future__ import annotations

import asyncio
import json
import wave
from collections.abc import AsyncIterator, Iterable, Iterator
from pathlib import Path
from typing import Any

from ..logging import get_logger
from .base import AudioChunk, ChatAdapter, ChatMessage, Frame, StreamCaptureAdapter

log = get_logger(__name__)

JsonRow = dict[str, Any]


def iter_jsonl(path: Path) -> Iterator[JsonRow]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def _scaled_delay(seconds: float, speed: float) -> float:
    return min(max(0.0, seconds) / max(speed, 0.001), 5.0)


async def paced_rows(rows: Iterable[JsonRow], speed: float) -> AsyncIterator[JsonRow]:
    """Yield rows honoring their relative 't' offsets, scaled by `speed`."""
    prev_t = 0.0
    for row in rows:
        row_t = float(row.get("t", prev_t))
        delay = _scaled_delay(row_t - prev_t, speed)
        if delay:
            await asyncio.sleep(delay)
        prev_t = row_t
        yield row


class FileReplayCaptureAdapter(StreamCaptureAdapter):
    def __init__(self, segment_path: str, speed: float = 10.0) -> None:
        self._dir = Path(segment_path)
        self._speed = speed

    async def start(self) -> None:
        log.info("file_replay capture from %s", self._dir)

    async def stop(self) -> None:
        pass

    async def audio_frames(self) -> AsyncIterator[AudioChunk]:
        wav_path = self._dir / "audio.wav"
        if not wav_path.exists():
            return
        with wave.open(str(wav_path), "rb") as wf:
            rate = wf.getframerate()
            frames_per_chunk = rate // 2  # ~0.5s chunks
            ts = 0.0
            while True:
                pcm = wf.readframes(frames_per_chunk)
                if not pcm:
                    break
                yield AudioChunk(pcm=pcm, sample_rate=rate, ts=ts)
                ts += frames_per_chunk / rate
                await asyncio.sleep(_scaled_delay(frames_per_chunk / rate, self._speed))

    async def video_frames(self) -> AsyncIterator[Frame]:
        # Video replay arrives in Phase 4.
        return
        yield  # pragma: no cover  (makes this an async generator)


class FileReplayChatAdapter(ChatAdapter):
    def __init__(self, segment_path: str, speed: float = 10.0) -> None:
        self._dir = Path(segment_path)
        self._speed = speed

    async def start(self) -> None:
        log.info("file_replay chat from %s", self._dir)

    async def stop(self) -> None:
        pass

    async def incoming(self) -> AsyncIterator[ChatMessage]:
        rows = iter_jsonl(self._dir / "chat.jsonl")
        async for row in paced_rows(rows, self._speed):
            yield ChatMessage(
                author=row.get("author", "anon"),
                text=row.get("text", ""),
                ts=row.get("t", 0.0),
                is_moderator=row.get("is_moderator", False),
                is_owner=row.get("is_owner", False),
                raw=row,
            )

    async def post(self, text: str) -> None:
        # Offline: posting is just logged so replays are side-effect free.
        log.info("[BOT would post] %s", text)


def read_transcript(segment_path: str) -> Iterator[JsonRow]:
    """Phase-0 convenience: pre-transcribed speech lines, replayed without ASR."""
    return iter_jsonl(Path(segment_path) / "transcript.jsonl")


def read_scene(segment_path: str) -> Iterator[JsonRow]:
    """Pre-captioned scene states, replayed without a VLM."""
    return iter_jsonl(Path(segment_path) / "scene.jsonl")
