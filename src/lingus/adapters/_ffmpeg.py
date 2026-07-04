"""Shared ffmpeg-backed capture base for URL-resolvable live platforms.

YouTube and Twitch differ only in *how they resolve a playable media URL*
(yt-dlp vs streamlink); once a direct HLS/DASH URL exists, the audio→PCM and
video→RGB pipe is identical. That pipe — subprocess lifecycle, stderr draining,
16 kHz mono PCM chunking, lazy RGB frame capture — lives here once. A subclass
supplies `_resolve_audio_url()` / `_resolve_video_url()` and a human label; it
inherits everything else.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from ..logging import get_logger
from .base import AudioChunk, Frame, StreamCaptureAdapter

log = get_logger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
_BYTES_PER_SAMPLE = 2
# 0.5 s chunks: small enough to keep capture latency low, big enough that the
# ASR buffer fills in a couple of reads.
_CHUNK_SECONDS = 0.5
_CHUNK_BYTES = int(SAMPLE_RATE * _CHUNK_SECONDS) * _BYTES_PER_SAMPLE


class FFmpegCaptureAdapter(StreamCaptureAdapter):
    """Pulls audio/video from a resolvable media URL via ffmpeg.

    Subclasses implement URL resolution; this class owns the two ffmpeg
    processes and their lifecycles. The audio process starts eagerly in
    `start()`; the video process starts lazily on the first `video_frames()`
    read so speech-only runs never pay for it.
    """

    def __init__(
        self,
        *,
        label: str,
        desc: str,
        frame_width: int = 512,
        frame_height: int = 288,
        frame_fps: float = 1.0,
    ) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        if frame_fps <= 0:
            raise ValueError("frame_fps must be positive")
        self._label = label
        self._desc = desc
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._frame_fps = frame_fps
        self._ff: asyncio.subprocess.Process | None = None
        self._ff_stderr_task: asyncio.Task[None] | None = None
        self._vf: asyncio.subprocess.Process | None = None
        self._vf_stderr_task: asyncio.Task[None] | None = None
        # Once we start tearing down, ffmpeg dumps benign shutdown lines
        # ("Immediate exit requested") to stderr. Keep draining them so the
        # pipe never blocks ffmpeg's exit, but stop treating them as warnings.
        self._stopping = False

    # --- URL resolution: supplied by the platform subclass -------------------
    async def _resolve_audio_url(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    async def _resolve_video_url(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    # --- capture lifecycle ---------------------------------------------------
    async def start(self) -> None:
        # Resolve the direct audio URL first, then let ffmpeg read it live. One
        # process to manage, and ffmpeg handles HLS reconnection.
        audio_url = await self._resolve_audio_url()
        self._ff = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-nostdin",
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            "-i", audio_url,
            "-vn",
            "-f", "s16le",
            "-ac", str(CHANNELS),
            "-ar", str(SAMPLE_RATE),
            "-loglevel", "error",
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if self._ff.stderr is not None:
            self._ff_stderr_task = asyncio.create_task(
                self._drain_ffmpeg_stderr(self._ff.stderr), name="ffmpeg_stderr"
            )
        log.info("%s capture started: %s", self._label, self._desc)

    async def stop(self) -> None:
        self._stopping = True
        await self._stop_process(self._ff)
        if self._ff_stderr_task is not None:
            if not self._ff_stderr_task.done():
                self._ff_stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ff_stderr_task
            self._ff_stderr_task = None
        self._ff = None
        await self._stop_process(self._vf)
        if self._vf_stderr_task is not None:
            if not self._vf_stderr_task.done():
                self._vf_stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._vf_stderr_task
            self._vf_stderr_task = None
        self._vf = None

    async def _stop_process(self, proc: asyncio.subprocess.Process | None) -> None:
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()

    async def audio_frames(self) -> AsyncIterator[AudioChunk]:
        if self._ff is None or self._ff.stdout is None:
            raise RuntimeError("capture not started")
        ts = 0.0
        stdout = self._ff.stdout
        while True:
            try:
                pcm = await stdout.readexactly(_CHUNK_BYTES)
            except asyncio.IncompleteReadError as exc:
                if exc.partial:
                    yield AudioChunk(pcm=exc.partial, sample_rate=SAMPLE_RATE, ts=ts)
                break
            yield AudioChunk(pcm=pcm, sample_rate=SAMPLE_RATE, ts=ts)
            ts += _CHUNK_SECONDS

    async def _drain_ffmpeg_stderr(self, stderr: asyncio.StreamReader) -> None:
        try:
            while True:
                chunk = await stderr.read(4096)
                if not chunk:
                    return
                text = chunk.decode(errors="replace").strip()
                if text:
                    # During teardown ffmpeg's stderr is expected exit noise.
                    level = logging.DEBUG if self._stopping else logging.WARNING
                    log.log(level, "ffmpeg: %s", text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("ffmpeg stderr drain failed: %s", exc)

    async def video_frames(self) -> AsyncIterator[Frame]:
        if self._vf is None:
            await self._start_video_capture()
        if self._vf is None or self._vf.stdout is None:
            raise RuntimeError("video capture not started")
        frame_bytes = self._frame_width * self._frame_height * 3
        ts = 0.0
        stdout = self._vf.stdout
        while True:
            try:
                data = await stdout.readexactly(frame_bytes)
            except asyncio.IncompleteReadError:
                break
            yield Frame(data=data, width=self._frame_width, height=self._frame_height, ts=ts)
            ts += 1.0 / self._frame_fps

    async def _start_video_capture(self) -> None:
        video_url = await self._resolve_video_url()
        vf = (
            f"fps={self._frame_fps},"
            f"scale={self._frame_width}:{self._frame_height}:force_original_aspect_ratio=decrease,"
            f"pad={self._frame_width}:{self._frame_height}:(ow-iw)/2:(oh-ih)/2"
        )
        self._vf = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-nostdin",
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            "-i", video_url,
            "-an",
            "-vf", vf,
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "-loglevel", "error",
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if self._vf.stderr is not None:
            self._vf_stderr_task = asyncio.create_task(
                self._drain_ffmpeg_stderr(self._vf.stderr), name="ffmpeg_video_stderr"
            )
        log.info(
            "%s video frame capture started: %sx%s @ %.2ffps",
            self._label,
            self._frame_width,
            self._frame_height,
            self._frame_fps,
        )
