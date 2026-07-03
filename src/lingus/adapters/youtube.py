"""Live YouTube capture + an observe-mode chat sink.

`YouTubeCaptureAdapter` pulls a live stream's audio with yt-dlp (URL resolution)
piped through ffmpeg into 16 kHz mono PCM — exactly what the ASR backend wants —
and yields it as `AudioChunk`s. Its video side lazily starts a second ffmpeg
process and yields sampled raw RGB frames for the Phase 4 local scene analyzer.

`ObserveChatAdapter` is the read-only chat side: `incoming()` yields the
stream's real live chat (keyless InnerTube reader, see youtube_chat.py) so the
arbiter and trend detector see the actual firehose, while `post()` just logs
what the bot *would* say — we watch the bot react to a real stream without
writing into a third-party chat (which would need OAuth and is poor etiquette
besides).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from ..logging import get_logger
from ..yt_dlp_api import YtDlpError, resolve_media_url, watch_url
from .base import AudioChunk, ChatAdapter, ChatMessage, Frame, StreamCaptureAdapter

log = get_logger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
_BYTES_PER_SAMPLE = 2
# 0.5 s chunks: small enough to keep capture latency low, big enough that the
# ASR buffer fills in a couple of reads.
_CHUNK_SECONDS = 0.5
_CHUNK_BYTES = int(SAMPLE_RATE * _CHUNK_SECONDS) * _BYTES_PER_SAMPLE


class YouTubeCaptureAdapter(StreamCaptureAdapter):
    def __init__(
        self,
        video: str,
        *,
        resolve_timeout: float = 30.0,
        frame_width: int = 512,
        frame_height: int = 288,
        frame_fps: float = 1.0,
    ) -> None:
        if not video:
            raise ValueError("YouTubeCaptureAdapter needs a video id or URL")
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        if frame_fps <= 0:
            raise ValueError("frame_fps must be positive")
        self._url = watch_url(video)
        self._resolve_timeout = resolve_timeout
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._frame_fps = frame_fps
        self._ff: asyncio.subprocess.Process | None = None
        self._ff_stderr_task: asyncio.Task[None] | None = None
        self._vf: asyncio.subprocess.Process | None = None
        self._vf_stderr_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        # Resolve the direct audio URL first (yt-dlp), then let ffmpeg read it
        # live. One process to manage, and ffmpeg handles HLS reconnection.
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
        log.info("youtube capture started: %s", self._url)

    async def _resolve_audio_url(self) -> str:
        return await self._resolve_media_url("bestaudio/best", "audio")

    async def _resolve_video_url(self) -> str:
        # Prefer a moderate-height video-only rendition so frame decoding stays
        # cheap; fall back to muxed/best formats for streams without DASH splits.
        return await self._resolve_media_url(
            "bestvideo[height<=720]/best[height<=720]/best", "video"
        )

    async def _resolve_media_url(self, fmt: str, label: str) -> str:
        try:
            return await resolve_media_url(
                self._url,
                fmt,
                label=label,
                timeout=self._resolve_timeout,
            )
        except YtDlpError as exc:
            raise RuntimeError(str(exc)) from exc

    async def stop(self) -> None:
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
                    log.warning("ffmpeg: %s", text)
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
            "youtube video frame capture started: %sx%s @ %.2ffps",
            self._frame_width,
            self._frame_height,
            self._frame_fps,
        )


class ObserveChatAdapter(ChatAdapter):
    """Reads the real live chat; logs what the bot would post, never posts."""

    def __init__(self, video: str | None = None) -> None:
        # No video -> no ingestion (speech-only observe, or youtube.chat_enabled
        # off). The client is built lazily in start() so constructing the
        # adapter never needs aiohttp.
        self._video = video
        self._client = None

    async def start(self) -> None:
        if self._video:
            try:
                from .youtube_chat import YouTubeLiveChatClient
            except ImportError:
                log.warning(
                    "live-chat ingestion needs aiohttp (pip install -e '.[youtube]'); "
                    "running without chat"
                )
            else:
                self._client = YouTubeLiveChatClient(self._video)
        log.info(
            "observe-mode chat: reading live chat=%s; replies will be logged, not posted",
            "on" if self._client is not None else "off",
        )

    async def stop(self) -> None:
        pass

    async def incoming(self) -> AsyncIterator[ChatMessage]:
        if self._client is None:
            return
        try:
            async for msg in self._client.messages():
                yield msg
        except Exception:
            # Chat is a perception channel, not the spine: if ingestion dies
            # (chat disabled mid-stream, protocol drift, network gone), log and
            # carry on speech-only rather than letting the task failure stop
            # the whole loop.
            log.exception("live chat ingestion failed; continuing without chat")

    async def post(self, text: str) -> None:
        log.info("[BOT would post] %s", text)
