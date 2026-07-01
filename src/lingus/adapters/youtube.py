"""Live YouTube capture + an observe-mode chat sink.

`YouTubeCaptureAdapter` pulls a live stream's audio with yt-dlp (URL resolution)
piped through ffmpeg into 16 kHz mono PCM — exactly what the ASR backend wants —
and yields it as `AudioChunk`s. Video frames arrive in Phase 4; for now the
speech spine is the point.

`ObserveChatAdapter` is the read-only chat side: `incoming()` yields the
stream's real live chat (keyless InnerTube reader, see youtube_chat.py) so the
arbiter and trend detector see the actual firehose, while `post()` just logs
what the bot *would* say — we watch the bot react to a real stream without
writing into a third-party chat (which would need OAuth and is poor etiquette
besides).
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator

from ..logging import get_logger
from .base import AudioChunk, ChatAdapter, ChatMessage, Frame, StreamCaptureAdapter

log = get_logger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
_BYTES_PER_SAMPLE = 2
# 0.5 s chunks: small enough to keep capture latency low, big enough that the
# ASR buffer fills in a couple of reads.
_CHUNK_SECONDS = 0.5
_CHUNK_BYTES = int(SAMPLE_RATE * _CHUNK_SECONDS) * _BYTES_PER_SAMPLE


def _watch_url(video: str) -> str:
    """Accept a bare video id or any youtube URL; normalize to a watch URL."""
    if video.startswith("http://") or video.startswith("https://"):
        return video
    return f"https://www.youtube.com/watch?v={video}"


class YouTubeCaptureAdapter(StreamCaptureAdapter):
    def __init__(self, video: str) -> None:
        if not video:
            raise ValueError("YouTubeCaptureAdapter needs a video id or URL")
        self._url = _watch_url(video)
        self._ff: asyncio.subprocess.Process | None = None

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
        log.info("youtube capture started: %s", self._url)

    async def _resolve_audio_url(self) -> str:
        # Invoke as a module via the running interpreter so it resolves whether or
        # not the venv's bin/ is on PATH.
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "yt_dlp",
            "-g", "-f", "bestaudio/best", "--no-warnings", "-q", self._url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"yt-dlp failed to resolve {self._url}: {err.decode(errors='replace').strip()}"
            )
        urls = [line for line in out.decode().splitlines() if line.strip()]
        if not urls:
            raise RuntimeError(f"yt-dlp returned no media URL for {self._url}")
        # bestaudio yields a single audio URL; take the last line defensively.
        return urls[-1]

    async def stop(self) -> None:
        if self._ff is not None and self._ff.returncode is None:
            self._ff.terminate()
            try:
                await asyncio.wait_for(self._ff.wait(), timeout=5.0)
            except TimeoutError:
                self._ff.kill()
        self._ff = None

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
        await self._log_ffmpeg_error()

    async def _log_ffmpeg_error(self) -> None:
        if self._ff is not None and self._ff.stderr is not None:
            err = (await self._ff.stderr.read()).decode(errors="replace").strip()
            if err:
                log.warning("ffmpeg: %s", err)

    async def video_frames(self) -> AsyncIterator[Frame]:
        return
        yield  # pragma: no cover  (makes this an async generator)


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
