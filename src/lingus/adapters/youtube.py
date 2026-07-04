"""Live YouTube capture + an observe-mode chat sink.

`YouTubeCaptureAdapter` pulls a live stream's audio with yt-dlp (URL resolution)
piped through ffmpeg into 16 kHz mono PCM — exactly what the ASR backend wants —
and yields it as `AudioChunk`s. Its video side lazily starts a second ffmpeg
process and yields sampled raw RGB frames for the Phase 4 local scene analyzer.
The ffmpeg pipe itself lives in `FFmpegCaptureAdapter`; this class only supplies
yt-dlp URL resolution.

`ObserveChatAdapter` is the read-only chat side: `incoming()` yields the
stream's real live chat (keyless InnerTube reader, see youtube_chat.py) so the
arbiter and trend detector see the actual firehose, while `post()` just logs
what the bot *would* say — we watch the bot react to a real stream without
writing into a third-party chat (which would need OAuth and is poor etiquette
besides).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ..logging import get_logger
from ..yt_dlp_api import YtDlpError, resolve_media_url, watch_url
from ._ffmpeg import SAMPLE_RATE, FFmpegCaptureAdapter  # noqa: F401 (SAMPLE_RATE re-exported)
from .base import ChatAdapter, ChatMessage

log = get_logger(__name__)


class YouTubeCaptureAdapter(FFmpegCaptureAdapter):
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
        self._url = watch_url(video)
        self._resolve_timeout = resolve_timeout
        super().__init__(
            label="youtube",
            desc=self._url,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_fps=frame_fps,
        )

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
        # No client means chat was never enabled (no video, chat_enabled off, or
        # aiohttp missing) — a clean, empty stream, not a failure. But once chat
        # IS live, an ingestion error (protocol drift, network gone) is an
        # unrecoverable capture failure: let it propagate so the loop crashes
        # rather than running on silently deaf to chat. `_ingest_chat` decides
        # whether that propagation is fatal (it is, unless we're shutting down).
        if self._client is None:
            return
        async for msg in self._client.messages():
            yield msg

    async def post(self, text: str) -> None:
        log.info("[BOT would post] %s", text)
