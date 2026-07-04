"""Live Twitch capture + a twitchio-backed chat adapter.

`TwitchCaptureAdapter` resolves a channel's live HLS URL with **streamlink**
(the maintained Twitch resolver) and then reuses the shared `FFmpegCaptureAdapter`
pipe — so audio→16 kHz PCM and video→RGB are byte-for-byte the same path as
YouTube; only URL resolution differs.

`TwitchChatAdapter` wraps **twitchio** (per user preference: use the maintained
library, not a hand-rolled IRC client). twitchio is callback-driven and owns its
own connection loop, so we bridge it to our `incoming()` async-generator through
an `asyncio.Queue`: the bot enqueues each chat message, `incoming()` drains the
queue. `post()` writes back via the channel object.

Posting is **opt-in** (`post_enabled`, default off): like YouTube observe mode,
the default is to react to a real chat while only logging what we *would* say, so
running against someone else's channel never writes without an explicit choice.
Reading and posting both need a chat OAuth token; with no token the adapter
degrades to a silent observer (no reads, posts logged) rather than failing.

The twitchio dependency is kept behind a lazy import (`.[twitch]` extra) and
isolated behind the `ChatAdapter` ABC, so the library — and its exact major
version — is a swappable detail the core loop never sees.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from typing import Any

from ..logging import get_logger
from ._ffmpeg import FFmpegCaptureAdapter
from .base import ChatAdapter, ChatMessage

log = get_logger(__name__)

# Sentinel pushed onto the queue to unblock incoming() at shutdown.
_SHUTDOWN = object()


def channel_url(channel: str) -> str:
    """Accept a bare channel name, an @handle, or a full twitch.tv URL."""
    channel = channel.strip()
    if channel.startswith(("http://", "https://")):
        return channel
    return f"https://www.twitch.tv/{channel.lstrip('@').lower()}"


def channel_name(channel: str) -> str:
    """Normalize any of the accepted channel forms to a bare lowercase login."""
    channel = channel.strip()
    if channel.startswith(("http://", "https://")):
        channel = channel.rstrip("/").rsplit("/", 1)[-1]
    return channel.lstrip("@").lower()


class TwitchStreamError(RuntimeError):
    """streamlink could not resolve a playable stream for the channel."""


class TwitchCaptureAdapter(FFmpegCaptureAdapter):
    def __init__(
        self,
        channel: str,
        *,
        quality: str = "best",
        frame_width: int = 512,
        frame_height: int = 288,
        frame_fps: float = 1.0,
    ) -> None:
        if not channel:
            raise ValueError("TwitchCaptureAdapter needs a channel name or URL")
        self._channel = channel_name(channel)
        self._url = channel_url(channel)
        self._quality = quality
        super().__init__(
            label="twitch",
            desc=self._url,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_fps=frame_fps,
        )

    async def _resolve_audio_url(self) -> str:
        # Twitch muxes audio+video in one HLS rendition; ffmpeg's `-vn` drops the
        # video, so the same resolved URL feeds both the audio and video pipes.
        return await self._resolve_stream_url()

    async def _resolve_video_url(self) -> str:
        return await self._resolve_stream_url()

    async def _resolve_stream_url(self) -> str:
        return await asyncio.to_thread(self._resolve_stream_url_blocking)

    def _resolve_stream_url_blocking(self) -> str:
        try:
            import streamlink
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise TwitchStreamError(
                "streamlink is not installed; install lingus[twitch] for Twitch capture"
            ) from exc
        try:
            streams = streamlink.streams(self._url)
        except Exception as exc:
            raise TwitchStreamError(
                f"streamlink failed to resolve {self._url!r}: {exc}"
            ) from exc
        if not streams:
            raise TwitchStreamError(
                f"no live stream for {self._url!r} (channel offline or unknown)"
            )
        stream = streams.get(self._quality) or streams.get("best") or next(iter(streams.values()))
        url = getattr(stream, "url", None)
        if not url:
            raise TwitchStreamError(
                f"streamlink returned no playable URL for {self._url!r}"
            )
        return str(url)


class TwitchChatAdapter(ChatAdapter):
    """Reads a Twitch channel's chat via twitchio; posts only when enabled."""

    def __init__(
        self,
        channel: str,
        token: str,
        *,
        post_enabled: bool = False,
        max_queue: int = 1024,
    ) -> None:
        if not channel:
            raise ValueError("TwitchChatAdapter needs a channel name or URL")
        self._channel = channel_name(channel)
        self._token = token.strip()
        self._post_enabled = post_enabled
        # Bounded: a chat firehose must not grow memory without limit if the
        # consumer stalls. Oldest messages drop (a perception channel, not the
        # spine — losing a few lines beats unbounded growth).
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max_queue)
        self._bot: Any = None
        self._bot_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if not self._token:
            log.warning(
                "twitch chat: no chat OAuth token (set TWITCH_OAUTH_TOKEN); "
                "running as a silent observer (no reads, replies logged not posted)"
            )
            return
        try:
            bot = _build_bot(self._token, self._channel, self._queue)
        except ImportError:
            log.warning(
                "twitch chat needs twitchio (pip install -e '.[twitch]'); "
                "running without chat"
            )
            return
        self._bot = bot
        self._bot_task = asyncio.create_task(self._run_bot(), name="twitchio_bot")
        log.info(
            "twitch chat: reading #%s; posting %s",
            self._channel,
            "ENABLED" if self._post_enabled else "disabled (replies logged)",
        )

    async def _run_bot(self) -> None:
        # twitchio owns the connection loop; if it dies, chat is a perception
        # channel (not the spine), so surface the failure by unblocking the
        # consumer rather than hanging incoming() forever.
        try:
            await self._bot.start()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - twitchio raises broad errors
            log.warning("twitch chat ingestion stopped: %s", exc)
        finally:
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(_SHUTDOWN)

    async def stop(self) -> None:
        if self._bot is not None:
            with contextlib.suppress(Exception):
                await self._bot.close()
        if self._bot_task is not None:
            self._bot_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._bot_task
            self._bot_task = None
        self._bot = None

    async def incoming(self) -> AsyncIterator[ChatMessage]:
        # No token / no twitchio -> no client -> a clean empty stream, not a
        # failure (mirrors the YouTube observe adapter).
        if self._token == "" or self._bot is None:
            return
        while True:
            item = await self._queue.get()
            if item is _SHUTDOWN:
                return
            yield item

    async def post(self, text: str) -> None:
        if not self._post_enabled or self._bot is None:
            log.info("[BOT would post] %s", text)
            return
        channel = self._bot.get_channel(self._channel)
        if channel is None:
            log.warning("twitch chat: channel #%s not joined yet; dropped reply", self._channel)
            return
        try:
            await channel.send(text)
        except Exception as exc:  # noqa: BLE001 - twitchio raises broad errors
            log.warning("twitch chat: failed to post reply: %s", exc)


def _build_bot(token: str, channel: str, queue: asyncio.Queue[Any]) -> Any:
    """Construct a twitchio Bot that enqueues incoming chat as `ChatMessage`s.

    Defined as a factory (not a module-level class) so twitchio stays a lazy,
    optional import — importing this module never requires the [twitch] extra.
    """
    from twitchio.ext import commands

    class _LingusTwitchBot(commands.Bot):
        def __init__(self) -> None:
            super().__init__(token=token, prefix="!", initial_channels=[channel])

        async def event_ready(self) -> None:
            log.info("twitch chat: connected as %s", self.nick)

        async def event_message(self, message: Any) -> None:
            # `echo` is True for the bot's own messages — self-memory already
            # tracks those; re-ingesting would double-count and risk a feedback
            # loop (see CLAUDE.md: the bot's posts re-enter world state once).
            if getattr(message, "echo", False):
                return
            author = getattr(message, "author", None)
            msg = ChatMessage(
                author=str(getattr(author, "name", None) or "anon"),
                text=str(getattr(message, "content", "") or ""),
                ts=time.monotonic(),
                is_moderator=bool(getattr(author, "is_mod", False)),
                is_owner=bool(getattr(author, "is_broadcaster", False)),
                raw={"channel": channel},
            )
            if not msg.text:
                return
            try:
                queue.put_nowait(msg)
            except asyncio.QueueFull:
                # Firehose outran the consumer: drop the oldest, keep the newest.
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(msg)

    return _LingusTwitchBot()
