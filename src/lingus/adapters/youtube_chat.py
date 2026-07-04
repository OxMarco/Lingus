"""Keyless YouTube live-chat ingestion over the InnerTube web API.

A live stream's chat is readable without OAuth or a Data API key from the same
endpoints the web player uses: fetch the popout chat page once to lift the
InnerTube api key, client context and a chat continuation token out of the
embedded config, then poll ``youtubei/v1/live_chat/get_live_chat`` following the
continuation chain like a browser tab would. Each poll returns a batch of chat
actions plus the next token and how long to wait before asking again.

This replaces pytchat (archived upstream) with a couple hundred lines we
control. Parsing is split into pure functions so the protocol can be regression
-tested from fixtures without a network. Posting is NOT here: writing to chat
needs OAuth and lands with the posting adapter; observe mode only reads.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import urllib.parse
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import aiohttp
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from ..logging import get_logger
from .base import ChatMessage

log = get_logger(__name__)

# Poll pacing: YouTube tells us how long to wait (timeoutMs); clamp it so a
# weird response can neither hot-loop the poller nor stall it for minutes.
_MIN_POLL_SECONDS = 0.5
_MAX_POLL_SECONDS = 30.0
_DEFAULT_POLL_SECONDS = 5.0
# Message-id dedup window: polls can overlap at the edges, so remember the last
# N ids. Bounded, or an all-day stream would grow the set forever.
_SEEN_IDS_WINDOW = 2048

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    # Pre-answered consent cookie: without it EU IPs get the consent
    # interstitial instead of the chat page.
    "Cookie": "SOCS=CAI",
}

_API_KEY_RE = re.compile(r'"INNERTUBE_API_KEY"\s*:\s*"([^"]+)"')


def _is_transient_http_error(exc: BaseException) -> bool:
    if isinstance(exc, aiohttp.ClientResponseError):
        return not (400 <= exc.status < 500)
    return isinstance(exc, aiohttp.ClientError | TimeoutError)


class YouTubeLiveChatError(RuntimeError):
    """The chat page/protocol didn't give us what we need (not live, chat
    disabled, layout drift, or the network gave up)."""


@dataclass(slots=True)
class ChatPage:
    """What one fetch of the popout chat page yields: everything a poll needs."""

    api_key: str
    context: dict[str, Any]
    continuation: str


def extract_video_id(video: str) -> str:
    """Accept a bare id or any youtube URL (watch?v=, youtu.be/, /live/)."""
    if not video.startswith(("http://", "https://")):
        return video
    parsed = urllib.parse.urlparse(video)
    v = urllib.parse.parse_qs(parsed.query).get("v", [""])[0]
    if v:
        return v
    parts = [p for p in parsed.path.split("/") if p]
    if parts:
        return parts[-1]
    raise YouTubeLiveChatError(f"cannot extract a video id from {video!r}")


def _dig(obj: Any, *path: str | int) -> Any:
    """Walk nested dicts/lists, returning None the moment a step is missing."""
    for key in path:
        if isinstance(obj, dict) and isinstance(key, str):
            obj = obj.get(key)
        elif isinstance(obj, list) and isinstance(key, int) and -len(obj) <= key < len(obj):
            obj = obj[key]
        else:
            return None
    return obj


def _json_after(text: str, marker: str) -> Any:
    """Decode the first balanced JSON object following `marker` in `text`."""
    idx = text.find(marker)
    if idx == -1:
        return None
    start = text.find("{", idx + len(marker))
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
    except ValueError:
        return None
    return obj


def parse_chat_page(html: str) -> ChatPage:
    """Lift the InnerTube config + initial continuation out of the popout page."""
    key_match = _API_KEY_RE.search(html)
    if key_match is None:
        raise YouTubeLiveChatError("no INNERTUBE_API_KEY on the chat page (layout drift?)")
    context = _json_after(html, '"INNERTUBE_CONTEXT":')
    if not isinstance(context, dict):
        raise YouTubeLiveChatError("no INNERTUBE_CONTEXT on the chat page (layout drift?)")
    initial = _json_after(html, "ytInitialData")
    continuation = _initial_continuation(initial)
    if not continuation:
        # When there's no chat, YouTube says why in a messageRenderer
        # ("Chat is disabled for this live stream.") — surface its words.
        why = _runs_text(_dig(initial, "contents", "messageRenderer", "text", "runs"))
        raise YouTubeLiveChatError(
            why or "no live chat continuation (video not live, chat disabled, or stream over)"
        )
    return ChatPage(api_key=key_match.group(1), context=context, continuation=continuation)


def _initial_continuation(initial: Any) -> str | None:
    renderer = _dig(initial, "contents", "liveChatRenderer")
    if not isinstance(renderer, dict):
        return None
    # Prefer the "Live chat" (all messages) view — the last view-selector entry —
    # over the default "Top chat", which hides most of the firehose the trend
    # detector wants to see.
    items = _dig(
        renderer,
        "header",
        "liveChatHeaderRenderer",
        "viewSelector",
        "sortFilterSubMenuRenderer",
        "subMenuItems",
    )
    if isinstance(items, list) and items:
        token = _dig(items[-1], "continuation", "reloadContinuationData", "continuation")
        if token:
            return str(token)
    for cont in renderer.get("continuations", []):
        if not isinstance(cont, dict):
            continue
        for data in cont.values():
            if isinstance(data, dict) and data.get("continuation"):
                return str(data["continuation"])
    return None


def parse_live_chat_response(data: Any, now: float) -> tuple[list[ChatMessage], str | None, float]:
    """One poll's worth of chat: (messages, next continuation, seconds to wait).

    A None continuation means YouTube stopped handing out tokens — the stream
    (or its chat) is over, or the token rotted and the page must be re-fetched.
    """
    live = _dig(data, "continuationContents", "liveChatContinuation")
    if not isinstance(live, dict):
        return [], None, 0.0
    token: str | None = None
    timeout = _DEFAULT_POLL_SECONDS
    for cont in live.get("continuations", []):
        if not isinstance(cont, dict):
            continue
        # The wrapper key varies (invalidation/timed/reload...ContinuationData);
        # all shapes carry `continuation` and usually `timeoutMs`.
        for value in cont.values():
            if isinstance(value, dict) and value.get("continuation"):
                token = str(value["continuation"])
                timeout_ms = value.get("timeoutMs")
                if isinstance(timeout_ms, int | float):
                    timeout = min(max(timeout_ms / 1000.0, _MIN_POLL_SECONDS), _MAX_POLL_SECONDS)
                break
        if token:
            break
    messages = []
    for action in live.get("actions", []):
        message = _message_from_item(_dig(action, "addChatItemAction", "item"), now)
        if message is not None:
            messages.append(message)
    return messages, token, timeout


# Renderers that carry viewer text. Memberships, stickers, mode banners and
# placeholders are deliberately skipped — they're chrome, not conversation.
_MESSAGE_RENDERERS = ("liveChatTextMessageRenderer", "liveChatPaidMessageRenderer")


def _message_from_item(item: Any, now: float) -> ChatMessage | None:
    if not isinstance(item, dict):
        return None
    for kind in _MESSAGE_RENDERERS:
        renderer = item.get(kind)
        if isinstance(renderer, dict):
            break
    else:
        return None
    text = _runs_text(_dig(renderer, "message", "runs"))
    if not text:  # e.g. a super chat sent with no message
        return None
    is_moderator, is_owner = _badges(renderer.get("authorBadges"))
    raw: dict[str, Any] = {
        "id": renderer.get("id", ""),
        "renderer": kind,
        "timestamp_usec": renderer.get("timestampUsec", ""),
    }
    amount = _dig(renderer, "purchaseAmountText", "simpleText")
    if amount:
        raw["paid_amount"] = str(amount)
    return ChatMessage(
        author=str(_dig(renderer, "authorName", "simpleText") or "anon"),
        text=text,
        ts=now,
        is_moderator=is_moderator,
        is_owner=is_owner,
        raw=raw,
    )


def _runs_text(runs: Any) -> str:
    """Flatten message runs; unicode emoji keep their glyph, custom channel
    emotes render as their :shortcut: so the text stays meaningful."""
    if not isinstance(runs, list):
        return ""
    parts = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        if "text" in run:
            parts.append(str(run["text"]))
        elif "emoji" in run:
            emoji = run["emoji"]
            if isinstance(emoji, dict):
                if emoji.get("isCustomEmoji"):
                    shortcuts = emoji.get("shortcuts") or []
                    parts.append(str(shortcuts[0]) if shortcuts else "")
                else:
                    parts.append(str(emoji.get("emojiId", "")))
    return "".join(parts).strip()


def _badges(badges: Any) -> tuple[bool, bool]:
    is_moderator = is_owner = False
    if isinstance(badges, list):
        for badge in badges:
            icon = _dig(badge, "liveChatAuthorBadgeRenderer", "icon", "iconType")
            if icon == "MODERATOR":
                is_moderator = True
            elif icon == "OWNER":
                is_owner = True
    return is_moderator, is_owner


class YouTubeLiveChatClient:
    """Polls a live stream's chat and yields `ChatMessage`s as they arrive.

    Transient network errors retry with backoff; a rotted continuation gets one
    page re-resolve before we conclude the chat is really over. Raises
    `YouTubeLiveChatError` when ingestion can't continue — the adapter decides
    whether that kills anything (it shouldn't: chat is a perception channel,
    not the spine).
    """

    def __init__(
        self,
        video: str,
        *,
        session: aiohttp.ClientSession | None = None,
        max_retries: int = 5,
    ) -> None:
        self._video_id = extract_video_id(video)
        self._session = session
        self._owns_session = session is None
        self._max_retries = max_retries

    async def messages(self) -> AsyncIterator[ChatMessage]:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers=_HEADERS, timeout=aiohttp.ClientTimeout(total=30)
            )
        seen: set[str] = set()
        seen_order: deque[str] = deque(maxlen=_SEEN_IDS_WINDOW)
        try:
            page = await self._load_page()
            continuation: str | None = page.continuation
            refreshed = False
            history = True  # first poll after a page load replays the backlog
            while continuation:
                data = await self._poll(page, continuation)
                messages, continuation, timeout = parse_live_chat_response(
                    data, now=time.monotonic()
                )
                for message in messages:
                    key = str(message.raw.get("id") or "")
                    if key:
                        if key in seen:
                            continue
                        if len(seen_order) == seen_order.maxlen:
                            seen.discard(seen_order[0])
                        seen_order.append(key)
                        seen.add(key)
                    # The backlog is what already happened, not what's happening:
                    # yielding it would stamp minutes-old messages "now" and fake
                    # a hype spike / trend wave at boot. Record ids, drop text.
                    if not history:
                        yield message
                if history:
                    log.debug("skipped %d backlog chat messages", len(messages))
                    history = False
                if continuation:
                    refreshed = False  # healthy again; re-arm the one-shot refresh
                    await asyncio.sleep(timeout)
                elif not refreshed:
                    # No token can mean "chat over" OR "this token rotted".
                    # Re-resolve from the page once; if the page has no chat
                    # either, it's genuinely over.
                    refreshed = True
                    try:
                        page = await self._load_page()
                        continuation = page.continuation
                        history = True  # the fresh reload replays backlog again
                    except YouTubeLiveChatError:
                        continuation = None
            log.info("live chat ended for %s", self._video_id)
        finally:
            if self._owns_session and self._session is not None:
                await self._session.close()
                self._session = None

    async def _load_page(self) -> ChatPage:
        url = f"https://www.youtube.com/live_chat?is_popout=1&v={self._video_id}"

        async def fetch() -> str:
            assert self._session is not None
            async with self._session.get(url) as resp:
                resp.raise_for_status()
                return await resp.text()

        return parse_chat_page(await self._retry(fetch))

    async def _poll(self, page: ChatPage, continuation: str) -> Any:
        url = (
            "https://www.youtube.com/youtubei/v1/live_chat/get_live_chat"
            f"?key={page.api_key}&prettyPrint=false"
        )
        payload = {"context": page.context, "continuation": continuation}

        async def fetch() -> Any:
            assert self._session is not None
            async with self._session.post(url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()

        return await self._retry(fetch)

    async def _retry(self, op):
        """Run one HTTP op with exponential backoff on transient failures.
        Client errors (4xx) are protocol problems, not weather — fail fast."""
        retryer = AsyncRetrying(
            retry=retry_if_exception(_is_transient_http_error),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=16),
            reraise=True,
            before_sleep=_log_retry,
        )
        try:
            async for attempt in retryer:
                with attempt:
                    return await op()
        except aiohttp.ClientResponseError as exc:
            if 400 <= exc.status < 500:
                raise YouTubeLiveChatError(f"chat request rejected ({exc.status})") from exc
            raise YouTubeLiveChatError(
                f"chat request failed after {self._max_retries} attempts: {exc}"
            ) from exc
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise YouTubeLiveChatError(
                f"chat request failed after {self._max_retries} attempts: {exc}"
            ) from exc
        raise AssertionError("unreachable")  # pragma: no cover


def _log_retry(retry_state) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome is not None else None
    if exc is not None:
        log.debug(
            "chat request failed (attempt %d): %s; retrying",
            retry_state.attempt_number,
            exc,
        )
