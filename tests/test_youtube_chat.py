"""InnerTube live-chat protocol parsing + the polling client's control flow.

The parsers are pure, so they're regression-tested from fixtures shaped like
real (trimmed) YouTube payloads. The client tests stub the two HTTP methods
(_load_page/_poll) and exercise dedup, the rot-refresh path, and the clean end.
"""

import asyncio

import pytest

from lingus.adapters.youtube import ObserveChatAdapter
from lingus.adapters.youtube_chat import (
    ChatPage,
    YouTubeLiveChatClient,
    YouTubeLiveChatError,
    extract_video_id,
    parse_chat_page,
    parse_live_chat_response,
)

# --- page fixture (the popout live_chat page, heavily trimmed) ---

PAGE = (
    '<script>ytcfg.set({"INNERTUBE_API_KEY":"test-key",'
    '"INNERTUBE_CONTEXT":{"client":{"clientName":"WEB","clientVersion":"2.20240101"}},'
    '"OTHER":1});</script>'
    '<script>window["ytInitialData"] = {"contents":{"liveChatRenderer":{'
    '"continuations":[{"invalidationContinuationData":'
    '{"continuation":"default-token","timeoutMs":10000}}],'
    '"header":{"liveChatHeaderRenderer":{"viewSelector":{"sortFilterSubMenuRenderer":'
    '{"subMenuItems":['
    '{"title":"Top chat","continuation":{"reloadContinuationData":{"continuation":"top-token"}}},'
    '{"title":"Live chat","continuation":{"reloadContinuationData":{"continuation":"all-token"}}}'
    "]}}}}}}};</script>"
)


def _text_item(msg_id, author, text, badges=None):
    runs = [{"text": text}]
    item = {
        "liveChatTextMessageRenderer": {
            "id": msg_id,
            "timestampUsec": "1700000000000000",
            "authorName": {"simpleText": author},
            "message": {"runs": runs},
        }
    }
    if badges:
        item["liveChatTextMessageRenderer"]["authorBadges"] = [
            {"liveChatAuthorBadgeRenderer": {"icon": {"iconType": b}}} for b in badges
        ]
    return item


def _response(items, token="next-token", timeout_ms=2000):
    continuations = []
    if token is not None:
        continuations = [
            {"invalidationContinuationData": {"continuation": token, "timeoutMs": timeout_ms}}
        ]
    return {
        "continuationContents": {
            "liveChatContinuation": {
                "continuations": continuations,
                "actions": [{"addChatItemAction": {"item": item}} for item in items],
            }
        }
    }


# --- video id extraction ---


def test_extract_video_id_accepts_ids_and_urls():
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=1") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/live/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


# --- page parsing ---


def test_parse_chat_page_prefers_all_messages_view():
    page = parse_chat_page(PAGE)
    assert page.api_key == "test-key"
    assert page.context["client"]["clientName"] == "WEB"
    # "Live chat" (all messages), not "Top chat" and not the default continuation.
    assert page.continuation == "all-token"


def test_parse_chat_page_falls_back_to_default_continuation():
    html = PAGE.replace('"header":{"liveChatHeaderRenderer"', '"ignored":{"liveChatHeaderRenderer"')
    assert parse_chat_page(html).continuation == "default-token"


def test_parse_chat_page_not_live_raises():
    html = (
        '<script>ytcfg.set({"INNERTUBE_API_KEY":"k","INNERTUBE_CONTEXT":{"client":{}}});'
        '</script><script>var ytInitialData = {"contents":{}};</script>'
    )
    with pytest.raises(YouTubeLiveChatError, match="continuation"):
        parse_chat_page(html)


def test_parse_chat_page_surfaces_youtubes_reason():
    # Seen live on streams with chat off: contents is just a messageRenderer.
    html = (
        '<script>ytcfg.set({"INNERTUBE_API_KEY":"k","INNERTUBE_CONTEXT":{"client":{}}});'
        '</script><script>window["ytInitialData"] = {"contents":{"messageRenderer":'
        '{"text":{"runs":[{"text":"Chat is disabled for this live stream."}]}}}};</script>'
    )
    with pytest.raises(YouTubeLiveChatError, match="Chat is disabled"):
        parse_chat_page(html)


# --- poll response parsing ---


def test_parse_response_maps_text_messages_and_badges():
    data = _response(
        [
            _text_item("id1", "viewer1", "hello there"),
            _text_item("id2", "modguy", "behave", badges=["MODERATOR"]),
            _text_item("id3", "the_streamer", "hi chat", badges=["OWNER"]),
        ]
    )
    messages, token, timeout = parse_live_chat_response(data, now=42.0)
    assert [m.text for m in messages] == ["hello there", "behave", "hi chat"]
    assert messages[0].author == "viewer1"
    assert messages[0].ts == 42.0
    assert messages[0].raw["id"] == "id1"
    assert not messages[0].is_moderator
    assert messages[1].is_moderator and not messages[1].is_owner
    assert messages[2].is_owner
    assert token == "next-token"
    assert timeout == 2.0


def test_parse_response_flattens_emoji_runs():
    item = _text_item("id1", "viewer", "")
    item["liveChatTextMessageRenderer"]["message"]["runs"] = [
        {"text": "nice "},
        {"emoji": {"emojiId": "🔥", "isCustomEmoji": False}},
        {"emoji": {"emojiId": "UCabc/xyz", "isCustomEmoji": True, "shortcuts": [":_pog:"]}},
    ]
    messages, _, _ = parse_live_chat_response(_response([item]), now=0.0)
    assert messages[0].text == "nice 🔥:_pog:"


def test_parse_response_keeps_paid_messages_and_skips_chrome():
    paid = {
        "liveChatPaidMessageRenderer": {
            "id": "paid1",
            "authorName": {"simpleText": "bigspender"},
            "message": {"runs": [{"text": "take my money"}]},
            "purchaseAmountText": {"simpleText": "$5.00"},
        }
    }
    membership = {"liveChatMembershipItemRenderer": {"id": "m1"}}
    empty_paid = {"liveChatPaidMessageRenderer": {"id": "paid2", "authorName": {"simpleText": "x"}}}
    messages, _, _ = parse_live_chat_response(_response([paid, membership, empty_paid]), now=0.0)
    assert len(messages) == 1
    assert messages[0].raw["paid_amount"] == "$5.00"


def test_parse_response_ended_chat_has_no_token():
    messages, token, _ = parse_live_chat_response({"responseContext": {}}, now=0.0)
    assert messages == [] and token is None


def test_parse_response_clamps_timeout():
    _, _, fast = parse_live_chat_response(_response([], timeout_ms=1), now=0.0)
    _, _, slow = parse_live_chat_response(_response([], timeout_ms=600000), now=0.0)
    assert fast == 0.5
    assert slow == 30.0


# --- client control flow (HTTP stubbed out) ---


class _FakeClient(YouTubeLiveChatClient):
    """Serves canned pages/polls; `session=object()` so no aiohttp is touched."""

    def __init__(self, pages, polls):
        super().__init__("vid123", session=object())
        self._pages = list(pages)
        self._polls = list(polls)

    async def _load_page(self):
        result = self._pages.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def _poll(self, page, continuation):
        return self._polls.pop(0)


def _page(token="tok"):
    return ChatPage(api_key="k", context={}, continuation=token)


async def _collect(client):
    return [msg async for msg in client.messages()]


async def test_client_skips_backlog_then_yields_and_dedups():
    polls = [
        # First poll after a page load replays the backlog: never yielded, but
        # its ids count as seen ("a" must not re-yield when it comes back live).
        _response(
            [_text_item("h", "old", "history"), _text_item("a", "v1", "one")],
            token="t2",
            timeout_ms=1,
        ),
        _response(
            [_text_item("a", "v1", "one"), _text_item("b", "v2", "two")], token="t3", timeout_ms=1
        ),
        # Poll windows overlap: "b" comes back again and must not re-yield.
        _response([_text_item("b", "v2", "two"), _text_item("c", "v3", "three")], token=None),
    ]
    client = _FakeClient(
        pages=[_page(), YouTubeLiveChatError("stream over")],  # end-of-chat refresh fails
        polls=polls,
    )
    messages = await asyncio.wait_for(_collect(client), timeout=10)
    assert [m.text for m in messages] == ["two", "three"]


async def test_client_refreshes_once_on_rotted_continuation():
    polls = [
        _response([], token="t1", timeout_ms=1),  # backlog (empty) under page 1
        {"responseContext": {}},  # token rotted: no continuationContents
        _response([], token="t3", timeout_ms=1),  # backlog again under refreshed page
        _response([_text_item("a", "v1", "after refresh")], token=None),
    ]
    client = _FakeClient(
        pages=[_page("tok1"), _page("tok2"), YouTubeLiveChatError("over")],
        polls=polls,
    )
    messages = await asyncio.wait_for(_collect(client), timeout=10)
    assert [m.text for m in messages] == ["after refresh"]


# --- adapter behavior ---


async def test_observe_adapter_without_video_yields_nothing():
    adapter = ObserveChatAdapter()
    await adapter.start()
    assert [msg async for msg in adapter.incoming()] == []


async def test_observe_adapter_propagates_ingestion_failure():
    class _ExplodingClient:
        async def messages(self):
            raise YouTubeLiveChatError("chat disabled")
            yield  # pragma: no cover  (makes this an async generator)

    adapter = ObserveChatAdapter("vid123")
    adapter._client = _ExplodingClient()
    # Chat is a capture channel: an unrecoverable ingestion failure must
    # propagate so the loop can crash rather than run on silently deaf to chat.
    with pytest.raises(YouTubeLiveChatError):
        [msg async for msg in adapter.incoming()]
