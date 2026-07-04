import asyncio
import sys
import types

import pytest

from lingus.adapters import twitch
from lingus.adapters.base import ChatMessage


# --- channel normalization ---------------------------------------------------
@pytest.mark.parametrize(
    "raw, name, url",
    [
        ("shroud", "shroud", "https://www.twitch.tv/shroud"),
        ("@Shroud", "shroud", "https://www.twitch.tv/shroud"),
        ("https://www.twitch.tv/xQc", "xqc", "https://www.twitch.tv/xQc"),
        ("https://twitch.tv/pokimane/", "pokimane", "https://twitch.tv/pokimane/"),
    ],
)
def test_channel_normalization(raw, name, url):
    assert twitch.channel_name(raw) == name
    assert twitch.channel_url(raw) == url


# --- capture: streamlink URL resolution --------------------------------------
class FakeStream:
    def __init__(self, url: str) -> None:
        self.url = url


def _install_streamlink(monkeypatch, streams):
    def fake_streams(url):
        fake_streams.called_with = url
        return streams

    module = types.SimpleNamespace(streams=fake_streams)
    monkeypatch.setitem(sys.modules, "streamlink", module)
    return fake_streams


def test_capture_rejects_empty_channel():
    with pytest.raises(ValueError, match="channel"):
        twitch.TwitchCaptureAdapter("")


@pytest.mark.asyncio
async def test_resolve_stream_url_prefers_configured_quality(monkeypatch):
    fake = _install_streamlink(
        monkeypatch,
        {"best": FakeStream("hls://best"), "720p": FakeStream("hls://720")},
    )
    adapter = twitch.TwitchCaptureAdapter("shroud", quality="720p")
    assert await adapter._resolve_audio_url() == "hls://720"
    # Video reuses the same resolution (Twitch muxes A/V into one rendition).
    assert await adapter._resolve_video_url() == "hls://720"
    assert fake.called_with == "https://www.twitch.tv/shroud"


@pytest.mark.asyncio
async def test_resolve_stream_url_falls_back_to_best(monkeypatch):
    _install_streamlink(monkeypatch, {"best": FakeStream("hls://best")})
    adapter = twitch.TwitchCaptureAdapter("shroud", quality="1080p")
    assert await adapter._resolve_audio_url() == "hls://best"


@pytest.mark.asyncio
async def test_resolve_stream_url_errors_when_offline(monkeypatch):
    _install_streamlink(monkeypatch, {})
    adapter = twitch.TwitchCaptureAdapter("shroud")
    with pytest.raises(twitch.TwitchStreamError, match="offline or unknown"):
        await adapter._resolve_audio_url()


@pytest.mark.asyncio
async def test_resolve_stream_url_errors_when_streamlink_raises(monkeypatch):
    def boom(url):
        raise RuntimeError("cloudflare said no")

    monkeypatch.setitem(sys.modules, "streamlink", types.SimpleNamespace(streams=boom))
    adapter = twitch.TwitchCaptureAdapter("shroud")
    with pytest.raises(twitch.TwitchStreamError, match="cloudflare said no"):
        await adapter._resolve_audio_url()


# --- chat: twitchio bridge ---------------------------------------------------
class FakeBaseBot:
    """Stand-in for twitchio.ext.commands.Bot: records init, no network."""

    def __init__(self, *, token, prefix, initial_channels) -> None:
        self.token = token
        self.prefix = prefix
        self.initial_channels = initial_channels
        self.nick = "lingusbot"


def _install_twitchio(monkeypatch, base=FakeBaseBot):
    commands_mod = types.ModuleType("twitchio.ext.commands")
    commands_mod.Bot = base
    ext_mod = types.ModuleType("twitchio.ext")
    ext_mod.commands = commands_mod
    root_mod = types.ModuleType("twitchio")
    root_mod.ext = ext_mod
    monkeypatch.setitem(sys.modules, "twitchio", root_mod)
    monkeypatch.setitem(sys.modules, "twitchio.ext", ext_mod)
    monkeypatch.setitem(sys.modules, "twitchio.ext.commands", commands_mod)
    return commands_mod


class FakeAuthor:
    def __init__(self, name, *, is_mod=False, is_broadcaster=False) -> None:
        self.name = name
        self.is_mod = is_mod
        self.is_broadcaster = is_broadcaster


class FakeIncoming:
    def __init__(self, content, author, *, echo=False) -> None:
        self.content = content
        self.author = author
        self.echo = echo


@pytest.mark.asyncio
async def test_build_bot_enqueues_chat_messages(monkeypatch):
    _install_twitchio(monkeypatch)
    queue: asyncio.Queue = asyncio.Queue()
    bot = twitch._build_bot("tok", "shroud", queue)

    assert bot.token == "tok"
    assert bot.initial_channels == ["shroud"]

    await bot.event_message(
        FakeIncoming("hello bot", FakeAuthor("viewer1", is_mod=True))
    )
    msg = queue.get_nowait()
    assert isinstance(msg, ChatMessage)
    assert (msg.author, msg.text, msg.is_moderator, msg.is_owner) == (
        "viewer1",
        "hello bot",
        True,
        False,
    )
    assert msg.raw["channel"] == "shroud"


@pytest.mark.asyncio
async def test_build_bot_skips_echo_and_empty(monkeypatch):
    _install_twitchio(monkeypatch)
    queue: asyncio.Queue = asyncio.Queue()
    bot = twitch._build_bot("tok", "shroud", queue)

    await bot.event_message(FakeIncoming("my own line", FakeAuthor("lingusbot"), echo=True))
    await bot.event_message(FakeIncoming("", FakeAuthor("viewer")))  # e.g. emote-only
    assert queue.empty()


@pytest.mark.asyncio
async def test_build_bot_drops_oldest_when_queue_full(monkeypatch):
    _install_twitchio(monkeypatch)
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    bot = twitch._build_bot("tok", "shroud", queue)

    await bot.event_message(FakeIncoming("first", FakeAuthor("a")))
    await bot.event_message(FakeIncoming("second", FakeAuthor("b")))
    # Oldest dropped, newest kept.
    assert queue.get_nowait().text == "second"
    assert queue.empty()


@pytest.mark.asyncio
async def test_chat_adapter_without_token_is_silent_observer(caplog):
    adapter = twitch.TwitchChatAdapter("shroud", "")
    await adapter.start()
    # incoming() yields nothing.
    assert [m async for m in adapter.incoming()] == []
    # post() only logs.
    with caplog.at_level("INFO", logger="lingus.adapters.twitch"):
        await adapter.post("hi chat")
    assert any("[BOT would post]" in r.message for r in caplog.records)
    await adapter.stop()


@pytest.mark.asyncio
async def test_chat_adapter_incoming_drains_queue_until_shutdown(monkeypatch):
    _install_twitchio(monkeypatch)
    adapter = twitch.TwitchChatAdapter("shroud", "tok")
    # Simulate a connected bot by hand (don't spin the real connection loop).
    adapter._bot = object()
    m1 = ChatMessage(author="a", text="one", ts=0.0)
    m2 = ChatMessage(author="b", text="two", ts=1.0)
    adapter._queue.put_nowait(m1)
    adapter._queue.put_nowait(m2)
    adapter._queue.put_nowait(twitch._SHUTDOWN)

    got = [m async for m in adapter.incoming()]
    assert [m.text for m in got] == ["one", "two"]


@pytest.mark.asyncio
async def test_chat_adapter_posts_when_enabled(monkeypatch):
    sent = []

    class FakeChannel:
        async def send(self, text):
            sent.append(text)

    class FakeBot:
        def get_channel(self, name):
            self.asked = name
            return FakeChannel()

    adapter = twitch.TwitchChatAdapter("shroud", "tok", post_enabled=True)
    adapter._bot = FakeBot()
    await adapter.post("nice clutch")
    assert sent == ["nice clutch"]


@pytest.mark.asyncio
async def test_chat_adapter_post_drops_when_channel_not_joined(caplog):
    class FakeBot:
        def get_channel(self, name):
            return None

    adapter = twitch.TwitchChatAdapter("shroud", "tok", post_enabled=True)
    adapter._bot = FakeBot()
    with caplog.at_level("WARNING", logger="lingus.adapters.twitch"):
        await adapter.post("dropped")
    assert any("not joined yet" in r.message for r in caplog.records)
