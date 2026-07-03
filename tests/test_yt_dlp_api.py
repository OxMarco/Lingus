import asyncio

import pytest

from lingus import yt_dlp_api


def test_watch_url_accepts_id_or_url():
    assert yt_dlp_api.watch_url("abc123") == "https://www.youtube.com/watch?v=abc123"
    assert yt_dlp_api.watch_url("https://youtu.be/abc123") == "https://youtu.be/abc123"


@pytest.mark.asyncio
async def test_extract_info_times_out(monkeypatch):
    async def hang(*args, **kwargs):
        await asyncio.sleep(3600)

    monkeypatch.setattr(yt_dlp_api.asyncio, "to_thread", hang)

    with pytest.raises(yt_dlp_api.YtDlpError, match="timed out"):
        await yt_dlp_api.extract_info("https://example.test/video", timeout=0.01)


@pytest.mark.asyncio
async def test_resolve_media_url_uses_format_selection(monkeypatch):
    seen = {}

    async def fake_extract_info(url, *, timeout, ydl_opts):
        seen.update({"url": url, "timeout": timeout, "ydl_opts": ydl_opts})
        return {
            "requested_formats": [
                {"url": "https://media.example/video"},
                {"url": "https://media.example/audio"},
            ]
        }

    monkeypatch.setattr(yt_dlp_api, "extract_info", fake_extract_info)

    url = await yt_dlp_api.resolve_media_url(
        "https://youtube.example/watch?v=x",
        "bestaudio/best",
        label="audio",
        timeout=7.0,
    )

    assert url == "https://media.example/audio"
    assert seen == {
        "url": "https://youtube.example/watch?v=x",
        "timeout": 7.0,
        "ydl_opts": {"format": "bestaudio/best", "noplaylist": True},
    }


@pytest.mark.asyncio
async def test_resolve_media_url_rejects_missing_url(monkeypatch):
    async def fake_extract_info(*args, **kwargs):
        return {"title": "no direct url"}

    monkeypatch.setattr(yt_dlp_api, "extract_info", fake_extract_info)

    with pytest.raises(yt_dlp_api.YtDlpError, match="did not return"):
        await yt_dlp_api.resolve_media_url("https://youtube.example/watch?v=x", "best")
