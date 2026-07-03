"""Small async wrapper around yt-dlp's Python API."""

from __future__ import annotations

import asyncio
from typing import Any


class YtDlpError(RuntimeError):
    """Raised when yt-dlp cannot resolve metadata or media URLs."""


def watch_url(video_or_url: str) -> str:
    """Return a YouTube watch URL for a video id or passthrough URL."""

    if video_or_url.startswith(("http://", "https://")):
        return video_or_url
    return f"https://www.youtube.com/watch?v={video_or_url}"


async def extract_info(
    url: str,
    *,
    timeout: float = 30.0,
    ydl_opts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run yt-dlp metadata extraction off the event loop."""

    try:
        info = await asyncio.wait_for(
            asyncio.to_thread(_extract_info_blocking, url, ydl_opts or {}),
            timeout=timeout,
        )
    except TimeoutError as exc:
        raise YtDlpError(
            f"yt-dlp timed out extracting metadata for {url!r} after {timeout:g}s"
        ) from exc
    except ModuleNotFoundError as exc:
        raise YtDlpError(
            "yt-dlp is not installed; install lingus[youtube] to use YouTube inputs"
        ) from exc
    except Exception as exc:
        raise YtDlpError(f"yt-dlp failed extracting metadata for {url!r}: {exc}") from exc

    if not isinstance(info, dict):
        raise YtDlpError(f"yt-dlp returned unexpected metadata for {url!r}")
    return info


async def resolve_media_url(
    url: str,
    fmt: str,
    *,
    label: str = "media",
    timeout: float = 30.0,
) -> str:
    """Resolve a direct media URL using yt-dlp format selection."""

    info = await extract_info(
        url,
        timeout=timeout,
        ydl_opts={
            "format": fmt,
            "noplaylist": True,
        },
    )
    urls = _media_urls(info)
    if not urls:
        raise YtDlpError(f"yt-dlp did not return a direct {label} URL for {url!r}")
    return urls[-1]


def _extract_info_blocking(url: str, ydl_opts: dict[str, Any]) -> dict[str, Any]:
    import yt_dlp

    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    opts.update(ydl_opts)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return ydl.sanitize_info(info)


def _media_urls(info: dict[str, Any]) -> list[str]:
    requested_formats = info.get("requested_formats")
    if isinstance(requested_formats, list):
        urls = [
            str(fmt["url"])
            for fmt in requested_formats
            if isinstance(fmt, dict) and fmt.get("url")
        ]
        if urls:
            return urls

    requested_downloads = info.get("requested_downloads")
    if isinstance(requested_downloads, list):
        urls = [
            str(fmt["url"])
            for fmt in requested_downloads
            if isinstance(fmt, dict) and fmt.get("url")
        ]
        if urls:
            return urls

    url = info.get("url")
    if isinstance(url, str) and url:
        return [url]
    return []
