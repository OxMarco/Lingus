"""Resolve *who* the bot is about to watch.

A `ChannelIdentity` is the raw, factual starting point for research: the
channel's name, what it says about itself, its tags, and a few recent video
titles. On YouTube this is free — `yt-dlp` already ships the metadata alongside
the media URL we resolve for capture, so one Python API extraction gets us
everything without touching the Data API or an OAuth flow. On other
platforms (or offline replay) there's no feed to introspect, so identity comes
from a channel name configured in `research.channel`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..logging import get_logger
from ..yt_dlp_api import YtDlpError, extract_info, watch_url

log = get_logger(__name__)


@dataclass(slots=True)
class ChannelIdentity:
    """Factual, pre-research description of a channel."""

    platform: str
    name: str
    channel_id: str = ""  # stable id when we have one (yt channel_id); else ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    recent_titles: list[str] = field(default_factory=list)
    url: str = ""

    def cache_key(self) -> str:
        """Stable key for the profile cache — the channel id when we have it,
        else the (lower-cased) name, always namespaced by platform."""
        ident = self.channel_id or self.name
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in ident.lower())
        return f"{self.platform}_{safe}" or self.platform


async def _dump_json(url: str, *, timeout: float = 30.0) -> dict | None:
    """Best-effort yt-dlp metadata. Research must never stop stream startup."""
    try:
        return await extract_info(
            url,
            timeout=timeout,
            ydl_opts={
                # Channel/playlist URLs: fetch metadata only, no per-video crawl.
                "playlist_items": "0",
            },
        )
    except YtDlpError as exc:
        log.warning("research: %s", exc)
        return None


async def _resolve_youtube(video: str) -> ChannelIdentity | None:
    watch = watch_url(video)
    data = await _dump_json(watch)
    if data is None:
        return None
    name = str(data.get("channel") or data.get("uploader") or "").strip()
    if not name:
        log.warning("research: could not determine channel name for %s", watch)
        return None
    tags = data.get("tags") or data.get("categories") or []
    tags = [str(t) for t in tags if str(t).strip()][:12] if isinstance(tags, list) else []
    # `--playlist-items 0` on a channel yields `entries: []`; on a single video
    # we don't get siblings. Recent titles are a best-effort bonus, so we take the
    # video's own title as the one guaranteed data point.
    recent = [str(data.get("title", "")).strip()] if data.get("title") else []
    return ChannelIdentity(
        platform="youtube",
        name=name,
        channel_id=str(data.get("channel_id") or "").strip(),
        description=str(data.get("channel_description") or "").strip(),
        tags=tags,
        recent_titles=[t for t in recent if t],
        url=str(data.get("channel_url") or data.get("uploader_url") or watch),
    )


async def resolve_identity(
    platform: str,
    *,
    video: str | None = None,
    channel_name: str = "",
) -> ChannelIdentity | None:
    """Resolve the channel identity for the configured run, or None if there's
    nothing to research (no video id, no configured channel name)."""
    if platform == "youtube" and video:
        return await _resolve_youtube(video)
    # Twitch / file_replay / a YouTube run with no video: fall back to a plain
    # configured name. There's no feed to introspect, so research leans entirely
    # on web search + the LLM.
    if channel_name.strip():
        return ChannelIdentity(platform=platform, name=channel_name.strip())
    return None
