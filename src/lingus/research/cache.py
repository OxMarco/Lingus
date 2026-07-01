"""Persist channel profiles so a channel is researched once, not every startup.

Web search + an LLM call on every boot would be slow, wasteful, and rude to the
search endpoint. A channel's profile is stable over days, so we cache it on disk
keyed by channel and only re-research after `refresh_days`. The cache is a plain
JSON file per channel under `research.cache_dir`; a corrupt or missing file just
means "not cached" (re-research), never an error.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from ..logging import get_logger
from .profiler import ChannelProfile

log = get_logger(__name__)


def _path(cache_dir: str, key: str) -> Path:
    return Path(cache_dir) / f"{key}.json"


def load_cached(
    cache_dir: str,
    key: str,
    *,
    refresh_days: float,
    now: float | None = None,
) -> ChannelProfile | None:
    """Return the cached profile if it exists and is fresh, else None.

    `refresh_days == 0` always misses (forces re-research); a fresh, well-formed
    cache hit returns the stored profile.
    """
    if refresh_days <= 0:
        return None
    path = _path(cache_dir, key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    researched_ts = float(data.get("researched_ts", 0.0))
    current = time.time() if now is None else now
    if current - researched_ts > refresh_days * 86400.0:
        log.info("research: cached profile for '%s' is stale, will refresh", key)
        return None
    profile = data.get("profile", {})
    return ChannelProfile(
        channel=str(profile.get("channel", key)),
        facts=[str(f) for f in profile.get("facts", [])],
        summary=str(profile.get("summary", "")),
        source_urls=[str(u) for u in profile.get("source_urls", [])],
    )


def save_cache(
    cache_dir: str,
    key: str,
    profile: ChannelProfile,
    *,
    now: float | None = None,
) -> None:
    path = _path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "researched_ts": time.time() if now is None else now,
        "profile": asdict(profile),
    }
    path.write_text(json.dumps(payload, indent=2))
    log.info("research: cached profile for '%s' (%d facts)", key, len(profile.facts))
