"""Cold-start channel research.

Before the perception loop starts, profile the streamer we're about to watch and
seed the durable (semantic) memory layer, so the bot walks in already knowing the
channel instead of learning it from scratch each stream. All of it is best-effort
and cached: research failing must never stop the stream from starting.

`research_channel()` is the one-call orchestrator app.py uses; the individual
pieces (`resolve_identity`, `build_search_backend`, `ChannelResearcher`, the
cache) are exported for testing and reuse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..logging import get_logger
from .cache import load_cached, save_cache
from .identity import ChannelIdentity, resolve_identity
from .profiler import ChannelProfile, ChannelResearcher
from .search import (
    DuckDuckGoSearch,
    NullSearch,
    SearchResult,
    WebSearchBackend,
    build_search_backend,
)

if TYPE_CHECKING:
    from ..models.base import LLMBackend

__all__ = [
    "ChannelIdentity",
    "ChannelProfile",
    "ChannelResearcher",
    "DuckDuckGoSearch",
    "NullSearch",
    "SearchResult",
    "WebSearchBackend",
    "build_search_backend",
    "load_cached",
    "research_channel",
    "resolve_identity",
    "save_cache",
]

log = get_logger(__name__)


async def research_channel(
    identity: ChannelIdentity,
    *,
    web_backend: str,
    cache_dir: str,
    refresh_days: float,
    max_facts: int,
    max_queries: int,
    max_results: int,
    llm: LLMBackend | None,
    force: bool = False,
) -> ChannelProfile | None:
    """Resolve → cache → research → cache. Returns the profile (cached or fresh),
    or None if nothing was produced. Never raises: any failure logs and yields
    None so the caller can carry on and start the stream."""
    key = identity.cache_key()
    if not force:
        cached = load_cached(cache_dir, key, refresh_days=refresh_days)
        if cached is not None:
            log.info("research: using cached profile for '%s' (%d facts)", key, len(cached.facts))
            return cached
    try:
        researcher = ChannelResearcher(
            build_search_backend(web_backend),
            llm=llm,
            max_facts=max_facts,
            max_queries=max_queries,
            max_results=max_results,
        )
        profile = await researcher.research(identity)
    except Exception as exc:  # noqa: BLE001 - research is best-effort, never fatal
        log.warning("research: profiling '%s' failed (%s)", identity.name, exc)
        return None
    if profile.facts:
        save_cache(cache_dir, key, profile)
    return profile
