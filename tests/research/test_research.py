"""Cold-start channel research: identity, search seam, profiler, cache, seeding."""

from __future__ import annotations

import json

import pytest

from lingus.research import (
    ChannelIdentity,
    ChannelProfile,
    ChannelResearcher,
    NullSearch,
    build_search_backend,
    load_cached,
    research_channel,
    save_cache,
)
from lingus.research.profiler import _search_queries
from lingus.research.search import SearchResult, WebSearchBackend


class FakeSearch(WebSearchBackend):
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.queries: list[str] = []

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.queries.append(query)
        return self.results[:max_results]


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.seen: str = ""

    async def generate(self, messages, **opts) -> str:
        self.seen = messages[-1].content
        return self.reply


# --- identity -----------------------------------------------------------------
def test_cache_key_is_stable_and_filesystem_safe():
    ident = ChannelIdentity(platform="youtube", name="Some Streamer!", channel_id="UC_abc/123")
    key = ident.cache_key()
    assert key == "youtube_uc_abc_123"
    assert "/" not in key and "!" not in key


def test_cache_key_falls_back_to_name_without_id():
    ident = ChannelIdentity(platform="twitch", name="Cool Guy")
    assert ident.cache_key() == "twitch_cool_guy"


# --- search backend build -----------------------------------------------------
def test_build_search_backend_none_is_null():
    assert isinstance(build_search_backend("none"), NullSearch)
    assert isinstance(build_search_backend(""), NullSearch)


def test_build_search_backend_unknown_degrades_to_null():
    assert isinstance(build_search_backend("altavista"), NullSearch)


async def test_null_search_returns_nothing():
    assert await NullSearch().search("anything", 5) == []


# --- profiler: LLM path -------------------------------------------------------
async def test_researcher_llm_path_returns_parsed_facts():
    search = FakeSearch([SearchResult("Bio", "http://x", "streams retro games")])
    llm = FakeLLM(json.dumps({"facts": ["the streamer plays retro games"], "summary": "Retro."}))
    researcher = ChannelResearcher(search, llm=llm)
    ident = ChannelIdentity(platform="youtube", name="RetroKid", description="old games")
    profile = await researcher.research(ident)
    assert profile.facts == ["the streamer plays retro games"]
    assert profile.summary == "Retro."
    # Web snippet + metadata both reach the model.
    assert "streams retro games" in llm.seen
    assert "old games" in llm.seen


async def test_researcher_llm_path_tolerates_chatty_json():
    llm = FakeLLM('Sure! Here you go:\n{"facts": ["fact one"], "summary": "s"}\nHope that helps')
    researcher = ChannelResearcher(FakeSearch([]), llm=llm)
    profile = await researcher.research(ChannelIdentity(platform="twitch", name="X"))
    assert profile.facts == ["fact one"]


async def test_researcher_falls_back_when_llm_returns_no_facts():
    llm = FakeLLM(json.dumps({"facts": [], "summary": ""}))
    researcher = ChannelResearcher(FakeSearch([]), llm=llm)
    ident = ChannelIdentity(platform="youtube", name="Nova", description="I speedrun Mario.")
    profile = await researcher.research(ident)
    # Fallback synthesizes facts from metadata rather than returning nothing.
    assert any("Nova" in f for f in profile.facts)


async def test_researcher_fallback_when_llm_raises():
    class Boom:
        async def generate(self, messages, **opts):
            raise RuntimeError("no key")

    researcher = ChannelResearcher(FakeSearch([]), llm=Boom())
    ident = ChannelIdentity(platform="youtube", name="Nova", tags=["speedrun", "mario"])
    profile = await researcher.research(ident)
    assert any("speedrun" in f for f in profile.facts)


# --- profiler: deterministic fallback ----------------------------------------
async def test_researcher_no_llm_uses_metadata_and_snippets():
    search = FakeSearch([SearchResult("T", "http://src", "Known for a duck mascot. Extra.")])
    researcher = ChannelResearcher(search, llm=None)
    ident = ChannelIdentity(
        platform="youtube", name="Ducky", description="A cozy art stream. Come hang.", tags=["art"]
    )
    profile = await researcher.research(ident)
    assert profile.facts[0] == "the streamer's channel is Ducky on youtube"
    assert any("cozy art stream" in f for f in profile.facts)
    assert any("art" in f for f in profile.facts)
    assert any("duck mascot" in f for f in profile.facts)
    assert "http://src" in profile.source_urls


def test_search_queries_respect_limit_and_include_name():
    qs = _search_queries(ChannelIdentity(platform="twitch", name="Zed"), limit=2)
    assert len(qs) == 2
    assert all("Zed" in q for q in qs)


async def test_gather_dedups_by_url():
    dup = SearchResult("A", "http://same", "one")
    other = SearchResult("B", "http://other", "two")
    search = FakeSearch([dup, other, dup])
    researcher = ChannelResearcher(search, llm=None, max_queries=1)
    profile = await researcher.research(ChannelIdentity(platform="twitch", name="N"))
    assert profile.source_urls.count("http://same") == 1


# --- cache --------------------------------------------------------------------
def test_cache_round_trip(tmp_path):
    prof = ChannelProfile(channel="A", facts=["f1", "f2"], summary="s", source_urls=["u"])
    save_cache(str(tmp_path), "youtube_a", prof, now=1000.0)
    got = load_cached(str(tmp_path), "youtube_a", refresh_days=14, now=1000.0)
    assert got is not None
    assert got.facts == ["f1", "f2"]
    assert got.summary == "s"


def test_cache_miss_when_absent(tmp_path):
    assert load_cached(str(tmp_path), "missing", refresh_days=14) is None


def test_cache_expires_after_refresh_days(tmp_path):
    prof = ChannelProfile(channel="A", facts=["f1"])
    save_cache(str(tmp_path), "k", prof, now=0.0)
    fresh = load_cached(str(tmp_path), "k", refresh_days=1, now=1000.0)
    assert fresh is not None
    stale = load_cached(str(tmp_path), "k", refresh_days=1, now=2 * 86400.0)
    assert stale is None


def test_cache_zero_refresh_always_misses(tmp_path):
    save_cache(str(tmp_path), "k", ChannelProfile(channel="A", facts=["f"]), now=0.0)
    assert load_cached(str(tmp_path), "k", refresh_days=0, now=0.0) is None


def test_cache_corrupt_file_is_a_miss(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    assert load_cached(str(tmp_path), "bad", refresh_days=14) is None


# --- orchestrator -------------------------------------------------------------
async def test_research_channel_uses_cache_without_researching(tmp_path):
    save_cache(str(tmp_path), "youtube_cached", ChannelProfile(channel="C", facts=["cached"]))
    ident = ChannelIdentity(platform="youtube", name="C", channel_id="cached")

    async def _fail(*a, **k):  # pragma: no cover - must not be called on cache hit
        raise AssertionError("should not research on a fresh cache hit")

    # No web/LLM needed: a fresh cache short-circuits before the researcher runs.
    profile = await research_channel(
        ident,
        web_backend="none",
        cache_dir=str(tmp_path),
        refresh_days=14,
        max_facts=12,
        max_queries=4,
        max_results=6,
        llm=None,
    )
    assert profile is not None
    assert profile.facts == ["cached"]


async def test_research_channel_force_bypasses_cache(tmp_path):
    save_cache(str(tmp_path), "youtube_x", ChannelProfile(channel="X", facts=["old"]))
    ident = ChannelIdentity(
        platform="youtube", name="X", channel_id="x", description="new bio here"
    )
    profile = await research_channel(
        ident,
        web_backend="none",
        cache_dir=str(tmp_path),
        refresh_days=14,
        max_facts=12,
        max_queries=4,
        max_results=6,
        llm=None,
        force=True,
    )
    assert profile is not None
    assert profile.facts != ["old"]  # re-researched from metadata
    # And the fresh profile overwrote the cache.
    reloaded = load_cached(str(tmp_path), "youtube_x", refresh_days=14)
    assert reloaded is not None and reloaded.facts == profile.facts


async def test_research_channel_writes_cache_on_fresh_research(tmp_path):
    ident = ChannelIdentity(platform="twitch", name="Fresh", description="cooking streams")
    profile = await research_channel(
        ident,
        web_backend="none",
        cache_dir=str(tmp_path),
        refresh_days=14,
        max_facts=12,
        max_queries=4,
        max_results=6,
        llm=None,
    )
    assert profile is not None and profile.facts
    assert (tmp_path / "twitch_fresh.json").exists()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
