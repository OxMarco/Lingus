"""The app-level cold-start seeding: research facts land in the semantic store."""

from __future__ import annotations

import argparse

import pytest

import lingus.app as app
from lingus.config import Settings
from lingus.memory import SemanticStore


def _args(**over) -> argparse.Namespace:
    base = {"video": None, "research": False, "no_research": False}
    base.update(over)
    return argparse.Namespace(**base)


def _settings(tmp_path, **research_over) -> Settings:
    research = {"channel": "TestStreamer", "cache_dir": str(tmp_path / "research")}
    research.update(research_over)
    return Settings.model_validate(
        {
            "platform": "twitch",  # no yt-dlp resolution; uses research.channel
            "research": research,
            "memory": {"semantic_path": str(tmp_path / "semantic.json")},
        }
    )


@pytest.mark.asyncio
async def test_seed_research_writes_facts_to_semantic_store(tmp_path):
    settings = _settings(tmp_path)
    # No LLM key -> deterministic fallback facts from the identity metadata.
    await app._seed_research(_args(), settings, llm_backend=None)

    store = SemanticStore(max_facts=settings.memory.semantic_max_facts)
    store.load_file(settings.memory.semantic_path)
    texts = store.texts()
    assert any("TestStreamer" in t for t in texts)


@pytest.mark.asyncio
async def test_seed_research_disabled_writes_nothing(tmp_path):
    settings = _settings(tmp_path, enabled=False)
    await app._seed_research(_args(), settings, llm_backend=None)
    assert not (tmp_path / "semantic.json").exists()


@pytest.mark.asyncio
async def test_seed_research_no_research_flag_skips(tmp_path):
    settings = _settings(tmp_path)
    await app._seed_research(_args(no_research=True), settings, llm_backend=None)
    assert not (tmp_path / "semantic.json").exists()


@pytest.mark.asyncio
async def test_seed_research_no_identity_is_noop(tmp_path):
    # Twitch with no configured channel -> nothing to research.
    settings = _settings(tmp_path, channel="")
    await app._seed_research(_args(), settings, llm_backend=None)
    assert not (tmp_path / "semantic.json").exists()


@pytest.mark.asyncio
async def test_seed_research_merges_with_existing_facts(tmp_path):
    settings = _settings(tmp_path)
    # A prior stream already persisted a fact; research must add, not clobber.
    store = SemanticStore()
    store.add("the streamer's name is Marco", source="stream")
    store.save_file(settings.memory.semantic_path)

    await app._seed_research(_args(), settings, llm_backend=None)

    reloaded = SemanticStore()
    reloaded.load_file(settings.memory.semantic_path)
    texts = reloaded.texts()
    assert any("Marco" in t for t in texts)  # old fact survived
    assert any("TestStreamer" in t for t in texts)  # new facts added


@pytest.mark.asyncio
async def test_seed_research_marks_facts_with_research_source(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    await app._seed_research(_args(), settings, llm_backend=None)
    store = SemanticStore()
    store.load_file(settings.memory.semantic_path)
    sources = {f.source for f in store._facts}  # noqa: SLF001 - test introspection
    assert "research" in sources
