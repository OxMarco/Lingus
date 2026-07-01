"""Semantic / long-term memory: store, retrieval, persistence, extraction."""

import asyncio

from lingus.memory import HeuristicFactExtractor, SemanticStore


def test_add_and_dedup_reworded_fact():
    s = SemanticStore()
    s.add("the streamer's name is Marco", now=1.0)
    s.add("the streamer's name is Marco", now=2.0)  # exact repeat -> reinforce
    s.add("the streamer's name is Marco apparently", now=3.0)  # near-dup -> same fact
    assert len(s) == 1


def test_retrieve_ranks_by_token_overlap():
    s = SemanticStore()
    s.add("the streamer loves jazz music", now=1.0)
    s.add("the streamer is from London", now=1.0)
    s.add("the streamer always drinks coffee", now=1.0)
    top = s.retrieve("do they like jazz", k=1)
    assert top[0].text == "the streamer loves jazz music"


def test_retrieve_falls_back_to_recency_when_no_overlap():
    s = SemanticStore()
    s.add("the streamer is from London", now=1.0)
    # No query overlap, but we still surface durable facts rather than nothing.
    assert len(s.retrieve("xyzzy", k=3)) == 1


def test_eviction_keeps_most_used_facts():
    s = SemanticStore(max_facts=2)
    s.add("fact about apples", now=1.0)
    s.add("fact about bananas", now=2.0)
    s.retrieve("apples", k=1)  # bump apples' hit count
    s.add("fact about cherries", now=3.0)  # over cap -> evict least-used
    texts = s.texts()
    assert "fact about apples" in texts  # retrieved, so it survived
    assert len(texts) == 2


def test_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "sem.json")
    s = SemanticStore()
    s.add("the streamer's name is Marco", now=1.0)
    s.add("the streamer is from London", now=1.0)
    s.save_file(path)

    loaded = SemanticStore()
    loaded.load_file(path)
    assert sorted(loaded.texts()) == sorted(s.texts())


def test_load_missing_file_is_noop(tmp_path):
    s = SemanticStore()
    s.load_file(str(tmp_path / "does-not-exist.json"))
    assert len(s) == 0


def test_heuristic_extractor_pulls_durable_first_person_facts():
    ex = HeuristicFactExtractor()
    lines = [
        "My name is Marco",
        "I'm from London originally",
        "I always drink coffee before streaming",
        "I love jazz",
        "yeah ok whatever happened there",  # no durable fact
    ]
    facts = {f.text for f in asyncio.run(ex.extract(lines))}
    assert "the streamer's name is Marco" in facts
    assert "the streamer is from London" in facts
    assert any("coffee" in f for f in facts)
    assert any("jazz" in f for f in facts)
    assert not any("whatever" in f for f in facts)
