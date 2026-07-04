"""Semantic / long-term memory: store, retrieval, persistence, extraction."""

import asyncio

from lingus.memory import HeuristicFactExtractor, LLMFactExtractor, SemanticStore


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


def test_scoped_store_hides_other_channels_facts(tmp_path):
    path = str(tmp_path / "sem.json")
    rakai = SemanticStore(channel="youtube_rakai")
    rakai.add("the streamer's channel is Rakai Live", now=1.0)
    rakai.save_file(path)

    spizee = SemanticStore(channel="youtube_spizee")
    spizee.load_file(path)
    spizee.add("the streamer's channel is Spizee 2", now=2.0)

    assert spizee.texts() == ["the streamer's channel is Spizee 2"]
    assert len(spizee) == 1
    assert all(f.text != "the streamer's channel is Rakai Live" for f in spizee.retrieve("", k=5))


def test_scoped_store_hides_legacy_unscoped_facts():
    s = SemanticStore()  # unscoped, like pre-channel-tagging data
    s.add("the streamer's channel is Rakai Live", now=1.0)
    s.channel = "youtube_spizee"
    assert s.texts() == []


def test_unscoped_store_sees_all_channels(tmp_path):
    path = str(tmp_path / "sem.json")
    rakai = SemanticStore(channel="youtube_rakai")
    rakai.add("fact about rakai", now=1.0)
    rakai.save_file(path)

    unscoped = SemanticStore()
    unscoped.load_file(path)
    unscoped.add("unscoped fact", now=2.0)
    assert sorted(unscoped.texts()) == ["fact about rakai", "unscoped fact"]


def test_dedup_is_per_channel(tmp_path):
    path = str(tmp_path / "sem.json")
    text = "the streamer loves spicy food"
    a = SemanticStore(channel="chan_a")
    a.add(text, now=1.0)
    a.save_file(path)

    b = SemanticStore(channel="chan_b")
    b.load_file(path)
    b.add(text, now=2.0)  # same words, different streamer -> a second fact
    b.save_file(path)

    merged = SemanticStore()
    merged.load_file(path)
    assert merged.texts() == [text, text]


def test_eviction_only_trims_own_channel(tmp_path):
    path = str(tmp_path / "sem.json")
    other = SemanticStore(channel="chan_a", max_facts=5)
    other.add("fact about apples", now=1.0)
    other.add("fact about bananas", now=2.0)
    other.save_file(path)

    mine = SemanticStore(channel="chan_b", max_facts=1)
    mine.load_file(path)
    mine.add("fact about cherries", now=3.0)
    mine.add("fact about dates", now=4.0)  # over chan_b's cap -> evict cherries only
    mine.save_file(path)

    merged = SemanticStore()
    merged.load_file(path)
    assert sorted(merged.texts()) == [
        "fact about apples",
        "fact about bananas",
        "fact about dates",
    ]


def test_persistence_roundtrip_keeps_channel(tmp_path):
    path = str(tmp_path / "sem.json")
    s = SemanticStore(channel="youtube_spizee")
    s.add("the streamer's channel is Spizee 2", now=1.0)
    s.save_file(path)

    loaded = SemanticStore()
    loaded.load_file(path)
    assert [f.channel for f in loaded.retrieve("", k=1)] == ["youtube_spizee"]


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


def test_load_malformed_fact_entries_skips_bad_rows(tmp_path):
    path = tmp_path / "sem.json"
    path.write_text(
        """
        {
          "facts": [
            "not an object",
            {"text": 123},
            {"text": "the streamer likes jazz", "subject": "streamer"},
            {"text": "bad", "unexpected": true}
          ]
        }
        """,
        encoding="utf-8",
    )
    s = SemanticStore()

    s.load_file(str(path))

    assert s.texts() == ["the streamer likes jazz"]


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


def test_llm_extractor_accepts_legacy_json_array():
    class FakeLLM:
        async def generate(self, messages, **opts):
            return '[{"fact": "the streamer likes jazz", "subject": "streamer"}]'

    facts = asyncio.run(LLMFactExtractor(FakeLLM()).extract(["I love jazz"]))

    assert [(f.text, f.subject) for f in facts] == [("the streamer likes jazz", "streamer")]


def test_llm_extractor_prefers_structured_backend():
    class FakeStructuredLLM:
        async def generate_structured(self, response_model, messages, **opts):
            return response_model(
                facts=[{"fact": "chat regular sam is from Malta", "subject": "sam"}]
            )

        async def generate(self, messages, **opts):  # pragma: no cover - should not run
            raise AssertionError("raw JSON fallback should not be used")

    facts = asyncio.run(LLMFactExtractor(FakeStructuredLLM()).extract(["Sam is from Malta"]))

    assert [(f.text, f.subject) for f in facts] == [("chat regular sam is from Malta", "sam")]
