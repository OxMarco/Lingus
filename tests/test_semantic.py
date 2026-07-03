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
