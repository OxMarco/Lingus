"""Episodic memory: eviction capture + extractive summarization."""

import asyncio
import json

from lingus.memory import EpisodicArchive, ExtractiveSummarizer
from lingus.world_state import Event, WorldState


def _speech(text: str) -> Event:
    return Event(source="speech", kind="transcript", payload={"text": text})


def test_working_window_evicts_into_pending_summary():
    ws = WorldState(transcript_window=3)
    for i in range(5):
        ws.add_event(_speech(f"line {i}"))
    # Window holds the last 3; the first 2 were captured for summarization.
    assert ws.recent_transcript() == "line 2 line 3 line 4"
    assert ws.drain_pending_summary() == ["line 0", "line 1"]
    assert ws.pending_summary_count() == 0  # drained


def test_no_eviction_means_nothing_pending():
    ws = WorldState(transcript_window=10)
    ws.add_event(_speech("only line"))
    assert ws.pending_summary_count() == 0


def test_extractive_summarizer_keeps_salient_lines_and_dedupes():
    s = ExtractiveSummarizer(max_chars=800)
    lines = [
        "yeah ok",  # filler, dropped (no entity/number/?)
        "Tuchel might resign",  # entity -> kept
        "should England drop Kane?",  # question -> kept
        "Tuchel might resign",  # duplicate -> not piled up
    ]
    out = asyncio.run(s.summarize("", lines))
    assert "Tuchel might resign" in out
    assert "should England drop Kane?" in out
    assert out.count("Tuchel might resign") == 1
    assert "yeah ok" not in out


def test_eviction_to_summary_roundtrip():
    # Mirrors BotLoop._summarize_pending: long stream evicts -> drain -> summarize.
    # Window of 2 means all but the last two lines age out into pending.
    ws = WorldState(transcript_window=2)
    for line in [
        "welcome back everyone",
        "Tuchel named the squad today",
        "is Bellingham starting?",
        "they lost 3 nil last week",
        "yeah",  # still in working window? no — pushed out below
        "ok",
        "the Wembley crowd was loud",
    ]:
        ws.add_event(_speech(line))
    assert ws.recent_transcript() == "ok the Wembley crowd was loud"  # last 2 retained
    s = ExtractiveSummarizer(max_chars=800)
    summary = asyncio.run(s.summarize(ws.episodic_summary, ws.drain_pending_summary()))
    ws.set_episodic_summary(summary)
    assert "Tuchel named the squad today" in ws.episodic_summary
    assert "Bellingham" in ws.episodic_summary  # question -> salient
    assert "3 nil" in ws.episodic_summary  # number -> salient
    assert "yeah" not in ws.episodic_summary  # filler dropped


def test_extractive_summarizer_folds_into_prior_and_bounds_length():
    s = ExtractiveSummarizer(max_chars=40)
    prior = "earlier: they talked about the Euros"
    out = asyncio.run(s.summarize(prior, ["now Pulis joined the panel?"]))
    assert len(out) <= 40
    assert "Pulis" in out  # most recent salient content is preserved


def test_extractive_summarizer_hard_caps_single_long_salient_line():
    s = ExtractiveSummarizer(max_chars=30)
    out = asyncio.run(
        s.summarize("", ["Bellingham said something extremely detailed about the final?"])
    )
    assert len(out) <= 30
    assert out.endswith("…")


def test_episodic_archive_upserts_stream_summary_and_persists(tmp_path):
    path = str(tmp_path / "episodes.json")
    archive = EpisodicArchive(max_entries=5)
    archive.add("first summary", stream_id="stream-a", now=1.0)
    archive.add("updated summary", stream_id="stream-a", now=2.0)
    archive.save_file(path)

    loaded = EpisodicArchive(max_entries=5)
    loaded.load_file(path)

    assert len(loaded) == 1
    assert loaded.summaries() == ["updated summary"]


def test_episodic_archive_retrieves_by_relevance_then_recency():
    archive = EpisodicArchive(max_entries=5)
    archive.add("streamer baked a chocolate cake and spilled batter", stream_id="cake", now=1.0)
    archive.add("streamer played chess badly", stream_id="chess", now=2.0)

    top = archive.retrieve("cake stain", k=1)

    assert top[0].stream_id == "cake"
    assert top[0].hits == 1


def test_episodic_archive_skips_malformed_rows(tmp_path):
    path = tmp_path / "episodes.json"
    path.write_text(
        json.dumps(
            {
                "episodes": [
                    "bad",
                    {"summary": 123, "stream_id": "bad"},
                    {"summary": "valid summary", "stream_id": "stream-a"},
                    {"summary": "unexpected", "stream_id": "bad", "extra": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    archive = EpisodicArchive()

    archive.load_file(str(path))

    assert archive.summaries() == ["valid summary"]
