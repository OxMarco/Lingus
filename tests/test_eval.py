import pytest

from lingus.arbiter import ArbiterDecision
from lingus.config import Settings
from lingus.context import ChatLine
from lingus.eval import (
    CollectingMonitor,
    EvalSample,
    HeuristicJudge,
    LLMJudge,
    Score,
    _parse_llm_score,
    evaluate_segment,
)
from lingus.models.base import ChatTurn
from lingus.monitor import TickReport
from lingus.persona.schema import Lexicon, PersonaSpec


def _persona(**kw):
    base = dict(name="Gremlin", voice="dry chat goblin")
    base.update(kw)
    return PersonaSpec(**base)


def _sample(text, *, context="", reasons=None):
    return EvalSample(
        t=0.0,
        text=text,
        reasons=reasons or [],
        transcript_tail=context,
    )


# --- Score ---
def test_overall_is_mean_of_axes():
    s = Score(in_character=1.0, not_generic=0.4, not_repetitive=0.7)
    assert s.overall == pytest.approx((1.0 + 0.4 + 0.7) / 3)


# --- HeuristicJudge: in_character ---
@pytest.mark.asyncio
async def test_assistant_voice_tanks_in_character():
    judge = HeuristicJudge()
    good = await judge.score(_sample("called it, humility speedrun"), _persona(), [])
    bad = await judge.score(
        _sample("Great question! I'm happy to help, how can I help?"), _persona(), []
    )
    assert bad.in_character < good.in_character
    assert bad.in_character < 0.4


@pytest.mark.asyncio
async def test_avoided_phrase_penalized():
    persona = _persona(lexicon=Lexicon(avoid=["As an AI"]))
    judge = HeuristicJudge()
    score = await judge.score(_sample("As an AI I think that's neat"), persona, [])
    assert "avoided" in score.notes
    assert score.in_character < 0.6


@pytest.mark.asyncio
async def test_persona_lexicon_rewarded():
    persona = _persona(lexicon=Lexicon(use=["ngl"], catchphrases=["called it"]))
    judge = HeuristicJudge()
    plain = await judge.score(_sample("that boss went down hard"), persona, [])
    voiced = await judge.score(_sample("called it ngl that boss went down"), persona, [])
    assert voiced.in_character >= plain.in_character


# --- HeuristicJudge: not_generic ---
@pytest.mark.asyncio
async def test_pure_filler_scores_low_not_generic():
    judge = HeuristicJudge()
    filler = await judge.score(_sample("lol nice cool wow"), _persona(), [])
    assert filler.not_generic < 0.5
    assert "filler" in filler.notes


@pytest.mark.asyncio
async def test_grounded_message_scores_higher_not_generic():
    judge = HeuristicJudge()
    grounded = await judge.score(
        _sample("that chocolate cake is doomed", context="streamer bakes a chocolate cake"),
        _persona(),
        [],
    )
    assert grounded.not_generic > 0.8
    assert "grounded" in grounded.notes


# --- HeuristicJudge: not_repetitive ---
@pytest.mark.asyncio
async def test_repeated_line_scores_low_not_repetitive():
    judge = HeuristicJudge()
    line = "the cleanup arc is about to have so much lore"
    score = await judge.score(_sample(line), _persona(), [line])
    assert score.not_repetitive < 0.2
    assert "echoes" in score.notes


@pytest.mark.asyncio
async def test_distinct_line_scores_high_not_repetitive():
    judge = HeuristicJudge()
    score = await judge.score(
        _sample("humility speedrun any%"), _persona(), ["totally unrelated words here"]
    )
    assert score.not_repetitive > 0.8


# --- CollectingMonitor ---
def _tick(*, posted=None, dropped=None, reasons=None):
    return TickReport(
        t=0.0,
        decision=ArbiterDecision(
            should_reply=posted is not None, score=1.0, reasons=reasons or [], threshold=1.0
        ),
        mood=0.0,
        n_events=1,
        transcript_tail="stuff happened",
        recent_chat=[ChatLine(author="viewer", text="hi")],
        scene_summary="a scene",
        posted=posted,
        dropped=dropped,
    )


def test_collecting_monitor_captures_posts_and_counts():
    mon = CollectingMonitor()
    mon.on_tick(_tick())  # nothing posted
    mon.on_tick(_tick(posted="hello", reasons=["hype"]))
    mon.on_tick(_tick(dropped="dupe"))
    assert mon.n_ticks == 3
    assert mon.n_dropped == 1
    assert [s.text for s in mon.samples] == ["hello"]
    assert mon.samples[0].reasons == ["hype"]
    assert mon.samples[0].recent_chat == ["viewer: hi"]


# --- LLM judge parsing ---
def test_parse_llm_score_maps_1_to_10_onto_0_1():
    score = _parse_llm_score(
        '{"in_character": 10, "not_generic": 1, "not_repetitive": 5, "note": "ok"}'
    )
    assert score.in_character == pytest.approx(1.0)
    assert score.not_generic == pytest.approx(0.0)
    assert score.not_repetitive == pytest.approx((5 - 1) / 9)
    assert score.notes == "ok"


def test_parse_llm_score_tolerates_surrounding_prose():
    score = _parse_llm_score(
        'Sure! {"in_character": 8, "not_generic": 7, "not_repetitive": 9} hope that helps'
    )
    assert score.in_character == pytest.approx((8 - 1) / 9)


def test_parse_llm_score_rejects_out_of_range():
    with pytest.raises(ValueError):
        _parse_llm_score('{"in_character": 11, "not_generic": 5, "not_repetitive": 5}')


class _FakeBackend:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    async def generate(self, messages, **opts):
        self.calls += 1
        return self.reply


@pytest.mark.asyncio
async def test_llm_judge_uses_backend_reply():
    backend = _FakeBackend('{"in_character": 9, "not_generic": 8, "not_repetitive": 10}')
    judge = LLMJudge(backend)
    score = await judge.score(_sample("nice line"), _persona(), [])
    assert backend.calls == 1
    assert score.in_character == pytest.approx((9 - 1) / 9)


@pytest.mark.asyncio
async def test_llm_judge_falls_back_on_bad_json():
    backend = _FakeBackend("I refuse to answer in JSON")
    judge = LLMJudge(backend)  # default fallback = HeuristicJudge
    score = await judge.score(_sample("lol nice cool wow"), _persona(), [])
    # heuristic fallback ran: pure filler -> low not_generic
    assert score.not_generic < 0.5


# --- end-to-end over a recorded segment ---
@pytest.mark.asyncio
async def test_evaluate_segment_scores_template_output():
    settings = Settings.model_validate(
        {"platform": "file_replay", "arbiter": {"weights": {"streamer_mishap": 1.1}}}
    )
    persona = PersonaSpec(name="Gremlin", voice="brief")
    report = await evaluate_segment(settings, persona, "tests/samples/cake", speed=100.0)

    assert report.n_posted == 1
    scored = report.scored[0]
    assert "chocolate" in scored.sample.text
    # the canned line is grounded, in-character (no assistant voice), and unique
    assert scored.score.overall > 0.6
    assert 0.0 <= report.mean_overall <= 1.0


@pytest.mark.asyncio
async def test_evaluate_segment_does_not_touch_semantic_file(tmp_path):
    marker = tmp_path / "semantic.json"
    settings = Settings.model_validate(
        {
            "platform": "file_replay",
            "memory": {"semantic_enabled": True, "semantic_path": str(marker)},
        }
    )
    persona = PersonaSpec(name="Gremlin", voice="brief")
    await evaluate_segment(settings, persona, "tests/samples/cake", speed=100.0)
    # eval disables semantic memory for reproducibility -> no file written
    assert not marker.exists()


@pytest.mark.asyncio
async def test_llm_judge_accepts_chat_turn_messages():
    # guard: the judge must send ChatTurns, not raw dicts
    seen = {}

    class Recorder:
        async def generate(self, messages, **opts):
            seen["messages"] = messages
            return '{"in_character": 5, "not_generic": 5, "not_repetitive": 5}'

    await LLMJudge(Recorder()).score(_sample("hi"), _persona(), [])
    assert all(isinstance(m, ChatTurn) for m in seen["messages"])
