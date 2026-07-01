"""RepetitionGuard: self-memory dedup + bit-fatigue."""

from lingus.memory import RepetitionGuard, jaccard, normalize
from lingus.persona.schema import Lexicon, PersonaSpec


def _persona(catchphrases: list[str]) -> PersonaSpec:
    return PersonaSpec(name="T", voice="dry", lexicon=Lexicon(catchphrases=catchphrases))


def test_normalize_strips_punctuation_and_case():
    assert normalize("OK, that's CLEAN!!") == "ok that's clean"


def test_jaccard_bounds():
    assert jaccard(frozenset(), frozenset({"a"})) == 0.0
    assert jaccard(frozenset({"a", "b"}), frozenset({"a", "b"})) == 1.0


def test_exact_repeat_is_caught():
    g = RepetitionGuard()
    assert g.is_repetitive("noted, deeply suspicious", ["noted, deeply suspicious"])


def test_reworded_near_duplicate_is_caught():
    # The naive `reply in own_messages` check misses this; Jaccard catches it.
    g = RepetitionGuard(similarity_threshold=0.6)
    recent = ["ok that was actually clean, i take back nothing"]
    assert g.is_repetitive("ok that was clean, i take back nothing", recent)


def test_genuinely_different_reply_passes():
    g = RepetitionGuard(similarity_threshold=0.6)
    recent = ["ok that was actually clean, i take back nothing"]
    assert not g.is_repetitive("heat management boss fight, apparently", recent)


def test_empty_or_whitespace_is_repetitive():
    g = RepetitionGuard()
    assert g.is_repetitive("   ", [])
    assert g.is_repetitive("", ["anything"])


def test_bit_fatigue_blocks_reused_catchphrase_within_window():
    g = RepetitionGuard(fatigue_seconds=100.0)
    persona = _persona(["where's the fun in that"])
    line = "real answer? probably. but where's the fun in that"
    # First use is fine; record it.
    assert not g.is_repetitive(line, [], persona, now=0.0)
    g.note_post(line, persona, now=0.0)
    # A *differently worded* reply that reuses the same bit is now fatigued.
    assert g.is_repetitive("sure but where's the fun in that", [], persona, now=30.0)


def test_bit_fatigue_expires_after_window():
    g = RepetitionGuard(fatigue_seconds=100.0)
    persona = _persona(["where's the fun in that"])
    g.note_post("but where's the fun in that", persona, now=0.0)
    assert not g.is_repetitive("ok but where's the fun in that", [], persona, now=150.0)
