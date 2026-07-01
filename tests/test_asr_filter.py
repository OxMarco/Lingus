"""The ASR hallucination filter is pure and model-free, so we test it directly."""

from lingus.models.faster_whisper_asr import _is_hallucination


def test_keeps_confident_real_speech():
    assert not _is_hallucination("per me Carlo meritava di vincere", 0.05, -0.3)


def test_drops_known_phantom_phrase_even_when_confident():
    # The subtitle-credit family is emitted with high confidence on music, so the
    # denylist must override the logprob/no-speech check.
    assert _is_hallucination("Sottotitoli a cura di QTSS", 0.02, -0.2)
    assert _is_hallucination("Thanks for watching!", 0.1, -0.4)


def test_drops_low_confidence_segment():
    assert _is_hallucination("garbled", 0.9, -0.2)  # high no-speech prob
    assert _is_hallucination("garbled", 0.1, -1.5)  # very low avg logprob


def test_marker_match_is_case_insensitive():
    assert _is_hallucination("SOTTOTITOLI E REVISIONE", 0.0, 0.0)
