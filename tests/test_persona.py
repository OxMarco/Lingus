from lingus.persona.loader import load_persona
from lingus.persona.schema import Mood


def test_default_persona_loads_with_exemplars():
    p = load_persona("src/lingus/persona/personas/default.yaml")
    assert p.name
    assert p.exemplar_bank, "persona should ship with concrete exemplars, not just adjectives"


def test_mood_is_bounded_and_decays_toward_baseline():
    m = Mood(value=0.0, baseline=0.0, minimum=-1.0, maximum=1.0, decay_per_minute=0.2)
    m.nudge(5.0)
    assert m.value == 1.0  # clamped to max
    m.decay(minutes=10.0)
    assert m.value == 0.0  # decayed all the way back to baseline
