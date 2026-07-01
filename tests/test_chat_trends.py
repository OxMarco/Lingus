import random

from lingus.chat_trends import ChatTrendDetector, canonicalize


def _detector(**kwargs):
    # follow_probability=1.0 + a seeded rng keeps the participation roll out of the
    # way for the threshold/phase/fatigue tests; the roll has its own test.
    defaults = dict(
        window_seconds=10.0,
        min_senders=4,
        min_fraction=0.35,
        follow_probability=1.0,
        cooldown_seconds=20.0,
        fatigue_seconds=90.0,
        rng=random.Random(0),
    )
    defaults.update(kwargs)
    return ChatTrendDetector(**defaults)


def _wave(detector, token, senders, now, *, prefix="viewer"):
    for i in range(senders):
        detector.observe(f"{prefix}{i}", token, now)


# --- canonicalization ---

def test_canonicalize_collapses_repeats_and_elongation():
    assert canonicalize("Pog Pog Pog") == "pog"
    assert canonicalize("POG") == "pog"
    assert canonicalize("loool") == "lol"
    assert canonicalize("loooool") == "lol"
    assert canonicalize("poggg") == "pog"
    assert canonicalize("x x x") == "x"
    assert canonicalize("   ") == ""
    # ordinary doubled letters survive (threshold is 3+)
    assert canonicalize("cool") == "cool"


# --- detection core ---

def test_follows_a_genuine_wave():
    det = _detector()
    _wave(det, "Pog", senders=5, now=100.0)
    trend = det.poll(101.0)
    assert trend is not None
    assert trend.token == "pog"
    assert trend.message == "Pog"  # verbatim original spelling, not the canonical key
    assert trend.senders == 5


def test_single_spammer_is_not_a_trend():
    # One viewer posting the same emote 20 times is spam, not convergence.
    det = _detector()
    for _ in range(20):
        det.observe("spammer", "Pog", 100.0)
    assert det.poll(101.0) is None


def test_below_fraction_threshold_does_not_fire():
    # 4 distinct senders say Pog, but they're drowned out by other chatter, so
    # Pog is well under min_fraction of the window.
    det = _detector(min_fraction=0.5)
    _wave(det, "Pog", senders=4, now=100.0)
    for i in range(20):
        det.observe(f"talker{i}", f"unrelated message {i}", 100.0)
    assert det.poll(101.0) is None


def test_picks_the_strongest_of_two_waves():
    det = _detector()
    _wave(det, "Pog", senders=4, now=100.0)
    _wave(det, "KEKW", senders=7, now=100.0, prefix="other")
    trend = det.poll(101.0)
    assert trend is not None
    assert trend.token == "kekw"


def test_does_not_follow_a_decaying_wave():
    det = _detector(window_seconds=10.0)
    # Big burst early in the window, nothing in the recent half → decaying.
    _wave(det, "Pog", senders=6, now=100.0)
    trend = det.poll(106.0)  # midpoint is 101.0; all 6 are in the older half
    assert trend is None


def test_stale_messages_fall_out_of_the_window():
    det = _detector(window_seconds=10.0)
    _wave(det, "Pog", senders=6, now=100.0)
    assert det.poll(120.0) is None  # window has fully advanced past the wave


# --- fatigue + cooldown + participation ---

def test_bit_fatigue_blocks_repeating_the_same_line():
    det = _detector(fatigue_seconds=90.0, cooldown_seconds=0.0)
    _wave(det, "Pog", senders=6, now=100.0)
    first = det.poll(101.0)
    assert first is not None
    det.mark_followed(first, 101.0)

    # Same wave still cresting a beat later, but we just did this bit.
    _wave(det, "Pog", senders=6, now=102.0)
    assert det.poll(103.0) is None


def test_cooldown_blocks_back_to_back_follows_of_different_trends():
    det = _detector(cooldown_seconds=20.0)
    _wave(det, "Pog", senders=6, now=100.0)
    first = det.poll(101.0)
    assert first is not None
    det.mark_followed(first, 101.0)

    # A fresh, strictly-strongest, rising wave — so the only thing stopping it is
    # the cooldown, not the decay or strength checks.
    _wave(det, "KEKW", senders=8, now=105.0, prefix="other")
    assert det.poll(106.0) is None  # within cooldown of the first follow


def test_participation_probability_zero_never_follows():
    det = _detector(follow_probability=0.0)
    _wave(det, "Pog", senders=6, now=100.0)
    assert det.poll(101.0) is None


def test_disabled_via_probability_is_decided_once_per_wave():
    # A "no" roll sticks for the wave's lifetime — it isn't re-rolled into a
    # "yes" on a later tick of the same wave.
    det = _detector(follow_probability=0.5, rng=random.Random(1))
    _wave(det, "Pog", senders=6, now=100.0)
    det.poll(101.0)
    decided = det._rolled.get("pog")
    assert decided is not None
    # Poll again on the same wave: the decision is unchanged.
    _wave(det, "Pog", senders=6, now=101.5)
    second = det.poll(102.0)
    assert (second is not None) == decided
