from lingus.output_governor import OutputGovernor


def _gov(**kw):
    base = dict(max_chars=40, min_seconds_between_posts=8.0, burst=2, posts_per_minute=6.0)
    base.update(kw)
    # Pin the bucket's clock so tests don't depend on wall time.
    return OutputGovernor(now=0.0, **base)


# --- length / truncation ---
def test_short_message_is_untouched():
    gov = _gov()
    text, truncated = gov.fit("be careful, stains love a main quest")
    assert text == "be careful, stains love a main quest"
    assert truncated is False


def test_truncates_at_sentence_boundary():
    gov = _gov(max_chars=40)
    text, truncated = gov.fit("That was clean. Honestly unreal. I take it all back now.")
    assert truncated is True
    assert text == "That was clean. Honestly unreal."
    assert len(text) <= 40


def test_truncates_at_word_boundary_with_ellipsis_when_no_sentence_end():
    gov = _gov(max_chars=20)
    text, truncated = gov.fit("the cleanup arc is about to have so much lore")
    assert truncated is True
    assert text.endswith("…")
    assert len(text) <= 20
    # never a mid-word cut: the char before the ellipsis is part of a whole word
    assert " " not in text[-2:-1] or text[:-1].endswith(text[:-1].split()[-1])


def test_truncation_never_exceeds_cap_even_for_one_giant_token():
    gov = _gov(max_chars=10)
    text, truncated = gov.fit("supercalifragilisticexpialidocious")
    assert truncated is True
    assert len(text) <= 10


# --- rate limiting ---
def test_first_post_is_admitted():
    gov = _gov()
    out = gov.admit("hello there", now=0.0)
    assert out.action == "post"
    assert out.reason == "ok"


def test_min_interval_blocks_immediate_second_post():
    gov = _gov(min_seconds_between_posts=8.0)
    assert gov.admit("first", now=0.0).action == "post"
    blocked = gov.admit("second", now=3.0)
    assert blocked.action == "drop"
    assert blocked.reason == "rate_limited_interval"
    # after the interval, it's allowed again
    assert gov.admit("third", now=9.0).action == "post"


def test_token_bucket_caps_a_burst():
    # burst=2, refills 6/min = 0.1/sec; min_interval tiny so the bucket is what bites.
    gov = _gov(burst=2, posts_per_minute=6.0, min_seconds_between_posts=0.0)
    assert gov.admit("a", now=0.0).action == "post"
    assert gov.admit("b", now=0.0).action == "post"
    third = gov.admit("c", now=0.0)
    assert third.action == "drop"
    assert third.reason == "rate_limited_bucket"
    # one token refills after 10s (0.1/sec)
    assert gov.admit("d", now=10.0).action == "post"


def test_dropped_post_does_not_consume_rate_budget():
    gov = _gov(burst=1, posts_per_minute=6.0, min_seconds_between_posts=0.0)
    assert gov.admit("a", now=0.0).action == "post"
    # bucket empty -> drop; that drop must not push _last_post forward either
    assert gov.admit("b", now=0.0).action == "drop"
    # token refills after 10s and we can post again
    assert gov.admit("c", now=10.0).action == "post"


def test_empty_message_is_dropped():
    gov = _gov()
    out = gov.admit("   ", now=0.0)
    assert out.action == "drop"
    assert out.reason == "empty"


# --- temporizer ---
def test_typing_delay_is_disabled_by_default():
    gov = _gov(
        typing_cps=10.0, typing_base_seconds=0.4, typing_min_seconds=0.8, typing_max_seconds=5.0
    )

    assert gov.typing_delay("a" * 1000) == 0.0


def test_typing_delay_grows_with_length_and_is_clamped():
    gov = _gov(
        typing_enabled=True,
        typing_cps=10.0,
        typing_base_seconds=0.4,
        typing_min_seconds=0.8,
        typing_max_seconds=5.0,
    )
    short = gov.typing_delay("hi")
    medium = gov.typing_delay("a" * 30)
    long = gov.typing_delay("a" * 1000)
    assert short == 0.8  # clamped up to the floor
    assert short < medium < 5.0
    assert long == 5.0  # clamped down to the ceiling
