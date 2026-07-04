import random

from lingus.config import HumanizerConfig
from lingus.humanizer import _QWERTY_NEIGHBORS, Humanizer, build_humanizer


def test_em_dash_becomes_comma_by_default():
    h = Humanizer()
    assert h.humanize("it was great — really great") == "it was great, really great"
    # unspaced em-dash (the classic AI style) too
    assert h.humanize("wait—what") == "wait, what"


def test_em_dash_replacement_is_configurable():
    h = Humanizer(em_dash_replacement=" - ")
    assert h.humanize("done—next") == "done - next"


def test_spaced_en_dash_treated_as_clause_break_but_ranges_preserved():
    h = Humanizer()
    assert h.humanize("nice – then bad") == "nice, then bad"
    # a numeric range uses an unspaced en-dash and must survive untouched
    assert h.humanize("scored 3–4 last night") == "scored 3–4 last night"


def test_smart_quotes_straightened():
    h = Humanizer()
    assert h.humanize("he said “hi” to ‘us’") == 'he said "hi" to \'us\''


def test_ellipsis_normalized():
    h = Humanizer()
    assert h.humanize("well… maybe") == "well... maybe"


def test_no_double_or_dangling_punctuation_after_replacement():
    h = Humanizer()
    # replacement sitting next to existing terminal punctuation should not double up
    assert h.humanize("sure—.") == "sure."
    assert h.humanize("ok — , then") == "ok, then"


def test_disabled_is_identity():
    h = Humanizer(enabled=False)
    text = "great — really “great”… ok"
    assert h.humanize(text) == text


def test_empty_and_plain_text_untouched():
    h = Humanizer()
    assert h.humanize("") == ""
    assert h.humanize("just normal chat, nothing fancy") == "just normal chat, nothing fancy"


def test_build_from_config_matches_defaults():
    h = build_humanizer(HumanizerConfig())
    assert h.enabled is True
    assert h.humanize("a—b") == "a, b"
    # typos are off by default, so a config-built humanizer never mangles words
    assert h.typo_enabled is False
    assert h.humanize("recognizable understanding conversation") == (
        "recognizable understanding conversation"
    )


def test_typos_only_touch_long_words_and_preserve_ends():
    h = Humanizer(typo_enabled=True, typo_rate=1.0, typo_min_word_len=7, rng=random.Random(0))
    out = h.humanize("i love recognizable stuff")
    words = out.split(" ")
    # short words untouched
    assert words[0] == "i" and words[1] == "love" and words[3] == "stuff"
    typoed = words[2]
    assert typoed != "recognizable"
    # first + last letter stay intact so the word remains readable, and the slip
    # only touches the interior (length changes by at most one — a drop or a
    # double — so it is never a wholesale corruption)
    assert typoed[0] == "r" and typoed[-1] == "e"
    assert abs(len(typoed) - len("recognizable")) <= 1
    assert typoed.isalpha()


def test_typo_slip_kinds_are_human_shaped():
    # Drive each slip kind directly and assert it looks like a real mistype.
    # drop: one interior letter vanishes, ends intact (word is one shorter and a
    # subsequence of the original — e.g. "healtiest" / "healthist")
    drop = Humanizer._apply_slip("drop", "healthiest", random.Random(0))
    assert len(drop) == len("healthiest") - 1
    assert drop[0] == "h" and drop[-1] == "t"
    it = iter("healthiest")
    assert all(c in it for c in drop)  # order-preserving subsequence
    # substitute: interior letter becomes a QWERTY neighbour (never a random one)
    sub = Humanizer._apply_slip("substitute", "healthiest", random.Random(1))
    assert len(sub) == len("healthiest") and sub != "healthiest"
    diff = [i for i, (a, b) in enumerate(zip(sub, "healthiest", strict=True)) if a != b]
    assert len(diff) == 1 and 0 < diff[0] < len(sub) - 1
    j = diff[0]
    assert sub[j] in _QWERTY_NEIGHBORS["healthiest"[j]]
    # double: an interior letter repeats, growing the word by one
    dbl = Humanizer._apply_slip("double", "healthiest", random.Random(2))
    assert len(dbl) == len("healthiest") + 1 and dbl[0] == "h" and dbl[-1] == "t"
    # transpose: two adjacent interior letters swap, length preserved
    tr = Humanizer._apply_slip("transpose", "healthiest", random.Random(3))
    assert len(tr) == len("healthiest") and sorted(tr) == sorted("healthiest")


def test_typos_capped_per_message():
    h = Humanizer(
        typo_enabled=True, typo_rate=1.0, typo_min_word_len=5,
        typo_max_per_message=1, rng=random.Random(1),
    )
    out = h.humanize("something different happened yesterday")
    changed = sum(
        1 for a, b in zip(
            out.split(" "), "something different happened yesterday".split(" "), strict=True
        )
        if a != b
    )
    assert changed == 1


def test_typos_disabled_when_rate_zero_or_flag_off():
    text = "recognizable understanding"
    assert Humanizer(typo_enabled=True, typo_rate=0.0).humanize(text) == text
    assert Humanizer(typo_enabled=False, typo_rate=1.0).humanize(text) == text


def test_typos_skip_non_alpha_tokens():
    # a URL and a mention are long but must never be corrupted into garbage
    h = Humanizer(
        typo_enabled=True, typo_rate=1.0, typo_min_word_len=5, rng=random.Random(3)
    )
    out = h.humanize("see https://example.com/page and @streamer_name ok")
    assert "https://example.com/page" in out
    assert "@streamer_name" in out


def test_introduce_typos_flag_gates_only_the_typo_pass():
    h = Humanizer(typo_enabled=True, typo_rate=1.0, typo_min_word_len=7, rng=random.Random(0))
    # punctuation cleanup still runs with typos suppressed (the trend-mirror path)
    assert h.humanize("great — recognizable", introduce_typos=False) == "great, recognizable"
