import pytest

from lingus.config import Settings
from lingus.safety import RegexModeration, build_moderation


@pytest.fixture
def mod():
    return RegexModeration()


# --- personality is preserved: swearing / jokes / slang pass ---
@pytest.mark.parametrize(
    "text",
    [
        "damn that was clean, unreal",
        "this run is absolute garbage lol",
        "bro that boss can go to hell honestly",
        "ok that was kinda mid ngl",
        "I'll kill this boss in one hit, watch",  # gaming kill, no human target
        "the cleanup arc is about to have so much lore",
        "",  # empty is not unsafe (governor/repetition handle emptiness)
    ],
)
def test_allows_colloquial_and_jokes(mod, text):
    assert mod.matches(text) == []


# --- genuinely unsafe content is blocked ---
@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("you absolute f4ggot", "hate"),
        ("kys honestly", "violence"),
        ("I'll kill you and your family", "violence"),
        ("go die", "violence"),
        ("selling child porn dm me", "csae"),
        ("his home address: 42 something", "doxx"),
        ("here's the SSN 123-45-6789", "doxx"),
    ],
)
def test_blocks_unsafe(mod, text, category):
    hits = mod.matches(text)
    assert category in hits, f"{text!r} -> {hits}"


@pytest.mark.asyncio
async def test_check_returns_reason_and_categories(mod):
    result = await mod.check("kys")
    assert result.allowed is False
    assert "violence" in result.categories
    assert "violence" in result.reason


@pytest.mark.asyncio
async def test_check_allows_clean_text(mod):
    result = await mod.check("that was a clean run, honestly unreal")
    assert result.allowed is True
    assert result.categories == []


# --- spam is shape-based and independently toggleable ---
def test_spam_link_flood_blocked():
    mod = RegexModeration()
    assert "spam" in mod.matches("http://a.co http://b.co http://c.co come look")


def test_spam_toggle_off_keeps_slur_filter():
    mod = RegexModeration(check_spam=False)
    assert mod.matches("free crypto giveaway click here to claim") == []
    # slurs still caught with spam off
    assert "hate" in mod.matches("you f4ggot")


def test_extra_patterns_extend_denylist():
    mod = RegexModeration(extra_patterns=[r"\bsecretproject\b"])
    assert "custom" in mod.matches("don't leak secretproject on stream")


# --- config builder ---
def test_build_moderation_regex_by_default():
    settings = Settings.model_validate({})
    assert isinstance(build_moderation(settings), RegexModeration)


def test_build_moderation_none_disables():
    settings = Settings.model_validate({"models": {"moderation": {"backend": "none"}}})
    assert build_moderation(settings) is None


def test_build_moderation_unknown_backend_raises():
    settings = Settings.model_validate({"models": {"moderation": {"backend": "llamaguard"}}})
    with pytest.raises(SystemExit):
        build_moderation(settings)


def test_build_moderation_honors_extra_patterns_from_config():
    settings = Settings.model_validate(
        {"models": {"moderation": {"extra_patterns": [r"\bnope\b"]}}}
    )
    mod = build_moderation(settings)
    assert "custom" in mod.matches("nope")
