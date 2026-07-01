from lingus.config import Settings
from lingus.control import ControlState
from lingus.output_governor import OutputGovernor


def _settings() -> Settings:
    return Settings.model_validate(
        {
            "arbiter": {"fire_threshold": 1.0, "cooldown_seconds": 20.0},
            "output": {
                "max_chars": 200,
                "min_seconds_between_posts": 8.0,
                "posts_per_minute": 6.0,
            },
            "models": {"llm": {"temperature": 0.9}},
        }
    )


def test_frequency_midpoint_is_neutral():
    c = ControlState(_settings())
    c.frequency = 0.5
    assert c.values()["_effective_threshold"] == 1.0
    assert c.values()["_effective_cooldown"] == 20.0


def test_higher_frequency_lowers_threshold_and_cooldown():
    c = ControlState(_settings())
    c.frequency = 1.0
    v = c.values()
    assert v["_effective_threshold"] < 1.0  # chattier => easier to fire
    assert v["_effective_cooldown"] < 20.0
    c.frequency = 0.0
    v = c.values()
    assert v["_effective_threshold"] > 1.0  # reticent => higher bar
    assert v["_effective_cooldown"] > 20.0


def test_set_clamps_to_schema_bounds_and_coerces_int():
    c = ControlState(_settings())
    c.set("max_chars", 9999)  # above max (500)
    assert c.max_chars == 500
    c.set("max_chars", 33.7)  # coerced to int
    assert c.max_chars == 34
    c.set("frequency", 5.0)  # above max (1.0)
    assert c.frequency == 1.0


def test_set_chat_enabled_bool():
    c = ControlState(_settings())
    c.set("chat_enabled", False)
    assert c.chat_enabled is False


def test_trends_enabled_is_a_toggle_seeded_from_config():
    c = ControlState(_settings())
    assert c.trends_enabled is True  # config default
    assert "trends_enabled" in c.values()
    assert any(s["key"] == "trends_enabled" and s["kind"] == "bool" for s in c.schema())
    c.set("trends_enabled", False)
    assert c.trends_enabled is False


def test_set_unknown_key_is_ignored():
    c = ControlState(_settings())
    c.set("nonexistent", 123)
    assert not hasattr(c, "nonexistent")


def test_apply_pushes_onto_governor():
    c = ControlState(_settings())
    gov = OutputGovernor(max_chars=200, min_seconds_between_posts=8.0, posts_per_minute=6.0)

    class _Arb:
        fire_threshold = 0.0
        cooldown_seconds = 0.0
        min_seconds_between_posts = 0.0

    captured = {}

    class _Gen:
        def set_temperature(self, t):
            captured["t"] = t

    c.set("max_chars", 120)
    c.set("posts_per_minute", 12.0)
    c.set("temperature", 1.3)
    arb = _Arb()
    c.apply(arb, gov, _Gen())

    assert gov.max_chars == 120
    assert gov._bucket.rate == 12.0 / 60.0
    assert captured["t"] == 1.3
    assert arb.fire_threshold == c.values()["_effective_threshold"]
