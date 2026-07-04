from lingus.arbiter import SimpleArbiter
from lingus.context import build_context_snapshot
from lingus.world_state import Event, SceneState, WorldState


def test_arbiter_fires_for_streamer_mishap_with_scene_context():
    world = WorldState()
    world.update_scene(SceneState(activity="streamer is cooking a chocolate cake"))
    world.add_event(
        Event(source="speech", kind="transcript", payload={"text": "ah no i got a stain from it"})
    )
    snapshot = build_context_snapshot(world)
    arbiter = SimpleArbiter(
        fire_threshold=1.0,
        cooldown_seconds=20.0,
        min_seconds_between_posts=8.0,
        weights={"streamer_mishap": 1.1},
    )

    decision = arbiter.decide(
        snapshot,
        persona_name="Lingus",
        seconds_since_own_message=float("inf"),
    )

    assert decision.should_reply
    assert "streamer_mishap" in decision.reasons


def test_arbiter_respects_post_cooldown():
    world = WorldState()
    world.add_event(
        Event(
            source="chat",
            kind="message",
            payload={"author": "viewer", "text": "@Lingus can you explain that?"},
        )
    )
    snapshot = build_context_snapshot(world)
    arbiter = SimpleArbiter(
        fire_threshold=1.0,
        cooldown_seconds=20.0,
        min_seconds_between_posts=8.0,
    )

    decision = arbiter.decide(
        snapshot,
        persona_name="Lingus",
        seconds_since_own_message=1.0,
    )

    assert not decision.should_reply
    assert "rate_limited" in decision.reasons
    assert "cooldown" in decision.reasons


def _direct_address_world():
    world = WorldState()
    world.add_event(
        Event(
            source="chat",
            kind="message",
            payload={"author": "viewer", "text": "@Lingus can you explain that?"},
        )
    )
    return build_context_snapshot(world)


def test_direct_address_accepts_punctuation_in_chat_and_speech():
    arbiter = SimpleArbiter(
        fire_threshold=1.0,
        cooldown_seconds=20.0,
        min_seconds_between_posts=8.0,
    )

    for source, text in (
        ("chat", "Lingus, I think you're right"),
        ("chat", "hey @Lingus! you called it"),
        ("speech", "Lingus, I think you're right"),
        ("speech", "hey Lingus. good call"),
    ):
        world = WorldState()
        kind = "message" if source == "chat" else "transcript"
        payload = {"text": text}
        if source == "chat":
            payload["author"] = "viewer"
        world.add_event(Event(source=source, kind=kind, payload=payload))

        decision = arbiter.decide(
            build_context_snapshot(world),
            persona_name="Lingus",
            seconds_since_own_message=float("inf"),
        )

        assert decision.should_reply
        assert "direct_address" in decision.reasons


def test_direct_address_does_not_match_inside_other_words():
    world = WorldState()
    world.add_event(
        Event(
            source="speech",
            kind="transcript",
            payload={"text": "that linguistics take is right"},
        )
    )
    snapshot = build_context_snapshot(world)
    arbiter = SimpleArbiter(
        fire_threshold=1.0,
        cooldown_seconds=20.0,
        min_seconds_between_posts=8.0,
    )

    decision = arbiter.decide(
        snapshot,
        persona_name="Lingus",
        seconds_since_own_message=float("inf"),
    )

    assert not decision.should_reply
    assert "direct_address" not in decision.reasons


def test_effective_threshold_decays_after_speaking():
    arbiter = SimpleArbiter(
        fire_threshold=1.0,
        cooldown_seconds=20.0,
        min_seconds_between_posts=8.0,
        cooldown_bump=1.0,
    )
    just_spoke = arbiter.effective_threshold(0.0)
    a_while = arbiter.effective_threshold(20.0)
    rested = arbiter.effective_threshold(float("inf"))

    assert just_spoke > a_while > rested
    assert rested == 1.0  # decays back to the baseline bar
    assert just_spoke == 2.0  # baseline + full bump right after speaking


def test_moderate_signal_fires_when_rested_but_is_suppressed_right_after_speaking():
    # A bare mention (1.5) clears the baseline bar (1.0) but not the raised one (~1.64).
    world = WorldState()
    world.add_event(
        Event(
            source="chat",
            kind="message",
            payload={"author": "viewer", "text": "@Lingus that was wild"},
        )
    )
    snapshot = build_context_snapshot(world)
    arbiter = SimpleArbiter(
        fire_threshold=1.0,
        cooldown_seconds=20.0,
        min_seconds_between_posts=8.0,
        cooldown_bump=1.0,
    )

    rested = arbiter.decide(
        snapshot, persona_name="Lingus", seconds_since_own_message=float("inf")
    )
    # 9s: past the hard rate-limit floor, but the bar is still elevated.
    fresh = arbiter.decide(snapshot, persona_name="Lingus", seconds_since_own_message=9.0)

    assert rested.should_reply
    assert not fresh.should_reply  # same signal, suppressed by the raised bar
    assert "rate_limited" not in fresh.reasons  # blocked by threshold, not the floor


def test_direct_question_breaks_through_an_active_cooldown():
    # A high-salience signal (mention + question) should clear even the raised bar.
    snapshot = _direct_address_world()
    arbiter = SimpleArbiter(
        fire_threshold=1.0,
        cooldown_seconds=20.0,
        min_seconds_between_posts=8.0,
        cooldown_bump=1.0,
    )

    decision = arbiter.decide(snapshot, persona_name="Lingus", seconds_since_own_message=10.0)

    assert decision.should_reply
    assert "cooldown" in decision.reasons  # still within cooldown, but broke through


def test_mood_lowers_the_bar():
    arbiter = SimpleArbiter(
        fire_threshold=1.0,
        cooldown_seconds=20.0,
        min_seconds_between_posts=8.0,
        mood_threshold_gain=0.3,
    )
    calm = arbiter.effective_threshold(float("inf"), mood=0.0)
    energized = arbiter.effective_threshold(float("inf"), mood=1.0)

    assert energized < calm


def test_lull_pressure_needs_active_stream_and_builds_with_silence():
    arbiter = SimpleArbiter(
        fire_threshold=1.0,
        cooldown_seconds=20.0,
        min_seconds_between_posts=8.0,
        lull_after_seconds=25.0,
        weights={"conversational_lull": 0.4},
    )

    # Active stream (transcript present) + a modest, non-question trigger.
    active = WorldState()
    active.add_event(Event(source="speech", kind="transcript", payload={"text": "ok so anyway"}))
    active_snap = build_context_snapshot(active)
    quiet = arbiter.decide(active_snap, persona_name="Lingus", seconds_since_own_message=10.0)
    long_lull = arbiter.decide(active_snap, persona_name="Lingus", seconds_since_own_message=45.0)

    assert "lull" not in quiet.reasons  # not silent long enough yet
    assert "lull" in long_lull.reasons
    assert long_lull.score > quiet.score  # silence adds pressure


def test_lull_does_not_fire_before_the_bot_has_ever_spoken():
    arbiter = SimpleArbiter(
        fire_threshold=1.0,
        cooldown_seconds=20.0,
        min_seconds_between_posts=8.0,
        lull_after_seconds=25.0,
    )
    world = WorldState()
    world.add_event(Event(source="speech", kind="transcript", payload={"text": "ok so anyway"}))
    snapshot = build_context_snapshot(world)

    decision = arbiter.decide(
        snapshot,
        persona_name="Lingus",
        seconds_since_own_message=float("inf"),
    )

    assert "lull" not in decision.reasons


def test_chat_engagement_lets_the_bot_banter_with_other_viewers():
    # A substantive line from a viewer (not aimed at the bot) is an occasion to
    # jump in. On its own 0.7 < 1.0, so it needs company — here a lull tips it.
    world = WorldState()
    world.add_event(
        Event(
            source="chat",
            kind="message",
            payload={"author": "viewer", "text": "honestly that build makes no sense to me"},
        )
    )
    snapshot = build_context_snapshot(world)
    arbiter = SimpleArbiter(
        fire_threshold=1.0,
        cooldown_seconds=20.0,
        min_seconds_between_posts=8.0,
        weights={"chat_engagement": 0.7},
    )

    decision = arbiter.decide(
        snapshot, persona_name="Lingus", seconds_since_own_message=float("inf")
    )
    assert "chat_engagement" in decision.reasons
    # High mood lowers the bar to 0.7, so a lone engageable line can clear it.
    hyped = arbiter.decide(
        snapshot, persona_name="Lingus", seconds_since_own_message=float("inf"), mood=1.0
    )
    assert hyped.should_reply


def test_chat_engagement_skips_short_pileons_and_direct_address():
    arbiter = SimpleArbiter(
        fire_threshold=1.0,
        cooldown_seconds=20.0,
        min_seconds_between_posts=8.0,
        weights={"chat_engagement": 0.7},
    )
    # Emote / one-word pile-on: the trend mirror's job, not chat_engagement.
    pileon = WorldState()
    pileon.add_event(
        Event(source="chat", kind="message", payload={"author": "a", "text": "POG POG"})
    )
    d1 = arbiter.decide(
        build_context_snapshot(pileon),
        persona_name="Lingus",
        seconds_since_own_message=float("inf"),
    )
    assert "chat_engagement" not in d1.reasons

    # Directed at the bot: that's direct_address, not butting into others' chat.
    addressed = WorldState()
    addressed.add_event(
        Event(
            source="chat",
            kind="message",
            payload={"author": "a", "text": "@Lingus what do you make of this"},
        )
    )
    d2 = arbiter.decide(
        build_context_snapshot(addressed),
        persona_name="Lingus",
        seconds_since_own_message=float("inf"),
    )
    assert "direct_address" in d2.reasons
    assert "chat_engagement" not in d2.reasons


def test_curiosity_is_tagged_on_a_lull_over_the_streamers_voice():
    arbiter = SimpleArbiter(
        fire_threshold=1.0,
        cooldown_seconds=20.0,
        min_seconds_between_posts=8.0,
        lull_after_seconds=25.0,
    )
    world = WorldState()
    world.add_event(Event(source="speech", kind="transcript", payload={"text": "ok so anyway"}))
    snapshot = build_context_snapshot(world)

    quiet = arbiter.decide(snapshot, persona_name="Lingus", seconds_since_own_message=10.0)
    lulled = arbiter.decide(snapshot, persona_name="Lingus", seconds_since_own_message=45.0)

    assert "curiosity" not in quiet.reasons  # no lull yet
    assert "lull" in lulled.reasons and "curiosity" in lulled.reasons


def test_mishap_scoring_uses_the_current_trigger_not_stale_transcript_tail():
    arbiter = SimpleArbiter(
        fire_threshold=1.0,
        cooldown_seconds=20.0,
        min_seconds_between_posts=8.0,
        weights={"streamer_mishap": 1.1},
    )
    world = WorldState()
    world.add_event(Event(source="speech", kind="transcript", payload={"text": "i got a stain"}))
    world.add_event(
        Event(source="chat", kind="message", payload={"author": "viewer", "text": "hello there"})
    )
    snapshot = build_context_snapshot(world)

    decision = arbiter.decide(
        snapshot,
        persona_name="Lingus",
        seconds_since_own_message=float("inf"),
    )

    assert "streamer_mishap" not in decision.reasons
    assert not decision.should_reply
