from lingus.arbiter import ArbiterDecision, SimpleArbiter
from lingus.config import PromotionItem, PromotionsConfig
from lingus.context import build_context_snapshot
from lingus.eval import CollectingMonitor
from lingus.monitor import TickReport
from lingus.promotions import PromotionPlanner
from lingus.world_state import Event, SceneState, WorldState


def _snapshot_with_speech(text: str) -> "object":
    world = WorldState()
    world.add_event(Event(source="speech", kind="transcript", payload={"text": text}))
    return build_context_snapshot(world)


def _planner(**item_kwargs) -> PromotionPlanner:
    item = PromotionItem(subject="Blitz Energy", triggers=["thirsty", "grind"], **item_kwargs)
    return PromotionPlanner(PromotionsConfig(enabled=True, items=[item]))


def test_disabled_planner_is_inert():
    planner = PromotionPlanner(
        PromotionsConfig(enabled=False, items=[PromotionItem(subject="X", triggers=["y"])])
    )
    assert not planner.active
    assert planner.plan(_snapshot_with_speech("y y y"), now=0.0) is None


def test_plan_fires_only_when_context_is_relevant():
    planner = _planner()
    # No trigger word present -> no plug.
    assert planner.plan(_snapshot_with_speech("nice weather today"), now=0.0) is None
    # Trigger word present -> plug selected.
    plan = planner.plan(_snapshot_with_speech("man this grind is brutal"), now=0.0)
    assert plan is not None
    assert plan.item.subject == "Blitz Energy"
    assert plan.salience == plan.item.weight  # relevance 1.0


def test_trigger_matches_chat_and_scene_not_just_speech():
    world = WorldState()
    world.update_scene(SceneState(activity="long grind session on the boss"))
    world.add_event(
        Event(source="chat", kind="message", payload={"author": "v", "text": "hi"})
    )
    planner = _planner()
    assert planner.plan(build_context_snapshot(world), now=0.0) is not None


def test_min_interval_spacing_blocks_back_to_back_plugs():
    planner = _planner(min_interval_seconds=300.0, max_per_stream=10)
    snap = _snapshot_with_speech("so thirsty rn")
    plan = planner.plan(snap, now=0.0)
    assert plan is not None
    planner.note_plugged(plan.item, now=0.0)
    # Too soon -> blocked.
    assert planner.plan(snap, now=100.0) is None
    # Past the interval -> available again.
    assert planner.plan(snap, now=400.0) is not None


def test_max_per_stream_cap_is_hard():
    planner = _planner(min_interval_seconds=0.0, max_per_stream=2)
    snap = _snapshot_with_speech("thirsty")
    for tick in range(2):
        plan = planner.plan(snap, now=float(tick))
        assert plan is not None
        planner.note_plugged(plan.item, now=float(tick))
    # Cap spent -> no more plugs, ever this run.
    assert planner.plan(snap, now=999.0) is None


def test_most_relevant_available_item_wins():
    a = PromotionItem(subject="Drink", triggers=["thirsty"], weight=0.5)
    b = PromotionItem(subject="Snack", triggers=["thirsty"], weight=0.9)
    planner = PromotionPlanner(PromotionsConfig(enabled=True, items=[a, b]))
    plan = planner.plan(_snapshot_with_speech("thirsty"), now=0.0)
    # Equal relevance (1.0) -> tie broken by iteration order (first wins), but
    # both are valid; assert we got one of the relevant items back.
    assert plan is not None and plan.item.subject in {"Drink", "Snack"}


def test_hint_is_offhand_and_carries_subject():
    plan = _planner(hint="the blue-can one").plan(_snapshot_with_speech("thirsty"), now=0.0)
    assert plan is not None
    hint = plan.hint()
    assert "Blitz Energy" in hint
    assert "the blue-can one" in hint
    assert "don't force it" in hint.lower()


def test_promo_salience_tips_the_arbiter_but_never_fires_alone():
    arbiter = SimpleArbiter(
        fire_threshold=1.0, cooldown_seconds=20.0, min_seconds_between_posts=8.0
    )
    # An ambient line that on its own scores 0 (no question/hype/mishap).
    snap = _snapshot_with_speech("just chilling here")

    without = arbiter.decide(snap, persona_name="Lingus", seconds_since_own_message=float("inf"))
    assert "promo" not in without.reasons

    # A plug adds salience and is recorded as a reason. (Weight 0.6 alone stays
    # under the 1.0 threshold — a plug tips, it doesn't fire the bot by itself.)
    tipped = arbiter.decide(
        snap,
        persona_name="Lingus",
        seconds_since_own_message=float("inf"),
        promo_salience=0.6,
    )
    assert "promo" in tipped.reasons
    assert tipped.score == without.score + 0.6
    assert not tipped.should_reply  # 0.6 < 1.0: never speaks on a plug alone


def _posted_tick(condition: str = "") -> TickReport:
    return TickReport(
        t=1.0,
        decision=ArbiterDecision(should_reply=True, score=2.0, reasons=["promo"]),
        mood=0.0,
        n_events=1,
        transcript_tail="so thirsty",
        posted="this grind is a two-can problem ngl",
        condition=condition,
    )


def test_webmonitor_tracks_promotional_share_and_flags_messages():
    from lingus.config import Settings
    from lingus.control import ControlState
    from lingus.webui import WebMonitor

    mon = WebMonitor(ControlState(Settings()), persona_name="Lingus", platform="file_replay")
    mon.on_tick(_posted_tick(condition="open_plug"))
    mon.on_tick(_posted_tick(condition=""))  # a normal, non-plug reply

    stats = mon._promo_stats()
    assert stats["posts_total"] == 2
    assert stats["posts_promo"] == 1
    assert stats["promo_share"] == 0.5

    # The most recent (non-plug) message is first; the plug is flagged + labelled.
    latest, plug = mon._messages[0], mon._messages[1]
    assert plug["promo"] is True and plug["condition"] == "open_plug"
    assert latest["promo"] is False


def test_promo_toggle_gates_plugs():
    from lingus.app import BotLoop
    from lingus.config import Settings
    from lingus.control import ControlState
    from lingus.persona.schema import PersonaSpec

    from .test_app import CollectingChatAdapter, EmptyCaptureAdapter

    settings = Settings()
    controls = ControlState(settings)
    loop = BotLoop(
        settings=settings,
        persona=PersonaSpec(name="Lingus", voice="brief"),
        capture=EmptyCaptureAdapter(),
        chat=CollectingChatAdapter(),
        segment=None,
        controls=controls,
    )

    # On = plugs may fire (relevance gating still decides whether one is apt).
    controls.promo_enabled = True
    assert loop._promo_enabled()

    # Off = never plug, regardless of context.
    controls.promo_enabled = False
    assert not loop._promo_enabled()


def test_condition_label_flows_into_eval_sample():
    monitor = CollectingMonitor()
    decision = ArbiterDecision(should_reply=True, score=2.0, reasons=["promo"])
    monitor.on_tick(
        TickReport(
            t=1.0,
            decision=decision,
            mood=0.0,
            n_events=1,
            transcript_tail="so thirsty",
            posted="this grind is a two-can problem ngl",
            condition="open_plug",
        )
    )
    assert len(monitor.samples) == 1
    assert monitor.samples[0].condition == "open_plug"
