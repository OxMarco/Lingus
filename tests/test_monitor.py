import pytest

from lingus.app import BotLoop
from lingus.arbiter import ArbiterDecision
from lingus.config import Settings
from lingus.monitor import NullMonitor, TickReport
from lingus.persona.schema import PersonaSpec
from lingus.world_state import Event

from .test_app import CollectingChatAdapter, EmptyCaptureAdapter


class RecordingMonitor:
    """Monitor protocol stand-in that captures every tick for assertions."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.reports: list[TickReport] = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def on_tick(self, report: TickReport) -> None:
        self.reports.append(report)


def test_null_monitor_is_a_noop():
    monitor = NullMonitor()
    monitor.start()
    monitor.on_tick(
        TickReport(
            t=0.0,
            decision=ArbiterDecision(should_reply=False, score=0.0),
            mood=0.0,
            n_events=0,
            transcript_tail="",
        )
    )
    monitor.stop()  # must not raise


@pytest.mark.asyncio
async def test_loop_emits_tick_reports_with_post():
    settings = Settings.model_validate(
        {"platform": "file_replay", "arbiter": {"weights": {"streamer_mishap": 1.1}}}
    )
    monitor = RecordingMonitor()
    loop = BotLoop(
        settings=settings,
        persona=PersonaSpec(name="Gremlin", voice="brief"),
        capture=EmptyCaptureAdapter(),
        chat=CollectingChatAdapter(),
        segment="tests/samples/cake",
        replay_speed=100.0,
        monitor=monitor,
    )

    await loop._ingest_scene()
    await loop._ingest_transcript()
    await loop._cognition_tick()

    assert len(monitor.reports) == 1
    report = monitor.reports[0]
    assert report.posted == "be careful next time, chocolate stains are hard to remove"
    assert report.dropped is None
    assert report.decision.score >= report.decision.threshold


@pytest.mark.asyncio
async def test_loop_emits_report_even_when_holding():
    settings = Settings.model_validate({"platform": "file_replay"})
    monitor = RecordingMonitor()
    loop = BotLoop(
        settings=settings,
        persona=PersonaSpec(name="Gremlin", voice="brief"),
        capture=EmptyCaptureAdapter(),
        chat=CollectingChatAdapter(),
        segment=None,
        monitor=monitor,
    )
    # A plain bit of chatter that shouldn't clear the bar.
    loop.world.add_event(
        Event(source="chat", kind="message", payload={"author": "v", "text": "hello there"})
    )

    await loop._cognition_tick()

    assert len(monitor.reports) == 1
    assert monitor.reports[0].posted is None


def test_dashboard_renders_synthetic_ticks_headless():
    pytest.importorskip("rich")  # skip if the optional extra isn't installed
    from rich.console import Console

    from lingus.context import ChatLine
    from lingus.dashboard import RichDashboard, _mood_bar, _sparkline

    assert _sparkline([0.0, 0.5, 1.0], 0.0, 1.0)[0] != _sparkline([0.0, 0.5, 1.0], 0.0, 1.0)[-1]
    assert len(_mood_bar(0.0, width=10)) == 10

    dash = RichDashboard("Gremlin", "file_replay")
    for i in range(3):
        dash.on_tick(
            TickReport(
                t=float(i),
                decision=ArbiterDecision(
                    should_reply=i == 2,
                    score=float(i),
                    threshold=1.5,
                    reasons=["direct_address", "cooldown"],
                ),
                mood=0.3,
                n_events=i,
                transcript_tail="streamer says something",
                recent_chat=[ChatLine(author="viewer", text="lol")],
                scene_summary="cooking a cake",
                posted="hi" if i == 2 else None,
            )
        )

    # Render to a headless console; it must produce non-empty output without error.
    console = Console(file=None, width=100, record=True)
    console.print(dash._render())
    assert console.export_text().strip()
