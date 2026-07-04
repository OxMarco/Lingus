"""The observability seam between the loop and any dashboard.

The cognition loop emits one `TickReport` per tick to a `Monitor`. This keeps
the core dependency-free: the default `NullMonitor` does nothing, and the Rich
terminal dashboard (see `dashboard.py`, behind the optional `dashboard` extra)
is just one implementation of the protocol. A future web overlay would be
another — both read the same report stream, neither couples to the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from .arbiter import ArbiterDecision
from .context import ChatLine

ReasonKind = Literal["positive", "blocking", "info"]

POSITIVE_REASONS = frozenset(
    {
        "direct_address",
        "question",
        "hype",
        "scene_event",
        "streamer_mishap",
        "lull",
        "promo",
    }
)
BLOCKING_REASONS = frozenset({"rate_limited", "stale_trigger"})


def reason_kind(reason: str) -> ReasonKind:
    if reason in POSITIVE_REASONS:
        return "positive"
    if reason in BLOCKING_REASONS:
        return "blocking"
    return "info"


def format_clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


@dataclass(slots=True)
class TickReport:
    """Everything a dashboard needs to render one cognition tick."""

    t: float  # monotonic timestamp of the tick
    decision: ArbiterDecision
    mood: float
    n_events: int
    transcript_tail: str
    recent_chat: list[ChatLine] = field(default_factory=list)
    episodic_summary: str = ""
    episodic_history: list[str] = field(default_factory=list)
    semantic_facts: list[str] = field(default_factory=list)
    scene_summary: str = ""
    posted: str | None = None  # message posted this tick, if any
    dropped: str | None = None  # message generated then dropped (stale/dup), if any
    condition: str = ""  # experiment-arm label when the post went out under a plug


@runtime_checkable
class Monitor(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def on_tick(self, report: TickReport) -> None: ...


class NullMonitor:
    """Default no-op monitor: the loop runs identically with no dashboard."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def on_tick(self, report: TickReport) -> None:
        pass
