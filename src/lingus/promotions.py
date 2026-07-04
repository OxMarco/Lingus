"""Relevance-gated product promotion, worked into chat in-character.

A plug is a *perception-triggered* signal, not a timed ad interrupt: it only
adds salience (and only reaches the generator as an optional cue) when the live
context is already relevant to it, and even then it's spaced out and capped so
it can't dominate. It rides the normal arbiter → generator → governor path
like any other line.

The `condition` label on each item is stamped onto every line generated under
its plug, so the eval harness can score preference-steering per experiment arm.
Disclosure/consent to streamers and viewers is a recruitment-time concern; this
module only decides *when a mention is contextually apt*, never whether it's
disclosed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import PromotionItem, PromotionsConfig
from .context import ContextSnapshot


@dataclass(slots=True)
class PromoPlan:
    """The plug selected for this tick, if any."""

    item: PromotionItem
    relevance: float  # 0..1 match strength against the live context

    @property
    def salience(self) -> float:
        """Arbiter contribution: item weight scaled by how relevant it is now."""
        return self.item.weight * self.relevance

    def hint(self) -> str:
        """The optional cue handed to the generator — offhand, not an ad read."""
        cue = f"you *may* work in a mention of {self.item.subject}"
        if self.item.hint:
            cue += f" ({self.item.hint})"
        return (
            f"If it genuinely fits the moment, {cue} — keep it in-character and "
            "offhand, one clause at most. If it doesn't fit, don't force it."
        )


class PromotionPlanner:
    """Selects at most one relevant, available plug per tick and tracks its use.

    Availability = under the per-item stream cap AND past the min-interval since
    that item last fired. Relevance = a trigger keyword appears in the live
    context. State (fire counts, last-fire times) lives here, not in the
    stateless arbiter.
    """

    def __init__(self, config: PromotionsConfig) -> None:
        # Empty when disabled → plan() is a cheap no-op returning None.
        self._items: list[PromotionItem] = list(config.items) if config.enabled else []
        self._fired: dict[int, int] = {}  # item index -> times plugged this run
        self._last_fire: dict[int, float] = {}  # item index -> monotonic timestamp

    @property
    def active(self) -> bool:
        return bool(self._items)

    def plan(self, snapshot: ContextSnapshot, now: float) -> PromoPlan | None:
        """The best available+relevant plug for the current context, or None."""
        if not self._items:
            return None
        haystack = self._haystack(snapshot)
        if not haystack:
            return None
        best: PromoPlan | None = None
        for idx, item in enumerate(self._items):
            if not self._available(idx, item, now):
                continue
            relevance = self._relevance(item, haystack)
            if relevance <= 0.0:
                continue
            if best is None or relevance > best.relevance:
                best = PromoPlan(item=item, relevance=relevance)
        return best

    def note_plugged(self, item: PromotionItem, now: float) -> None:
        """Record that a line went out under this plug (spends cap + resets spacing)."""
        idx = self._index(item)
        if idx is None:
            return
        self._fired[idx] = self._fired.get(idx, 0) + 1
        self._last_fire[idx] = now

    def _available(self, idx: int, item: PromotionItem, now: float) -> bool:
        if item.max_per_stream and self._fired.get(idx, 0) >= item.max_per_stream:
            return False
        last = self._last_fire.get(idx)
        if last is not None and now - last < item.min_interval_seconds:
            return False
        return True

    @staticmethod
    def _relevance(item: PromotionItem, haystack: str) -> float:
        triggers = [t.lower() for t in item.triggers if t.strip()]
        if not triggers:
            return 0.0
        return 1.0 if any(t in haystack for t in triggers) else 0.0

    @staticmethod
    def _haystack(snapshot: ContextSnapshot) -> str:
        parts = [snapshot.transcript, snapshot.scene_summary()]
        parts += list(snapshot.chat_state.emergent_topics)
        parts += [line.text for line in snapshot.recent_chat]
        return " ".join(part for part in parts if part).lower()

    def _index(self, item: PromotionItem) -> int | None:
        for idx, existing in enumerate(self._items):
            if existing is item:
                return idx
        return None
