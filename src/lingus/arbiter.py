"""Cheap "should I speak?" scoring for the hot loop.

Two decisions, kept separate from "what do I say?":

  * **salience** — a weighted sum of pressure signals (direct address, open
    question, hype, scene event, mishap, lull). The weights *are* the
    temperament.
  * **threshold** — the bar salience must clear. It is *dynamic*: it jumps up
    right after the bot speaks and decays back to baseline, so the bot can't
    dominate, yet a strong signal (a direct question) can still break through a
    cooldown. Mood nudges the bar too — more energy lowers it.

speak  ⟺  salience(now)  >  effective_threshold(now)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .context import ContextSnapshot
from .world_state import Event, event_text

QUESTION_PREFIXES = (
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "can",
    "could",
    "should",
    "do",
    "does",
    "did",
    "is",
    "are",
)
MISHAP_TERMS = (
    "stain",
    "spilled",
    "spill",
    "burned",
    "burnt",
    "dropped",
    "broke",
    "broken",
    "oops",
    "oh no",
    "messed up",
    "stuck",
    "lost",
    "died",
    "fail",
)
HYPE_TERMS = ("pog", "pogg", "let's go", "lets go", "clutch", "clean", "insane")


@dataclass(slots=True)
class ArbiterDecision:
    should_reply: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    trigger_event: Event | None = None
    threshold: float = 0.0  # the (dynamic) bar `score` had to clear this tick


class SimpleArbiter:
    def __init__(
        self,
        *,
        fire_threshold: float,
        cooldown_seconds: float,
        min_seconds_between_posts: float,
        weights: dict[str, float] | None = None,
        max_trigger_age: float = 8.0,
        cooldown_bump: float = 1.0,
        lull_after_seconds: float = 25.0,
        mood_threshold_gain: float = 0.3,
    ) -> None:
        self.fire_threshold = fire_threshold
        self.cooldown_seconds = cooldown_seconds
        self.min_seconds_between_posts = min_seconds_between_posts
        self.weights = weights or {}
        self.max_trigger_age = max_trigger_age
        # How high the bar jumps the instant after the bot speaks; it decays back
        # to `fire_threshold` with time-constant `cooldown_seconds`.
        self.cooldown_bump = cooldown_bump
        # Lull pressure only starts building once the bot has been quiet this long.
        self.lull_after_seconds = lull_after_seconds
        # How strongly mood (energy, in [-1, 1]) moves the bar: +mood lowers it.
        self.mood_threshold_gain = mood_threshold_gain

    def effective_threshold(self, seconds_since_own_message: float, mood: float = 0.0) -> float:
        """The bar salience must clear right now.

        Baseline `fire_threshold`, plus a cooldown bump that decays since the
        bot last spoke, minus a mood term (more energy → readier to speak).
        """
        bump = self.cooldown_bump * math.exp(-seconds_since_own_message / self.cooldown_seconds)
        mood_adjust = self.mood_threshold_gain * mood
        return max(0.0, self.fire_threshold + bump - mood_adjust)

    def decide(
        self,
        snapshot: ContextSnapshot,
        *,
        persona_name: str,
        seconds_since_own_message: float,
        mood: float = 0.0,
    ) -> ArbiterDecision:
        threshold = self.effective_threshold(seconds_since_own_message, mood)
        event = snapshot.latest_event
        if event is None or event.source == "bot":
            return ArbiterDecision(False, 0.0, trigger_event=event, threshold=threshold)

        reasons: list[str] = []
        score = 0.0
        text = event_text(event)
        trigger_text = text.lower()

        if event.age(snapshot.now) > self.max_trigger_age:
            reasons.append("stale_trigger")

        if self._mentions_persona(text, persona_name):
            score += self._weight("direct_address", 1.5)
            reasons.append("direct_address")

        if self._looks_like_question(text):
            score += self._weight("unanswered_question", 1.2)
            reasons.append("question")

        if self._has_hype(text) or snapshot.chat_state.hype_level >= 0.7:
            score += self._weight("chat_hype_spike", 0.8)
            reasons.append("hype")

        if event.source == "scene" and snapshot.scene.last_event:
            score += self._weight("scene_event", 0.7)
            reasons.append("scene_event")

        if self._has_mishap(trigger_text):
            score += self._weight("streamer_mishap", 1.0)
            reasons.append("streamer_mishap")

        # Conversational lull: once the bot has been quiet a while AND the stream
        # is still active (there's speech to react to), a standing pressure ramps
        # up so the bot is readier to chime in on an otherwise-modest trigger.
        # It never fires into pure silence — there must still be a real event.
        lull = self._lull_pressure(seconds_since_own_message, snapshot)
        if lull > 0.0:
            score += self._weight("conversational_lull", 0.4) * lull
            reasons.append("lull")

        # Hard rate-limit floor: never post faster than this, regardless of score.
        if seconds_since_own_message < self.min_seconds_between_posts:
            reasons.append("rate_limited")
        # Informational only: the bar is still elevated from the last post.
        if seconds_since_own_message < self.cooldown_seconds:
            reasons.append("cooldown")

        blocked = {"stale_trigger", "rate_limited"} & set(reasons)
        should_reply = score >= threshold and not blocked
        return ArbiterDecision(should_reply, score, reasons, event, threshold)

    def _lull_pressure(self, seconds_since_own_message: float, snapshot: ContextSnapshot) -> float:
        """0..1 ramp: how much the silence is pushing the bot to speak."""
        if math.isinf(seconds_since_own_message):
            return 0.0
        if seconds_since_own_message < self.lull_after_seconds:
            return 0.0
        if not snapshot.transcript:  # nothing to react to → don't talk to the void
            return 0.0
        over = seconds_since_own_message - self.lull_after_seconds
        return min(1.0, over / self.cooldown_seconds)

    def _weight(self, key: str, default: float) -> float:
        return self.weights.get(key, default)

    @staticmethod
    def _mentions_persona(text: str, persona_name: str) -> bool:
        name = persona_name.strip()
        if not name:
            return False
        pattern = rf"(?<!\w)@?{re.escape(name)}(?!\w)"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        lowered = text.lower().strip()
        return lowered.endswith("?") or lowered.startswith(QUESTION_PREFIXES)

    @staticmethod
    def _has_hype(text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in HYPE_TERMS)

    @staticmethod
    def _has_mishap(text: str) -> bool:
        return any(term in text for term in MISHAP_TERMS)
