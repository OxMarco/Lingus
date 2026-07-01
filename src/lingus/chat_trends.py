"""Chat trend detection — "everyone's saying it, so the bot says it too."

When a chat converges on a single emote or short phrase (the classic Twitch
pile-on: a wall of ``Pog``, ``X``, ``LULW``, ``+2``), the characterful move is to
*join in* with the same line. Crucially this is a **mirror**, not a "what do I
say": routing it through the LLM would be slower, costlier, and would paraphrase
the emote instead of echoing it. So this module decides *which* line is trending
and the loop posts it verbatim, bypassing the generator (see
``BotLoop._maybe_follow_trend``).

The detector is pure and side-effect-free: it owns a sliding window of recent
chat and answers one question on demand — "is a wave cresting right now that we
should pile onto?". It never touches the world-state or the network. Time and
randomness are injected so the replay/eval harness can drive it deterministically.

What separates a real wave from spam is **distinct senders**: one viewer spamming
``LULW`` twenty times is not a trend (and is what the raid/spam detector is for);
twenty viewers each saying it once is. So the gate is on the count of *unique*
authors plus the share of all chat the line occupies, never on raw volume.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Literal

TrendPhase = Literal["rising", "crest", "decaying"]

# Collapse a run of 3+ identical characters to a single one, so elongation
# variants converge: "loool"/"loooool" → "lol", "poggg" → "pog", "!!!" → "!".
# Threshold 3 leaves ordinary doubled letters ("cool", "chill") untouched.
_RUN = re.compile(r"(.)\1{2,}")


def canonicalize(text: str) -> str:
    """Map a chat message to the key it should be bucketed under.

    Lowercases, collapses character-elongation, and reduces a message that is
    just one token repeated (``"pog pog pog"``, ``"x x x"``) down to that single
    token — so all the natural spellings of one pile-on land in the same bucket.
    Returns ``""`` for whitespace-only input.
    """
    collapsed = _RUN.sub(r"\1", text.strip().lower())
    tokens = collapsed.split()
    if not tokens:
        return ""
    if len(set(tokens)) == 1:  # "pog pog pog" / "x x x" → one canonical token
        return tokens[0]
    return " ".join(tokens)


@dataclass(slots=True)
class ChatTrend:
    """A cresting pile-on the bot may choose to echo."""

    token: str  # canonical bucket key
    message: str  # the verbatim line to post (most common original spelling)
    senders: int  # distinct authors saying it in the window
    count: int  # total messages in the window matching the bucket
    fraction: float  # share of all windowed chat this bucket occupies
    phase: TrendPhase


@dataclass(slots=True)
class _Entry:
    author: str
    text: str
    key: str
    ts: float


@dataclass(slots=True)
class _Bucket:
    count: int = 0
    senders: set[str] = field(default_factory=set)
    originals: dict[str, int] = field(default_factory=dict)  # original spelling → count


class ChatTrendDetector:
    """Sliding-window detector for chat convergence (pile-ons).

    Feed every real chat message to :meth:`observe`; call :meth:`poll` once per
    cognition tick to ask whether a wave is worth joining *right now*. On a
    successful post, call :meth:`mark_followed` so the same bit isn't echoed
    again until it goes stale (bit fatigue).
    """

    def __init__(
        self,
        *,
        window_seconds: float = 12.0,
        min_senders: int = 4,
        min_fraction: float = 0.35,
        max_token_chars: int = 24,
        follow_probability: float = 0.6,
        fatigue_seconds: float = 90.0,
        cooldown_seconds: float = 20.0,
        rng: random.Random | None = None,
    ) -> None:
        self.window_seconds = window_seconds
        self.min_senders = min_senders
        self.min_fraction = min_fraction
        self.max_token_chars = max_token_chars
        self.follow_probability = follow_probability
        self.fatigue_seconds = fatigue_seconds
        self.cooldown_seconds = cooldown_seconds
        self._rng = rng or random.Random()
        self._window: list[_Entry] = []
        # Decision memory: one participation roll per wave (so a single low roll
        # doesn't get re-rolled to a "yes" tick after tick), the last time each
        # token was actually followed (fatigue), and the last follow of anything.
        self._rolled: dict[str, bool] = {}
        self._fatigue: dict[str, float] = {}
        self._last_follow_ts: float = float("-inf")

    # --- ingest ---
    def observe(self, author: str, text: str, now: float) -> None:
        """Record one chat message into the sliding window."""
        key = canonicalize(text)
        if not key:
            return
        self._window.append(_Entry(author=author, text=text.strip(), key=key, ts=now))

    # --- decide ---
    def poll(self, now: float) -> ChatTrend | None:
        """Return a trend to pile onto right now, or ``None``.

        A trend qualifies when, within the window, a single bucket is carried by
        at least ``min_senders`` distinct authors and occupies at least
        ``min_fraction`` of all chat — and is not already decaying. The caller
        decides nothing about salience here; this *is* the salience.
        """
        self._prune(now)
        if not self._window:
            return None

        total = len(self._window)
        buckets = self._buckets()
        # Forget participation rolls for waves that have fully left the window, so
        # a genuinely new occurrence of the same emote later gets a fresh roll.
        self._rolled = {k: v for k, v in self._rolled.items() if k in buckets}

        qualifying = [
            (key, b)
            for key, b in buckets.items()
            if len(b.senders) >= self.min_senders and b.count / total >= self.min_fraction
        ]
        if not qualifying:
            return None
        # Strongest wave wins: most distinct senders, then most messages.
        key, bucket = max(qualifying, key=lambda kb: (len(kb[1].senders), kb[1].count))

        phase = self._phase(key, now)
        if phase == "decaying":  # joining a dying wave reads as lagging the room
            return None
        if now - self._last_follow_ts < self.cooldown_seconds:
            return None
        last_followed = self._fatigue.get(key)
        if last_followed is not None and now - last_followed < self.fatigue_seconds:
            return None  # bit fatigue: don't echo the same line twice in a row

        # Participation: a character doesn't join *every* wave (that's a bot).
        # Roll once per wave and remember it for the wave's lifetime.
        if key not in self._rolled:
            self._rolled[key] = self._rng.random() < self.follow_probability
        if not self._rolled[key]:
            return None

        message = max(bucket.originals.items(), key=lambda kv: kv[1])[0]
        return ChatTrend(
            token=key,
            message=message,
            senders=len(bucket.senders),
            count=bucket.count,
            fraction=bucket.count / total,
            phase=phase,
        )

    def mark_followed(self, trend: ChatTrend, now: float) -> None:
        """Record that we posted ``trend`` — starts its cooldown and bit-fatigue."""
        self._last_follow_ts = now
        self._fatigue[trend.token] = now

    # --- internals ---
    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        if self._window and self._window[0].ts < cutoff:
            self._window = [e for e in self._window if e.ts >= cutoff]

    def _buckets(self) -> dict[str, _Bucket]:
        buckets: dict[str, _Bucket] = {}
        for entry in self._window:
            if len(entry.key) > self.max_token_chars:
                continue  # long sentences aren't pile-ons
            bucket = buckets.get(entry.key)
            if bucket is None:
                bucket = buckets[entry.key] = _Bucket()
            bucket.count += 1
            bucket.senders.add(entry.author)
            bucket.originals[entry.text] = bucket.originals.get(entry.text, 0) + 1
        return buckets

    def _phase(self, key: str, now: float) -> TrendPhase:
        """Rising / crest / decaying from the bucket's count in each window half."""
        midpoint = now - self.window_seconds / 2.0
        recent = older = 0
        for entry in self._window:
            if entry.key != key:
                continue
            if entry.ts >= midpoint:
                recent += 1
            else:
                older += 1
        if recent > older:
            return "rising"
        if recent == older:
            return "crest"
        return "decaying"
