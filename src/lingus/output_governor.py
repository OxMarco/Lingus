"""The output governor — the deterministic last line before a message is posted.

Everything upstream of here is *probabilistic*: the arbiter decides whether a
moment is worth reacting to (temperament, tunable, allowed to be wrong) and the
generator decides what to say (an LLM that will sometimes ignore a length
instruction). Spam and message length, by contrast, are *invariants*: there is
no acceptable rate at which the bot floods chat, and no acceptable length past
the cap. So they are enforced here, in code, where no model decision can talk
its way past them.

Three responsibilities, all deterministic and time-injectable for tests:

  * **rate** — a token bucket (sustained + burst ceiling) plus a hard minimum
    spacing between posts. This is the authority; the arbiter's matching
    pre-filter only exists to avoid generating a reply we'd drop anyway.
  * **length** — a hard character cap with *sentence-aware* truncation, so an
    over-long reply is cut at a sentence (or at worst a word) boundary with an
    ellipsis — never mid-word, never over the cap.
  * **temporizer** — a human-like "typing time" derived from the message
    length, so a full sentence can't land instantly on the heels of the prior
    message. The caller awaits this delay before posting (and re-checks
    staleness afterwards, since the world moves while the bot "types").

`admit()` both checks the rate limit and *commits* (consumes a token, stamps the
post time) when it returns `post` — so call it only when you actually intend to
post, after the staleness re-check.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

_SENTENCE_ENDINGS = ".!?…"
_ELLIPSIS = "…"


@dataclass(slots=True)
class GovernedOutput:
    action: Literal["post", "drop"]
    text: str
    reason: str  # "ok" | "truncated" | "empty" | "rate_limited_interval" | "rate_limited_bucket"
    truncated: bool = False


class _TokenBucket:
    """Classic token bucket: `capacity` posts of burst, refilling at `rate`/sec."""

    def __init__(self, capacity: float, rate_per_second: float, now: float) -> None:
        self.capacity = float(capacity)
        self.rate = rate_per_second
        self.tokens = float(capacity)
        self._t = now

    def _refill(self, now: float) -> None:
        if now > self._t:
            self.tokens = min(self.capacity, self.tokens + (now - self._t) * self.rate)
            self._t = now

    def available(self, now: float) -> bool:
        self._refill(now)
        return self.tokens >= 1.0

    def consume(self, now: float) -> bool:
        self._refill(now)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class OutputGovernor:
    def __init__(
        self,
        *,
        max_chars: int,
        min_seconds_between_posts: float,
        burst: int = 2,
        posts_per_minute: float = 6.0,
        typing_cps: float = 15.0,
        typing_base_seconds: float = 0.4,
        typing_min_seconds: float = 0.8,
        typing_max_seconds: float = 7.0,
        now: float | None = None,
    ) -> None:
        self.max_chars = max_chars
        self.min_interval = min_seconds_between_posts
        self.typing_cps = typing_cps
        self.typing_base_seconds = typing_base_seconds
        self.typing_min_seconds = typing_min_seconds
        self.typing_max_seconds = typing_max_seconds
        start = now if now is not None else time.monotonic()
        self._bucket = _TokenBucket(burst, posts_per_minute / 60.0, start)
        self._last_post: float | None = None

    # --- live tuning (web UI) ---
    def set_posts_per_minute(self, posts_per_minute: float) -> None:
        """Retune the sustained rate without resetting accumulated tokens."""
        self._bucket.rate = max(0.0, posts_per_minute) / 60.0

    def set_burst(self, burst: int) -> None:
        self._bucket.capacity = float(max(1, burst))
        self._bucket.tokens = min(self._bucket.tokens, self._bucket.capacity)

    # --- length ---
    def over_length(self, text: str) -> bool:
        return len(text.strip()) > self.max_chars

    def fit(self, text: str) -> tuple[str, bool]:
        """Hard-cap `text` to `max_chars`, cutting at the nicest boundary available.

        Prefer a sentence ending; fall back to a word boundary + ellipsis; only
        ever cut mid-token for a single word longer than the whole budget. The
        result is guaranteed to be `<= max_chars`.
        """
        text = text.strip()
        if len(text) <= self.max_chars:
            return text, False

        window = text[: self.max_chars]
        # Prefer the last sentence ending, but not so early that we return a stub.
        sentence_cut = max(window.rfind(c) for c in _SENTENCE_ENDINGS)
        if sentence_cut >= self.max_chars * 0.4:
            return window[: sentence_cut + 1].rstrip(), True

        # Otherwise cut on a word boundary, leaving room for the ellipsis.
        limit = self.max_chars - len(_ELLIPSIS)
        word_cut = text.rfind(" ", 0, limit + 1)
        if word_cut <= 0:  # one giant token with no space to break on
            word_cut = limit
        return text[:word_cut].rstrip() + _ELLIPSIS, True

    # --- temporizer (human-like typing time) ---
    def typing_delay(self, text: str) -> float:
        """Seconds a human would plausibly take to type `text`.

        A base reaction beat plus per-character time, clamped so a one-word
        reaction still feels deliberate and a long one doesn't stall forever.
        The caller awaits this before posting (scaling by replay speed offline).
        """
        raw = self.typing_base_seconds + len(text.strip()) / self.typing_cps
        return max(self.typing_min_seconds, min(self.typing_max_seconds, raw))

    # --- rate (check + commit) ---
    def admit(self, text: str, *, now: float | None = None) -> GovernedOutput:
        """Final gate: enforce spacing + bucket, hard-cap length, and commit on accept.

        Consumes a rate token and stamps the post time *only* when it returns
        ``post`` — so a drop here costs nothing toward the rate budget.
        """
        now = now if now is not None else time.monotonic()
        text = (text or "").strip()
        if not text:
            return GovernedOutput("drop", "", "empty")

        if self._last_post is not None and now - self._last_post < self.min_interval:
            return GovernedOutput("drop", text, "rate_limited_interval")
        if not self._bucket.available(now):
            return GovernedOutput("drop", text, "rate_limited_bucket")

        final, truncated = self.fit(text)
        self._bucket.consume(now)
        self._last_post = now
        return GovernedOutput("post", final, "truncated" if truncated else "ok", truncated)
