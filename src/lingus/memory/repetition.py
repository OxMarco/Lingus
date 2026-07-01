"""Self-memory dedup + bit-fatigue.

The generator is an LLM (or, offline, a template) and will happily say the same
thing twice — verbatim or, worse, lightly reworded ("ok that was clean" vs "ok
that was actually clean"). Feeding the bot's own recent messages into the prompt
helps but doesn't *enforce* anything. This is the deterministic gate that does:

- **Near-duplicate detection** — normalized token Jaccard against recent self-
  memory, so paraphrases are caught, not just exact repeats.
- **Bit-fatigue** — a catchphrase used recently is "fatigued"; leaning on it
  again inside the window is treated as repetition. A personality with one joke
  it tells every minute has no character (CLAUDE.md §5).

Pure and deterministic (no model, no I/O), so it sits in the post path next to
the output governor and is cheap to unit-test.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..persona.schema import PersonaSpec

_WORD_RE = re.compile(r"[a-z0-9']+")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — a canonical form."""
    return " ".join(_WORD_RE.findall(text.lower()))


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(text.lower()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Token-set overlap in [0, 1]. 0 if either side is empty."""
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


class RepetitionGuard:
    def __init__(
        self,
        *,
        similarity_threshold: float = 0.6,
        fatigue_seconds: float = 180.0,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.fatigue_seconds = fatigue_seconds
        # normalized catchphrase -> last-used monotonic timestamp
        self._bit_last_used: dict[str, float] = {}

    def is_repetitive(
        self,
        candidate: str,
        recent_messages: Iterable[str],
        persona: PersonaSpec | None = None,
        *,
        now: float | None = None,
    ) -> bool:
        """True if `candidate` repeats recent self-memory or a fatigued bit."""
        cand_norm = normalize(candidate)
        if not cand_norm:
            return True  # nothing to say; don't post whitespace
        cand_tokens = _tokens(candidate)
        for prev in recent_messages:
            if normalize(prev) == cand_norm:
                return True
            if jaccard(cand_tokens, _tokens(prev)) >= self.similarity_threshold:
                return True
        if persona is not None:
            current = time.monotonic() if now is None else now
            for phrase in self._fatigued_phrases(persona, current):
                if phrase and phrase in cand_norm:
                    return True
        return False

    def note_post(
        self, text: str, persona: PersonaSpec, *, now: float | None = None
    ) -> None:
        """Record which catchphrases a just-posted message leaned on, so the next
        reuse inside the fatigue window is caught."""
        current = time.monotonic() if now is None else now
        text_norm = normalize(text)
        for phrase in persona.lexicon.catchphrases:
            key = normalize(phrase)
            if key and key in text_norm:
                self._bit_last_used[key] = current

    def _fatigued_phrases(self, persona: PersonaSpec, now: float) -> list[str]:
        fatigued = []
        for phrase in persona.lexicon.catchphrases:
            key = normalize(phrase)
            last = self._bit_last_used.get(key)
            if last is not None and now - last < self.fatigue_seconds:
                fatigued.append(key)
        return fatigued
