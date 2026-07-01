"""Episodic memory — the running "stream so far" narrative.

Working memory is a rolling window; on a long stream its early content is lost as
the buffer evicts. Episodic memory periodically folds those evicted transcript
lines into a compact narrative the generator can still see, so the bot remembers
how the stream got here (and can make callbacks) without holding every raw line.

Two summarizers behind one protocol, mirroring the generator:
- `LLMSummarizer` — the real path; reuses the generator's LLM backend.
- `ExtractiveSummarizer` — a deterministic, model-free fallback so the loop (and
  its tests) run offline without an API key. It keeps a bounded, deduplicated
  reel of the *salient* evicted lines rather than a true abstractive summary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..models.base import LLMBackend


class EpisodicSummarizer(Protocol):
    async def summarize(self, prior: str, new_lines: list[str]) -> str:
        """Fold `new_lines` into `prior`, returning the updated narrative."""
        ...


def _is_salient(line: str) -> bool:
    """A cheap signal heuristic: questions, numbers, or substantive lines naming
    something carry the stream's narrative; short filler ("yeah ok", "right",
    "no") mostly doesn't. Coarse on purpose — the LLM summarizer is the real path;
    this just keeps the offline fallback from piling up noise."""
    if "?" in line or any(ch.isdigit() for ch in line):
        return True
    words = line.split()
    has_name = any(w[:1].isupper() for w in words)
    return len(words) >= 3 and has_name


class ExtractiveSummarizer:
    def __init__(self, max_chars: int = 800) -> None:
        self.max_chars = max_chars

    async def summarize(self, prior: str, new_lines: list[str]) -> str:
        picks = [line.strip() for line in new_lines if line.strip() and _is_salient(line)]
        parts = [prior.strip()] if prior.strip() else []
        for pick in picks:
            if pick not in parts:  # cheap dedup so a repeated beat isn't piled up
                parts.append(pick)
        digest = " · ".join(parts)
        if len(digest) <= self.max_chars:
            return digest
        # Over budget: drop oldest segments (after a leading ellipsis) until it fits.
        while len(parts) > 1 and len(" · ".join(parts)) > self.max_chars:
            parts.pop(0)
        return "… · " + " · ".join(parts)


class LLMSummarizer:
    def __init__(self, backend: LLMBackend, max_chars: int = 800) -> None:
        self._backend = backend
        self.max_chars = max_chars

    async def summarize(self, prior: str, new_lines: list[str]) -> str:
        from ..models.base import ChatTurn

        transcript = "\n".join(new_lines)
        system = (
            "You maintain a running summary of a live stream so a chat bot can "
            "remember how it got here and make callbacks. Keep it factual and "
            "compact — events, topics, names, running jokes — not commentary."
        )
        user = (
            f"Summary so far:\n{prior or '(nothing yet)'}\n\n"
            f"New transcript to fold in:\n{transcript}\n\n"
            f"Return the updated summary in under {self.max_chars} characters."
        )
        out = await self._backend.generate(
            [ChatTurn(role="system", content=system), ChatTurn(role="user", content=user)]
        )
        return out.strip()[: self.max_chars]
