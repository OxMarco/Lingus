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

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .repetition import _tokens, jaccard  # noqa: PLC2701 (sibling helper reuse)

if TYPE_CHECKING:
    from ..models.base import LLMBackend


class EpisodicSummarizer(Protocol):
    async def summarize(self, prior: str, new_lines: list[str]) -> str:
        """Fold `new_lines` into `prior`, returning the updated narrative."""
        ...


@dataclass(slots=True)
class EpisodicMemory:
    summary: str
    stream_id: str
    kind: str = "stream_summary"
    source: str = "stream"
    created_ts: float = 0.0
    updated_ts: float = 0.0
    last_seen_ts: float = 0.0
    hits: int = 0


class EpisodicArchive:
    """Durable per-stream summaries.

    Short-term memory stays in `WorldState`; this archive stores compact stream
    summaries across runs so the bot can make callbacks without carrying raw
    transcripts or introducing a vector DB.
    """

    def __init__(self, *, max_entries: int = 20) -> None:
        self.max_entries = max_entries
        self._episodes: list[EpisodicMemory] = []

    def __len__(self) -> int:
        return len(self._episodes)

    def add(
        self,
        summary: str,
        *,
        stream_id: str,
        source: str = "stream",
        now: float | None = None,
    ) -> EpisodicMemory | None:
        summary = summary.strip()
        stream_id = stream_id.strip() or "unknown-stream"
        if not summary:
            return None
        current = time.time() if now is None else now
        for episode in self._episodes:
            if episode.stream_id == stream_id:
                episode.summary = summary
                episode.source = source
                episode.updated_ts = current
                episode.last_seen_ts = current
                self._evict()
                return episode
        episode = EpisodicMemory(
            summary=summary,
            stream_id=stream_id,
            source=source,
            created_ts=current,
            updated_ts=current,
            last_seen_ts=current,
        )
        self._episodes.append(episode)
        self._evict()
        return episode

    def retrieve(
        self,
        query: str,
        k: int = 3,
        *,
        now: float | None = None,
    ) -> list[EpisodicMemory]:
        """Top-k summaries by token overlap, then prior usefulness and recency."""
        if not self._episodes or k <= 0:
            return []
        qt = _tokens(query)
        ranked = sorted(
            self._episodes,
            key=lambda e: (jaccard(qt, _tokens(e.summary)), e.hits, e.updated_ts),
            reverse=True,
        )
        current = time.time() if now is None else now
        top = ranked[:k]
        for episode in top:
            episode.hits += 1
            episode.last_seen_ts = current
        return top

    def summaries(self) -> list[str]:
        return [episode.summary for episode in self._episodes]

    def _evict(self) -> None:
        if len(self._episodes) <= self.max_entries:
            return
        self._episodes.sort(key=lambda e: (e.hits, e.updated_ts))
        self._episodes = self._episodes[len(self._episodes) - self.max_entries :]

    def load_file(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict):
            return
        raw_episodes = data.get("episodes", [])
        if not isinstance(raw_episodes, list):
            return
        episodes: list[EpisodicMemory] = []
        for raw in raw_episodes:
            if not isinstance(raw, dict):
                continue
            try:
                episode = EpisodicMemory(**raw)
            except TypeError:
                continue
            if not isinstance(episode.summary, str) or not isinstance(episode.stream_id, str):
                continue
            episode.summary = episode.summary.strip()
            episode.stream_id = episode.stream_id.strip()
            if episode.summary and episode.stream_id:
                episodes.append(episode)
        self._episodes = episodes[: self.max_entries]

    def save_file(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"episodes": [asdict(e) for e in self._episodes]}, indent=2),
            encoding="utf-8",
        )


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
        return _fit_text("… · " + " · ".join(parts), self.max_chars)


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


def _fit_text(text: str, max_chars: int) -> str:
    """Hard-cap a summary without cutting past the configured budget."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    ellipsis = "…"
    if max_chars <= len(ellipsis):
        return text[:max_chars]
    limit = max_chars - len(ellipsis)
    cut = text.rfind(" ", 0, limit + 1)
    if cut <= 0:
        cut = limit
    return text[:cut].rstrip() + ellipsis
