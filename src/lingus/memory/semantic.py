"""Semantic / long-term memory — durable facts that persist across streams.

The streamer's name, where regulars are from, running jokes — the things that
make the bot feel like it *knows* the channel rather than meeting it fresh every
time. This is the one memory layer that survives a process restart (persisted to
JSON) and the only one RAG-shaped: facts are stored, then *retrieved* by
relevance to the current moment.

Hand-rolled on purpose (CLAUDE.md §6/§10): at one-streamer scale a real vector DB
+ embedding model is overkill, so retrieval is token-overlap, not cosine. The
seams (a `FactExtractor` protocol, a `retrieve()` method) are where Chroma/Mem0
slot in later if it ever creaks.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .repetition import _tokens, jaccard, normalize  # noqa: PLC2701 (sibling helper reuse)

if TYPE_CHECKING:
    from ..models.base import LLMBackend


@dataclass(slots=True)
class SemanticFact:
    text: str
    subject: str = "streamer"  # who/what it's about
    source: str = "stream"  # "stream" | "manual"
    created_ts: float = 0.0
    updated_ts: float = 0.0
    hits: int = 0  # times retrieved — a cheap popularity signal for eviction


@dataclass(slots=True)
class ExtractedFact:
    text: str
    subject: str = "streamer"


class FactExtractor(Protocol):
    async def extract(self, lines: list[str]) -> list[ExtractedFact]:
        """Pull durable facts out of a batch of transcript lines."""
        ...


class SemanticStore:
    def __init__(self, *, max_facts: int = 50, dedup_threshold: float = 0.7) -> None:
        self.max_facts = max_facts
        self.dedup_threshold = dedup_threshold
        self._facts: list[SemanticFact] = []

    def __len__(self) -> int:
        return len(self._facts)

    def add(
        self,
        text: str,
        subject: str = "streamer",
        *,
        source: str = "stream",
        now: float | None = None,
    ) -> SemanticFact | None:
        text = text.strip()
        if not text:
            return None
        current = time.time() if now is None else now
        cand = _tokens(text)
        norm = normalize(text)
        for fact in self._facts:
            # Same fact again (verbatim or reworded): reinforce, don't duplicate.
            same = normalize(fact.text) == norm
            if same or jaccard(cand, _tokens(fact.text)) >= self.dedup_threshold:
                fact.updated_ts = current
                return fact
        fact = SemanticFact(
            text=text, subject=subject, source=source, created_ts=current, updated_ts=current
        )
        self._facts.append(fact)
        self._evict()
        return fact

    def retrieve(self, query: str, k: int = 5, *, now: float | None = None) -> list[SemanticFact]:
        """Top-k facts by token overlap with `query`; ties fall back to
        popularity then recency, so we always surface *something* durable."""
        if not self._facts:
            return []
        qt = _tokens(query)
        ranked = sorted(
            self._facts,
            key=lambda f: (jaccard(qt, _tokens(f.text)), f.hits, f.updated_ts),
            reverse=True,
        )
        top = ranked[:k]
        for fact in top:
            fact.hits += 1
        return top

    def texts(self) -> list[str]:
        return [f.text for f in self._facts]

    def _evict(self) -> None:
        if len(self._facts) <= self.max_facts:
            return
        # Drop the least useful first: fewest retrievals, then least recently seen.
        self._facts.sort(key=lambda f: (f.hits, f.updated_ts))
        self._facts = self._facts[len(self._facts) - self.max_facts :]

    # --- persistence (the layer that survives across streams) ---
    def load_file(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return
        self._facts = [SemanticFact(**f) for f in data.get("facts", [])][: self.max_facts]

    def save_file(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"facts": [asdict(f) for f in self._facts]}, indent=2))


# --- extraction --------------------------------------------------------------


def _name(m: re.Match[str]) -> str:
    return f"the streamer's name is {m.group(1)}"


def _from(m: re.Match[str]) -> str:
    return f"the streamer is from {m.group(1).strip()}"


def _habit(m: re.Match[str]) -> str:
    return f"the streamer {m.group(1)} {m.group(2).strip()}"


def _taste(m: re.Match[str]) -> str:
    return f"the streamer {m.group(1)}s {m.group(2).strip()}"


# Conservative first-person patterns — these are exactly the "durable fact"
# shape, and matchable without a model. Whisper capitalizes sentence starts and
# names, which the entity captures rely on.
# `(?i:…)` makes only the lead-in case-insensitive ("My"/"my") while the entity
# capture keeps requiring a capital, so we grab "Marco"/"London", not "the".
_PATTERNS: list[tuple[re.Pattern[str], object]] = [
    (re.compile(r"(?i:\bmy name(?:'s| is)\s+)([A-Z][a-zA-Z]+)"), _name),
    (re.compile(r"(?i:\bi'?m from\s+)([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?)"), _from),
    (re.compile(r"(?i:\bi (always|never|usually) )([a-z][a-z ]{3,40}?)\b(?:[.,!?]|$)"), _habit),
    (re.compile(r"(?i:\bi (love|hate|prefer) )([a-z][a-z ]{2,40}?)\b(?:[.,!?]|$)"), _taste),
]


class HeuristicFactExtractor:
    """Deterministic, model-free extraction so the long-term store works offline.

    Catches conservative first-person statements ("my name is…", "I'm from…",
    "I always…", "I love…"). Coarse, but real — the LLM extractor is the rich path."""

    async def extract(self, lines: list[str]) -> list[ExtractedFact]:
        out: list[ExtractedFact] = []
        for line in lines:
            for pattern, build in _PATTERNS:
                match = pattern.search(line)
                if match:
                    out.append(ExtractedFact(text=build(match)))  # type: ignore[operator]
        return out


class LLMFactExtractor:
    def __init__(self, backend: LLMBackend, max_facts: int = 5) -> None:
        self._backend = backend
        self.max_facts = max_facts

    async def extract(self, lines: list[str]) -> list[ExtractedFact]:
        from ..models.base import ChatTurn

        transcript = "\n".join(lines)
        system = (
            "Extract durable facts about the streamer or chat regulars from this "
            "transcript — names, places, preferences, recurring jokes. Only lasting "
            "facts, not momentary events. Reply as a JSON array of "
            '{"fact": str, "subject": str}. Empty array if none.'
        )
        try:
            raw = await self._backend.generate(
                [ChatTurn(role="system", content=system), ChatTurn(role="user", content=transcript)]
            )
            data = json.loads(_json_slice(raw))
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            return []
        facts = []
        for item in data[: self.max_facts]:
            text = str(item.get("fact", "")).strip()
            if text:
                facts.append(ExtractedFact(text=text, subject=str(item.get("subject", "streamer"))))
        return facts


def _json_slice(text: str) -> str:
    """Best-effort: pull the JSON array out of a chatty LLM response."""
    start, end = text.find("["), text.rfind("]")
    return text[start : end + 1] if start != -1 and end > start else "[]"
