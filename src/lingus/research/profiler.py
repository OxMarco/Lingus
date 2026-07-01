"""Turn a raw channel identity + web snippets into durable memory.

`ChannelResearcher.research()` is the cold-start brain: it forms a few search
queries, gathers snippets, and distills everything the LLM can find into a small
set of *durable facts* — phrased like the facts the semantic store already holds
("the streamer's name is…", "the channel is about…", "a running bit is…"). Those
facts are what get seeded into long-term memory so the bot walks in warm.

Two paths, same shape out:
- **LLM path** (a key is set): the model reads identity + snippets and writes
  facts + a short prose profile. This is the rich one.
- **Deterministic fallback** (no key): facts are synthesized straight from the
  yt-dlp metadata (name, description, tags). Coarse, but real and offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..logging import get_logger
from .identity import ChannelIdentity
from .search import SearchResult, WebSearchBackend

if TYPE_CHECKING:
    from ..models.base import LLMBackend

log = get_logger(__name__)


@dataclass(slots=True)
class ChannelProfile:
    """Research output: durable facts to seed memory, plus a human-readable
    summary and the sources it drew on (kept in the cache for auditing)."""

    channel: str
    facts: list[str] = field(default_factory=list)
    summary: str = ""
    source_urls: list[str] = field(default_factory=list)


def _search_queries(identity: ChannelIdentity, limit: int) -> list[str]:
    name = identity.name
    plat = identity.platform
    base = [
        f"{name} {plat} streamer",
        f"{name} {plat} about",
        f"who is {name} {plat}",
        f"{name} streamer running jokes catchphrases community",
    ]
    return base[: max(1, limit)]


def _first_sentences(text: str, n: int = 2) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    parts = text.replace("!", ".").replace("?", ".").split(".")
    keep = [p.strip() for p in parts if p.strip()][:n]
    return ". ".join(keep)


class ChannelResearcher:
    def __init__(
        self,
        search: WebSearchBackend,
        *,
        llm: LLMBackend | None = None,
        max_facts: int = 12,
        max_queries: int = 4,
        max_results: int = 6,
    ) -> None:
        self._search = search
        self._llm = llm
        self.max_facts = max_facts
        self.max_queries = max_queries
        self.max_results = max_results

    async def research(self, identity: ChannelIdentity) -> ChannelProfile:
        hits = await self._gather(identity)
        if self._llm is not None:
            profile = await self._distill_llm(identity, hits)
            if profile.facts:
                return profile
            log.info("research: LLM returned no facts, using metadata fallback")
        return self._distill_fallback(identity, hits)

    async def _gather(self, identity: ChannelIdentity) -> list[SearchResult]:
        out: list[SearchResult] = []
        seen: set[str] = set()
        for query in _search_queries(identity, self.max_queries):
            for hit in await self._search.search(query, self.max_results):
                if hit.url and hit.url in seen:
                    continue
                seen.add(hit.url)
                out.append(hit)
        log.info("research: gathered %d web snippets for '%s'", len(out), identity.name)
        return out

    # --- LLM distillation ---------------------------------------------------
    async def _distill_llm(
        self, identity: ChannelIdentity, hits: list[SearchResult]
    ) -> ChannelProfile:
        from ..models.base import ChatTurn

        material = self._material(identity, hits)
        system = (
            "You are building a durable memory profile of a live streamer so a chat "
            "bot can feel like it already knows the channel. From the metadata and "
            "web snippets, extract lasting facts — real name/handle, where they're "
            "from, what they stream, their persona, recurring bits, catchphrases, "
            "notable community in-jokes. Only durable facts, not momentary events, "
            "and do not invent anything unsupported by the material. Reply as JSON: "
            '{"facts": [str, ...], "summary": str}. Each fact one short sentence. '
            f"At most {self.max_facts} facts. Empty facts list if the material is too thin."
        )
        try:
            raw = await self._llm.generate(  # type: ignore[union-attr]
                [ChatTurn(role="system", content=system), ChatTurn(role="user", content=material)]
            )
            data = json.loads(_json_object(raw))
        except Exception as exc:  # noqa: BLE001 - any LLM/parse failure → fallback
            log.warning("research: LLM distillation failed (%s)", exc)
            return ChannelProfile(channel=identity.name)
        facts = [str(f).strip() for f in data.get("facts", []) if str(f).strip()]
        return ChannelProfile(
            channel=identity.name,
            facts=facts[: self.max_facts],
            summary=str(data.get("summary", "")).strip(),
            source_urls=[h.url for h in hits if h.url],
        )

    def _material(self, identity: ChannelIdentity, hits: list[SearchResult]) -> str:
        lines = [
            f"Platform: {identity.platform}",
            f"Channel name: {identity.name}",
        ]
        if identity.description:
            lines.append(f"Channel description: {identity.description}")
        if identity.tags:
            lines.append(f"Tags: {', '.join(identity.tags)}")
        if identity.recent_titles:
            lines.append(f"Recent titles: {'; '.join(identity.recent_titles)}")
        if hits:
            lines.append("\nWeb search results:")
            lines += [f"- {h.as_context()}" for h in hits]
        return "\n".join(lines)

    # --- deterministic fallback --------------------------------------------
    def _distill_fallback(
        self, identity: ChannelIdentity, hits: list[SearchResult]
    ) -> ChannelProfile:
        """Model-free facts straight from the metadata, so cold-start research
        still seeds *something* durable with no API key and no web access."""
        facts: list[str] = [f"the streamer's channel is {identity.name} on {identity.platform}"]
        desc = _first_sentences(identity.description, 2)
        if desc:
            facts.append(f"the channel describes itself as: {desc}")
        if identity.tags:
            facts.append(f"the channel's topics include {', '.join(identity.tags[:6])}")
        # A couple of web snippets, lightly folded in, so the no-LLM path still
        # benefits from search when it's available.
        for hit in hits[:2]:
            snippet = _first_sentences(hit.snippet, 1)
            if snippet:
                facts.append(f"per {hit.url}: {snippet}")
        summary = desc or f"{identity.name} streams on {identity.platform}."
        return ChannelProfile(
            channel=identity.name,
            facts=facts[: self.max_facts],
            summary=summary,
            source_urls=[h.url for h in hits if h.url],
        )


def _json_object(text: str) -> str:
    """Pull the JSON object out of a chatty LLM response."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else "{}"
