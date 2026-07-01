"""Live web search behind a swappable backend.

The researcher needs a few paragraphs of "what does the internet say about this
channel" to distill into facts. That's a classic search-then-read task, so it
sits behind a tiny `WebSearchBackend` seam: the default is a keyless DuckDuckGo
scrape (`ddgs`, the `[research]` extra) so it works with zero credentials, but a
keyed provider (Tavily/Brave/SerpAPI) can drop in behind the same interface.

Every backend degrades to "no results" rather than raising — research is a
best-effort cold-start nicety, never a hard dependency of the run.
"""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass

from ..logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str

    def as_context(self) -> str:
        body = self.snippet.strip() or "(no snippet)"
        return f"{self.title.strip()} — {body} [{self.url}]"


class WebSearchBackend(abc.ABC):
    @abc.abstractmethod
    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Return up to `max_results` hits for `query`; [] on any failure."""


class NullSearch(WebSearchBackend):
    """No web access — the researcher falls back to yt-dlp metadata alone."""

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        return []


class DuckDuckGoSearch(WebSearchBackend):
    """Keyless DDG search via `ddgs`. The library is sync and network-bound, so
    each query runs in a thread to keep the event loop free."""

    def __init__(self) -> None:
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError:  # pragma: no cover - exercised via build_search_backend
            try:
                from duckduckgo_search import DDGS  # type: ignore  # older package name
            except ImportError as exc:
                raise ImportError(
                    "web search needs the 'research' extra: pip install -e '.[research]'"
                ) from exc
        self._ddgs = DDGS

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            rows = await asyncio.to_thread(self._blocking_search, query, max_results)
        except Exception as exc:  # noqa: BLE001 - network/parse errors must never bubble up
            log.warning("research: web search failed for %r: %s", query, exc)
            return []
        return rows

    def _blocking_search(self, query: str, max_results: int) -> list[SearchResult]:
        out: list[SearchResult] = []
        with self._ddgs() as ddgs:
            for hit in ddgs.text(query, max_results=max_results):
                out.append(
                    SearchResult(
                        title=str(hit.get("title", "")),
                        url=str(hit.get("href", hit.get("url", ""))),
                        snippet=str(hit.get("body", hit.get("snippet", ""))),
                    )
                )
        return out


def build_search_backend(backend: str) -> WebSearchBackend:
    """Construct the configured web-search backend, degrading to NullSearch when
    it's disabled or its optional dependency is missing."""
    name = (backend or "none").lower()
    if name in ("none", "", "null"):
        return NullSearch()
    if name in ("duckduckgo", "ddg"):
        try:
            return DuckDuckGoSearch()
        except ImportError as exc:
            log.warning("research: %s — web search disabled, using metadata only", exc)
            return NullSearch()
    log.warning("research: unknown web_search backend '%s' — disabling", backend)
    return NullSearch()
