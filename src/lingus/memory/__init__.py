"""Memory layers (Phase 3).

Working memory (the rolling transcript/chat buffers) and self-memory (the bot's
own recent messages) already live on the WorldState. This package adds the parts
that *act* on memory:

- `RepetitionGuard` — self-memory dedup + bit-fatigue, the deterministic defense
  against the #1 immersion-killer: a character that repeats itself.

Episodic summarization and semantic/long-term stores live here too.
"""

from __future__ import annotations

from .episodic import (
    EpisodicArchive,
    EpisodicMemory,
    EpisodicSummarizer,
    ExtractiveSummarizer,
    LLMSummarizer,
)
from .repetition import RepetitionGuard, jaccard, normalize
from .semantic import (
    ExtractedFact,
    FactExtractor,
    HeuristicFactExtractor,
    LLMFactExtractor,
    SemanticFact,
    SemanticStore,
)

__all__ = [
    "EpisodicSummarizer",
    "EpisodicArchive",
    "EpisodicMemory",
    "ExtractedFact",
    "ExtractiveSummarizer",
    "FactExtractor",
    "HeuristicFactExtractor",
    "LLMFactExtractor",
    "LLMSummarizer",
    "RepetitionGuard",
    "SemanticFact",
    "SemanticStore",
    "jaccard",
    "normalize",
]
