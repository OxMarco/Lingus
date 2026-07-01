"""Reply generation boundary.

The template generator is deliberately deterministic so offline replay tests can
exercise the full loop before a hosted LLM backend is wired in.
"""

from __future__ import annotations

from typing import Protocol

from .arbiter import ArbiterDecision
from .context import ContextSnapshot
from .persona.schema import PersonaSpec


class ReplyGenerator(Protocol):
    async def generate(
        self,
        snapshot: ContextSnapshot,
        decision: ArbiterDecision,
        persona: PersonaSpec,
        *,
        max_chars: int,
    ) -> str: ...


class TemplateReplyGenerator:
    async def generate(
        self,
        snapshot: ContextSnapshot,
        decision: ArbiterDecision,
        persona: PersonaSpec,
        *,
        max_chars: int,
    ) -> str:
        # `max_chars` is a soft target the generator aims for (and that a future
        # LLM backend gets as a token budget). The *hard* cap is enforced
        # downstream by the OutputGovernor, so we don't truncate here — a
        # mid-word cut at this layer would defeat the governor's sentence-aware
        # truncation.
        text = self._context_text(snapshot)
        return self._choose_reply(text, decision, persona)

    def _choose_reply(
        self, text: str, decision: ArbiterDecision, persona: PersonaSpec
    ) -> str:
        if "chocolate" in text and "stain" in text:
            return "be careful next time, chocolate stains are hard to remove"
        if "stain" in text:
            return "careful next time, stains love becoming the main quest"
        if "spill" in text or "spilled" in text:
            return "careful, that cleanup arc is about to have lore"
        if "burn" in text or "burnt" in text:
            return "heat management boss fight, apparently"
        if "direct_address" in decision.reasons or "question" in decision.reasons:
            exemplar = self._find_exemplar(persona, "direct question")
            return exemplar or "real answer? probably. but where's the fun in that"
        if "hype" in decision.reasons:
            exemplar = self._find_exemplar(persona, "hype")
            return exemplar or "ok that was actually clean, i take back nothing"
        if persona.exemplar_bank:
            return persona.exemplar_bank[0].line
        return "noted. deeply suspicious, but noted"

    @staticmethod
    def _context_text(snapshot: ContextSnapshot) -> str:
        return " ".join(
            [
                snapshot.scene_summary(),
                snapshot.transcript,
                " ".join(line.text for line in snapshot.recent_chat),
                snapshot.latest_event_summary(),
            ]
        ).lower()

    @staticmethod
    def _find_exemplar(persona: PersonaSpec, needle: str) -> str:
        for exemplar in persona.exemplar_bank:
            if needle in exemplar.situation.lower():
                return exemplar.line
        return ""
