"""Hosted generator over any OpenAI-compatible endpoint.

`OpenAICompatLLM` is the thin transport (OpenAI, Grok, or Claude's compat shim,
selected by base_url + api_key + model). `LLMReplyGenerator` is the cognition-
side `ReplyGenerator`: it turns the PersonaSpec into a system prompt and the
world-state snapshot into the user turn, then asks for one short in-character
line. The hard length cap and rate limits still live downstream in the
OutputGovernor — this layer only *aims* for brevity.
"""

from __future__ import annotations

from typing import Any, TypeVar

from ..arbiter import ArbiterDecision
from ..context import ContextSnapshot
from ..logging import get_logger
from ..persona.schema import PersonaSpec
from .base import ChatTurn, LLMBackend

log = get_logger(__name__)
T = TypeVar("T")


def _message_dicts(messages: list[ChatTurn]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


class OpenAICompatLLM(LLMBackend):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "",
        model: str = "gpt-5.5",
        temperature: float = 0.9,
        max_tokens: int = 120,
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._structured_client: Any | None = None
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def set_temperature(self, temperature: float) -> None:
        self._temperature = max(0.0, float(temperature))

    async def generate(self, messages: list[ChatTurn], **opts) -> str:
        resp = await self._client.chat.completions.create(
            model=opts.get("model", self._model),
            messages=_message_dicts(messages),
            temperature=opts.get("temperature", self._temperature),
            max_tokens=opts.get("max_tokens", self._max_tokens),
        )
        return (resp.choices[0].message.content or "").strip()

    async def generate_structured(
        self,
        response_model: type[T],
        messages: list[ChatTurn],
        **opts,
    ) -> T:
        """Generate and validate structured output via Instructor."""

        import instructor

        if self._structured_client is None:
            self._structured_client = instructor.from_openai(self._client)
        request = dict(opts)
        max_retries = request.pop("max_retries", 3)
        return await self._structured_client.create(
            response_model=response_model,
            messages=_message_dicts(messages),
            model=request.pop("model", self._model),
            temperature=request.pop("temperature", self._temperature),
            max_tokens=request.pop("max_tokens", self._max_tokens),
            max_retries=max_retries,
            **request,
        )


def build_system_prompt(persona: PersonaSpec, max_chars: int) -> str:
    p = persona
    lines: list[str] = [
        f"You are {p.name}, a character in a live stream's text chat.",
        f"Voice: {p.voice.strip()}",
    ]
    if p.lexicon.use:
        lines.append("Words/phrases you reach for: " + ", ".join(p.lexicon.use))
    if p.lexicon.avoid:
        lines.append("Never say: " + ", ".join(p.lexicon.avoid))
    if p.lexicon.catchphrases:
        lines.append("Catchphrases (use sparingly): " + ", ".join(p.lexicon.catchphrases))
    if p.interests:
        lines.append("Interests: " + "; ".join(p.interests))
    if p.opinions:
        lines.append("Opinions you hold (have takes, be playfully wrong): " + "; ".join(p.opinions))
    if p.relationships:
        lines.append("Relationships: " + "; ".join(f"{r.who}: {r.stance}" for r in p.relationships))
    if p.boundaries:
        lines.append("Boundaries (won't engage): " + "; ".join(p.boundaries))
    if p.exemplar_bank:
        lines.append("Sample reactions (match this energy, don't copy verbatim):")
        lines += [f'  - when {e.situation} -> "{e.line}"' for e in p.exemplar_bank]
    lines.append(f"Current mood/energy (−1 calm .. +1 hyped): {p.mood.value:+.2f}")
    lines += [
        "",
        "Rules:",
        f"- One short line, at most {max_chars} characters. No paragraphs.",
        "- Stay fully in character. Never sound like an assistant: no hedging, no "
        '"how can I help", no balanced both-sides answers, no trailing questions.',
        "- React to what's happening right now. If nothing's worth saying, be brief anyway.",
        "- Output only the chat line itself — no quotes, no name prefix, no narration.",
    ]
    return "\n".join(lines)


class LLMReplyGenerator:
    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend
        self._temperature: float | None = None

    def set_temperature(self, temperature: float) -> None:
        self._temperature = max(0.0, float(temperature))

    async def generate(
        self,
        snapshot: ContextSnapshot,
        decision: ArbiterDecision,
        persona: PersonaSpec,
        *,
        max_chars: int,
    ) -> str:
        system = build_system_prompt(persona, max_chars)
        trigger = ", ".join(decision.reasons) or "ambient"
        user = (
            snapshot.to_prompt_context()
            + f"\n\nWhy you're reacting now: {trigger}."
            + "\nReply with one short in-character line."
        )
        try:
            opts = {"max_tokens": max(32, max_chars // 2)}
            if self._temperature is not None:
                opts["temperature"] = self._temperature
            return await self._backend.generate(
                [ChatTurn("system", system), ChatTurn("user", user)],
                **opts,
            )
        except Exception as exc:  # a dropped reply beats crashing the loop
            log.warning("LLM generation failed, skipping reply: %s", exc)
            return ""
