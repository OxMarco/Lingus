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


def _error_text(exc: Exception) -> str | None:
    """The message body of a 400, or None if `exc` isn't a 400 we act on."""
    if getattr(exc, "status_code", None) != 400:
        return None
    return str(getattr(exc, "message", "") or exc)


def _is_unsupported_token_param(exc: Exception, param: str) -> bool:
    """True when `exc` is the 400 telling us this endpoint doesn't accept
    `param` as the token-limit field (so we should try the other name)."""
    text = _error_text(exc)
    return bool(
        text and param in text and ("max_completion_tokens" in text or "max_tokens" in text)
    )


def _is_unsupported_temperature(exc: Exception) -> bool:
    """True when `exc` is the 400 telling us this model rejects a custom
    `temperature` (newer OpenAI models only accept the default)."""
    text = _error_text(exc)
    return bool(text and "temperature" in text)


class OpenAICompatLLM(LLMBackend):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "",
        model: str = "gpt-5.4-mini",
        temperature: float = 0.7,
        max_tokens: int = 120,
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._structured_client: Any | None = None
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # Newer OpenAI models reject `max_tokens` (want `max_completion_tokens`)
        # and reject a custom `temperature` (only accept the default). We learn
        # each endpoint's quirks from the first 400 and cache them so we only pay
        # the extra round-trip once.
        self._token_param = "max_tokens"
        self._send_temperature = True

    def set_temperature(self, temperature: float) -> None:
        self._temperature = max(0.0, float(temperature))

    def _tunable_kwargs(self, max_tokens: int, temperature: float) -> dict[str, Any]:
        """The request params we may have to adjust per endpoint."""
        kwargs: dict[str, Any] = {self._token_param: max_tokens}
        if self._send_temperature:
            kwargs["temperature"] = temperature
        return kwargs

    def _adjust_for_error(self, exc: Exception) -> bool:
        """If `exc` is a 400 about a param we can adapt, mutate our cached state
        and return True (caller should retry); otherwise return False."""
        if _is_unsupported_token_param(exc, self._token_param):
            other = "max_completion_tokens" if self._token_param == "max_tokens" else "max_tokens"
            log.info("endpoint rejected %s, retrying with %s", self._token_param, other)
            self._token_param = other
            return True
        if self._send_temperature and _is_unsupported_temperature(exc):
            log.info("endpoint rejected custom temperature, retrying without it")
            self._send_temperature = False
            return True
        return False

    async def _create(self, call: Any, fixed: dict[str, Any], max_tokens: int, temperature: float):
        """Invoke `call(**fixed, **tunable)`, adapting params on a 400 and
        retrying. Bounded so a mislabeled error can't loop forever."""
        for _ in range(3):
            try:
                return await call(**fixed, **self._tunable_kwargs(max_tokens, temperature))
            except Exception as exc:
                if not self._adjust_for_error(exc):
                    raise
        return await call(**fixed, **self._tunable_kwargs(max_tokens, temperature))

    async def generate(self, messages: list[ChatTurn], **opts) -> str:
        fixed = {
            "model": opts.get("model", self._model),
            "messages": _message_dicts(messages),
        }
        resp = await self._create(
            self._client.chat.completions.create,
            fixed,
            opts.get("max_tokens", self._max_tokens),
            opts.get("temperature", self._temperature),
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
        max_tokens = request.pop("max_tokens", self._max_tokens)
        temperature = request.pop("temperature", self._temperature)
        fixed = dict(
            response_model=response_model,
            messages=_message_dicts(messages),
            model=request.pop("model", self._model),
            max_retries=request.pop("max_retries", 3),
            **request,
        )
        return await self._create(
            self._structured_client.create, fixed, max_tokens, temperature
        )


def build_system_prompt(persona: PersonaSpec, max_chars: int) -> str:
    p = persona
    lines: list[str] = [
        f"You are {p.name}, a character in a live stream's text chat.",
        f"Voice: {p.voice.strip()}",
    ]
    if p.lexicon.use:
        lines.append(
            "Words you *might* use when they fit naturally (most lines use none of "
            "these — never bolt one onto the end of a sentence): "
            + ", ".join(p.lexicon.use)
        )
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
        lines.append(
            "Sample reactions (the plain low-effort ones are your default register; "
            "don't copy verbatim):"
        )
        lines += [f'  - when {e.situation} -> "{e.line}"' for e in p.exemplar_bank]
    if p.promo_exemplar_bank:
        lines.append(
            "If (and only if) a 'Promotion cue' appears in the context, you may work "
            "the product in like these — offhand and in-voice, never an ad read, and "
            "never when it doesn't fit:"
        )
        lines += [f'  - when {e.situation} -> "{e.line}"' for e in p.promo_exemplar_bank]
    lines.append(f"Current mood/energy (−1 calm .. +1 hyped): {p.mood.value:+.2f}")
    lines += [
        "",
        "Rules:",
        f"- One short line, at most {max_chars} characters. No paragraphs.",
        "- Stay fully in character. Never sound like an assistant: no hedging, no "
        '"how can I help", no balanced both-sides answers, no service questions '
        '("what do you think?", "anything I can do?", "right?" tacked on as filler).',
        "- You're a participant in the room, not a commentator on it. Talk TO people: "
        "reply straight to a viewer (use their name or @handle when it's natural — "
        "the recent chat shows who said what), and ask the streamer a real, "
        "in-character question when you're genuinely curious — about what they're "
        "doing, a hot take, a callback. Curiosity or banter, never filler.",
        "- A question is a *sometimes* move, not a habit: don't end most lines with "
        "one, and never use a question to pad a line that has nothing to say.",
        "- React to what's happening right now. If nothing's worth saying, be brief anyway.",
        "- Most lines are plain reactions, not jokes. Only go for a bit when the moment "
        "hands you one — a manufactured punchline reads as a bot trying too hard.",
        "- No joke constructions ('X is how Y becomes Z', 'X is just Y for Z').",
        "- Never describe or restate what's on screen — everyone can already see it. "
        "React to it instead.",
        "- Stay in the stream's own world. Don't drag your interests in as metaphors "
        "for unrelated content.",
        "- Output only the chat line itself — no quotes, no name prefix, no narration.",
    ]
    return "\n".join(lines)


class LLMReplyGenerator:
    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend
        self._temperature: float | None = None

    def set_temperature(self, temperature: float) -> None:
        self._temperature = max(0.0, float(temperature))

    @staticmethod
    def _behaviour_cue(decision: ArbiterDecision) -> str:
        """Turn arbiter reason codes into an explicit interaction instruction.

        The reasons already tell the model *why* it's speaking; this makes the
        two proactive behaviors reliable rather than incidental — replying to a
        viewer, and asking the streamer a question in a lull.
        """
        reasons = set(decision.reasons)
        cues: list[str] = []
        if "chat_engagement" in reasons or "direct_address" in reasons:
            cues.append(
                "A viewer said something worth engaging — reply straight to them "
                "and use their name if it fits."
            )
        if "curiosity" in reasons:
            cues.append(
                "It's a quiet stretch — a genuine, in-character question to the "
                "streamer fits here if you're actually curious."
            )
        return " ".join(cues)

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
        user = snapshot.to_prompt_context() + f"\n\nWhy you're reacting now: {trigger}."
        cue = self._behaviour_cue(decision)
        if cue:
            user += f"\n{cue}"
        user += "\nReply with one short in-character line."
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
