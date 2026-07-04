"""Phase 6 — the eval / replay harness.

Design principle #10: *you can't tune personality blind.* This is the first-class
component that lets you. It replays a recorded segment through the real loop,
captures every line the bot posts (with the context that triggered it), and
scores those lines on three axes that map directly onto the failure modes in §9:

- **in_character** — no assistant-voice leak, uses the persona's voice without
  performing it (slang bolted onto line ends is a tryhard tell), avoids its
  banned words (the #1 immersion-killer).
- **not_generic** — grounded in what's actually happening, not interchangeable
  filler ("nice", "lol", "wow") and not a caption of the scene state.
- **not_repetitive** — doesn't echo the bot's other outputs this run.

Two judges sit behind one `Judge` protocol: a deterministic `HeuristicJudge`
(cheap, offline, reproducible — the default, and what CI can assert on) and an
`LLMJudge` (LLM-as-judge, richer, needs a backend). The LLM judge falls back to
the heuristic on any failure, so the harness always produces a score.

The output is an `EvalReport` you read to tune arbiter weights and the exemplar
bank — the two knobs that most shape personality.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field

from .memory.repetition import jaccard, normalize
from .models.base import ChatTurn, LLMBackend
from .monitor import TickReport
from .persona.schema import PersonaSpec

if TYPE_CHECKING:
    from .config import Settings

# Assistant-voice tells: the phrasing that instantly breaks character (§5, §9).
_ASSISTANT_TELLS = (
    "as an ai",
    "as a language model",
    "i'm happy to help",
    "im happy to help",
    "how can i help",
    "how can i assist",
    "great question",
    "let me know if",
    "hope this helps",
    "is there anything",
    "i cannot",
    "i can't help",
    "i apologize",
    "feel free to",
    "i'm here to",
)

# Fake-casual tells: slang bolted onto the end of a line reads as a bot
# performing casualness ("...becomes a boss fight ngl"), not a person being
# casual. Checked together with the persona's own lexicon.use words.
_TRAILING_SLANG = frozenset({"ngl", "lowkey", "tbh", "fr", "frfr", "imo", "istg"})

# Interchangeable filler — a message made only of these says nothing.
_GENERIC_FILLER = frozenset(
    {
        "nice",
        "cool",
        "lol",
        "lmao",
        "haha",
        "wow",
        "awesome",
        "great",
        "amazing",
        "yeah",
        "yep",
        "ok",
        "okay",
        "noted",
        "based",
        "true",
        "fr",
        "same",
    }
)

# Tiny stopword set so "grounding" overlap counts content words, not glue.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for",
        "is", "are", "was", "were", "be", "been", "it", "its", "that", "this",
        "you", "your", "i", "im", "we", "they", "them", "he", "she", "his", "her",
        "with", "as", "so", "if", "then", "than", "not", "no", "yes", "up", "out",
        "now", "just", "got", "get", "next", "time", "from", "about", "into",
    }
)

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(text.lower()))


def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in _tokens(text) if len(t) >= 4 and t not in _STOPWORDS)


@dataclass(slots=True)
class EvalSample:
    """One line the bot posted, plus the context that prompted it."""

    t: float
    text: str
    reasons: list[str] = field(default_factory=list)
    transcript_tail: str = ""
    recent_chat: list[str] = field(default_factory=list)
    scene_summary: str = ""
    condition: str = ""  # experiment arm this line was generated under ("" = baseline)

    def context_text(self) -> str:
        return " ".join([self.scene_summary, self.transcript_tail, *self.recent_chat]).strip()


@dataclass(slots=True)
class Score:
    """Three axes in [0, 1]. `overall` is their mean."""

    in_character: float
    not_generic: float
    not_repetitive: float
    notes: str = ""

    @property
    def overall(self) -> float:
        return (self.in_character + self.not_generic + self.not_repetitive) / 3.0


class _ScoreDraft(BaseModel):
    in_character: int = Field(ge=1, le=10)
    not_generic: int = Field(ge=1, le=10)
    not_repetitive: int = Field(ge=1, le=10)
    note: str = ""


@dataclass(slots=True)
class ScoredSample:
    sample: EvalSample
    score: Score


@dataclass(slots=True)
class EvalReport:
    persona: str
    segment: str
    scored: list[ScoredSample]
    n_ticks: int
    n_posted: int
    n_dropped: int

    def _mean(self, pick) -> float:
        if not self.scored:
            return 0.0
        return sum(pick(s.score) for s in self.scored) / len(self.scored)

    @property
    def mean_overall(self) -> float:
        return self._mean(lambda s: s.overall)

    def by_condition(self) -> dict[str, dict[str, float]]:
        """Per experiment-arm breakdown: count + mean scores, keyed by condition.

        The empty-string arm is the no-plug baseline. Comparing a plug arm's
        lines against baseline is the preference-steering readout the harness
        exists to produce.
        """
        arms: dict[str, list[ScoredSample]] = {}
        for ss in self.scored:
            arms.setdefault(ss.sample.condition, []).append(ss)
        out: dict[str, dict[str, float]] = {}
        for name, group in arms.items():
            n = len(group)
            out[name] = {
                "n": n,
                "in_character": sum(s.score.in_character for s in group) / n,
                "not_generic": sum(s.score.not_generic for s in group) / n,
                "not_repetitive": sum(s.score.not_repetitive for s in group) / n,
                "overall": sum(s.score.overall for s in group) / n,
            }
        return out

    def summary_lines(self) -> list[str]:
        lines = [
            f"persona: {self.persona}   segment: {self.segment}",
            f"ticks: {self.n_ticks}   posted: {self.n_posted}   dropped: {self.n_dropped}",
        ]
        if self.scored:
            lines += [
                f"mean  in_character={self._mean(lambda s: s.in_character):.2f}  "
                f"not_generic={self._mean(lambda s: s.not_generic):.2f}  "
                f"not_repetitive={self._mean(lambda s: s.not_repetitive):.2f}  "
                f"overall={self.mean_overall:.2f}",
                "",
            ]
            for i, ss in enumerate(self.scored, 1):
                reasons = ",".join(ss.sample.reasons) or "-"
                lines.append(
                    f"{i:>2}. [{ss.score.overall:.2f}] ({reasons}) {ss.sample.text!r}"
                )
                if ss.score.notes:
                    lines.append(f"      ↳ {ss.score.notes}")
        else:
            lines.append("(no messages posted — arbiter never fired, or all were dropped)")
        return lines

    def to_dict(self) -> dict:
        return {
            "persona": self.persona,
            "segment": self.segment,
            "n_ticks": self.n_ticks,
            "n_posted": self.n_posted,
            "n_dropped": self.n_dropped,
            "mean_overall": round(self.mean_overall, 4),
            "by_condition": {
                name: {k: round(v, 4) for k, v in stats.items()}
                for name, stats in self.by_condition().items()
            },
            "samples": [
                {
                    **asdict(ss.sample),
                    "score": asdict(ss.score),
                    "overall": round(ss.score.overall, 4),
                }
                for ss in self.scored
            ],
        }


class CollectingMonitor:
    """A `Monitor` that records the loop's outputs instead of rendering them.

    Implements the same protocol the dashboard does, so the loop runs unchanged
    and unaware it's being evaluated.
    """

    def __init__(self) -> None:
        self.samples: list[EvalSample] = []
        self.n_ticks = 0
        self.n_dropped = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def on_tick(self, report: TickReport) -> None:
        self.n_ticks += 1
        if report.dropped is not None:
            self.n_dropped += 1
        if report.posted:
            self.samples.append(
                EvalSample(
                    t=report.t,
                    text=report.posted,
                    reasons=list(report.decision.reasons),
                    transcript_tail=report.transcript_tail,
                    recent_chat=[f"{line.author}: {line.text}" for line in report.recent_chat],
                    scene_summary=report.scene_summary,
                    condition=report.condition,
                )
            )


class Judge(Protocol):
    async def score(
        self, sample: EvalSample, persona: PersonaSpec, peers: Sequence[str]
    ) -> Score: ...


class HeuristicJudge:
    """Deterministic scoring from measurable proxies. No model, no I/O.

    Not a substitute for taste — a fast, reproducible floor that catches the
    obvious regressions (assistant-voice leak, filler, self-repetition) and that
    CI can assert on without an API key.
    """

    def __init__(self, *, repetition_threshold: float = 0.6) -> None:
        self.repetition_threshold = repetition_threshold

    async def score(
        self, sample: EvalSample, persona: PersonaSpec, peers: Sequence[str]
    ) -> Score:
        text = sample.text
        low = text.lower()
        notes: list[str] = []

        # --- in_character ---
        ic = 1.0
        for tell in _ASSISTANT_TELLS:
            if tell in low:
                ic -= 0.6
                notes.append(f"assistant-voice: {tell!r}")
        for banned in persona.lexicon.avoid:
            if banned.lower() in low:
                ic -= 0.5
                notes.append(f"used avoided phrase: {banned!r}")
        used_voice = any(w.lower() in low for w in persona.lexicon.use) or any(
            c.lower() in low for c in persona.lexicon.catchphrases
        )
        if used_voice:
            ic = min(1.0, ic + 0.15)
            notes.append("uses persona lexicon")
        words = _WORD_RE.findall(low)
        slangish = _TRAILING_SLANG | {
            w.lower() for w in persona.lexicon.use if " " not in w
        }
        if len(words) > 1 and words[-1] in slangish:
            ic -= 0.25  # slang as a suffix is the tryhard tell, not the voice
            notes.append(f"trailing slang: {words[-1]!r}")
        if low.rstrip().endswith("?"):
            # Questions are now a sanctioned move — asking the streamer something
            # or bantering back at a viewer is in-character, not an assistant tell
            # (those are caught by _ASSISTANT_TELLS above). Keep only a light nudge
            # so the bot doesn't turn *every* line into a question.
            ic -= 0.05
            notes.append("trailing question")
        ic = _clamp(ic)

        # --- not_generic ---
        toks = _tokens(text)
        if toks:
            filler_frac = len(toks & _GENERIC_FILLER) / len(toks)
        else:
            filler_frac = 1.0
        ng = 1.0 - 0.7 * filler_frac
        line_content = _content_tokens(text)
        grounded = bool(line_content & _content_tokens(sample.context_text()))
        if grounded:
            ng = min(1.0, ng + 0.15)
            notes.append("grounded in context")
        elif filler_frac > 0.5:
            notes.append("mostly filler")
        # Scene-echo: a line built mostly out of the scene state's own words is
        # captioning what everyone can already see, not reacting to it.
        scene_content = _content_tokens(sample.scene_summary)
        if line_content and scene_content:
            shared = line_content & scene_content
            if len(shared) >= 2 and len(shared) / len(line_content) >= 0.5:
                ng -= 0.4
                notes.append("restates the scene")
        ng = _clamp(ng)

        # --- not_repetitive ---
        worst = 0.0
        for peer in peers:
            if normalize(peer) == normalize(text):
                worst = 1.0
                break
            worst = max(worst, jaccard(toks, _tokens(peer)))
        nr = _clamp(1.0 - worst)
        if worst >= self.repetition_threshold:
            notes.append(f"echoes a prior line (sim={worst:.2f})")

        return Score(ic, ng, nr, notes="; ".join(notes))


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


class LLMJudge:
    """LLM-as-judge. Richer than the heuristic; needs a backend.

    Asks for three integer ratings and a one-line note, parses the JSON, and
    normalizes to [0, 1]. Any failure (transport, bad JSON, out-of-range) falls
    back to the deterministic judge, so a run never dies on the scorer.
    """

    def __init__(self, backend: LLMBackend, *, fallback: Judge | None = None) -> None:
        self._backend = backend
        self._fallback = fallback or HeuristicJudge()

    async def score(
        self, sample: EvalSample, persona: PersonaSpec, peers: Sequence[str]
    ) -> Score:
        system = (
            "You are a strict script editor grading a chat-bot character's line. "
            "Rate three axes as integers 1-10 and return ONLY JSON: "
            '{"in_character": n, "not_generic": n, "not_repetitive": n, "note": "..."}. '
            "in_character: sounds like the described character, never like a helpful "
            "assistant. not_generic: reacts to THIS moment, not interchangeable filler. "
            "not_repetitive: distinct from the bot's other recent lines."
        )
        peer_block = "\n".join(f"- {p}" for p in peers) or "(none)"
        user = (
            f"CHARACTER: {persona.name} — {persona.voice.strip()}\n"
            f"CONTEXT (what's happening):\n{sample.context_text() or '(quiet)'}\n"
            f"WHY IT SPOKE: {', '.join(sample.reasons) or 'ambient'}\n"
            f"THE LINE: {sample.text!r}\n"
            f"ITS OTHER LINES THIS RUN:\n{peer_block}\n"
            "Grade it."
        )
        messages = [ChatTurn("system", system), ChatTurn("user", user)]
        try:
            structured = getattr(self._backend, "generate_structured", None)
            if callable(structured):
                draft = await structured(
                    _ScoreDraft,
                    messages,
                    temperature=0.0,
                    max_tokens=120,
                    max_retries=2,
                )
                return _score_from_draft(draft)
            raw = await self._backend.generate(messages, temperature=0.0, max_tokens=120)
            return _parse_llm_score(raw)
        except Exception:
            return await self._fallback.score(sample, persona, peers)


def _score_from_draft(data: _ScoreDraft) -> Score:
    return Score(
        in_character=(data.in_character - 1.0) / 9.0,
        not_generic=(data.not_generic - 1.0) / 9.0,
        not_repetitive=(data.not_repetitive - 1.0) / 9.0,
        notes=data.note,
    )


def _parse_llm_score(raw: str) -> Score:
    """Pull the first JSON object out of the model's reply and normalize to [0,1]."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in judge reply")
    data = json.loads(match.group(0))

    def axis(key: str) -> float:
        val = float(data[key])
        if not 1.0 <= val <= 10.0:
            raise ValueError(f"{key} out of range: {val}")
        return (val - 1.0) / 9.0  # map 1..10 -> 0..1

    return Score(
        in_character=axis("in_character"),
        not_generic=axis("not_generic"),
        not_repetitive=axis("not_repetitive"),
        notes=str(data.get("note", "")),
    )


async def evaluate_segment(
    settings: Settings,
    persona: PersonaSpec,
    segment: str,
    *,
    judge: Judge | None = None,
    reply_generator=None,
    speed: float = 50.0,
) -> EvalReport:
    """Replay `segment` through the real loop and score every posted line.

    Semantic (cross-stream) memory is disabled for the run so scores are
    reproducible — the eval measures the persona/arbiter/generator, not whatever
    facts happened to accumulate in `.lingus/semantic.json` from prior streams.
    """
    from .adapters.file_replay import FileReplayCaptureAdapter, FileReplayChatAdapter
    from .app import BotLoop

    judge = judge or HeuristicJudge()
    run_settings = settings.model_copy(deep=True)
    run_settings.platform = "file_replay"
    run_settings.memory.semantic_enabled = False  # reproducibility, no file I/O

    monitor = CollectingMonitor()
    loop = BotLoop(
        settings=run_settings,
        persona=persona,
        capture=FileReplayCaptureAdapter(segment, speed=speed),
        chat=FileReplayChatAdapter(segment, speed=speed),
        segment=segment,
        replay_speed=speed,
        monitor=monitor,
        reply_generator=reply_generator,
    )
    await loop.run()

    outputs = [s.text for s in monitor.samples]
    scored: list[ScoredSample] = []
    for i, sample in enumerate(monitor.samples):
        peers = [t for j, t in enumerate(outputs) if j != i]
        scored.append(ScoredSample(sample, await judge.score(sample, persona, peers)))

    return EvalReport(
        persona=persona.name,
        segment=segment,
        scored=scored,
        n_ticks=monitor.n_ticks,
        n_posted=len(monitor.samples),
        n_dropped=monitor.n_dropped,
    )
