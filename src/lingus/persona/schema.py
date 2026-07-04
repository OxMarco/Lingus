"""PersonaSpec — the character as a structured artifact, not a prose blurb.

Personality is distributed across three subsystems (arbiter timing, generator
voice, memory callbacks). This schema is the generator's half of that: voice,
lexicon, opinions, relationships, and — most importantly — a concrete
*exemplar bank* of sample reactions (not adjective lists). A bounded, decaying
mood nudges both phrasing and arbiter thresholds.

Reference design: SillyTavern character cards + lorebooks.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Lexicon(BaseModel):
    use: list[str] = Field(default_factory=list)  # words/phrases it reaches for
    avoid: list[str] = Field(default_factory=list)  # words/phrases it never uses
    catchphrases: list[str] = Field(default_factory=list)


class Exemplar(BaseModel):
    """A concrete sample reaction: 'in situation X, it might say Y'."""

    situation: str
    line: str


class Relationship(BaseModel):
    who: str  # "streamer", "regulars", a specific name
    stance: str  # how the bot relates to them


class Mood(BaseModel):
    """Bounded, decaying energy variable. Kept in [min, max]; decays to baseline."""

    value: float = 0.0
    baseline: float = 0.0
    minimum: float = -1.0
    maximum: float = 1.0
    decay_per_minute: float = 0.2

    def nudge(self, delta: float) -> None:
        self.value = max(self.minimum, min(self.maximum, self.value + delta))

    def decay(self, minutes: float) -> None:
        gap = self.value - self.baseline
        step = self.decay_per_minute * minutes
        if abs(gap) <= step:
            self.value = self.baseline
        else:
            self.value -= step if gap > 0 else -step


class PersonaSpec(BaseModel):
    name: str
    voice: str  # tone description
    lexicon: Lexicon = Field(default_factory=Lexicon)
    interests: list[str] = Field(default_factory=list)
    opinions: list[str] = Field(default_factory=list)  # it has takes; can be wrong in-character
    relationships: list[Relationship] = Field(default_factory=list)
    boundaries: list[str] = Field(default_factory=list)  # what it won't engage with
    exemplar_bank: list[Exemplar] = Field(default_factory=list)
    # How a plug should sound when a promotion cue is present: offhand, in-voice,
    # never an ad read. Only used when config.promotions is enabled.
    promo_exemplar_bank: list[Exemplar] = Field(default_factory=list)
    mood: Mood = Field(default_factory=Mood)
