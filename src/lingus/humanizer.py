"""The humanizer — a deterministic pass that strips AI-tell punctuation and,
optionally, sprinkles in human-shaped typos.

Hosted LLMs have a recognizable typographic accent: the em-dash used as a
free-floating clause break (``great — really great``), curly "smart" quotes,
the single-glyph ellipsis ``…`` — and, more subtly, *flawless spelling*. None of
these are wrong, but together they read as machine-written in a live chat where
humans type on a phone keyboard, and that read is the fastest way to break the
illusion the persona is sustaining (CLAUDE.md §5 — assistant-voice leak, at the
typographic layer).

Two passes, both deterministic and in the hot path so no model gets to overrule
them:

  * **punctuation** — em-dash / spaced en-dash → a natural substitute, smart
    quotes straightened, single-glyph ellipsis expanded. Safe on any text, so it
    runs on verbatim trend mirrors too.
  * **typos** — an opt-in flavor pass that introduces a human-shaped fingerslip
    into a *long* word with some probability. Off by default; the rate is
    live-tunable from the web UI. It runs only on the bot's *own generated*
    voice — never on mirrored chat, where a corrupted emote (``POGGERS`` →
    ``POGGRES``) would simply fail to render.

    The slip is drawn from the errors people actually make at speed, not a
    single mechanical transposition (which produces tells of its own —
    ``healthiest`` → ``healtihest`` reads as *fake*, because nobody's finger
    slips that way). The mix, weighted toward the common cases:
      - **drop** a letter — ``healthiest`` → ``healtiest`` (the finger never
        lands);
      - **substitute** a *QWERTY-adjacent* key — ``healthiest`` → ``healthirst``
        (``r`` sits next to ``e``);
      - **transpose** two adjacent letters — a real but rarer slip;
      - **double** a letter — a key that bounces.
    The first and last letter are always left intact, so the word stays
    readable (people misread scrambled interiors far more forgivingly than
    scrambled ends).

The humanizer runs *before* the output governor so the length cap measures the
final string (a substitution can change its length) — the exact string that
will be posted.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

# Em-dash in any spacing, plus a *spaced* en-dash used as a clause break. An
# unspaced en-dash (``3–4``) is a numeric range, not an AI tell, so it is left
# alone.
_EM_DASH = re.compile(r"\s*—\s*|\s+–\s+")
_SMART_DOUBLE = {"“": '"', "”": '"'}
_SMART_SINGLE = {"‘": "'", "’": "'"}
_ELLIPSIS = re.compile(r"…")
# After a comma-style replacement we can end up with doubled or misplaced
# punctuation (``done, , then`` / ``done ,then``); normalize those away.
_DOUBLE_PUNCT = re.compile(r"\s*,\s*(?=[,.!?])")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?])")
_MULTISPACE = re.compile(r"[ \t]{2,}")

# QWERTY key neighbours (letters only) — the physically-plausible mistypes for a
# given key. Used by the substitution slip so a fingerslip lands on an *adjacent*
# key (``e`` → ``r``/``w``/``s``/``d``), never a random letter across the board.
_QWERTY_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "rygf",
    "y": "tuhg", "u": "yijh", "i": "uokj", "o": "iplk", "p": "ol",
    "a": "qwsz", "s": "wedxza", "d": "erfcxs", "f": "rtgvcd", "g": "tyhbvf",
    "h": "yujnbg", "j": "uikmnh", "k": "iolmj", "l": "opk",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}


@dataclass(slots=True)
class Humanizer:
    """Strip AI-tell punctuation (and optionally add typos) before a message posts."""

    enabled: bool = True
    em_dash_replacement: str = ", "
    straighten_quotes: bool = True
    normalize_ellipsis: bool = True
    # Typo pass (opt-in flavor; rate live-tunable from the web UI).
    typo_enabled: bool = False
    typo_rate: float = 0.0  # per-eligible-word probability of a slip
    typo_min_word_len: int = 7  # only "long / complex" words are eligible
    typo_max_per_message: int = 2  # cap so a line never reads as garbled
    # Injectable for reproducible tests; falls back to the module RNG.
    rng: random.Random | None = None

    def humanize(self, text: str, *, introduce_typos: bool = True) -> str:
        """Return `text` with AI-tell punctuation stripped.

        `introduce_typos` gates *only* the typo pass — punctuation cleanup always
        runs. Pass ``False`` for verbatim mirrors so a copied emote is never
        mangled.
        """
        if not self.enabled or not text:
            return text

        out = _EM_DASH.sub(self.em_dash_replacement, text)

        if self.straighten_quotes:
            for src, dst in {**_SMART_DOUBLE, **_SMART_SINGLE}.items():
                out = out.replace(src, dst)

        if self.normalize_ellipsis:
            out = _ELLIPSIS.sub("...", out)

        # Clean up seams the replacement may have introduced.
        out = _DOUBLE_PUNCT.sub("", out)
        out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
        out = _MULTISPACE.sub(" ", out).strip()

        if introduce_typos and self.typo_enabled and self.typo_rate > 0.0:
            out = self._introduce_typos(out)
        return out

    def _introduce_typos(self, text: str) -> str:
        rng = self.rng or random
        budget = self.typo_max_per_message
        words = text.split(" ")
        for i, word in enumerate(words):
            if budget <= 0:
                break
            if not self._eligible(word) or rng.random() >= self.typo_rate:
                continue
            slipped = self._slip(word, rng)
            if slipped != word:
                words[i] = slipped
                budget -= 1
        return " ".join(words)

    def _eligible(self, word: str) -> bool:
        # Pure-alpha only: skips URLs, @mentions, digits and punctuation-glued
        # tokens, so a fingerslip can never break a link or a handle.
        return len(word) >= self.typo_min_word_len and word.isalpha()

    # Slip kinds and their weights — drop/substitute dominate because those are
    # the mistypes people actually make; transpose and double are the long tail.
    _SLIP_KINDS = ("drop", "substitute", "transpose", "double")
    _SLIP_WEIGHTS = (4, 4, 2, 1)

    @classmethod
    def _slip(cls, word: str, rng) -> str:
        """Introduce one human-shaped fingerslip, keeping the first/last letter
        intact for readability. Tries each slip kind (in a shuffled-by-weight
        order) until one actually changes the word, so an eligible word is never
        silently left untouched (e.g. a transpose on a doubled letter)."""
        if len(word) < 4:  # need at least one interior letter to slip
            return word
        kinds = list(cls._SLIP_KINDS)
        # Draw kinds in weighted-preference order, without replacement.
        order: list[str] = []
        weights = list(cls._SLIP_WEIGHTS)
        while kinds:
            pick = rng.choices(range(len(kinds)), weights=weights, k=1)[0]
            order.append(kinds.pop(pick))
            weights.pop(pick)
        for kind in order:
            slipped = cls._apply_slip(kind, word, rng)
            if slipped != word:
                return slipped
        return word

    @staticmethod
    def _apply_slip(kind: str, word: str, rng) -> str:
        chars = list(word)
        interior = range(1, len(chars) - 1)  # never the first or last letter
        if kind == "transpose":
            if len(chars) < 4:
                return word
            i = rng.randint(1, len(chars) - 3)  # i, i+1 both interior
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            return "".join(chars)
        i = rng.choice(interior)
        if kind == "drop":
            del chars[i]
            return "".join(chars)
        if kind == "double":
            chars.insert(i, chars[i])
            return "".join(chars)
        if kind == "substitute":
            neighbors = _QWERTY_NEIGHBORS.get(chars[i].lower())
            if not neighbors:
                return word
            repl = rng.choice(neighbors)
            chars[i] = repl.upper() if chars[i].isupper() else repl
            return "".join(chars)
        return word


def build_humanizer(cfg) -> Humanizer:
    """Construct the humanizer from a ``HumanizerConfig`` (or any duck-typed cfg)."""
    return Humanizer(
        enabled=cfg.enabled,
        em_dash_replacement=cfg.em_dash_replacement,
        straighten_quotes=cfg.straighten_quotes,
        normalize_ellipsis=cfg.normalize_ellipsis,
        typo_enabled=cfg.typo_enabled,
        typo_rate=cfg.typo_rate,
        typo_min_word_len=cfg.typo_min_word_len,
        typo_max_per_message=cfg.typo_max_per_message,
    )
