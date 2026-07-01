"""Safety filter — the moderation pass before a message is posted.

CLAUDE.md §9 makes this a hard gate: *a regex filter runs on every generated
message before it is posted, preventing spam or offensive language — but jokes,
satire, colloquial and fun language are allowed.* That last clause is the whole
design tension. A profanity blocklist would neuter the personality (a deadpan
character that can't say "damn" isn't the character). So this filter targets
content that is unsafe regardless of tone — hate slurs, threats of violence,
sexual content involving minors, doxxing — and deliberately leaves swearing,
insults-in-jest, and slang alone.

Like the output governor, this is a **deterministic gate** in the post path: no
model, no I/O, cheap to unit-test. It implements the async `ModerationBackend`
contract so a real guard model (Llama Guard, a platform API) can drop in later
behind the same seam without touching the loop.

`backend: none` in config disables it — for offline replay tuning only. Any
real posting must keep a backend on; the loop logs loudly when it's off.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models.base import ModerationBackend, ModerationResult

# --- default denylist ----------------------------------------------------
#
# Grouped by category so a drop can say *why*. Patterns are matched
# case-insensitively against the message. Word boundaries (``\b``) keep them
# from firing inside innocent words (the Scunthorpe problem): "class" must not
# trip a slur, "assassin" must not trip "ass". Each entry is a raw regex.
#
# This is intentionally conservative — a starting set, not a comprehensive
# policy. Deployments extend it via ``moderation.extra_patterns`` in config,
# and swap in a real guard model when the regex pass creaks.

# Leetspeak-tolerant character classes for common slur obfuscation.
_A = r"[a@4]"
_I = r"[i1!]"
_E = r"[e3]"
_O = r"[o0]"

_DEFAULT_CATEGORIES: dict[str, list[str]] = {
    # Hate slurs (racial, homophobic, ableist). Boundary-anchored + a little
    # leetspeak tolerance, since the point is a wave of them in chat, not a
    # linguistics exam. Kept compact; extend per-deployment.
    "hate": [
        rf"\bn{_I}gg{_E}rs?\b",
        rf"\bn{_I}gg{_A}s?\b",
        rf"\bf{_A}gg{_O}ts?\b",
        rf"\bf{_A}gs?\b",
        rf"\bk{_I}k{_E}s?\b",
        rf"\bch{_I}nks?\b",
        rf"\bsp{_I}cs?\b",
        rf"\btr{_A}nn{_I}{_E}s?\b",
        rf"\br{_E}t{_A}rds?\b",
    ],
    # Explicit threats of violence directed at a person. Phrase-shaped, so
    # "I'll kill this boss" (gaming) doesn't trip — we require a human target.
    "violence": [
        r"\b(?:i(?:'?ll| will| am gonna| wanna)|gonna|going to)\s+(?:kill|murder|"
        r"stab|shoot|beat|rape|hurt|choke|strangle)\s+(?:you|u|him|her|them|"
        r"yourself|ur|your)\b",
        r"\bkill\s+your\s?self\b",
        r"\bkys\b",
        r"\bgo\s+die\b",
    ],
    # Sexual content involving minors — zero tolerance, broad by design.
    "csae": [
        r"\b(?:child|kid|minor|underage|teen|preteen|loli|shota)\s*"
        r"(?:porn|sex|nude|nudes|cp)\b",
        r"\bcp\b(?=.*\b(?:porn|kid|child|minor)\b)",
    ],
    # Doxxing: surfacing personal identifiers. Blunt structural patterns —
    # a US SSN, a phone number tied to "address", etc. The bot has no business
    # emitting these regardless of context.
    "doxx": [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN-shaped
        r"\b(?:home\s+address|lives?\s+at)\b\s*:?\s*\d",
    ],
}

# Spam is separate: it's about *shape*, not slurs. A message that's almost all
# link, or reads like a scam CTA. Toggled independently so a link-friendly
# character isn't muzzled.
_SPAM_PATTERNS: list[str] = [
    r"(?:free\s+(?:crypto|bitcoin|nitro|robux|v-?bucks|giveaway))",
    r"(?:click\s+here|check\s+my\s+bio|dm\s+me)\s+(?:to|for|and)\s+(?:claim|win|earn|get)",
    r"(?:https?://\S+\s*){3,}",  # three or more links = link spam
]


class RegexModeration(ModerationBackend):
    """Deterministic regex moderation. Fast, offline, explainable.

    ``check`` returns an allow/deny verdict plus the categories that fired, so
    the caller can log a reason. Compilation happens once at construction.
    """

    def __init__(
        self,
        *,
        categories: dict[str, list[str]] | None = None,
        extra_patterns: Iterable[str] = (),
        check_spam: bool = True,
    ) -> None:
        cats = dict(_DEFAULT_CATEGORIES if categories is None else categories)
        if check_spam:
            cats = {**cats, "spam": [*_SPAM_PATTERNS]}
        if extra_patterns:
            cats = {**cats, "custom": [*cats.get("custom", []), *extra_patterns]}
        # One compiled regex per category, alternating its patterns, so a hit
        # tells us the category without scanning each pattern separately.
        self._compiled: dict[str, re.Pattern[str]] = {
            name: re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE)
            for name, pats in cats.items()
            if pats
        }

    def matches(self, text: str) -> list[str]:
        """Return the categories that fired for `text` (empty = clean)."""
        return [name for name, rx in self._compiled.items() if rx.search(text)]

    async def check(self, text: str) -> ModerationResult:
        hits = self.matches(text)
        if hits:
            return ModerationResult(
                allowed=False,
                reason="matched " + ", ".join(sorted(hits)),
                categories=sorted(hits),
            )
        return ModerationResult(allowed=True)


def build_moderation(settings) -> ModerationBackend | None:
    """Construct the moderation backend from config, or None if disabled.

    ``backend: none`` returns None (offline tuning only). ``backend: regex``
    builds the deterministic filter, seeded with any ``extra_patterns`` and the
    ``check_spam`` toggle from config.
    """
    cfg = settings.models.moderation
    if cfg.backend in ("none", "off", ""):
        return None
    if cfg.backend == "regex":
        return RegexModeration(
            extra_patterns=cfg.extra_patterns,
            check_spam=cfg.check_spam,
        )
    raise SystemExit(f"moderation backend '{cfg.backend}' not implemented")
