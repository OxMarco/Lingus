"""Live-tunable runtime controls — the write side of the web UI.

The Monitor stream lets a dashboard *watch* the loop; this lets one *steer* it.
`ControlState` is a single mutable bag of knobs the cognition loop consults at
the top of every tick (`apply()` pushes them onto the live arbiter / governor /
generator), so a slider move in the browser takes effect within one tick — no
restart, no config reload.

Two headline knobs plus a handful of raw parameters:
  * **chat_enabled** — the on/off switch. When off, the loop still perceives and
    scores (so you keep watching the bar move) but never posts.
  * **frequency** — a 0..1 "how chatty" macro. It scales the arbiter's fire
    threshold and cooldown around their configured baselines: higher frequency
    lowers the bar and shortens the cooldown, so the bot speaks more often.

`schema()` describes the knobs so the frontend can render generic controls and
stay in sync; nothing about the UI is hard-coded to a specific parameter set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .arbiter import SimpleArbiter
    from .config import Settings
    from .generator import ReplyGenerator
    from .output_governor import OutputGovernor


@dataclass(slots=True)
class ParamSpec:
    key: str
    label: str
    kind: str  # "bool" | "int" | "float"
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    help: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "min": self.minimum,
            "max": self.maximum,
            "step": self.step,
            "help": self.help,
        }


# The headline frequency macro maps [0,1] onto multipliers around the configured
# baseline: at 0.5 nothing changes; toward 1 the bar drops and the cooldown
# shrinks (chattier), toward 0 both rise (reticent).
_THRESHOLD_SPAN = 2.5  # threshold multiplier at frequency 0 (and its inverse at 1)
_COOLDOWN_SPAN = 2.0


class ControlState:
    def __init__(self, settings: Settings) -> None:
        self.chat_enabled: bool = True
        # Whether the bot piles onto chat trends (mirrors a cresting emote/phrase).
        # Independent of chat_enabled: you can leave normal chatting on but stop
        # the bot joining waves, or vice-versa.
        self.trends_enabled: bool = settings.chat_trends.enabled
        self.frequency: float = 0.5
        # Baselines the frequency macro scales around (captured once at startup).
        self._base_threshold: float = settings.arbiter.fire_threshold
        self._base_cooldown: float = settings.arbiter.cooldown_seconds
        # Directly tunable parameters, seeded from config.
        self.max_chars: int = settings.output.max_chars
        self.min_seconds_between_posts: float = settings.output.min_seconds_between_posts
        self.posts_per_minute: float = settings.output.posts_per_minute
        self.temperature: float = settings.models.llm.temperature

    # --- schema + values (for the frontend) ---
    @staticmethod
    def schema() -> list[dict[str, Any]]:
        specs = [
            ParamSpec("chat_enabled", "Chat enabled", "bool", help="Master on/off for posting"),
            ParamSpec(
                "trends_enabled", "Follow chat trends", "bool",
                help="Pile onto cresting emote/phrase waves in chat",
            ),
            ParamSpec(
                "frequency", "Interaction frequency", "float", 0.0, 1.0, 0.01,
                help="How chatty: scales the arbiter threshold + cooldown",
            ),
            ParamSpec(
                "max_chars", "Max message length", "int", 20, 500, 5,
                help="Hard length cap (sentence-aware truncation)",
            ),
            ParamSpec(
                "min_seconds_between_posts", "Min seconds between posts", "float", 0.0, 60.0, 0.5,
                help="Hard floor between any two posts",
            ),
            ParamSpec(
                "posts_per_minute", "Posts per minute (sustained)", "float", 0.5, 30.0, 0.5,
                help="Token-bucket sustained rate",
            ),
            ParamSpec(
                "temperature", "Generator temperature", "float", 0.0, 2.0, 0.05,
                help="LLM sampling temperature (ignored by the template generator)",
            ),
        ]
        return [s.as_dict() for s in specs]

    def values(self) -> dict[str, Any]:
        return {
            "chat_enabled": self.chat_enabled,
            "trends_enabled": self.trends_enabled,
            "frequency": self.frequency,
            "max_chars": self.max_chars,
            "min_seconds_between_posts": self.min_seconds_between_posts,
            "posts_per_minute": self.posts_per_minute,
            "temperature": self.temperature,
            # Derived read-outs so the UI can show what the macro resolved to.
            "_effective_threshold": round(self._effective_threshold(), 3),
            "_effective_cooldown": round(self._effective_cooldown(), 3),
        }

    # --- updates ---
    def set(self, key: str, value: Any) -> None:
        """Validate + apply one knob change from the UI. Unknown keys are ignored."""
        spec = next((s for s in self.schema() if s["key"] == key), None)
        if spec is None or not hasattr(self, key):
            return
        if spec["kind"] == "bool":
            setattr(self, key, bool(value))
            return
        if spec["kind"] == "int":
            value = int(round(float(value)))
        elif spec["kind"] == "float":
            value = float(value)
        if spec["min"] is not None:
            value = max(spec["min"], value)
        if spec["max"] is not None:
            value = min(spec["max"], value)
        setattr(self, key, value)

    # --- push onto the live objects (called each cognition tick) ---
    def apply(
        self,
        arbiter: SimpleArbiter,
        governor: OutputGovernor,
        reply_generator: ReplyGenerator,
    ) -> None:
        arbiter.fire_threshold = self._effective_threshold()
        arbiter.cooldown_seconds = self._effective_cooldown()
        arbiter.min_seconds_between_posts = self.min_seconds_between_posts
        governor.max_chars = self.max_chars
        governor.min_interval = self.min_seconds_between_posts
        governor.set_posts_per_minute(self.posts_per_minute)
        set_temp = getattr(reply_generator, "set_temperature", None)
        if callable(set_temp):
            set_temp(self.temperature)

    def _effective_threshold(self) -> float:
        return self._base_threshold * (_THRESHOLD_SPAN ** (1.0 - 2.0 * self.frequency))

    def _effective_cooldown(self) -> float:
        return max(1.0, self._base_cooldown * (_COOLDOWN_SPAN ** (1.0 - 2.0 * self.frequency)))
