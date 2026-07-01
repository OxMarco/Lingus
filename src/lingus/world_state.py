"""The shared world-state (blackboard).

Perception modules WRITE timestamped events here; cognition (arbiter, generator)
READS this state and never touches raw streams. This decouples the three
perception channels — which arrive at wildly different rates — and gives the
generator one coherent, time-aligned context.

Single-event-loop design: all mutation happens on the asyncio loop thread.
Perception workers that run off-loop hand events back via asyncio.Queue (see
perception/*), so WorldState itself needs no locking.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .chat_trends import ChatTrend

EventSource = Literal["speech", "scene", "chat", "bot"]


@dataclass(slots=True)
class Event:
    """One timestamped thing that happened on the stream."""

    source: EventSource
    kind: str  # e.g. "transcript", "question", "hype_spike", "scene_change", "post"
    payload: dict[str, Any] = field(default_factory=dict)
    salience_hints: dict[str, float] = field(default_factory=dict)
    ts: float = field(default_factory=time.monotonic)

    def age(self, now: float | None = None) -> float:
        return (now if now is not None else time.monotonic()) - self.ts


def event_text(event: Event) -> str:
    """Best-effort text representation used by cognition heuristics."""
    if event.source in ("chat", "speech", "bot"):
        return str(event.payload.get("text", ""))
    if event.source == "scene":
        return " ".join(
            str(event.payload.get(key, ""))
            for key in ("activity", "setting", "on_screen_text", "last_event")
        )
    return str(event.payload.get("text", ""))


def event_summary(event: Event | None) -> str:
    """Human-readable event label for prompts and monitor payloads."""
    if event is None:
        return ""
    if event.source == "chat":
        author = event.payload.get("author", "chat")
        return f"chat/{event.kind}: {author}: {event_text(event)}"
    if event.source == "speech":
        return f"speech/{event.kind}: {event_text(event)}"
    if event.source == "scene":
        scene_event = event.payload.get("last_event", "")
        activity = event.payload.get("activity", "")
        return f"scene/{event.kind}: {scene_event or activity}"
    if event.source == "bot":
        return f"bot/{event.kind}: {event_text(event)}"
    return f"{event.source}/{event.kind}"


@dataclass(slots=True)
class SceneState:
    """Structured running description of the video, updated on real change."""

    activity: str = ""
    setting: str = ""
    on_screen_text: str = ""
    salient_objects: list[str] = field(default_factory=list)
    last_event: str = ""
    updated_ts: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class ChatState:
    """Aggregated view of the chat firehose (not raw messages)."""

    questions_to_bot: list[str] = field(default_factory=list)
    hype_level: float = 0.0  # 0..1 rolling sentiment/volume spike indicator
    emergent_topics: list[str] = field(default_factory=list)
    spam_or_raid: bool = False
    # Most recent pile-on the bot chose to follow (observability for the dashboard).
    trend: ChatTrend | None = None
    updated_ts: float = field(default_factory=time.monotonic)


class WorldState:
    """In-memory blackboard. One instance per running bot."""

    def __init__(
        self,
        event_window: int = 500,
        transcript_window: int = 40,
        self_memory_window: int = 20,
    ) -> None:
        self.events: deque[Event] = deque(maxlen=event_window)
        self.transcript: deque[str] = deque(maxlen=transcript_window)
        self.own_messages: deque[str] = deque(maxlen=self_memory_window)
        self.scene = SceneState()
        self.chat = ChatState()
        # Episodic memory: the running "stream so far" narrative, plus transcript
        # lines that have aged out of the working window and await summarization
        # (so a long stream's early context isn't simply lost on eviction).
        self.episodic_summary: str = ""
        self._pending_summary: list[str] = []
        # Top durable facts currently surfaced from the semantic store (refreshed
        # by the consolidation loop; read into the generator's context).
        self.semantic_facts: list[str] = []

    # --- writes (called by perception) ---
    def add_event(self, event: Event) -> None:
        self.events.append(event)
        if event.source == "speech" and event.kind == "transcript":
            text = event.payload.get("text", "")
            if text:
                # Capture the line about to fall out of the working window before
                # the deque evicts it, so episodic summarization can fold it in.
                if len(self.transcript) == self.transcript.maxlen:
                    self._pending_summary.append(self.transcript[0])
                self.transcript.append(text)

    def record_own_message(self, text: str) -> None:
        """Self-memory: what the bot itself said, fed back in + used for dedup."""
        self.own_messages.append(text)
        self.add_event(Event(source="bot", kind="post", payload={"text": text}))

    def pending_summary_count(self) -> int:
        return len(self._pending_summary)

    def drain_pending_summary(self) -> list[str]:
        """Hand the evicted-but-unsummarized lines to the summarizer and clear them."""
        lines = self._pending_summary
        self._pending_summary = []
        return lines

    def set_episodic_summary(self, summary: str) -> None:
        self.episodic_summary = summary

    def update_scene(self, scene: SceneState) -> None:
        scene.updated_ts = time.monotonic()
        self.scene = scene

    def update_chat(self, chat: ChatState) -> None:
        chat.updated_ts = time.monotonic()
        self.chat = chat

    # --- reads (called by cognition) ---
    def recent_events(self, max_age: float | None = None) -> list[Event]:
        if max_age is None:
            return list(self.events)
        now = time.monotonic()
        return [e for e in self.events if e.age(now) <= max_age]

    def last_event(self) -> Event | None:
        return self.events[-1] if self.events else None

    def recent_transcript(self, n: int | None = None) -> str:
        items = list(self.transcript)
        if n is not None:
            items = items[-n:]
        return " ".join(items)

    def seconds_since_own_message(self) -> float:
        for event in reversed(self.events):
            if event.source == "bot":
                return event.age()
        return float("inf")
