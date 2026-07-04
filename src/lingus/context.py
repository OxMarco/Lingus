"""Build compact LLM-ready context from the shared world-state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .world_state import ChatState, Event, SceneState, WorldState, event_summary


@dataclass(slots=True)
class ChatLine:
    author: str
    text: str


@dataclass(slots=True)
class ContextSnapshot:
    """The cognition layer reads this, not raw stream firehoses."""

    transcript: str
    recent_chat: list[ChatLine]
    scene: SceneState
    chat_state: ChatState
    own_messages: list[str]
    latest_event: Event | None
    now: float
    episodic: str = ""  # the running "stream so far" narrative
    episodic_history: list[str] = field(default_factory=list)  # prior stream summaries
    semantic_facts: list[str] = field(default_factory=list)  # durable cross-stream facts
    promo_hint: str = ""  # optional, relevance-gated product cue (see promotions.py)

    def scene_summary(self) -> str:
        parts = [
            self.scene.activity,
            self.scene.setting,
            self.scene.on_screen_text,
            ", ".join(self.scene.salient_objects),
            self.scene.last_event,
        ]
        return " | ".join(part for part in parts if part)

    def latest_event_summary(self) -> str:
        return event_summary(self.latest_event)

    def to_prompt_context(self) -> str:
        chat = "\n".join(f"- {line.author}: {line.text}" for line in self.recent_chat)
        own_messages = "\n".join(f"- {text}" for text in self.own_messages)
        history = "\n".join(f"- {summary}" for summary in self.episodic_history)
        facts = "\n".join(f"- {fact}" for fact in self.semantic_facts)
        sections = [
            "Current stream context:",
            f"Known facts:\n{facts or '- none yet'}",
            f"Past stream memories:\n{history or '- none yet'}",
            f"Stream so far: {self.episodic or 'just started'}",
            f"Scene: {self.scene_summary() or 'unknown'}",
            f"Recent speech: {self.transcript or 'none'}",
            f"Recent chat:\n{chat or '- none'}",
            f"Recent bot messages:\n{own_messages or '- none'}",
            f"Reply trigger: {self.latest_event_summary() or 'none'}",
        ]
        if self.promo_hint:
            sections.append(f"Promotion cue: {self.promo_hint}")
        return "\n".join(sections)


def build_context_snapshot(
    world: WorldState,
    *,
    max_age: float | None = 30.0,
    transcript_turns: int = 6,
    chat_messages: int = 8,
    now: float | None = None,
) -> ContextSnapshot:
    current_time = time.monotonic() if now is None else now
    events = world.recent_events(max_age=max_age)
    latest_event = events[-1] if events else world.last_event()
    chat_events = [event for event in events if event.source == "chat" and event.kind == "message"][
        -chat_messages:
    ]
    recent_chat = [
        ChatLine(
            author=str(event.payload.get("author", "anon")),
            text=str(event.payload.get("text", "")),
        )
        for event in chat_events
    ]
    return ContextSnapshot(
        transcript=world.recent_transcript(transcript_turns),
        recent_chat=recent_chat,
        scene=world.scene,
        chat_state=world.chat,
        own_messages=list(world.own_messages),
        latest_event=latest_event,
        now=current_time,
        episodic=world.episodic_summary,
        episodic_history=list(world.episodic_history),
        semantic_facts=list(world.semantic_facts),
    )
