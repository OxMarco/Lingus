"""Template generator: deterministic stand-ins for the LLM's behaviors.

These exist so offline replay/eval can exercise the full loop — including the
two proactive interaction paths (banter with a viewer, ask the streamer) —
without a hosted key.
"""

import pytest

from lingus.arbiter import ArbiterDecision
from lingus.context import build_context_snapshot
from lingus.generator import TemplateReplyGenerator
from lingus.persona.schema import PersonaSpec
from lingus.world_state import Event, WorldState


def _snapshot_with_chat(author: str, text: str):
    world = WorldState()
    world.add_event(Event(source="chat", kind="message", payload={"author": author, "text": text}))
    return build_context_snapshot(world)


@pytest.mark.asyncio
async def test_template_replies_straight_at_a_chatter_by_name():
    gen = TemplateReplyGenerator()
    snapshot = _snapshot_with_chat("thatguy", "honestly that build makes no sense")
    decision = ArbiterDecision(should_reply=True, score=1.5, reasons=["chat_engagement"])

    line = await gen.generate(
        snapshot, decision, PersonaSpec(name="Lingus", voice="brief"), max_chars=120
    )

    assert "@thatguy" in line


@pytest.mark.asyncio
async def test_template_asks_the_streamer_on_curiosity():
    gen = TemplateReplyGenerator()
    world = WorldState()
    world.add_event(Event(source="speech", kind="transcript", payload={"text": "ok so anyway"}))
    snapshot = build_context_snapshot(world)
    decision = ArbiterDecision(should_reply=True, score=1.2, reasons=["lull", "curiosity"])

    line = await gen.generate(
        snapshot, decision, PersonaSpec(name="Lingus", voice="brief"), max_chars=120
    )

    assert line.endswith("?") or "wait" in line.lower()


@pytest.mark.asyncio
async def test_template_prefers_a_matching_persona_exemplar():
    gen = TemplateReplyGenerator()
    persona = PersonaSpec.model_validate(
        {
            "name": "Lingus",
            "voice": "brief",
            "exemplar_bank": [
                {"situation": "banters with a chatter by name", "line": "@you that's cap"}
            ],
        }
    )
    snapshot = _snapshot_with_chat("you", "this take is objectively correct")
    decision = ArbiterDecision(should_reply=True, score=1.5, reasons=["chat_engagement"])

    line = await gen.generate(snapshot, decision, persona, max_chars=120)

    assert line == "@you that's cap"
