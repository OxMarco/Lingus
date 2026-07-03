from lingus.context import build_context_snapshot
from lingus.world_state import Event, SceneState, WorldState


def test_context_snapshot_combines_scene_speech_chat_and_trigger():
    world = WorldState()
    world.update_scene(
        SceneState(
            activity="streamer is cooking a chocolate cake",
            setting="home kitchen",
            salient_objects=["chocolate batter"],
            last_event="streamer got batter on their shirt",
        )
    )
    world.add_event(Event(source="speech", kind="transcript", payload={"text": "i got a stain"}))
    world.add_event(
        Event(source="chat", kind="message", payload={"author": "viewer", "text": "oh no"})
    )
    world.set_episodic_history(["last stream ended with a cake disaster"])

    snapshot = build_context_snapshot(world)

    assert "chocolate cake" in snapshot.scene_summary()
    assert snapshot.transcript == "i got a stain"
    assert snapshot.recent_chat[0].author == "viewer"
    assert snapshot.episodic_history == ["last stream ended with a cake disaster"]
    assert snapshot.latest_event_summary() == "chat/message: viewer: oh no"
    assert "Past stream memories:" in snapshot.to_prompt_context()
    assert "last stream ended with a cake disaster" in snapshot.to_prompt_context()
    assert "Reply trigger: chat/message: viewer: oh no" in snapshot.to_prompt_context()
