from lingus.world_state import Event, WorldState


def test_transcript_accumulates_from_speech_events():
    ws = WorldState()
    ws.add_event(Event(source="speech", kind="transcript", payload={"text": "hello"}))
    ws.add_event(Event(source="speech", kind="transcript", payload={"text": "world"}))
    assert ws.recent_transcript() == "hello world"
    assert ws.recent_transcript(1) == "world"


def test_own_messages_recorded_as_self_memory_and_event():
    ws = WorldState()
    ws.record_own_message("called it")
    assert list(ws.own_messages) == ["called it"]
    last = ws.last_event()
    assert last is not None and last.source == "bot"


def test_seconds_since_own_message_is_inf_when_none():
    ws = WorldState()
    assert ws.seconds_since_own_message() == float("inf")


def test_recent_events_filters_by_age():
    ws = WorldState()
    ws.add_event(Event(source="chat", kind="message", payload={"text": "x"}, ts=0.0))
    # ancient event filtered out under a tiny window
    assert ws.recent_events(max_age=0.01) == []
