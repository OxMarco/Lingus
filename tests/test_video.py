from lingus.adapters.base import Frame
from lingus.video import FrameGate, frame_difference, scene_content_changed, scene_event_changed
from lingus.world_state import SceneState


def _frame(byte: int, ts: float = 0.0, width: int = 2, height: int = 2) -> Frame:
    return Frame(data=bytes([byte]) * width * height * 3, width=width, height=height, ts=ts)


def test_frame_difference_normalizes_rgb_delta():
    assert frame_difference(_frame(0), _frame(255)) == 1.0
    assert frame_difference(_frame(10), _frame(10)) == 0.0


def test_frame_gate_accepts_first_then_material_changes():
    gate = FrameGate(diff_threshold=0.5, min_interval_seconds=0.0)

    assert gate.accept(_frame(0, ts=0.0))
    assert not gate.accept(_frame(0, ts=1.0))
    assert gate.accept(_frame(255, ts=2.0))


def test_frame_gate_respects_min_interval():
    gate = FrameGate(diff_threshold=0.1, min_interval_seconds=5.0)

    assert gate.accept(_frame(0, ts=0.0))
    assert not gate.accept(_frame(255, ts=4.0))
    assert gate.accept(_frame(255, ts=5.0))


def test_scene_content_changed_ignores_timestamp_only_changes():
    first = SceneState(activity="streamer cooking", updated_ts=1.0)
    same = SceneState(activity="streamer cooking", updated_ts=2.0)
    changed = SceneState(activity="streamer plating cake", updated_ts=2.0)

    assert not scene_content_changed(first, same)
    assert scene_content_changed(first, changed)


def test_scene_content_changed_ignores_object_order_case_and_spacing():
    first = SceneState(
        activity="Car parked",
        salient_objects=["Road", " car "],
        last_event="Car parked",
        updated_ts=1.0,
    )
    same = SceneState(
        activity="  car   parked ",
        salient_objects=["CAR", "road"],
        last_event="car parked",
        updated_ts=2.0,
    )

    assert not scene_content_changed(first, same)


def test_scene_event_changed_compares_normalized_last_event_only():
    first = SceneState(activity="car stopped", last_event="A black car parked")
    same_event = SceneState(activity="vehicle idling", last_event=" a BLACK   car parked ")
    new_event = SceneState(activity="vehicle driving", last_event="car started moving")

    assert not scene_event_changed(first, same_event)
    assert scene_event_changed(first, new_event)
