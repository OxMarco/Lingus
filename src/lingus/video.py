"""Small video helpers for Phase 4.

Scene analysis should run only when something materially changes. This module
keeps that first gate deterministic and dependency-free, so it can run in tests
and before heavier OpenCV/PySceneDetect plumbing is installed.
"""

from __future__ import annotations

from .adapters.base import Frame
from .world_state import SceneState


class FrameGate:
    """Accept the first frame, then frames that differ enough from the last kept one."""

    def __init__(
        self,
        *,
        diff_threshold: float = 0.08,
        min_interval_seconds: float = 3.0,
        max_sample_pixels: int = 4096,
    ) -> None:
        if diff_threshold < 0.0 or diff_threshold > 1.0:
            raise ValueError("diff_threshold must be between 0 and 1")
        if min_interval_seconds < 0.0:
            raise ValueError("min_interval_seconds must be non-negative")
        if max_sample_pixels <= 0:
            raise ValueError("max_sample_pixels must be positive")
        self.diff_threshold = diff_threshold
        self.min_interval_seconds = min_interval_seconds
        self.max_sample_pixels = max_sample_pixels
        self._last: Frame | None = None

    def accept(self, frame: Frame) -> bool:
        if self._last is None:
            self._last = frame
            return True
        if frame.ts - self._last.ts < self.min_interval_seconds:
            return False
        if frame_difference(self._last, frame, self.max_sample_pixels) < self.diff_threshold:
            return False
        self._last = frame
        return True


def frame_difference(a: Frame, b: Frame, max_sample_pixels: int = 4096) -> float:
    """Mean absolute RGB difference normalized to 0..1."""
    if a.width != b.width or a.height != b.height:
        return 1.0
    expected = a.width * a.height * 3
    if len(a.data) != expected or len(b.data) != expected:
        return 1.0
    pixels = a.width * a.height
    if pixels <= 0:
        return 0.0
    step = max(1, pixels // max_sample_pixels)
    total = 0
    samples = 0
    for pixel in range(0, pixels, step):
        i = pixel * 3
        total += abs(a.data[i] - b.data[i])
        total += abs(a.data[i + 1] - b.data[i + 1])
        total += abs(a.data[i + 2] - b.data[i + 2])
        samples += 3
    return total / (samples * 255)


def scene_content_changed(before: SceneState, after: SceneState) -> bool:
    """Compare scene fields while ignoring timestamp churn."""
    return _scene_signature(before) != _scene_signature(after)


def scene_event_changed(before: SceneState, after: SceneState) -> bool:
    """Whether the user-visible scene event text is new."""
    return _norm(before.last_event) != _norm(after.last_event)


def _scene_signature(scene: SceneState) -> tuple[str, str, str, tuple[str, ...], str]:
    return (
        _norm(scene.activity),
        _norm(scene.setting),
        _norm(scene.on_screen_text),
        tuple(sorted({_norm(item) for item in scene.salient_objects if _norm(item)})),
        _norm(scene.last_event),
    )


def _norm(text: str) -> str:
    return " ".join(text.casefold().strip().split())
