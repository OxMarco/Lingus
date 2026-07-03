"""Local video analysis for Phase 4.

This is intentionally conservative: it never sends frames to a hosted model and
does not pretend to identify objects or read text. It extracts cheap visual
signals that are still useful for timing: major brightness, contrast, and color
changes in the live feed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from math import sqrt
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..adapters.base import Frame
from ..logging import get_logger
from ..world_state import SceneState
from .base import VLMBackend

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FrameStats:
    brightness: float
    contrast: float
    red: float
    green: float
    blue: float
    dominant_color: str
    tone: str


class _SceneResponse(BaseModel):
    changed: bool = True
    activity: str = ""
    setting: str = ""
    on_screen_text: str = ""
    salient_objects: list[str] = Field(default_factory=list)
    last_event: str = ""


class LocalFrameAnalyzer(VLMBackend):
    """Dependency-light local frame analyzer.

    The output is scene-state, but deliberately generic: "dark high-contrast
    blue-toned frame" is safe and useful; "streamer picked up a trophy" would
    require a real local vision model, so this backend does not invent it.
    """

    def __init__(
        self,
        *,
        max_sample_pixels: int = 4096,
        brightness_change_threshold: float = 0.16,
        contrast_change_threshold: float = 0.10,
    ) -> None:
        if max_sample_pixels <= 0:
            raise ValueError("max_sample_pixels must be positive")
        self.max_sample_pixels = max_sample_pixels
        self.brightness_change_threshold = brightness_change_threshold
        self.contrast_change_threshold = contrast_change_threshold
        self._last_stats: FrameStats | None = None

    async def describe_change(self, frame: Frame, prev: SceneState) -> SceneState:
        stats = analyze_frame(frame, max_sample_pixels=self.max_sample_pixels)
        last_event = self._describe_delta(stats)
        self._last_stats = stats
        if not last_event and prev.activity:
            return prev
        return SceneState(
            activity=_activity(stats),
            setting=prev.setting or "unknown",
            on_screen_text="",
            salient_objects=_visual_descriptors(stats),
            last_event=last_event or "video frame analyzed locally",
        )

    def _describe_delta(self, stats: FrameStats) -> str:
        prev = self._last_stats
        if prev is None:
            return "video feed became available for local analysis"
        reasons: list[str] = []
        brightness_delta = stats.brightness - prev.brightness
        if abs(brightness_delta) >= self.brightness_change_threshold:
            reasons.append(
                "scene became brighter" if brightness_delta > 0 else "scene became darker"
            )
        contrast_delta = stats.contrast - prev.contrast
        if abs(contrast_delta) >= self.contrast_change_threshold:
            reasons.append(
                "scene contrast increased" if contrast_delta > 0 else "scene contrast decreased"
            )
        if (
            stats.dominant_color != prev.dominant_color
            and stats.dominant_color != "neutral"
            and prev.dominant_color != "neutral"
        ):
            reasons.append(
                f"dominant color shifted from {prev.dominant_color} to {stats.dominant_color}"
            )
        return "; ".join(reasons) if reasons else "visual composition changed"


class MLXVLMSceneAnalyzer(VLMBackend):
    """Local MLX-VLM scene analyzer for Apple Silicon.

    This runs a local VLM via `mlx-vlm` and never sends frames off-machine. The
    dependency/model are loaded lazily on the first accepted frame so offline
    tests and non-video runs stay lightweight.
    """

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int = 180,
        temperature: float = 0.0,
        fallback: VLMBackend | None = None,
    ) -> None:
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.fallback = fallback
        self._loaded: tuple[Any, Any, Any, Any, Any] | None = None
        self._warned_fallback = False

    async def describe_change(self, frame: Frame, prev: SceneState) -> SceneState:
        try:
            return await asyncio.to_thread(self._describe_change_sync, frame, prev)
        except Exception as exc:
            if self.fallback is None:
                raise RuntimeError(
                    "mlx_vlm backend needs the 'video-mlx' extra: "
                    "pip install -e '.[video-mlx]' and a visible Metal device"
                ) from exc
            if not self._warned_fallback:
                log.warning(
                    "mlx_vlm unavailable (%s); falling back to local_cv scene analysis",
                    exc,
                )
                self._warned_fallback = True
            return await self.fallback.describe_change(frame, prev)

    def _describe_change_sync(self, frame: Frame, prev: SceneState) -> SceneState:
        load, generate, apply_chat_template, load_config, model_bundle = self._load()
        model, processor, config = model_bundle
        image_path = _write_ppm_frame(frame)
        try:
            prompt = _scene_prompt(prev)
            formatted = apply_chat_template(processor, config, prompt, num_images=1)
            try:
                output = generate(
                    model,
                    processor,
                    formatted,
                    [image_path],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    verbose=False,
                )
            except TypeError:
                output = generate(model, processor, formatted, [image_path])
        finally:
            with contextlib.suppress(OSError):
                os.unlink(image_path)
        parsed = _parse_scene_response(_generation_text(output))
        if not parsed.changed:
            return prev
        return _merge_scene(parsed, prev)

    def _load(self) -> tuple[Any, Any, Any, Any, tuple[Any, Any, Any]]:
        if self._loaded is None:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config

            model, processor = load(self.model_name)
            config = load_config(self.model_name)
            self._loaded = (
                load,
                generate,
                apply_chat_template,
                load_config,
                (model, processor, config),
            )
            log.info("local VLM loaded: %s", self.model_name)
        return self._loaded


def analyze_frame(frame: Frame, *, max_sample_pixels: int = 4096) -> FrameStats:
    expected = frame.width * frame.height * 3
    if frame.width <= 0 or frame.height <= 0:
        raise ValueError("frame dimensions must be positive")
    if len(frame.data) != expected:
        raise ValueError("frame data length does not match width*height*3")

    pixels = frame.width * frame.height
    step = max(1, pixels // max_sample_pixels)
    count = 0
    total_r = total_g = total_b = 0.0
    total_luma = 0.0
    total_luma_sq = 0.0
    for pixel in range(0, pixels, step):
        i = pixel * 3
        r = frame.data[i] / 255.0
        g = frame.data[i + 1] / 255.0
        b = frame.data[i + 2] / 255.0
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        total_r += r
        total_g += g
        total_b += b
        total_luma += luma
        total_luma_sq += luma * luma
        count += 1
    red = total_r / count
    green = total_g / count
    blue = total_b / count
    brightness = total_luma / count
    variance = max(0.0, (total_luma_sq / count) - (brightness * brightness))
    contrast = sqrt(variance)
    return FrameStats(
        brightness=brightness,
        contrast=contrast,
        red=red,
        green=green,
        blue=blue,
        dominant_color=_dominant_color(red, green, blue),
        tone=_tone(brightness),
    )


def _activity(stats: FrameStats) -> str:
    contrast = "high-contrast" if stats.contrast >= 0.22 else "low-contrast"
    if stats.dominant_color == "neutral":
        return f"local video analysis: {stats.tone} {contrast} frame"
    return (
        f"local video analysis: {stats.tone} {contrast} "
        f"{stats.dominant_color}-toned frame"
    )


def _visual_descriptors(stats: FrameStats) -> list[str]:
    descriptors = [f"{stats.tone} frame"]
    if stats.contrast >= 0.22:
        descriptors.append("high contrast")
    if stats.dominant_color != "neutral":
        descriptors.append(f"{stats.dominant_color}-toned image")
    return descriptors


def _tone(brightness: float) -> str:
    if brightness < 0.18:
        return "dark"
    if brightness > 0.78:
        return "bright"
    return "mid-brightness"


def _dominant_color(red: float, green: float, blue: float) -> str:
    hi = max(red, green, blue)
    lo = min(red, green, blue)
    if hi - lo < 0.08:
        return "neutral"
    if red >= green and red >= blue:
        if green > blue * 1.25:
            return "yellow"
        if blue > green * 1.25:
            return "magenta"
        return "red"
    if green >= red and green >= blue:
        if blue > red * 1.25:
            return "cyan"
        if red > blue * 1.25:
            return "yellow"
        return "green"
    if red > green * 1.25:
        return "magenta"
    if green > red * 1.25:
        return "cyan"
    return "blue"


def _scene_prompt(prev: SceneState) -> str:
    prev_state = {
        "activity": prev.activity,
        "setting": prev.setting,
        "on_screen_text": prev.on_screen_text,
        "salient_objects": prev.salient_objects,
        "last_event": prev.last_event,
    }
    return (
        "You are the local video perception module for a live-stream chat bot. "
        "Describe only visible facts from this frame; do not infer identity, "
        "private attributes, emotion, or intent. Prefer generic visible labels "
        "like vehicle, animal, road, menu, or text over uncertain specific nouns. "
        "If the frame is unclear or effectively unchanged, set changed=false. "
        "Compare with the previous scene state and return compact JSON only with "
        "keys: changed, activity, setting, on_screen_text, salient_objects, "
        "last_event. Keep activity and last_event under 12 words each, and use "
        "previous values for fields that are still true.\nPrevious scene state:\n"
        + json.dumps(prev_state, ensure_ascii=True)
    )


def _parse_scene_response(text: str) -> _SceneResponse:
    data = _loads_json_object(text)
    if not data:
        return _SceneResponse(changed=False)
    try:
        return _SceneResponse.model_validate(data)
    except ValidationError:
        return _SceneResponse(changed=False)


def _generation_text(output: Any) -> str:
    if isinstance(output, dict) and "text" in output:
        return str(output["text"]).strip()
    text = getattr(output, "text", None)
    if text is not None:
        return str(text).strip()
    return str(output).strip()


def _loads_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _merge_scene(update: _SceneResponse, prev: SceneState) -> SceneState:
    fields = update.model_fields_set
    return SceneState(
        activity=_field(update, "activity", prev.activity, fields),
        setting=_field(update, "setting", prev.setting, fields),
        on_screen_text=_field(update, "on_screen_text", prev.on_screen_text, fields),
        salient_objects=_objects(update, prev, fields),
        last_event=_field(update, "last_event", "", fields),
    )


def _field(update: _SceneResponse, name: str, fallback: str, fields: set[str]) -> str:
    if name not in fields:
        return fallback
    return _clean_scene_text(str(getattr(update, name)))


def _objects(update: _SceneResponse, prev: SceneState, fields: set[str]) -> list[str]:
    if "salient_objects" not in fields:
        return list(prev.salient_objects)
    objects: list[str] = []
    seen: set[str] = set()
    for item in update.salient_objects:
        cleaned = _clean_scene_text(str(item))
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            objects.append(cleaned)
        if len(objects) >= 6:
            break
    return objects


def _clean_scene_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    while cleaned.endswith((".", ";", ",")):
        cleaned = cleaned[:-1].rstrip()
    words = [
        word[:-2] if len(word) > 6 and word.casefold().endswith("eded") else word
        for word in cleaned.split()
    ]
    return " ".join(words)


def _write_ppm_frame(frame: Frame) -> str:
    expected = frame.width * frame.height * 3
    if frame.width <= 0 or frame.height <= 0:
        raise ValueError("frame dimensions must be positive")
    if len(frame.data) != expected:
        raise ValueError("frame data length does not match width*height*3")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ppm") as fh:
        fh.write(f"P6\n{frame.width} {frame.height}\n255\n".encode("ascii"))
        fh.write(frame.data)
        return fh.name
