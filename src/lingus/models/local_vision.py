"""Local video scene analysis for Phase 4.

Scene understanding runs entirely on-device via a local MLX-VLM: frames are
never sent to a hosted model. There is deliberately no colour-only fallback —
if the VLM cannot load, the run terminates rather than degrading to useless
brightness/contrast stats that cannot describe what is on screen.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..adapters.base import Frame
from ..logging import get_logger
from ..world_state import SceneState
from .base import VLMBackend

log = get_logger(__name__)


class _SceneResponse(BaseModel):
    changed: bool = True
    activity: str = ""
    setting: str = ""
    on_screen_text: str = ""
    salient_objects: list[str] = Field(default_factory=list)
    last_event: str = ""


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
    ) -> None:
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._loaded: tuple[Any, Any, Any, Any, Any] | None = None

    async def describe_change(self, frame: Frame, prev: SceneState) -> SceneState:
        try:
            # Keep MLX on the owning asyncio/main thread. libmlx keeps compiler
            # state in thread-local storage and has crashed on macOS when Python
            # worker threads exit during Ctrl+C shutdown.
            return self._describe_change_sync(frame, prev)
        except Exception as exc:
            # No colour-only fallback: colour stats are useless for scene
            # understanding, and silently degrading hides a broken VLM. Raise so
            # the task supervisor stops the run.
            raise RuntimeError(
                "mlx_vlm scene analysis failed; local video needs the "
                "'video-mlx' extra (pip install -e '.[video-mlx]') and a visible "
                "Metal device. Set models.vlm.backend=none to disable video."
            ) from exc

    def _describe_change_sync(self, frame: Frame, prev: SceneState) -> SceneState:
        load, generate, apply_chat_template, load_config, model_bundle = self._load()
        model, processor, config = model_bundle
        image_path = _write_ppm_frame(frame)
        try:
            prompt = _scene_prompt(prev)
            formatted = apply_chat_template(processor, config, prompt, num_images=1)
            started = time.perf_counter()
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
            log.debug(
                "vlm scene analysis: %.0fms (max_tokens=%d)",
                (time.perf_counter() - started) * 1000,
                self.max_tokens,
            )
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
        "Describe what is visibly happening in this frame in concrete, specific "
        "terms. Name the actual objects you see (a mug, a mechanical keyboard, a "
        "race car) rather than vague categories, and describe the subject's "
        "visible action and expression (e.g. sipping coffee, laughing, leaning "
        "into the mic). You may report a visible facial expression, but do not "
        "guess at hidden intent, private attributes, or a person's identity. "
        "For on_screen_text, transcribe only text that is actually legible in the "
        "frame; never invent captions, names, overlays, or notifications that are "
        "not clearly readable. If the frame is unclear or effectively unchanged "
        "from the previous state, set changed=false. Compare with the previous "
        "scene state and return compact JSON only with keys: changed, activity, "
        "setting, on_screen_text, salient_objects, last_event. Keep activity and "
        "last_event vivid but under 16 words each; set last_event to the single "
        "most notable thing that just changed. Reuse previous values for fields "
        "that are still true.\nPrevious scene state:\n"
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
