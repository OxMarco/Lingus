from types import SimpleNamespace

import pytest

from lingus.adapters.base import Frame
from lingus.models.local_vision import LocalFrameAnalyzer, MLXVLMSceneAnalyzer, analyze_frame
from lingus.world_state import SceneState


def _frame(rgb: tuple[int, int, int], ts: float = 0.0, width: int = 2, height: int = 2) -> Frame:
    return Frame(data=bytes(rgb) * width * height, width=width, height=height, ts=ts)


def test_analyze_frame_detects_local_visual_properties():
    stats = analyze_frame(_frame((250, 20, 20)))

    assert stats.brightness > 0
    assert stats.dominant_color == "red"
    assert stats.tone == "mid-brightness"


@pytest.mark.asyncio
async def test_local_frame_analyzer_describes_first_frame_without_remote_model():
    analyzer = LocalFrameAnalyzer()

    scene = await analyzer.describe_change(_frame((0, 0, 0)), SceneState())

    assert scene.activity == "local video analysis: dark low-contrast frame"
    assert scene.setting == "unknown"
    assert scene.on_screen_text == ""
    assert scene.salient_objects == ["dark frame"]
    assert scene.last_event == "video feed became available for local analysis"


@pytest.mark.asyncio
async def test_local_frame_analyzer_reports_brightness_shift():
    analyzer = LocalFrameAnalyzer(brightness_change_threshold=0.1)
    await analyzer.describe_change(_frame((0, 0, 0)), SceneState())

    scene = await analyzer.describe_change(
        _frame((255, 255, 255), ts=2.0),
        SceneState(activity="prior scene", setting="desk"),
    )

    assert scene.setting == "desk"
    assert "bright" in scene.activity
    assert "scene became brighter" in scene.last_event


@pytest.mark.asyncio
async def test_local_frame_analyzer_reports_generic_composition_change():
    analyzer = LocalFrameAnalyzer()
    prior = SceneState(activity="prior local analysis")
    analyzer._last_stats = analyze_frame(_frame((128, 128, 128)))  # noqa: SLF001

    scene = await analyzer.describe_change(_frame((128, 128, 128), ts=2.0), prior)

    assert scene.last_event == "visual composition changed"


@pytest.mark.asyncio
async def test_mlx_vlm_scene_analyzer_parses_structured_local_output(monkeypatch):
    analyzer = MLXVLMSceneAnalyzer(model="local-qwen")
    calls = {}

    def fake_template(processor, config, prompt, *, num_images):
        calls["prompt"] = prompt
        calls["num_images"] = num_images
        return "formatted prompt"

    def fake_generate(model, processor, prompt, images, **opts):
        calls["images"] = images
        calls["opts"] = opts
        return """
        {
          "changed": true,
          "activity": "streamer is showing a game menu",
          "setting": "game capture",
          "on_screen_text": "START",
          "salient_objects": ["menu", "cursor"],
          "last_event": "a start menu appeared"
        }
        """

    monkeypatch.setattr(
        analyzer,
        "_load",
        lambda: (
            None,
            fake_generate,
            fake_template,
            None,
            ("model", "processor", {"model_type": "qwen"}),
        ),
    )

    scene = await analyzer.describe_change(
        _frame((10, 20, 30)),
        SceneState(activity="prior scene"),
    )

    assert scene.activity == "streamer is showing a game menu"
    assert scene.setting == "game capture"
    assert scene.on_screen_text == "START"
    assert scene.salient_objects == ["menu", "cursor"]
    assert scene.last_event == "a start menu appeared"
    assert calls["num_images"] == 1
    assert calls["images"][0].endswith(".ppm")
    assert calls["opts"]["max_tokens"] == 180
    assert "Previous scene state" in calls["prompt"]


@pytest.mark.asyncio
async def test_mlx_vlm_scene_analyzer_reads_generation_result_text(monkeypatch):
    analyzer = MLXVLMSceneAnalyzer(model="local-qwen")

    def fake_template(processor, config, prompt, *, num_images):
        return "formatted prompt"

    def fake_generate(model, processor, prompt, images, **opts):
        return SimpleNamespace(
            text="""
            {
              "changed": true,
              "activity": "vehicle driving through a city street",
              "setting": "gameplay",
              "last_event": "the street view changed"
            }
            """
        )

    monkeypatch.setattr(
        analyzer,
        "_load",
        lambda: (
            None,
            fake_generate,
            fake_template,
            None,
            ("model", "processor", {"model_type": "qwen"}),
        ),
    )

    scene = await analyzer.describe_change(_frame((40, 50, 60)), SceneState())

    assert scene.activity == "vehicle driving through a city street"
    assert scene.setting == "gameplay"
    assert scene.last_event == "the street view changed"


@pytest.mark.asyncio
async def test_mlx_vlm_scene_analyzer_cleans_generated_scene_text(monkeypatch):
    analyzer = MLXVLMSceneAnalyzer(model="local-qwen")

    def fake_template(processor, config, prompt, *, num_images):
        return "formatted prompt"

    def fake_generate(model, processor, prompt, images, **opts):
        return SimpleNamespace(
            text="""
            {
              "changed": true,
              "activity": "car parkeded.",
              "setting": "gameplay,",
              "salient_objects": ["Car", "car", "road."],
              "last_event": "car parkeded."
            }
            """
        )

    monkeypatch.setattr(
        analyzer,
        "_load",
        lambda: (
            None,
            fake_generate,
            fake_template,
            None,
            ("model", "processor", {"model_type": "qwen"}),
        ),
    )

    scene = await analyzer.describe_change(_frame((10, 10, 10)), SceneState())

    assert scene.activity == "car parked"
    assert scene.setting == "gameplay"
    assert scene.salient_objects == ["Car", "road"]
    assert scene.last_event == "car parked"


@pytest.mark.asyncio
async def test_mlx_vlm_scene_analyzer_ignores_malformed_json(monkeypatch):
    analyzer = MLXVLMSceneAnalyzer(model="local-qwen")

    def fake_template(processor, config, prompt, *, num_images):
        return "formatted prompt"

    def fake_generate(model, processor, prompt, images, **opts):
        return '{"activity": "a bear walks on the road", "salient_objects": ["bear"'

    monkeypatch.setattr(
        analyzer,
        "_load",
        lambda: (
            None,
            fake_generate,
            fake_template,
            None,
            ("model", "processor", {"model_type": "qwen"}),
        ),
    )

    prior = SceneState(activity="prior scene", last_event="prior event")

    assert await analyzer.describe_change(_frame((10, 10, 10)), prior) == prior


@pytest.mark.asyncio
async def test_mlx_vlm_falls_back_to_local_cv_when_dependency_missing(monkeypatch):
    fallback = LocalFrameAnalyzer()
    analyzer = MLXVLMSceneAnalyzer(model="local-qwen", fallback=fallback)

    def missing():
        raise ImportError("no mlx_vlm")

    monkeypatch.setattr(analyzer, "_load", missing)

    scene = await analyzer.describe_change(_frame((0, 0, 0)), SceneState())

    assert scene.last_event == "video feed became available for local analysis"


@pytest.mark.asyncio
async def test_mlx_vlm_falls_back_to_local_cv_when_metal_unavailable(monkeypatch):
    fallback = LocalFrameAnalyzer()
    analyzer = MLXVLMSceneAnalyzer(model="local-qwen", fallback=fallback)

    def no_metal():
        raise RuntimeError("No Metal device available")

    monkeypatch.setattr(analyzer, "_load", no_metal)

    scene = await analyzer.describe_change(_frame((0, 0, 0)), SceneState())

    assert scene.last_event == "video feed became available for local analysis"
