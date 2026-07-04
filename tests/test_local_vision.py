import threading
from types import SimpleNamespace

import pytest

from lingus.adapters.base import Frame
from lingus.models.local_vision import MLXVLMSceneAnalyzer
from lingus.world_state import SceneState


def _frame(rgb: tuple[int, int, int], ts: float = 0.0, width: int = 2, height: int = 2) -> Frame:
    return Frame(data=bytes(rgb) * width * height, width=width, height=height, ts=ts)


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
async def test_mlx_vlm_scene_analyzer_keeps_mlx_on_event_loop_thread(monkeypatch):
    analyzer = MLXVLMSceneAnalyzer(model="local-qwen")
    event_loop_thread = threading.get_ident()
    calls = {}

    def fake_template(processor, config, prompt, *, num_images):
        return "formatted prompt"

    def fake_generate(model, processor, prompt, images, **opts):
        calls["thread"] = threading.get_ident()
        return '{"changed": false}'

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

    await analyzer.describe_change(_frame((10, 20, 30)), SceneState())

    assert calls["thread"] == event_loop_thread


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
async def test_mlx_vlm_raises_when_dependency_missing(monkeypatch):
    analyzer = MLXVLMSceneAnalyzer(model="local-qwen")

    def missing():
        raise ImportError("no mlx_vlm")

    monkeypatch.setattr(analyzer, "_load", missing)

    # No colour-only fallback: a broken VLM must terminate the run.
    with pytest.raises(RuntimeError, match="mlx_vlm scene analysis failed"):
        await analyzer.describe_change(_frame((0, 0, 0)), SceneState())


@pytest.mark.asyncio
async def test_mlx_vlm_raises_when_metal_unavailable(monkeypatch):
    analyzer = MLXVLMSceneAnalyzer(model="local-qwen")

    def no_metal():
        raise RuntimeError("No Metal device available")

    monkeypatch.setattr(analyzer, "_load", no_metal)

    with pytest.raises(RuntimeError, match="mlx_vlm scene analysis failed"):
        await analyzer.describe_change(_frame((0, 0, 0)), SceneState())
