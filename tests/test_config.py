import pytest
from pydantic import ValidationError

from lingus.config import Settings


def test_loads_repo_config_yaml():
    s = Settings.load("config.yaml")
    assert s.platform == "file_replay"
    assert s.models.asr.backend == "faster_whisper"
    assert s.arbiter.weights.get("direct_address", 0) > 0


def test_schema_defaults_are_available_without_yaml():
    s = Settings.model_validate({})
    assert s.platform == "file_replay"  # schema default
    assert s.output.max_chars == 200
    assert s.output.typing_enabled is False
    assert s.models.asr.model_size == "turbo"
    assert s.models.audio_gate.backend == "spectral"
    assert s.models.vlm.backend == "mlx_vlm"
    assert s.models.vlm.model == "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    assert s.models.vlm.max_tokens == 180


def test_missing_explicit_config_raises(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="config file not found"):
        Settings.load(missing)


def test_arbiter_cooldown_seconds_must_be_positive():
    with pytest.raises(ValidationError):
        Settings.model_validate({"arbiter": {"cooldown_seconds": 0}})


def test_audio_gate_thresholds_are_bounded():
    with pytest.raises(ValidationError):
        Settings.model_validate({"models": {"audio_gate": {"music_threshold": 1.5}}})


def test_vlm_thresholds_are_validated():
    with pytest.raises(ValidationError):
        Settings.model_validate({"models": {"vlm": {"frame_diff_threshold": 1.5}}})
