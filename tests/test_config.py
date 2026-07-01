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


def test_missing_explicit_config_raises(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="config file not found"):
        Settings.load(missing)


def test_arbiter_cooldown_seconds_must_be_positive():
    with pytest.raises(ValidationError):
        Settings.model_validate({"arbiter": {"cooldown_seconds": 0}})
