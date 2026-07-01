"""Load a PersonaSpec from a YAML character card."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import PersonaSpec


def load_persona(path: str | Path) -> PersonaSpec:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return PersonaSpec.model_validate(data)
