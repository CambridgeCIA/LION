from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dictionary config in {path}, got {type(data)}")
    return data


def make_dataclass(cls: type, data: object | None) -> Any:
    """Instantiate a dataclass from a dictionary."""
    if data is None:
        return None

    data = cast(dict[str, Any], data)
    field_names = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(data) - field_names)
    if unknown:
        raise ValueError(f"Unknown fields for {cls.__name__}: {unknown}")

    return cls(**data)


def make_dataclass_default(cls: type, data: dict[str, Any], key: str) -> Any:
    return make_dataclass(cls, data.get(key, {}))
