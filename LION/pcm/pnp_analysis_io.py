"""YAML import/export helpers for PCM PnP iteration-analysis plans."""

from __future__ import annotations

import copy
import dataclasses
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

ANALYSIS_SCHEMA = "lion.pcm.pnp_iteration_analysis"
ANALYSIS_SCHEMA_VERSION = 1


def _yaml_value(value: Any) -> Any:
    """Convert common LION config values to human-readable YAML values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _yaml_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_yaml_value(item) for item in value]
    if isinstance(value, list):
        return [_yaml_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _yaml_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"Cannot serialize {type(value).__name__} to analysis YAML: {value!r}"
    )


def config_to_mapping(config: object) -> dict[str, Any]:
    """Serialize one reconstruction-method dataclass to a YAML-safe mapping."""
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        raise TypeError("Method configuration must be a dataclass instance.")
    return _yaml_value(config)


def _resolved_field_type(owner: object, field: dataclasses.Field) -> object:
    try:
        return get_type_hints(type(owner)).get(field.name, field.type)
    except Exception:
        return field.type


def _non_none_candidates(annotation: object) -> tuple[object, ...]:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return tuple(arg for arg in get_args(annotation) if arg is not type(None))
    return (annotation,)


def _coerce_value(raw: Any, annotation: object, current: Any) -> Any:
    if raw is None:
        return None

    candidates = _non_none_candidates(annotation)
    if Path in candidates:
        return Path(raw).expanduser()

    scalar = (
        candidates[0]
        if len(candidates) == 1
        else type(current)
        if current is not None
        else None
    )
    origin = get_origin(scalar)
    if origin is not None:
        scalar = origin

    if isinstance(scalar, type) and issubclass(scalar, Enum):
        try:
            return scalar(raw)
        except ValueError:
            return scalar[str(raw)]

    if scalar is tuple or isinstance(current, tuple):
        return tuple(raw)
    if scalar is list or isinstance(current, list):
        return list(raw)
    if scalar is dict or isinstance(current, dict):
        return dict(raw)
    if scalar in (bool, int, float, str):
        return scalar(raw)
    return raw


def apply_config_mapping(config: object, values: dict[str, Any]) -> object:
    """Apply a YAML mapping to a copy of a preset method dataclass."""
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        raise TypeError("Method configuration must be a dataclass instance.")

    updated = copy.deepcopy(config)
    known_fields = {field.name: field for field in dataclasses.fields(updated)}
    unknown = set(values) - set(known_fields)
    if unknown:
        raise ValueError(f"Unknown method configuration fields: {sorted(unknown)}")

    for name, raw in values.items():
        field = known_fields[name]
        current = getattr(updated, name)
        if dataclasses.is_dataclass(current) and not isinstance(current, type):
            if not isinstance(raw, dict):
                raise TypeError(f"Expected a mapping for nested field {name!r}.")
            setattr(updated, name, apply_config_mapping(current, raw))
            continue

        annotation = _resolved_field_type(updated, field)
        setattr(updated, name, _coerce_value(raw, annotation, current))

    return updated


def validate_analysis_definition(definition: dict[str, Any]) -> None:
    """Validate the small stable envelope used by exported analysis YAML files."""
    if definition.get("schema") != ANALYSIS_SCHEMA:
        raise ValueError(
            f"Not a LION PCM PnP iteration-analysis YAML file: schema={definition.get('schema')!r}."
        )
    version = int(definition.get("version", -1))
    if version != ANALYSIS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported analysis YAML version {version}; expected {ANALYSIS_SCHEMA_VERSION}."
        )
    if not definition.get("preset"):
        raise ValueError("Analysis YAML is missing 'preset'.")
    if not isinstance(definition.get("methods"), list) or not definition["methods"]:
        raise ValueError(
            "Analysis YAML must contain at least one method configuration."
        )


def save_analysis_definition(path: Path, definition: dict[str, Any]) -> Path:
    """Save one re-importable PnP iteration-analysis plan."""
    validate_analysis_definition(definition)
    path = Path(path).expanduser()
    if path.suffix.lower() not in {".yaml", ".yml"}:
        path = path.with_suffix(".yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            _yaml_value(definition), file, sort_keys=False, allow_unicode=True
        )
    return path


def load_analysis_definition(path: Path) -> dict[str, Any]:
    """Load and validate a PnP iteration-analysis plan."""
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Analysis YAML root must be a mapping.")
    validate_analysis_definition(data)
    return data
