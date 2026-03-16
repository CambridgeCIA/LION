"""Utility function to convert LIONParameter to custom dataclasses."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

from LION.utils.parameter import LIONParameter


def lion_to_dataclass(cls: type, param: LIONParameter | dict[str, Any]) -> object:
    """
    Convert a LIONParameter or dict into an instance of ``cls``.

    Motivation:
    ``LIONParameter`` is a convenient way to store and pass around parameters
    without needing to define a custom dataclass for each experiment.
    On the other hand, custom dataclasses have several advantages:
    - type checking and IDE support
    - default values and optional fields
    - better error messages for missing or extra fields
    - type hints
    - clearer defaults and nested structure

    This function is intended for devs who define their experiment parameters
    as a dataclass, but want an interface that also accepts ``LIONParameter``
    which users might be more used to using.

    Supported:
    - plain scalar fields
    - nested dataclasses
    - pathlib.Path

    Behavior:
    - extra fields in ``param`` are ignored
    - missing fields use the dataclass defaults
    - missing required fields raise when ``cls(...)`` is called
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type.")

    data = vars(param) if isinstance(param, LIONParameter) else param
    type_hints = get_type_hints(cls)

    kwargs: dict[str, Any] = {}
    for field in fields(cls):
        if field.name not in data:
            continue

        value = data[field.name]
        field_type = type_hints.get(field.name, field.type)
        kwargs[field.name] = _convert_value(field_type, value)

    return cls(**kwargs)


def _convert_value(field_type: type, value: Any) -> None | Path | object:
    """Convert one field value."""
    if value is None:
        return None

    if field_type is Path:
        return Path(value)

    if isinstance(field_type, type) and is_dataclass(field_type):
        if isinstance(value, (LIONParameter, dict)):
            return lion_to_dataclass(field_type, value)

    return value
