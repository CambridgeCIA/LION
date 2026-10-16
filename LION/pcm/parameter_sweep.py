"""Generic parameter-sweep helpers for PCM reconstruction methods."""

from __future__ import annotations

import copy
import itertools
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterSweep:
    """Values to sweep for one nested configuration field."""

    path: str
    values: list[Any]


@dataclass(frozen=True)
class MethodVariant:
    """One concrete reconstruction-method configuration."""

    id: str
    label: str
    config: object


def set_nested_value(config: object, path: str, value: Any) -> None:
    """Set one dot-separated field on a nested dataclass/config object."""
    parts = path.split(".")
    target = config
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _value_label(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._=-]+", "-", text.strip())
    return slug.strip("-") or "variant"


def expand_parameter_sweeps(
    method_id: str,
    base_config: object,
    sweeps: list[ParameterSweep],
    *,
    base_label: str | None = None,
) -> list[MethodVariant]:
    """Expand Cartesian-product sweeps into concrete method configurations."""
    label = method_id if base_label is None else base_label
    active_sweeps = [sweep for sweep in sweeps if sweep.values]
    if not active_sweeps:
        return [
            MethodVariant(id=method_id, label=label, config=copy.deepcopy(base_config))
        ]

    variants: list[MethodVariant] = []
    value_lists = [sweep.values for sweep in active_sweeps]

    for values in itertools.product(*value_lists):
        config = copy.deepcopy(base_config)
        assignments = []
        id_parts = []

        for sweep, value in zip(active_sweeps, values, strict=True):
            set_nested_value(config, sweep.path, value)
            value_text = _value_label(value)
            assignments.append(f"{sweep.path}={value_text}")
            id_parts.append(f"{_slug(sweep.path)}={_slug(value_text)}")

        variants.append(
            MethodVariant(
                id=f"{method_id}__{'__'.join(id_parts)}",
                label=f"{label} [{', '.join(assignments)}]",
                config=config,
            )
        )

    return variants
