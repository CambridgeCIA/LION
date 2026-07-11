"""Canonical and legacy PaDIS reconstruction identifiers.

New manifests and outputs use names which describe the algorithms and data.
The aliases remain part of the public interface so historical commands,
metrics, manifests, and result directories continue to be readable.
"""

from __future__ import annotations


METHOD_ALIASES = {
    "admm_tv": "cp_tv",
}

EXPERIMENT_ALIASES = {
    "ct_fanbeam_180": "ct_20_limited_angle_120",
    "fanbeam_180": "ct_20_limited_angle_120",
    "ct_fan_180": "ct_20_limited_angle_120",
    "180": "ct_20_limited_angle_120",
    "fanbeam_120": "ct_20_limited_angle_120",
    "ct_fan_120": "ct_20_limited_angle_120",
}


def canonical_method(value: str) -> str:
    """Return the canonical reconstruction-method identifier."""

    return METHOD_ALIASES.get(value, value)


def canonical_experiment(value: str) -> str:
    """Return the canonical reconstruction-experiment identifier."""

    return EXPERIMENT_ALIASES.get(value, value)


def method_storage_names(value: str) -> tuple[str, ...]:
    """Return canonical then legacy method names usable in stored paths."""

    canonical = canonical_method(value)
    return (
        canonical,
        *(old for old, new in METHOD_ALIASES.items() if new == canonical),
    )


def experiment_storage_names(value: str) -> tuple[str, ...]:
    """Return canonical then legacy experiment names usable in stored paths."""

    canonical = canonical_experiment(value)
    aliases = tuple(
        old
        for old, new in EXPERIMENT_ALIASES.items()
        if new == canonical and old == "ct_fanbeam_180"
    )
    return (canonical, *aliases)
