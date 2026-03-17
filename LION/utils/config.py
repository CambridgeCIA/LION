"""Utility function to convert LIONParameter to custom dataclasses."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, TypeVar, cast

from LION.utils.parameter import LIONParameter

DataClass = TypeVar("DataClass")


def lion_parameter_to_dataclass(
    lion_parameter: LIONParameter, data_cls: type[DataClass]
) -> DataClass:
    """
    Convert a ``LIONParameter`` into an instance of ``data_cls``.

    Motivation:
    ``LIONParameter`` is a convenient way to store and pass around parameters
    without needing to define a custom dataclass for each experiment.
    On the other hand, custom dataclasses have several advantages:
    - type hints, type checking, and IDE support
    - clearer default values, required/optional fields, and nested structure
    - better error messages for missing or extra fields

    This function is intended for devs who define their experiment parameters
    as a dataclass, but want an interface that also accepts ``LIONParameter``
    which users might be more used to using.

    This function is intended to be minimal and assumes that
    - all necessary dataclass fields are present in the ``LIONParameter``
    - all copied values already have the correct runtime types
    For more customized checks and conversions, it is recommended to define a
    ``from_lion_parameter`` method in the dataclass and use that instead.

    Behavior:
    - extra fields in ``lion_parameter`` are ignored
    - missing fields use the dataclass defaults
    - missing required fields raise errors when ``data_cls(...)`` is called

    Parameters
    ----------
    lion_parameter : LIONParameter
        The LIONParameter instance to convert.
    data_cls : type[DataClass]
        The dataclass type to convert to.

    Returns
    -------
    DataClass
        An instance of ``data_cls`` with values from ``lion_parameter``.
    """
    if not isinstance(data_cls, type) or not is_dataclass(data_cls):
        raise TypeError(f"{data_cls!r} is not a dataclass type.")

    lion_parameter_data: dict[str, Any] = vars(lion_parameter)
    dataclass_kwargs: dict[str, Any] = {}
    for field in fields(data_cls):
        if field.name in lion_parameter_data:
            dataclass_kwargs[field.name] = lion_parameter_data[field.name]

    return cast(DataClass, data_cls(**dataclass_kwargs))
