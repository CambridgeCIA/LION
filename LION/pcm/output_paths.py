"""Helpers for consistent timestamped PCM output paths."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

DATETIME_PREFIX_FORMAT = "%Y%m%d_%H%M%S"


def current_datetime_prefix() -> str:
    """Return the timestamp format already used by PCM single-sample runs."""
    return datetime.now().strftime(DATETIME_PREFIX_FORMAT)


def with_datetime_prefix(path: Path | str, enabled: bool) -> Path:
    """Prefix the final path component with the current datetime when requested."""
    result = Path(path)
    if not enabled:
        return result
    return result.with_name(f"{current_datetime_prefix()}_{result.name}")


def name_with_datetime_prefix(name: str, enabled: bool) -> str:
    """Prefix a filename or output-directory name with the current datetime."""
    if not enabled:
        return name
    return f"{current_datetime_prefix()}_{name}"
