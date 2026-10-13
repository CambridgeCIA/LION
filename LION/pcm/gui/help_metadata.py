"""Shared help-text extraction for dataclass-driven PCM GUI controls."""

from __future__ import annotations

import dataclasses
import inspect
import re


def dataclass_help(config: object) -> str:
    """Return the class docstring used by GUI help and future generated docs."""
    return inspect.getdoc(type(config)) or ""


def _attribute_docs(config: object) -> dict[str, str]:
    """Parse a simple ``Attributes:`` section from a class docstring."""
    doc = dataclass_help(config)
    if not doc:
        return {}

    lines = doc.splitlines()
    try:
        start = (
            next(
                index
                for index, line in enumerate(lines)
                if line.strip() == "Attributes:"
            )
            + 1
        )
    except StopIteration:
        return {}

    docs: dict[str, str] = {}
    current_name: str | None = None
    for line in lines[start:]:
        if not line.strip():
            continue
        match = re.match(r"^\s{0,8}([A-Za-z_]\w*)\s*:\s*(.+)$", line)
        if match:
            current_name = match.group(1)
            docs[current_name] = match.group(2).strip()
            continue
        if current_name is not None and line.startswith((" ", "\t")):
            docs[current_name] += " " + line.strip()
            continue
        if not line.startswith((" ", "\t")):
            break
    return docs


def field_help(config: object, field: dataclasses.Field) -> str:
    """Return help for one field from metadata, falling back to class docstrings."""
    metadata_help = field.metadata.get("help") if field.metadata is not None else None
    if metadata_help:
        return str(metadata_help)
    return _attribute_docs(config).get(field.name, "")
