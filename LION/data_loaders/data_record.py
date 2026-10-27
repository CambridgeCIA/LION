# =============================================================================
# This file is part of LION library
# License : GPL-3
# =============================================================================
"""The record of the data a model is trained or tested on.

``data_record(dataset)`` describes a LION dataset so it can be made again and
checked, from the record alone:

- ``dataset``: its spec (class, mode, geometry, parameters; ``LION.utils.spec``);
- ``sinogram_transform`` and its seed: how the measurements are simulated, e.g.
  ``functools.partial(sinogram_add_noise, I0=...)``. A function that can't be
  written as a spec (a lambda) is recorded by name only, with a warning;
- ``experiment`` and ``split``: for a dataset made by an experiment, the
  experiment's spec and which split this is;
- ``samples``: what each sample is (e.g. ``"LIDC-IDRI-0912/slice_37"``), in
  order, their number and a hash of the list;
- ``files`` (``hash_files=True``): a hash of the contents of the files the
  samples are read from, to check that two copies of the data are the same.
  It reads the data, so it is only computed when asked.

A record is plain JSON. It is saved with a model as it is
(``model.save(f, dataset=data_record(dataset))``) and read back as a dict: the
dataset is only made again on request, ``build(record["dataset"])`` or
``build(record["experiment"]).get_dataset(record["split"])``.
"""

import hashlib
import warnings

from LION.utils import spec

RECORD_KEY = "lion_data_record"
RECORD_VERSION = 1


def is_data_record(value) -> bool:
    return isinstance(value, dict) and RECORD_KEY in value


def _unsaved(obj, error) -> dict:
    warnings.warn(
        f"{obj!r} can't be written as a spec, so the data record only names it "
        f"({error}). Use a LION class or functools.partial of a LION function.",
        stacklevel=3,
    )
    return {"unsaved": repr(obj), "reason": str(error)}


def _sha256_of_lines(lines) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def file_contents_hash(paths) -> str:
    """A hash of the contents of the files, in order (not of their names or
    locations, which differ between machines)."""
    total = hashlib.sha256()
    for path in paths:
        content = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                content.update(block)
        total.update(content.digest())
    return total.hexdigest()


def data_record(dataset, hash_files=False) -> dict:
    """The record of a LION dataset (see the module docstring)."""
    record = {RECORD_KEY: RECORD_VERSION}

    experiment = getattr(dataset, "experiment", None)
    if experiment is not None:
        try:
            record["experiment"] = spec.to_spec(experiment)
        except spec.SpecError as error:
            record["experiment"] = _unsaved(experiment, error)
        record["split"] = getattr(dataset, "split", None)

    try:
        record["dataset"] = spec.to_spec(dataset)
    except spec.SpecError as error:
        record["dataset"] = _unsaved(dataset, error)

    transform = getattr(dataset, "sinogram_transform", None)
    if transform is not None:
        try:
            record["sinogram_transform"] = spec.encode(transform)
        except spec.SpecError as error:
            record["sinogram_transform"] = _unsaved(transform, error)
        record["sinogram_transform_seed"] = getattr(
            dataset, "sinogram_transform_seed", None
        )

    if hasattr(dataset, "sample_ids"):
        ids = list(dataset.sample_ids())
        record["samples"] = {
            "count": len(ids),
            "sha256": _sha256_of_lines(ids),
            "ids": ids,
        }

    if hash_files:
        if not hasattr(dataset, "sample_files"):
            raise ValueError(
                f"{type(dataset).__name__} does not list its files (sample_files())"
            )
        files = list(dataset.sample_files())
        record["files"] = {"count": len(files), "sha256": file_contents_hash(files)}
    return record
