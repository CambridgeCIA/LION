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
- ``subset``: for a ``torch.utils.data.Subset``, its ``indices`` into the
  dataset it is taken from (whose record this is); ``samples`` lists the
  subset's samples;
- ``files`` (``hash_files=True``): a hash of the contents of the files the
  samples are read from, to check that two copies of the data are the same.
  It reads the data, so it is only computed when asked.

A record is a ``LIONParameter`` (``record.samples.ids``, ``record.split``) in
which the specs (``record.dataset``, ``record.experiment``) stay dicts: they
describe objects, which are only made on request. It is saved with a model as
plain JSON (``model.save(f, dataset=data_record(dataset))``) and read back as a
``LIONParameter``. The data are made again from the record or the model's file:

- ``LIONDataset.instantiate(source)``: the dataset alone, without noise (the
  noise is part of the experiment), checked against the record's samples;
- ``Experiment.instantiate(source).get_dataset(record.split)``: the data
  as the experiment made them, noise included.
"""

import hashlib

from torch.utils.data import Subset

from LION.utils import spec
from LION.utils.parameter import LIONParameter

RECORD_KEY = "lion_data_record"
RECORD_VERSION = 1


def is_data_record(value) -> bool:
    """A data record, as a ``LIONParameter`` or as the dict read from JSON."""
    if isinstance(value, LIONParameter):
        return hasattr(value, RECORD_KEY)
    return isinstance(value, dict) and RECORD_KEY in value


def read_record(source) -> LIONParameter:
    """The data record in ``source``: a record, a model's metadata holding
    one (as its ``dataset_params``), or the path of a ``.json`` with either."""
    saved = spec.read(source)
    if is_data_record(saved):
        return LIONParameter(**saved)
    if is_data_record(saved.get("dataset_params")):
        return LIONParameter(**saved["dataset_params"])
    raise ValueError(
        "no data record found: give a record (data_record(dataset)) or the file "
        "of a model saved with one (model.save(f, dataset=record))"
    )


def recorded_spec(record: LIONParameter, key: str) -> dict:
    """The spec of ``record.<key>`` ("dataset", "experiment"), or an error if
    the record could only name it."""
    if not hasattr(record, key):
        raise ValueError(f"the data record has no {key}")
    value = getattr(record, key)
    if spec.is_unsaved(value):
        raise ValueError(
            f"the data record only names its {key}, {value.unsaved}: it couldn't "
            f"be saved ({value.reason}), so it can't be made again"
        )
    return value


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


def data_record(dataset, hash_files=False) -> LIONParameter:
    """The record of a LION dataset (see the module docstring)."""
    return LIONParameter(**_record(dataset, hash_files))


def _record(dataset, hash_files) -> dict:
    """The record as JSON values."""
    if isinstance(dataset, Subset):
        return _subset_record(dataset, hash_files)
    record = {RECORD_KEY: RECORD_VERSION}

    experiment = getattr(dataset, "experiment", None)
    if experiment is not None:
        record["experiment"] = spec.encode_or_name(experiment, "the experiment")
        record["split"] = getattr(dataset, "split", None)

    record["dataset"] = spec.encode_or_name(dataset, "the dataset")

    transform = getattr(dataset, "sinogram_transform", None)
    if transform is not None:
        record["sinogram_transform"] = spec.encode_or_name(transform, "the noise")
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


def _subset_record(subset, hash_files) -> dict:
    """A ``torch.utils.data.Subset``: the record of the dataset it is taken
    from, with the subset's ``indices`` and its samples only. The file hash, if
    asked, is of the whole dataset's files."""
    record = _record(subset.dataset, hash_files)
    indices = [int(i) for i in subset.indices]
    record["subset"] = {"indices": indices}
    if "samples" in record:
        ids = [record["samples"]["ids"][i] for i in indices]
        record["samples"] = {
            "count": len(ids),
            "sha256": _sha256_of_lines(ids),
            "ids": ids,
        }
    return record
