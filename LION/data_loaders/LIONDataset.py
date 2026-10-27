# =============================================================================
# This file is part of LION library
# License : GPL-3
# =============================================================================
"""``LIONDataset``: the base class of LION's datasets."""

import warnings

from torch.utils.data import Dataset

from LION.utils import spec
from LION.data_loaders.data_record import (
    file_contents_hash,
    read_record,
    recorded_spec,
)


class LIONDataset(Dataset):
    """A torch ``Dataset`` that LION can save and make again.

    It only adds ``instantiate()``: what a sample is, and how data are read,
    depend on the inverse problem, so they are left to each dataset. For its
    record (``LION.data_loaders.data_record``) a LION dataset keeps its
    constructor arguments as attributes of the same names, and may list its
    samples (``sample_ids()``) and the files they are read from
    (``sample_files()``).
    """

    @classmethod
    def instantiate(cls, source, *, check_files=False, allow=()):
        """The dataset a saved file describes.

        ``source``: a data record, a model's file (or the dict read from it)
        saved with one, or the dataset's spec. Called on a dataset class it
        makes that class; called on ``LIONDataset``, the class the record
        names (``allow``: see ``LION.utils.spec.build``).

        The dataset is made alone, **without noise**: noise is part of the
        experiment. If the record's data had noise, a warning says how to get
        it (``Experiment.instantiate(source)`` for data made by an experiment).

        The dataset's samples are compared with the record's, and with
        ``check_files=True`` the contents of their files too (the record
        must have been made with ``hash_files=True``); a difference gives a
        warning.
        """
        if isinstance(source, dict) and spec.TARGET in source:
            return spec.instantiate(source, cls, LIONDataset, allow)
        record = read_record(source)
        dataset = spec.instantiate(
            recorded_spec(record, "dataset"), cls, LIONDataset, allow
        )
        _check_samples(dataset, record)
        if check_files:
            _check_files(dataset, record)
        if record.get("sinogram_transform") is not None:
            _warn_noise_left_out(dataset, record)
        return dataset


def _check_samples(dataset, record):
    if "samples" not in record or not hasattr(dataset, "sample_ids"):
        return
    saved = record["samples"]["ids"]
    now = list(dataset.sample_ids())
    if now == saved:
        return
    missing = len(set(saved) - set(now))
    new = len(set(now) - set(saved))
    if missing or new:
        difference = f"{missing} of the record's are missing, {new} are not in it"
    else:
        difference = "the same samples, in another order"
    warnings.warn(
        f"{type(dataset).__name__}: its samples differ from the record's "
        f"({len(saved)} in the record, {len(now)} now: {difference}). The data, "
        "or how LION selects them, changed since the record was made.",
        stacklevel=3,
    )


def _check_files(dataset, record):
    if "files" not in record:
        raise ValueError(
            "the record has no hash of the files' contents: it was made without "
            "data_record(dataset, hash_files=True)"
        )
    if not hasattr(dataset, "sample_files"):
        raise ValueError(f"{type(dataset).__name__} does not list its files")
    if file_contents_hash(dataset.sample_files()) != record["files"]["sha256"]:
        warnings.warn(
            f"{type(dataset).__name__}: the contents of its files differ from the "
            "record's: these are not the same data (e.g. pre-processed with other "
            "software versions).",
            stacklevel=3,
        )


def _warn_noise_left_out(dataset, record):
    noise = record["sinogram_transform"]
    if "unsaved" in noise:
        name = noise["unsaved"]
    else:
        name = noise.get(spec.TARGET, "")
        if name == "functools.partial":
            name = noise[spec.ARGS][0][spec.TARGET]
    if "experiment" in record:
        how = (
            "For the data with their noise, make the experiment: "
            f"Experiment.instantiate(source).get_dataset({record.get('split')!r})."
        )
    else:
        how = (
            "To add it: dataset.set_sinogram_transform("
            "build(record['sinogram_transform']), "
            "seed=record['sinogram_transform_seed']), with "
            "LION.utils.spec.build and record = read_record(source)."
        )
    warnings.warn(
        f"{type(dataset).__name__} is made without the noise its recorded data had "
        f"({name}, seed {record.get('sinogram_transform_seed')}): noise is part of "
        f"the experiment, not the dataset. {how}",
        stacklevel=3,
    )
