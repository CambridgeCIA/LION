"""Where LION's data and results are: the ``LION_DATA_PATH`` and
``LION_RESULTS_PATH`` environment variables, read when a path is used.

- ``data_path(relative)``: a path inside ``LION_DATA_PATH``, e.g.
  ``data_path("processed/LIDC-IDRI")``; an error if it isn't set.
- ``results_path(relative)``: a path inside ``LION_RESULTS_PATH`` (optional:
  a solver without a ``save_folder`` saves there).

Saved parameters write paths inside these folders as calls to these functions
(``LION.utils.spec``), so they point to the same data on any machine. The
datasets' folders are named below (``paths.LIDC_IDRI_PROCESSED_DATASET_PATH``,
...), each read from ``LION_DATA_PATH`` when used, as are ``LION_DATA_PATH``
and ``LION_RESULTS_PATH`` themselves.
"""

import os
import pathlib

#: The datasets' folders, inside LION_DATA_PATH.
DATA_FOLDERS = {
    "LION_DATA_PATH": "",
    # as downloaded
    "LUNA_DATASET_PATH": "raw/LUNA16",
    "WALNUT_DATASET_PATH": "raw/walnuts",
    "LIDC_IDRI_PATH": "raw/LIDC-IDRI",
    "DETECT_PATH": "raw/2detect",
    # ready for training
    "LUNA_PROCESSED_DATASET_PATH": "processed/LUNA16",
    "LIDC_IDRI_PROCESSED_DATASET_PATH": "processed/LIDC-IDRI",
    "DETECT_PROCESSED_DATASET_PATH": "processed/2detect",
}


def _folder(variable):
    root = os.environ.get(variable)
    if root is None:
        return None
    return pathlib.Path(root).expanduser().resolve()


def data_path(relative="") -> pathlib.Path:
    """A path inside ``LION_DATA_PATH``, e.g. ``data_path("processed/LIDC-IDRI")``."""
    root = _folder("LION_DATA_PATH")
    if root is None:
        raise RuntimeError(
            "LION_DATA_PATH is not set. Set it to your LION data directory, e.g. "
            "export LION_DATA_PATH=/path/to/Data"
        )
    return root.joinpath(relative)


def results_path(relative="") -> pathlib.Path:
    """A path inside ``LION_RESULTS_PATH``, e.g. ``results_path("LPD/model")``."""
    root = _folder("LION_RESULTS_PATH")
    if root is None:
        raise RuntimeError(
            f"'{relative}' is inside LION_RESULTS_PATH, which is not set. Set it to "
            "your results folder, e.g. export LION_RESULTS_PATH=/path/to/results"
        )
    return root.joinpath(relative)


def __getattr__(name):
    """The named folders, read when used (``paths.LIDC_IDRI_PATH``)."""
    if name in DATA_FOLDERS:
        return data_path(DATA_FOLDERS[name])
    if name == "LION_RESULTS_PATH":  # None if not set
        return _folder("LION_RESULTS_PATH")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
