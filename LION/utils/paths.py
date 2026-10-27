import pathlib
import os

## The primary data path is set via the LION_DATA_PATH environment variable.

_lion_data_path = os.environ.get("LION_DATA_PATH")
if _lion_data_path is None:
    raise RuntimeError(
        "LION_DATA_PATH is not set. "
        "Set it to your LION data directory, e.g. "
        "export LION_DATA_PATH=/path/to/Data"
    )

LION_DATA_PATH = pathlib.Path(_lion_data_path).expanduser().resolve()

LUNA_DATASET_PATH = LION_DATA_PATH.joinpath("raw/LUNA16")
WALNUT_DATASET_PATH = LION_DATA_PATH.joinpath("raw/walnuts")
LIDC_IDRI_PATH = LION_DATA_PATH.joinpath("raw/LIDC-IDRI")
DETECT_PATH = LION_DATA_PATH.joinpath("raw/2detect")

## Data ready for training use
LUNA_PROCESSED_DATASET_PATH = LION_DATA_PATH.joinpath("processed/LUNA16")
LIDC_IDRI_PROCESSED_DATASET_PATH = LION_DATA_PATH.joinpath("processed/LIDC-IDRI")
DETECT_PROCESSED_DATASET_PATH = LION_DATA_PATH.joinpath("processed/2detect")


## Where results (trained models, checkpoints) go by default, set via the
## optional LION_RESULTS_PATH environment variable: a solver without a
## save_folder saves there. None if not set.
_lion_results_path = os.environ.get("LION_RESULTS_PATH")
LION_RESULTS_PATH = (
    None
    if _lion_results_path is None
    else pathlib.Path(_lion_results_path).expanduser().resolve()
)


def results_path(relative) -> pathlib.Path:
    """A path inside ``LION_RESULTS_PATH``, e.g. ``results_path("LPD/model")``.
    Saved parameters write paths inside it this way, as for ``data_path``."""
    root = os.environ.get("LION_RESULTS_PATH")
    if root is None:
        raise RuntimeError(
            f"'{relative}' is inside LION_RESULTS_PATH, which is not set. Set it to "
            "your results folder, e.g. export LION_RESULTS_PATH=/path/to/results"
        )
    return pathlib.Path(root).expanduser().resolve().joinpath(relative)


def data_path(relative) -> pathlib.Path:
    """A path inside ``LION_DATA_PATH``, e.g. ``data_path("processed/LIDC-IDRI")``.
    Saved parameters write paths inside ``LION_DATA_PATH`` this way (see
    ``LION.utils.spec``), so they point to the same data on any machine."""
    return LION_DATA_PATH.joinpath(relative)
