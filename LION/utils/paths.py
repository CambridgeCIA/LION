import os
import pathlib

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

# Optional PCM datasets. Override LION_TTV22_PATH on other machines.
ROOT = os.environ.get("LION_STORAGE_ROOT", "/")
TTV22_PATH = (
    pathlib.Path(
        os.environ.get("LION_TTV22_PATH", str(pathlib.Path(ROOT) / "store/LION/ttv22"))
    )
    .expanduser()
    .resolve()
)
PCM_SYNTHETIC_PNG_PATH = (
    TTV22_PATH / "pcm/data/PCM Synthetic Data Set/Synthetic Data/Synthetic Data"
)
PCM_EXAMPLE_DATA_PATH = TTV22_PATH / "pcm/data/photocurrent_data"
PCM_REAL_DATA_EXAMPLE_PATH = TTV22_PATH / "pcm/data/example_data"
