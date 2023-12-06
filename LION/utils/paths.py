import pathlib

## File to contain paths to data created/needed by this toolbox.
# If you don't have access to the paths and you are part of the research group, ask Ander Biguri about it.
# Otherwise, feel free to change those to your paths.

TOMOTOOLS_DATASET_PATH = pathlib.Path(
    "/store/DAMTP/ab2860/AItomotools/data/AItomotools/"
)
LUNA_DATASET_PATH = TOMOTOOLS_DATASET_PATH.joinpath("raw/LUNA16")
WALNUT_DATASET_PATH = TOMOTOOLS_DATASET_PATH.joinpath("raw/walnuts")
LIDC_IDRI_PATH = TOMOTOOLS_DATASET_PATH.joinpath("raw/LIDC-IDRI")

## Data ready for training use
LUNA_PROCESSED_DATASET_PATH = TOMOTOOLS_DATASET_PATH.joinpath("processed/LUNA16")
LIDC_IDRI_PROCESSED_DATASET_PATH = TOMOTOOLS_DATASET_PATH.joinpath(
    "processed/LIDC-IDRI"
)
