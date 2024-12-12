import pathlib

## File to contain paths to data created/needed by this toolbox.
# If you don't have access to the paths and you are part of the research group, ask Ander Biguri about it.
# Otherwise, feel free to change those to your paths.

LION_DATA_PATH = pathlib.Path("/home/cr661/rds/hpc-work/store/LION/data")
LUNA_DATASET_PATH = LION_DATA_PATH.joinpath("raw/LUNA16")
WALNUT_DATASET_PATH = LION_DATA_PATH.joinpath("raw/walnuts")
LIDC_IDRI_PATH = LION_DATA_PATH.joinpath("LIDC-IDRI")
DETECT_PATH = LION_DATA_PATH.joinpath("raw/2detect")

## Data ready for training use
LUNA_PROCESSED_DATASET_PATH = LION_DATA_PATH.joinpath("processed/LUNA16")
LIDC_IDRI_PROCESSED_DATASET_PATH = LION_DATA_PATH.joinpath("LIDC-IDRI")
DETECT_PROCESSED_DATASET_PATH = LION_DATA_PATH.joinpath("processed/2detect")
