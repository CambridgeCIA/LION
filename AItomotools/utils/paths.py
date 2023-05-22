import pathlib
## File to contain paths to data created/needed by this toolbox.
# If you don't have access to the paths, ask Ander Biguri about it.

TOMOTOOLS_DATASET_PATH = pathlib.Path("/local/scratch/public/AItomotools")
LUNA_DATASET_PATH   = TOMOTOOLS_DATASET_PATH.joinpath("raw/LUNA16")
WALNUT_DATASET_PATH = TOMOTOOLS_DATASET_PATH.joinpath("raw/walnuts")
LIDC_IDRI_PATH      = TOMOTOOLS_DATASET_PATH.joinpath("raw/LIDC-IDRI")

## DAta ready for training use
LUNA_PROCESSED_DATASET_PATH  = TOMOTOOLS_DATASET_PATH.joinpath("processed/LUNA16")
LIDC_IDRI_PROCESSED_DATASET_PATH = TOMOTOOLS_DATASET_PATH.joinpath("processed/LIDC-IDRI")




