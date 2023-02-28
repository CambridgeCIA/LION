import pathlib
## File to contain paths to data created/needed by this toolbox.
# If you don't have access to the paths, ask Ander Biguri about it.

TOMOTOOLS_DATASET_PATH = pathlib.Path("/local/scratch/public/AItomotools")
LUNA_DATASET_PATH   = TOMOTOOLS_DATASET_PATH.joinpath("raw/LUNA16")
WALNUT_DATASET_PATH = TOMOTOOLS_DATASET_PATH.joinpath("raw/walnuts")

LIDC_IDRI_PATH  = pathlib.Path("/local/scratch/public/ab2860/data/LIDC-IDRI")

## Do we keep this one?
LUNA_PROCESSED_DATASET_PATH   = TOMOTOOLS_DATASET_PATH.joinpath("processed/LUNA16")




