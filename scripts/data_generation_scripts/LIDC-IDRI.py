# =============================================================================
# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================

import subprocess

process = subprocess.Popen(
    "conda run -n lidc_idri python ./AItomotools/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py".split(),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,
)
output, error = process.communicate()
print(output)
if error:
    print(f"ERROR: {error}")
    raise RuntimeError(
        "Can't run the data processing code. Likely you don't have the 'lidc_idri' python enviroment created. \n create it with 'conda env create -f ./AItomotools/data_loaders/LIDC_IDRI/pre_process_lidc_idri_environment.yml'"
    )
