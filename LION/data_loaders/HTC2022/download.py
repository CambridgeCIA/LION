# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Downloads the Helsinki Tomography Challenge 2022 data (Meaney, Silva de
Moura, Siltanen and Juvonen; Zenodo, version 1.4.0, CC BY 4.0, 124 MB) into
``LION_DATA_PATH/raw/HTC2022``: the 5 training phantoms and the 21 test
phantoms, with their full and limited-angle sinograms."""

from LION.utils.paths import HTC2022_PATH
from LION.utils.utils import download_file, unzip_file

ZENODO_RECORD = 8041800
FILES = ["htc2022_teaching_data", "htc2022_test_data"]

if __name__ == "__main__":
    HTC2022_PATH.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}.zip"
        zipped = HTC2022_PATH.joinpath(f"{name}.zip")
        download_file(url, zipped)
        unzip_file(zipped, HTC2022_PATH.joinpath(name))
