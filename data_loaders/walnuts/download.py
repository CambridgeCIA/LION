# This file is part of LION library
# License : BSD-3
#
# Author  : Emilien Valat
# Modifications: Ander Biguri
# =============================================================================

from LION.utils.paths import WALNUT_DATASET_PATH
from LION.utils.utils import unzip_file, download_file

# Downloads the Walnut Dataset (https://www.nature.com/articles/s41597-019-0235-y) on a LINUX machine
storage_path = WALNUT_DATASET_PATH

file_ids = {
    2686726: [i for i in range(1, 9)],
    2686971: [i for i in range(9, 17)],
    2687387: [i for i in range(17, 25)],
    2687635: [i for i in range(25, 33)],
    2687897: [i for i in range(33, 38)],
    2688112: [i for i in range(38, 43)],
}

for series_name, series_ids in file_ids.items():
    print(f"Processing series {series_name}")
    for serie_id in series_ids:
        print(f"Processing Walnut {serie_id}")
        ## Check if folder exists
        file_name = f"Walnut{serie_id}"
        placeholder_url = (
            f"https://zenodo.org/record/{series_name}/files/Walnut{serie_id}.zip"
        )
        zipped_file_path = storage_path.joinpath(f"{file_name}.zip")
        unzipped_file_path = storage_path.joinpath(file_name)
        download_file(placeholder_url, zipped_file_path)
        unzip_file(zipped_file_path, unzipped_file_path)
