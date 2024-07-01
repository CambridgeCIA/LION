# This file is part of LION library
# License : BSD-3
#
# Author  : Emilien Valat
# Modifications: Ander Biguri
# =============================================================================

from LION.utils.paths import DETECT_PATH
from LION.utils.utils import download_file, unzip_file

storage_path = DETECT_PATH
file_ids = {3723295: [i for i in range(0, 7)], 4121926: [i for i in range(7, 10)]}

data_ids = {
    ### Sinogram
    "2DeteCT_slices1-1000": 8014758,
    "2DeteCT_slices1001-2000": 8014766,
    "2DeteCT_slices2001-3000": 8014787,
    "2DeteCT_slices3001-4000": 8014829,
    "2DeteCT_slices4001-5000": 8014874,
    "2DeteCT_slicesOOD": 8014907,
    ### Reconstructions
    "2DeteCT_slices1-1000_RecSeg": 8017583,
    "2DeteCT_slices1001-2000_RecSeg": 8017604,
    "2DeteCT_slices2001-3000_RecSeg": 8017612,
    "2DeteCT_slices3001-4000_RecSeg": 8017618,
    "2DeteCT_slices4001-5000_RecSeg": 8017624,
    "2DeteCT_slicesOOD_RecSeg": 8017653,
}  # recon, seg

if __name__ == '__main__':
    for series_name, zenodo_id in data_ids.items():
        placeholder_url = f"https://zenodo.org/records/{zenodo_id}/files/{series_name}.zip"
        zipped_file_path   = storage_path.joinpath(f"{series_name}.zip")
        unzipped_file_path = storage_path.joinpath(f"{series_name}")

        download_file(placeholder_url, zipped_file_path)
        unzip_file(zipped_file_path, unzipped_file_path)
    
    print("done!")