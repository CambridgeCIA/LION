# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Emilien Valat
# Modifications: Ander Biguri
# =============================================================================


import subprocess
from LION.utils.paths import DETECT_PATH
from LION.utils.utils import run_cmd


storage_path = DETECT_PATH
file_ids = {3723295: [i for i in range(0, 7)], 4121926: [i for i in range(7, 10)]}

sinograms_ids = {
    "2DeteCT_slices1-1000": 8014758,
    "2DeteCT_slices1001-2000": 8014766,
    "2DeteCT_slices2001-3000": 8014787,
    "2DeteCT_slices3001-4000": 8014829,
    "2DeteCT_slices4001-5000": 8014874,
    "2DeteCT_slicesOOD": 8014907,
}  # Sinogram
recon_ids = {
    "2DeteCT_slices1-1000_RecSeg": 8017583,
    "2DeteCT_slices1001-2000_RecSeg": 8017604,
    "2DeteCT_slices2001-3000_RecSeg": 8017612,
    "2DeteCT_slices3001-4000_RecSeg": 8017618,
    "2DeteCT_slices4001-5000_RecSeg": 8017624,
    "2DeteCT_slicesOOD_RecSeg": 8017653,
}  # recon, seg

for series_name, zenodo_id in sinograms_ids.items():

    print(f"Processing sinogram series {series_name}")

    placeholder_url = f"https://zenodo.org/records/{zenodo_id}/files/{series_name}.zip"
    zip_file_name = storage_path.joinpath(f"{series_name}.zip")
    print("Dowloading zip file....")
    bash_command = f"wget {placeholder_url} -P {storage_path}"
    run_cmd(bash_command)
    print("Dowload DONE!")
    print("Extracting files....")
    bash_command = f"7z x {zip_file_name} -o{storage_path} -y"
    run_cmd(bash_command)
    print("Extraction done!")
    zip_file_name.unlink()
    print("Zip file deleted.")

for series_name, zenodo_id in recon_ids.items():

    print(f"Processing reconstruction series {series_name}")

    placeholder_url = f"https://zenodo.org/records/{zenodo_id}/files/{series_name}.zip"
    zip_file_name = storage_path.joinpath(f"{series_name}.zip")
    print("Dowloading zip file....")
    bash_command = f"wget {placeholder_url} -P {storage_path}"
    run_cmd(bash_command)
    print("Dowload DONE!")
    print("Extracting files....")
    bash_command = f"7z x {zip_file_name} -o{storage_path} -y"
    run_cmd(bash_command)
    print("Extraction done!")
    zip_file_name.unlink()
    print("Zip file deleted.")


print("done!")
