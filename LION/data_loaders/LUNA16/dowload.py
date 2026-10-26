# This file is part of LION library
# License : GPL-3
#
# Author  : Emilien Valat
# Modifications: Ander Biguri
# =============================================================================


from LION.utils.paths import LUNA_DATASET_PATH
from LION.utils.utils import run_cmd


def download():
    """Download the LUNA16 subsets into LUNA_DATASET_PATH."""
    storage_path = LUNA_DATASET_PATH
    file_ids = {3723295: [i for i in range(0, 7)], 4121926: [i for i in range(7, 10)]}

    for series_name, series_ids in file_ids.items():
        print(f"Processing series {series_name}")
        for serie_id in series_ids:
            print(f"Processing subset {serie_id}")

            placeholder_url = (
                f"https://zenodo.org/record/{series_name}/files/subset{serie_id}.zip"
            )
            zip_file_name = storage_path.joinpath(f"subset{serie_id}.zip")
            run_cmd(f"wget {placeholder_url} -P {storage_path}")
            run_cmd(f"7z x {zip_file_name} -o{storage_path} -y")
            zip_file_name.unlink()


# only when run as a script, not when imported
if __name__ == "__main__":
    download()
