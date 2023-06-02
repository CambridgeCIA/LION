import subprocess
from AItomotools.utils.paths import LUNA_DATASET_PATH
from AItomotools.utils.utils import run_cmd


storage_path = LUNA_DATASET_PATH
storage_path = storage_path.joinpath("test")
file_ids = {3723295: [i for i in range(0, 7)], 4121926: [i for i in range(7, 10)]}

for series_name, series_ids in file_ids.items():
    print(f"Processing series {series_name}")
    for serie_id in series_ids:
        print(f"Processing subset {serie_id}")

        placeholder_url = (
            f"https://zenodo.org/record/{series_name}/files/subset{serie_id}.zip"
        )
        zip_file_name = storage_path.joinpath(f"subset{serie_id}.zip")
        bash_command = f"wget {placeholder_url} -P {storage_path}"
        run_cmd(bash_command)
        subfolder = torage_path.joinpath(zip_file_name)
        bash_command = f"7z e {zip_file_name} -o{storage_path}"
        run_cmd(bash_command)
        # zip_file_name.unlink()
        exit()
