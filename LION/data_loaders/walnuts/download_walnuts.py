import subprocess
from utils.paths import WALNUT_DATASET_PATH
from AItomotools.utils.utils import run_cmd

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
        save_file_path = storage_path.joinpath(f"walnut_{serie_id}")
        if save_file_path.is_dir():
            print(f"Walnut {serie_id} has already been processed, passing")
        else:
            placeholder_url = (
                f"https://zenodo.org/record/{series_name}/files/Walnut{serie_id}.zip"
            )
            zip_file_name = storage_path.joinpath(f"Walnut{serie_id}.zip")
            bash_command = f"wget {placeholder_url} -P {storage_path}"
            run_cmd(bash_command)
            bash_command = f"unzip {zip_file_name} -d {storage_path}"
            run_cmd(bash_command)
            zip_file_name.unlink()
