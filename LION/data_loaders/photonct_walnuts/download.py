import os
import urllib.request
import zipfile
from LION.utils.paths import PHOTONCT_WALNUTS_DATASET_PATH

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")

def unzip_file(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def download_photonct_walnut(index: int):
    # Zenodo records mapped based on the publication
    records = {
        0: "16163983", # Calibration Table
        1: "15738313",
        2: "15738361",
        3: "15748896",
        4: "15749216",
        5: "15750269",
        # Assuming users will fill the rest if needed or look up the record ID
    }
    
    PHOTONCT_WALNUTS_DATASET_PATH.mkdir(parents=True, exist_ok=True)
    
    if index == 0:
        file_name = "CalibrationTable.zip"
        record_id = records.get(0)
    else:
        file_name = f"Walnut_{index}.zip"
        record_id = records.get(index)
        
    if not record_id:
        print(f"Record ID for {file_name} is not provided in the script. Please update manually.")
        return
        
    url = f"https://zenodo.org/record/{record_id}/files/{file_name}"
    dest_path = PHOTONCT_WALNUTS_DATASET_PATH / file_name
    
    if not dest_path.exists():
        download_file(url, dest_path)
    else:
        print(f"{file_name} already exists.")
    
    extract_folder = PHOTONCT_WALNUTS_DATASET_PATH
    unzip_file(dest_path, extract_folder)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=1, help="Walnut index (1-15). 0 for Calibration Table.")
    args = parser.parse_args()
    download_photonct_walnut(args.index)
