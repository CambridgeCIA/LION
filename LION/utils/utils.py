import json
import numpy as np
import subprocess
from pathlib import Path
from typing import Dict
import pathlib
import argparse

## JSON numpy encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # NumPy scalar -> native Python scalar
        if isinstance(obj, np.generic):
            return obj.item()
        # NumPy array -> list
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_git_revision_hash() -> str:
    "Gets git commit hash"
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


# Run cmd line
def run_cmd(cmd, verbose=True, *args, **kwargs):
    if verbose:
        print(cmd)
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    std_out, std_err = process.communicate()

    if process.returncode:
        print("Error encountered in commad line call:")
        raise RuntimeError(std_err.strip())
    if verbose:
        print(std_out.strip(), std_err)
    return std_out.strip()


def check_if_file_changed_git(fname, hash1, hash2) -> bool:
    bash_command = f"git diff --name-only {hash1} {hash2} {fname}"
    out = run_cmd(bash_command, verbose=False)
    return bool(out)


# This replaces standard python warnings with a custom format
def custom_format_warning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + "\n"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def unzip_file(zipped_file_path: Path, unzipped_file_path: Path) -> None:
    """Unzip file. First tries unzip command then 7zip if unzip fails"""
    try:
        print(f"Unzipping {zipped_file_path} files with unzip...")
        bash_command = f"unzip {zipped_file_path} -d {unzipped_file_path}"
        run_cmd(bash_command)
    except RuntimeError:
        try:
            print("Unzipping with Unzip failed. Unzipping with 7zip")
            bash_command = f"7z x {zipped_file_path} -o {unzipped_file_path} -y"
            run_cmd(bash_command)
        except RuntimeError:
            raise RuntimeError("Both unzip and 7zip failed for unzipping")

    print("Extraction done!")
    zipped_file_path.unlink()
    print(f"{zipped_file_path} deleted.")


def download_file(placeholder_url: str, file_path: Path) -> None:
    """
    Downloads file from placeholder_url. Tries first with wget, then with curl.

    """
    if not file_path.is_file():
        try:
            print(f"Downloading {placeholder_url} with wget...")
            bash_command = (
                f"wget {placeholder_url} -P {file_path.parent} -O {file_path.stem}"
            )
            run_cmd(bash_command)

        except RuntimeError:
            try:
                print("Downloading with wget failed. Downloading with curl...")
                bash_command = f"curl {placeholder_url} > {file_path}"
                run_cmd(bash_command)
            except RuntimeError:
                raise RuntimeError("curl and wget failed, cannot download")

        print("Dowload DONE!")
    else:
        print(f"{file_path.stem} already exists in {file_path.parent}, passing")
