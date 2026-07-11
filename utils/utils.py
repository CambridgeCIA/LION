import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch


## JSON numpy encoder
class JSONParamEncoder(json.JSONEncoder):
    def default(self, obj):
        """Default converter for JSON serialization.

        Handles common non-JSON types used in LION and converts them to
        JSON-serialisable Python types.
        """
        # Handle pathlib.Path
        if isinstance(obj, Path):
            return str(obj)

        # Handle torch.Tensor
        if isinstance(obj, torch.Tensor):
            # Scalar tensor -> Python number
            if obj.dim() == 0:
                return obj.item()
            # Non-scalar tensor -> nested lists
            return obj.detach().cpu().tolist()

        # Handle NumPy scalar types
        if isinstance(obj, np.generic):
            return obj.item()

        # Handle NumPy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Fallback to the standard behaviour (will raise TypeError if unknown)
        return super().default(obj)


def get_git_revision_hash() -> str:
    """Gets git commit hash"""
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
