# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri, Emilien Valat, Thanh Trung (Troy) Vu
# =============================================================================
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np
import torch


class JSONParamEncoder(json.JSONEncoder):
    """A JSON encoder for LION's parameters: paths become strings, tensors and
    numpy arrays lists (numbers when scalar). Use as ``json.dump(...,
    cls=JSONParamEncoder)``."""

    def default(self, obj):
        """``obj`` as something JSON can write; the standard encoder's
        TypeError for other types."""
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


# Run cmd line
def run_cmd(cmd, verbose=True):
    """Runs the shell command ``cmd`` and returns its output; a RuntimeError
    with its error output if it fails. ``verbose``: print the command and its
    output."""
    if verbose:
        print(cmd)
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    std_out, std_err = process.communicate()

    if process.returncode:
        raise RuntimeError(f"{cmd} failed: {std_err.strip()}")
    if verbose:
        print(std_out.strip(), std_err)
    return std_out.strip()


def unzip_file(zipped_file_path: Path, unzipped_file_path: Path) -> None:
    """Unzips ``zipped_file_path`` into ``unzipped_file_path`` with ``unzip``
    (``7z`` if that fails), then deletes the zip file."""
    try:
        print(f"Unzipping {zipped_file_path} files with unzip...")
        bash_command = f"unzip {shlex.quote(str(zipped_file_path))} -d {shlex.quote(str(unzipped_file_path))}"
        run_cmd(bash_command)
    except RuntimeError:
        try:
            print("Unzipping with Unzip failed. Unzipping with 7zip")
            # 7z takes the output folder glued to the flag: -o<folder>
            bash_command = f"7z x {shlex.quote(str(zipped_file_path))} -o{shlex.quote(str(unzipped_file_path))} -y"
            run_cmd(bash_command)
        except RuntimeError:
            raise RuntimeError("Both unzip and 7zip failed for unzipping")

    print("Extraction done!")
    zipped_file_path.unlink()
    print(f"{zipped_file_path} deleted.")


def download_file(placeholder_url: str, file_path: Path) -> None:
    """Downloads ``placeholder_url`` to ``file_path`` with ``wget`` (``curl``
    if that fails), unless the file is there already."""
    if not file_path.is_file():
        try:
            print(f"Downloading {placeholder_url} with wget...")
            # -O sets the full output path (it overrides -P)
            bash_command = (
                f"wget {shlex.quote(placeholder_url)} -O {shlex.quote(str(file_path))}"
            )
            run_cmd(bash_command)

        except RuntimeError:
            try:
                print("Downloading with wget failed. Downloading with curl...")
                bash_command = f"curl -fL {shlex.quote(placeholder_url)} -o {shlex.quote(str(file_path))}"
                run_cmd(bash_command)
            except RuntimeError:
                raise RuntimeError("curl and wget failed, cannot download")

        print("Download done!")
    else:
        print(f"{file_path.stem} already exists in {file_path.parent}, passing")
