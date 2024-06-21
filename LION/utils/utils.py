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
