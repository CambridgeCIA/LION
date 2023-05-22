
import json
import numpy as np
import subprocess
from pathlib import Path

## JSON numpy encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_git_commit():
    return subprocess.check_output(["git", "describe", "--always"], cwd=Path(__file__).resolve().parent).strip().decode()