"""Run LION tests, including CUDA tests only when CUDA is available."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch


def pytest_args(args: list[str], *, cuda_available: bool) -> list[str]:
    """Build pytest arguments for the available hardware."""

    result = list(args)
    if not cuda_available and not any(
        arg == "-m" or arg.startswith("-m=") for arg in result
    ):
        result.extend(["-m", "not cuda"])
    return result


def main() -> int:
    """Detect CUDA, invoke pytest, and return pytest's process status."""

    # Unit tests import path-resolving LION modules but never require a real
    # dataset.  Give clean CI machines an isolated root while preserving every
    # explicitly configured development/production path.
    test_data_root = Path(tempfile.gettempdir()) / "lion-test-data"
    os.environ.setdefault("LION_DATA_PATH", str(test_data_root))
    os.environ.setdefault("LION_EXPERIMENTS_PATH", str(test_data_root / "experiments"))
    Path(os.environ["LION_DATA_PATH"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["LION_EXPERIMENTS_PATH"]).mkdir(parents=True, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    mode = "full suite (CUDA available)" if cuda_available else "CPU suite"
    print(f"Running LION {mode}.", flush=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        *pytest_args(sys.argv[1:], cuda_available=cuda_available),
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
