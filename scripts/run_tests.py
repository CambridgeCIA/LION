"""Run LION tests, including CUDA tests only when CUDA is available."""

from __future__ import annotations

import subprocess
import sys

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
