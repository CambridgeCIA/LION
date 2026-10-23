from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def read_pcm_current_map_from_txt(
    filepath: str | Path,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Read a ground-truth current map from a text file.

    Args:
        filepath: Path to the map file.
        device: Device for the output tensor.
        dtype: Data type for the output tensor.

    Returns:
        Two-dimensional floating-point array.
    """
    return torch.as_tensor(
        np.loadtxt(filepath, dtype=np.float64), device=device, dtype=dtype
    )


def read_pcm_measurement_data_from_txt(filepath: str | Path) -> pd.DataFrame:
    """Read a PCM measurement file.

    Args:
        filepath: Path to the PCM text file.

    Returns:
        DataFrame with columns `PatternIndex` and `MeasuredCurrentA`.
    """
    return pd.read_csv(
        filepath,
        sep=r"\t",
        skiprows=3,
        header=None,
        names=["PatternIndex", "MeasuredCurrentA"],
        usecols=[0, 1],
        engine="python",
    )
