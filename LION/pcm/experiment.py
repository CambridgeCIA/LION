from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm as std_tqdm

from LION.pcm.config import ExperimentConfig
from LION.utils.pcm_sampling import multilevel_sample, uniform_random_sample

tqdm = partial(std_tqdm, dynamic_ncols=True)


METRICS_HEADER = (
    "sampling_percentage,coarse_J,"
    "mse_zero_filled,psnr_zero_filled,ssim_zero_filled,pearson_corr_zero_filled,"
    "mse_recon,psnr_recon,ssim_recon,pearson_corr_recon\n"
)


def resolve_device(device_name: str) -> torch.device:
    """Resolve the torch device from a string.

    Parameters
    ----------
    device_name : str
        Requested device string, or ``'auto'``.

    Returns
    -------
    torch.device
        Resolved torch device.
    """
    if device_name != "auto":
        return torch.device(device_name)

    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )


def make_csv(method_name: str, log_dir: Path | str) -> None:
    """Create the metrics CSV file for one reconstruction method.

    Parameters
    ----------
    method_name : str
        Method-specific output directory name.
    log_dir : Path | str
        Trial log directory.
    """
    method_dir = Path(log_dir) / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    csv_path = method_dir / "metrics.csv"
    csv_path.write_text(METRICS_HEADER)


# Designs: small-but-many (this) vs. big-but-few (current LION)
#   - Support both?
#   - Turn small-but-many into big-but-few with custom data loader logics:
#     - Pass a list of (sampling_ratio, coarse_J) pairs to the data loader,
#       then iterate over them using the data loader.
#       For logging, data loader can also return the sub-experiment name.
def make_test_cases(config: ExperimentConfig) -> list[tuple[float, int]]:
    """Generate sampling/coarse-level test cases.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.

    Returns
    -------
    list[tuple[float, int]]
        List of ``(sampling_ratio, coarse_J)`` pairs.
    """
    test_cases: list[tuple[float, int]] = []
    j_order = config.data.j_order

    if config.sampling.coarse_j_values is not None:
        coarse_j_values = list(config.sampling.coarse_j_values)
    elif config.sampling.coarse_j_offset_from_j_order is not None:
        coarse_j_values = [j_order - config.sampling.coarse_j_offset_from_j_order]
    else:
        coarse_j_values = list(range(j_order))

    for sampling_ratio in config.sampling.sampling_ratios:
        for coarse_j in coarse_j_values:
            test_cases.append((sampling_ratio, coarse_j))

    if config.sampling.reverse_test_cases:
        test_cases.reverse()
    return test_cases


def sample_indices(
    j_order: int,
    sampling_ratio: float,
    coarse_j: int,
    randomising_scheme: str,
    seed: int,
) -> np.ndarray:
    """Generate sampling indices for one test case.

    Parameters
    ----------
    j_order : int
        Walsh-Hadamard order.
    sampling_ratio : float
        Ratio of sampled measurements.
    coarse_j : int
        Number of coarse levels to keep deterministically.
    randomising_scheme : str
        Sampling scheme name.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Sampled measurement indices.
    """
    n = 1 << j_order
    num_samples = int(sampling_ratio * n * n)
    rng = np.random.default_rng(seed)

    if randomising_scheme == "multilevel":
        return multilevel_sample(
            J=j_order,
            num_samples=num_samples,
            coarse_J=coarse_j,
            alpha=1.0,
            rng=rng,
        )
    if randomising_scheme == "uniform":
        return uniform_random_sample(
            J=j_order,
            num_samples=num_samples,
            coarse_J=coarse_j,
            rng=rng,
        )

    raise ValueError(f"Unknown sampling scheme '{randomising_scheme}'.")
