"""Build a curve of SSIM and PSNR as a function of sampling percentage."""

from dataclasses import dataclass

import numpy as np
import torch
from spyrit.core.torch import ifwht

from LION.pcm_tools.noise_metrics import compute_ssim_torch
from LION.pcm_tools.noise_utils import torch_reshape_fortran_2d


@dataclass
class SubsamplingCurve:
    sampling_percent: np.ndarray
    ssim_vals: np.ndarray
    psnr_vals: np.ndarray


@dataclass
class SubsamplingCurveBuilder:

    iy: torch.Tensor
    ground_truth: torch.Tensor
    pixel_min: float
    n_row: int
    n_col: int
    n_subsamples: int = 128
    batch_size: int = 8
    compute_ssim: bool = True
    ssim_stride: int = 1

    def build_curve(
        self, y_sorted: torch.Tensor, transpose_after_reshape: bool, skip: bool
    ) -> SubsamplingCurve:
        """
        Compute SSIM and PSNR as a function of sampling percentage.

        Parameters
        ----------
        y_sorted : torch.Tensor
            Sorted measurement vector.
        iy : torch.Tensor
            Ordered-to-Hadamard permutation.
        ground_truth : torch.Tensor
            Reference image.
        pixel_min : float
            Minimum pixel value used for MATLAB-style scaling.
        n_row : int
            Padded row count.
        n_col : int
            Padded column count.
        n_subsamples : int, default=128
            Number of sampling levels.
        batch_size : int, default=8
            Number of sampling levels processed together.
        transpose_after_reshape : bool, default=False
            Whether to use the MATLAB real-measurement reshape convention.
        compute_ssim : bool, default=True
            Whether to compute SSIM.
        ssim_stride : int, default=1
            Compute SSIM every `ssim_stride` sampling levels.
        skip : bool, default=False
            If True, skip and return empty curves.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Sampling percentages, SSIM values, and PSNR values.
        """
        if skip:
            return SubsamplingCurve(np.array([]), np.array([]), np.array([]))

        iy = self.iy
        ground_truth = self.ground_truth
        pixel_min = self.pixel_min
        n_row = self.n_row
        n_col = self.n_col
        n_subsamples = self.n_subsamples
        batch_size = self.batch_size
        compute_ssim = self.compute_ssim
        ssim_stride = self.ssim_stride

        n = int(iy.numel())
        n_rows_orig, n_cols_orig = ground_truth.shape

        sample_ids = np.arange(1, n_subsamples + 1, dtype=np.int64)
        n_keep_values = (n * sample_ids) // n_subsamples

        ssim_vals = np.full(n_subsamples, np.nan, dtype=np.float64)
        psnr_vals = np.zeros(n_subsamples, dtype=np.float64)

        data_range = torch.max(ground_truth) - torch.min(ground_truth)

        for batch_start in range(0, n_subsamples, batch_size):
            batch_stop = min(batch_start + batch_size, n_subsamples)
            counts = n_keep_values[batch_start:batch_stop]

            y_batch = torch.zeros(
                (counts.size, n), dtype=y_sorted.dtype, device=y_sorted.device
            )

            for j, n_keep in enumerate(counts):
                y_batch[j, iy[:n_keep]] = y_sorted[:n_keep]

            x_batch = ifwht(y_batch, order=False, dim=-1)

            for j in range(counts.size):
                if transpose_after_reshape:
                    image = torch_reshape_fortran_2d(x_batch[j], n_col, n_row).T
                else:
                    image = torch_reshape_fortran_2d(x_batch[j], n_row, n_col)

                image = image[:n_rows_orig, :n_cols_orig]

                mse = torch.mean((image - ground_truth) ** 2)
                if float(mse.detach().cpu()) == 0.0:
                    psnr_vals[batch_start + j] = float("inf")
                else:
                    psnr = 20.0 * torch.log10(data_range) - 10.0 * torch.log10(mse)
                    psnr_vals[batch_start + j] = float(psnr.detach().cpu())

                idx = batch_start + j
                if compute_ssim and idx % ssim_stride == 0:
                    ssim_vals[idx] = compute_ssim_torch(
                        image=image,
                        reference=ground_truth,
                        pixel_min=pixel_min,
                    )

        if compute_ssim:
            known = np.isfinite(ssim_vals)
            if np.any(known) and not np.all(known):
                ssim_vals = np.interp(
                    np.arange(n_subsamples),
                    np.flatnonzero(known),
                    ssim_vals[known],
                )
        else:
            ssim_vals[:] = np.nan

        sampling_percent = 100.0 * sample_ids / n_subsamples

        return SubsamplingCurve(sampling_percent, ssim_vals, psnr_vals)
