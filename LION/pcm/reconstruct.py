from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from tqdm import tqdm as std_tqdm

from LION.classical_algorithms.spgl1_torch import spgl1_torch
from LION.operators.CompositeOp import CompositeOp
from LION.operators.DebiasOp import debias_ls
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.operators.Wavelet2D import Wavelet2D
from LION.pcm.config import DataConfig, PnPConfig, SPGL1Config
from LION.pcm.types import GrayscaleImage2D, Measurement1D
from LION.reconstructors.PnP import PnP

ReconFn = Callable[
    [PhotocurrentMapOp, Measurement1D, GrayscaleImage2D], GrayscaleImage2D
]

tqdm = partial(std_tqdm, dynamic_ncols=True)


def make_pnp_admm_reconstructor(
    config: PnPConfig,
    data_config: DataConfig,
    denoiser: torch.nn.Module,
) -> ReconFn:
    """Create a PnP-ADMM reconstruction callable.

    Parameters
    ----------
    config : PnPConfig
        PnP reconstruction configuration.
    data_config : DataConfig
        Data-dependent scaling configuration.
    denoiser : torch.nn.Module
        Loaded denoiser.

    Returns
    -------
    ReconFn
        Reconstruction function with the expected PCM signature.
    """
    if data_config.is_out_of_distribution:
        if data_config.r_high is None or data_config.r_low is None:
            raise ValueError(
                "r_high and r_low must be provided when is_out_of_distribution is True."
            )
        scale_a = max(data_config.r_high - data_config.r_low, data_config.scale_eps)
    else:
        scale_a = None

    def run_pnp_admm(
        pcm_op: PhotocurrentMapOp,
        pcm_measurement: Measurement1D,
        initial_image: GrayscaleImage2D,
    ) -> GrayscaleImage2D:
        """Reconstruct PCM using PnP-ADMM."""
        del initial_image

        def denoiser_fn(x: GrayscaleImage2D) -> GrayscaleImage2D:
            with torch.no_grad():
                if data_config.is_out_of_distribution:
                    model_input = (x - data_config.r_low) / scale_a
                else:
                    model_input = x

                model_output = (
                    denoiser(
                        model_input.unsqueeze(0).unsqueeze(0), sigma=config.drunet_sigma
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

                if data_config.is_out_of_distribution:
                    model_output = scale_a * model_output + data_config.r_low
                return model_output

        pnp = PnP(physics=pcm_op, prior_fn=denoiser_fn, default_algorithm="ADMM")
        return pnp.admm_algorithm(
            measurement=pcm_measurement,
            eta=config.eta,
            max_iter=config.iters,
            cg_max_iter=config.cg_iters,
            cg_eps=config.cg_eps,
            cg_rel_tol=config.cg_rel_tol,
            prog_bar=tqdm,
        )

    return run_pnp_admm


def make_spgl1_reconstructor(
    config: SPGL1Config,
    device: torch.device,
) -> ReconFn:
    """Create an SPGL1 reconstruction callable.

    Parameters
    ----------
    config : SPGL1Config
        SPGL1 configuration.
    device : torch.device
        Device used by the wavelet and operators.

    Returns
    -------
    ReconFn
        Reconstruction function with the expected PCM signature.
    """

    def run_spgl1(
        pcm_op: PhotocurrentMapOp,
        pcm_measurement: Measurement1D,
        initial_image: GrayscaleImage2D,
    ) -> GrayscaleImage2D:
        """Reconstruct PCM using SPGL1."""
        del initial_image

        height, width = pcm_op.domain_shape
        wavelet = Wavelet2D(
            (height, width), wavelet_name=config.wavelet_name, device=device
        )
        a_op = CompositeOp(wavelet, pcm_op, device=device)

        scaled_measurement = pcm_measurement * config.factor
        w_hat, _ = spgl1_torch(
            op=a_op,
            y=scaled_measurement,
            iter_lim=config.max_iter,
            verbosity=0,
        )
        w_debias = debias_ls(
            op=a_op,
            y=scaled_measurement,
            w=w_hat,
            support_tol=config.debias_support_tol,
            max_iter=config.debias_max_iter,
            tol=config.debias_tol,
            prog_bar=tqdm,
        )

        cs_result_tensor = wavelet.inverse(w_debias)
        return cs_result_tensor / config.factor

    return run_spgl1
