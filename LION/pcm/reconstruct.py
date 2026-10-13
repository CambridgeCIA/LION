"""Reconstruction functions for PCM."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Protocol

import torch
from tqdm import tqdm as std_tqdm

from LION.classical_algorithms.conjugate_gradient import conjugate_gradient
from LION.classical_algorithms.pdhg_direct import tv_pdhg
from LION.classical_algorithms.spgl1_torch import SPGL1Parameter, spgl1_torch
from LION.config.ExperimentConfig import ConjugateGradientConfig
from LION.data_loaders.pcm.data_loader import GrayscaleImage2D
from LION.data_loaders.pcm.data_primitives import PCMScalingParameter
from LION.operators.CompositeOp import CompositeOp
from LION.operators.DebiasOp import debias_ls
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.operators.Wavelet2D import Wavelet2D
from LION.reconstructors.GSPnP import (
    DeepInvLinearPhysics,
    GSDenoiser,
    GSPnP,
    GSPnPDeepInv,
    GSPnPParameter,
    PCMDeepInvPhysics,
)
from LION.reconstructors.PnPADMM import PnPADMM, PnPADMMParameter
from LION.utils.parameter import LIONParameter


@dataclass(kw_only=True)
class ReconInput:
    """Input data for a PCM reconstruction.

    Attributes:
        op: Photocurrent map operator.
        y: PCM measurement.
        x0: Initial image estimate.
        i0: Optional initial intensity.
    """

    op: PhotocurrentMapOp
    y: torch.Tensor
    x0: torch.Tensor
    i0: float | None


@dataclass(kw_only=True)
class ReconOutput:
    reconstruction: torch.Tensor


@dataclass(kw_only=True)
class AdaptiveMapReconOutput(ReconOutput):
    lambda_maps: torch.Tensor


class ReconFn(Protocol):
    """Type for a PCM reconstruction function."""

    def __call__(self, recon_input: ReconInput) -> ReconOutput:
        """Reconstruct PCM."""
        ...


@dataclass
class GSPnPReconFn(ReconFn):
    """Create a GS-PnP reconstruction callable.

    Assume image pixels have been standardised to [0, 1].

    Attributes:
        config: GS-PnP reconstruction configuration.
        denoiser: Loaded denoiser.
        device: Device to run the reconstruction on.
    """

    config: GSPnPParameter
    denoiser: GSDenoiser
    device: torch.device | None = None
    iteration_callback: Callable[[int, torch.Tensor], None] | None = None
    early_stop_callback: Callable[[int, torch.Tensor], None] | None = None
    backtracking_stop_callback: Callable[..., None] | None = None

    def __call__(self, recon_input: ReconInput) -> ReconOutput:
        """Reconstruct PCM using GS-PnP."""

        if self.config.uses_deepinv:
            if isinstance(recon_input.op, PhotocurrentMapOp):
                physics = PCMDeepInvPhysics(recon_input.op)
            else:
                physics = DeepInvLinearPhysics(recon_input.op)
            pnp = GSPnPDeepInv(physics=physics, denoiser=self.denoiser)
            reconstruction = pnp.reconstruct_sample(
                recon_input.y,
                config=self.config,
                iteration_callback=self.iteration_callback,
                early_stop_callback=self.early_stop_callback,
                backtracking_stop_callback=self.backtracking_stop_callback,
            )
        else:
            if (
                self.iteration_callback is not None
                or self.early_stop_callback is not None
                or self.backtracking_stop_callback is not None
            ):
                raise NotImplementedError(
                    "Iteration callbacks are not yet supported for non-DeepInv GS-PnP."
                )
            pnp = GSPnP(physics=recon_input.op, denoiser=self.denoiser)
            reconstruction = pnp.reconstruct_sample(recon_input.y, config=self.config)

        return ReconOutput(reconstruction=reconstruction)


@dataclass
class PnPADMMReconFn(ReconFn):
    """Create a PnP-ADMM reconstruction callable.

    Attributes:
        config: PnP reconstruction configuration.
        scaling: Scaling parameters for the PCM data.
        denoiser: Loaded denoiser.
        precision_builder: Takes the current image estimate and returns a precision matrix for the data term.
        device: Device to run the reconstruction on.
        shows_prog_bar: Whether to show a progress bar during reconstruction.
    """

    config: PnPADMMParameter
    # scaling: PCMScalingParameter
    denoiser: torch.nn.Module
    # precision_builder: PrecisionBuilder | None
    device: torch.device | None = None
    shows_prog_bar: bool = False
    iteration_callback: Callable[[int, torch.Tensor], None] | None = None

    def __call__(self, recon_input: ReconInput) -> ReconOutput:
        """Reconstruct PCM using PnP-ADMM."""
        # if self.scaling.is_out_of_distribution:
        #     assert self.scaling.r_high is not None
        #     assert self.scaling.r_low is not None
        #     scale_a = max(self.scaling.r_high - self.scaling.r_low, self.scaling.scale_eps)
        # else:
        #     scale_a = None

        # if self.config.sigma_start is not None and self.config.sigma_end is not None:
        #     sigmas = deque(
        #         float(sigma)
        #         for sigma in np.logspace(
        #             np.log10(self.config.sigma_start), np.log10(self.config.sigma_end), num=self.config.iters
        #         )
        #     )
        #     print("Using sigma schedule")
        # else:
        #     sigmas = deque([self.config.sigma] * self.config.iters)

        # print(f"Type of denoiser: {type(self.denoiser)}")

        # def denoiser_fn(x: GrayscaleImage2D) -> GrayscaleImage2D:
        def denoiser_fn(x: torch.Tensor, sigma: float) -> torch.Tensor:
            with torch.no_grad():
                # if scaling_param.is_out_of_distribution:
                #     model_input = (x - scaling_param.r_low) / scale_a
                # else:
                #     model_input = x

                # Assume standardised to [0, 1]
                model_input = x

                # sigma = sigmas.popleft()
                # sigma = self.config.sigma
                model_output = self.denoiser(model_input, sigma=sigma)

                # if scaling_param.is_out_of_distribution:
                #     model_output = scale_a * model_output + scaling_param.r_low
                return model_output

        pnp = PnPADMM(physics=recon_input.op, denoiser=denoiser_fn)

        # if self.precision_builder is not None:
        #     assert recon_input.i0 is not None
        #     self.precision_builder.I0 = recon_input.i0
        prog_bar = (
            partial(std_tqdm, dynamic_ncols=True) if self.shows_prog_bar else None
        )

        return ReconOutput(
            reconstruction=pnp.admm_algorithm(
                recon_input.y,
                self.config,
                prog_bar=prog_bar,
                iteration_callback=self.iteration_callback,
            )
        )
        # return pnp.admm_algorithm_lagged_covariance(
        #     recon_input.y, self.config, self.precision_builder, prog_bar=prog_bar
        # )


# def make_tv_reconstructor(tv_config: TvPdhgTorchParameter) -> ReconFn:
#     """Create a TV reconstruction callable."""
#     tv_solver = TvPdhgTorch(param=tv_config)

#     def run_tv(
#         pcm_op: PhotocurrentMapOp,
#         pcm_measurement: Measurement1D,
#         initial_image: GrayscaleImage2D,
#         I0: float | None,
#     ) -> GrayscaleImage2D:
#         """Reconstruct PCM using TV-PDHG."""
#         return tv_solver.forward(
#             measurement=pcm_measurement,
#             x=initial_image,
#             forward_op=pcm_op,
#         )

#     return run_tv


@dataclass(kw_only=True)
class ScalarTVParameter(LIONParameter):
    """Anisotropic scalar-TV reconstruction using direct PDHG.

    The loss is 0.5 * ||A x - y||^2 + lambda_reg * sum(|D_x x| + |D_y x|).
    The image and measurement use the normalized PCM operator's scale.
    """

    lambda_reg: float = 0.1
    iters: int = 1024
    primal_stepsize: float = 0.30
    dual_stepsize: float = 0.30
    theta: float = 1.0
    metric_every: int = 4

    def __post_init__(self) -> None:
        if self.lambda_reg < 0:
            raise ValueError("Scalar TV lambda_reg must be nonnegative.")
        if self.iters < 1 or self.metric_every < 1:
            raise ValueError("Scalar TV iterations and metric_every must be positive.")
        if self.primal_stepsize <= 0 or self.dual_stepsize <= 0:
            raise ValueError("Scalar TV step sizes must be positive.")


@dataclass
class ScalarTVReconFn(ReconFn):
    """Reconstruct PCM with a fixed scalar TV penalty and zero-filled initialization."""

    config: ScalarTVParameter
    iteration_callback: Callable[[int, torch.Tensor], None] | None = None
    shows_prog_bar: bool = False

    def __call__(self, recon_input: ReconInput) -> ReconOutput:
        with torch.no_grad():
            reconstruction = tv_pdhg(
                measurement=recon_input.y,
                initial_image=recon_input.x0,
                forward_op=recon_input.op,
                lambda_reg=self.config.lambda_reg,
                max_iterations=self.config.iters,
                primal_stepsize=self.config.primal_stepsize,
                dual_stepsize=self.config.dual_stepsize,
                theta=self.config.theta,
                iteration_callback=self.iteration_callback,
                callback_every=self.config.metric_every,
                use_tqdm=self.shows_prog_bar,
            )
        return ReconOutput(reconstruction=reconstruction)


def cg_reconstruction(
    op: PhotocurrentMapOp,
    measurement: torch.Tensor,
    config: ConjugateGradientConfig,
    *,
    zero_filled: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct PCM by conjugate gradients on the normal equations."""
    rhs = op.adjoint(measurement)

    if config.initial_value == "zeros":
        x0 = torch.zeros_like(zero_filled)
    elif config.initial_value == "adjoint":
        x0 = rhs
    elif config.initial_value == "zero_filled":
        x0 = zero_filled
    else:
        raise ValueError(
            f"Unsupported CG initial_value {config.initial_value!r}; "
            "expected 'zeros', 'adjoint', or 'zero_filled'."
        )

    def matmul_closure(x: torch.Tensor) -> torch.Tensor:
        return op.adjoint(op(x))

    return conjugate_gradient(
        matmul_closure, rhs, x0=x0, max_iter=config.max_iters, rel_tol=config.tolerance
    )


def spgl1_reconstruction(
    op: PhotocurrentMapOp,
    measurement: torch.Tensor,
    config: SPGL1Parameter,
    *,
    device: torch.device | None = None,
    shows_prog_bar: bool = False,
) -> torch.Tensor:
    """Reconstruct a PCM batch using independent SPGL1 solves."""
    height, width = op.domain_shape
    wavelet = Wavelet2D(
        (height, width), wavelet_name=config.wavelet_name, device=device
    )
    a_op = CompositeOp(wavelet, op, device=device)
    prog_bar = partial(std_tqdm, dynamic_ncols=True) if shows_prog_bar else None

    reconstructions = []
    for measurement_i in measurement:
        scaled_measurement = measurement_i.squeeze(0) * config.factor
        w_hat, _ = spgl1_torch(
            op=a_op, y=scaled_measurement, iter_lim=config.iters, verbosity=0
        )
        w_debias = debias_ls(
            op=a_op,
            y=scaled_measurement,
            w=w_hat,
            support_tol=config.debias_support_tol,
            max_iter=config.debias_iters,
            tol=config.debias_tol,
            prog_bar=prog_bar,
        )
        reconstructions.append(wavelet.inverse(w_debias))

    return torch.stack(reconstructions, dim=0).unsqueeze(1) / config.factor


@dataclass
class SPGL1ReconFn(ReconFn):
    """Create an SPGL1 reconstruction callable.

    Attributes:
        config: SPGL1 reconstruction configuration.
        scaling: Scaling parameters for the PCM data.
        device: Device to run the reconstruction on.
        shows_prog_bar: Whether to show a progress bar during reconstruction.
    """

    config: SPGL1Parameter
    scaling: PCMScalingParameter
    device: torch.device | None = None
    shows_prog_bar: bool = False

    def __call__(self, recon_input: ReconInput) -> ReconOutput:
        """Reconstruct PCM using SPGL1."""
        return ReconOutput(
            reconstruction=spgl1_reconstruction(
                recon_input.op,
                recon_input.y,
                self.config,
                device=self.device,
                shows_prog_bar=self.shows_prog_bar,
            )
        )
