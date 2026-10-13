"""Plug-and-Play (PnP) Reconstructor using the ADMM algorithm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, cast

import torch

from LION.classical_algorithms.conjugate_gradient import (
    CGProxParameter,
    conjugate_gradient,
)
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm_tools.pcm_prox import (
    PCMProxParameter,
    closed_form_pcm_prox_with_precomputed_terms,
    precompute_pcm_prox_terms,
)
from LION.reconstructors.BasePnP import BasePnP, DenoiserParameter, DenoiserT
from LION.utils.parameter import LIONParameter


@dataclass(kw_only=True)
class PnPADMMParameter(LIONParameter):
    """Configuration for PnP-ADMM reconstruction.

    Attributes:
        denoiser: The configuration for the denoiser.
        iters: The number of iterations.
        prox: Configuration for the proximal operator for the data-fidelity term.
        sigma: The noise level for the denoiser.

        lagged_iters: The number of lagged iterations.
        lagged_damping: The damping factor for the lagged iterations.
        sigma_start: The starting noise level for the denoiser (if using a schedule).
        sigma_end: The ending noise level for the denoiser (if using a schedule).
    """

    denoiser: DenoiserParameter
    iters: int = 1
    prox: Any
    sigma: float = 0.05

    # lagged_iters: int = 1
    # lagged_damping: float = 1.0
    # sigma_start: float | None = None
    # sigma_end: float | None = None


class ADMMDenoiser(torch.nn.Module):
    """Convergent denoiser to be used in the ADMM algorithm."""

    def __init__(self, network: torch.nn.Module, b: float, eps: float = 1e-12):
        super().__init__()
        self.network = network
        self.b = b
        self.eps = eps

    def forward(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply the denoiser to the input tensor.

        Args:
            x: The input tensor.
            sigma: The noise level for the denoiser.

        Returns:
            The denoised tensor.
        """
        network_output = self.network(x, sigma=sigma)

        residual = network_output - x

        sizes = torch.full(
            (x.shape[0],), x[0].numel(), dtype=torch.long, device=x.device
        )

        q = torch.linalg.vector_norm(residual, ord=2) / torch.sqrt(sizes.float()).item()

        factor = 1.0
        if q > self.eps:
            factor = min(factor, self.b * sigma / q)

        return x + residual * factor


class PnPADMM(BasePnP[DenoiserT]):
    """Plug-and-Play (PnP) Reconstructor using the ADMM algorithm."""

    def __init__(
        self, physics, denoiser: DenoiserT, measurement_dims: tuple[int, ...] = (-1,)
    ):
        """Initialize the PnP-ADMM reconstructor.

        Args:
            physics: A CT Geometry or an Operator. If Geometry, CT operator will be inferred.
            denoiser: The denoiser prior used in the PnP-ADMM algorithm.
            measurement_dims: Dimensions belonging to one measurement sample. Useful for normalizing.
        """
        super().__init__(physics, denoiser)
        self.measurement_dims = measurement_dims

    @torch.no_grad()
    def reconstruct_sample(self, measurement: torch.Tensor, **kwargs) -> torch.Tensor:
        """Reconstruct a single sample using the PnP-ADMM algorithm.

        Args:
            measurement: The measurement tensor.
            **kwargs: Additional keyword arguments consisting of:
                - `config` (required): Configuration parameters for the PnP-ADMM algorithm.
                - `prog_bar` (optional): Progress bar callable for ADMM iterations.

        Returns:
            Reconstructed image tensor.
        """
        config: PnPADMMParameter = cast(PnPADMMParameter, kwargs["config"])
        prog_bar: Callable | None = kwargs.get("prog_bar")

        return self.admm_algorithm(measurement, config=config, prog_bar=prog_bar)

    def admm_algorithm(
        self,
        measurement: torch.Tensor,
        config: PnPADMMParameter,
        prog_bar: Callable | None = None,
        iteration_callback: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """ADMM algorithm implementation for PnP reconstruction.

        Args:
            measurement: The measurement tensor.
            config: Configuration parameters for the ADMM algorithm.
            prog_bar: Progress bar callable for ADMM iterations.
            iteration_callback: Optional callback called with the 1-based iteration number
                and current reconstruction after each ADMM iteration.

        Returns:
            Reconstructed image tensor.
        """
        # TODO: Use Chan's ADMM (also adaptive eta, assume bounded denoiser)
        #  ``Plug-and-Play ADMM for Image Restoration:  Fixed Point Convergence and Applications``
        # TODO: Explore auto-tuning eta during iterations (residual balancing)
        #       web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
        adjoint_recon = self.op.adjoint(measurement).detach()
        x = torch.zeros_like(adjoint_recon)
        v = torch.zeros_like(adjoint_recon)
        u = torch.zeros_like(adjoint_recon)
        del adjoint_recon
        prox = self.prep_prox(config.prox, measurement)
        iterator = prog_bar(range(config.iters)) if prog_bar else range(config.iters)
        for iteration in iterator:
            x = prox(v - u)
            self.debug_value(x, "x")
            denoiser_input = x + u
            self.debug_value(denoiser_input, "denoiser_input")
            v = self.denoiser(denoiser_input, sigma=config.sigma)
            # v = self.denoiser(denoiser_input)  # Assume sigma is already handled
            self.debug_value(v, "v")
            u = u + (x - v)
            self.debug_value(u, "u")
            if iteration_callback is not None:
                iteration_callback(iteration + 1, x)
        return x

    def debug_value(self, value: torch.Tensor, name: str) -> None:
        """Print debug information about a tensor value."""
        # print(
        #     f"{name} range: [{value.min()}, {value.max()}], has NaN: {torch.isnan(value).any()}, has Inf: {torch.isinf(value).any()}"
        # )

    def prep_prox(
        self, config, measurement: torch.Tensor
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Prepare the proximal operator for the data-fidelity term.

        Args:
            config: Configuration for the proximal operator.
            measurement: The measurement tensor.

        Returns:
            Callable that computes the proximal step.
        """
        op = self.op

        if isinstance(config, PCMProxParameter):
            assert isinstance(op, PhotocurrentMapOp), f"Got {type(op)}."
            ST_y, m = precompute_pcm_prox_terms(op, y=measurement)
            rho = config.rho
            assert rho is not None
            m_plus_rho = m + rho

            def prox(z: torch.Tensor) -> torch.Tensor:
                return closed_form_pcm_prox_with_precomputed_terms(
                    op, rho=rho, z=z, ST_y=ST_y, m_plus_rho=m_plus_rho
                )

        elif isinstance(config, CGProxParameter):

            # Normalize by number of measurements so that rho is independent of the measurement size
            measurement_size = math.prod(
                measurement.shape[d] for d in self.measurement_dims
            )

            def matmul_closure(q: torch.Tensor) -> torch.Tensor:
                # return op.adjoint(op(q)) + rho * q
                return op.adjoint(op(q)) / measurement_size + config.eta * q

            # AT_y = op.adjoint(measurement)
            AT_y = op.adjoint(measurement) / measurement_size

            def prox(z: torch.Tensor) -> torch.Tensor:
                rhs = AT_y + config.eta * z
                return conjugate_gradient(
                    matmul_closure,
                    rhs,
                    x0=z,
                    max_iter=config.iters,
                    eps=config.eps,
                    rel_tol=config.rel_tol,
                )

        else:
            raise ValueError(f"Unsupported prox configuration: {type(config)}")

        return prox


#     def admm_algorithm_lagged_covariance(
#         self,
#         measurement: torch.Tensor,
#         config: PnPADMMParameter,
#         precision_builder: PrecisionBuilder | None = None,
#         measurement_offset: torch.Tensor | float | None = None,
#         prog_bar: Callable | None = None,
#         cg_prog_bar: Callable | None = None,
#     ) -> torch.Tensor:
#         """ADMM algorithm with lagged covariance weighted data fidelity.

#         Args:
#             measurement: The measurement tensor.
#             config: Configuration parameters for the ADMM algorithm.
#             precision_builder: Takes the current image estimate and returns a precision matrix for the data term.
#             measurement_offset: Offset to subtract from the measurement before applying the precision operator.
#             prog_bar: Progress bar for outer iterations.
#             cg_prog_bar: Progress bar for conjugate gradient iterations.

#         Returns:
#             Reconstructed image tensor.
#         """
#         x = torch.zeros(self.op.domain_shape, device=measurement.device)
#         v = torch.zeros(self.op.domain_shape, device=measurement.device)
#         u = torch.zeros(self.op.domain_shape, device=measurement.device)

#         if measurement_offset is None:
#             measurement_tilde = measurement
#         else:
#             measurement_tilde = measurement - measurement_offset

#         # tensor_stats("measurement_tilde", measurement_tilde)

#         measurement_size = measurement.numel()

#         if precision_builder is not None:
#             print("Using precision builder")

#         iterator = prog_bar(range(config.iters), desc="ADMM iterations") if prog_bar else range(config.iters)
#         for _ in iterator:
#             z = v - u
#             x_lag = x
#             # tensor_stats("x_lag", x_lag)

#             for _ in range(config.lagged_iters):
#                 if precision_builder is None:

#                     def apply_precision(b: torch.Tensor) -> torch.Tensor:
#                         return b

#                 else:
#                     s_lag = self.op(x_lag)
#                     # tensor_stats("s_lag", s_lag)
#                     apply_precision = precision_builder(s_lag)

#                 P_y = apply_precision(measurement_tilde)
#                 # tensor_stats("P_y", P_y)

#                 AT_P_y = self.op.adjoint(P_y) / measurement_size
#                 # tensor_stats("AT_P_y / m", AT_P_y)

#                 rhs = config.data_weight * AT_P_y + config.rho * z
#                 # tensor_stats("rhs", rhs)

#                 def matmul_closure(q: torch.Tensor) -> torch.Tensor:
#                     return (
#                         config.data_weight * self.op.adjoint(apply_precision(self.op(q))) / measurement_size
#                         + config.rho * q
#                     )

#                 x_new = conjugate_gradient(
#                     matmul_closure,
#                     rhs,
#                     x0=x_lag,
#                     max_iter=config.cg_iters,
#                     eps=config.cg_eps,
#                     rel_tol=config.cg_rel_tol,
#                     prog_bar=cg_prog_bar,
#                 )
#                 # tensor_stats("x_new", x_new)

#                 x_lag = (1.0 - config.lagged_damping) * x_lag + config.lagged_damping * x_new

#             x = x_lag

#             v = self.denoiser(x + u)
#             u = u + (x - v)

#         return x


# @dataclass(kw_only=True)
# class PrecisionBuilder(LIONParameter):
#     """Building a PCM precision operator.

#     Attributes:
#         I0: Bias photocurrent scale.
#         sigma_eta_rel: Relative white bias-noise standard deviation.
#         sigma_g_rel: Relative bias-gradient standard deviation.
#         sigma_beam_rel: Relative beam-noise standard deviation in decimal units.
#         sigma_dark_abs: Absolute dark-noise standard deviation in amps.
#         time_grid: Differential time-grid vector. If None, uses 0, 1, ..., m - 1.
#         differential_factor: Variance multiplier for plus-minus differencing.
#         variance_floor_rel: Relative floor based on the median signal magnitude.
#         jitter: Small positive variance added for numerical stability.
#     """

#     I0: float
#     sigma_eta_rel: float = 2.9e-5
#     sigma_g_rel: float = 1.6e-7
#     sigma_beam_rel: float = 0.012 / 100.0
#     sigma_dark_abs: float = 0.0
#     time_grid: torch.Tensor | None = None
#     differential_factor: float = 2.0
#     variance_floor_rel: float = 1e-6
#     jitter: float = 1e-20

#     def __call__(self, s_lag: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
#         """Build a numerically safer PCM precision-operator factory.

#         The approximate differential covariance is

#             Sigma = D + alpha t t^T,

#         where

#             D_i = differential_factor * (I0^2 sigma_eta_rel^2 + sigma_dark_abs^2)
#                 + sigma_beam_rel^2 s_i^2
#                 + variance floor,

#             alpha = I0^2 sigma_g_rel^2.

#         The returned precision is normalised so that typical weights are order one.

#         Args:
#             s_lag: The lagged signal vector (H x_lag).

#         Returns:
#             Builder mapping s_lag = H x_lag to a frozen precision operator.

#         Raises:
#             ValueError: If the length of `s_lag` does not match the length of `time_grid` (if provided).
#         """

#         s = s_lag.reshape(-1)

#         dtype = s.dtype
#         device = s.device

#         if self.time_grid is None:
#             t = torch.arange(s.numel(), dtype=dtype, device=device)
#         else:
#             t = self.time_grid.to(dtype=dtype, device=device).reshape(-1)

#         if t.numel() != s.numel():
#             raise ValueError(f"time_grid has length {t.numel()}, but s_lag has length {s.numel()}.")

#         abs_var = self.differential_factor * ((self.I0 * self.sigma_eta_rel) ** 2 + self.sigma_dark_abs**2)

#         signal_scale = torch.median(torch.abs(s.detach()))
#         signal_scale = torch.clamp(signal_scale, min=abs(self.I0) * 1e-12 + 1e-30)

#         variance_floor = (self.sigma_beam_rel * self.variance_floor_rel * signal_scale) ** 2

#         diag_var = abs_var + (self.sigma_beam_rel * s) ** 2 + variance_floor + self.jitter

#         diag_var = torch.clamp(diag_var, min=self.jitter)

#         alpha = (self.I0 * self.sigma_g_rel) ** 2

#         # Normalise the precision so that typical weights are order one.
#         # This rescales the data-fidelity term and therefore changes the effective eta.
#         var_scale = torch.median(diag_var.detach())
#         var_scale = torch.clamp(var_scale, min=self.jitter)

#         D_inv_t = t / diag_var
#         denominator = 1.0 + alpha * torch.sum(t * D_inv_t)

#         denominator = torch.clamp(denominator, min=self.jitter)

#         def apply_precision(b: torch.Tensor) -> torch.Tensor:
#             b_shape = b.shape
#             b_flat = b.reshape(-1)

#             if b_flat.numel() != s.numel():
#                 raise ValueError(f"Input has length {b_flat.numel()}, expected {s.numel()}.")

#             D_inv_b = b_flat / diag_var
#             coeff = alpha * torch.sum(t * D_inv_b) / denominator
#             out = D_inv_b - coeff * D_inv_t

#             return (var_scale * out).reshape(b_shape)

#         return apply_precision
