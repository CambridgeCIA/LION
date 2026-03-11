"""Plug-and-Play (PnP) Reconstructor using a prior function."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import torch
from tqdm import tqdm

# from LION.classical_algorithms.fdk import fdk
from LION.classical_algorithms.conjugate_gradient import conjugate_gradient
from LION.CTtools.ct_geometry import Geometry
from LION.operators.Operator import Operator
from LION.reconstructors.LIONreconstructor import LIONReconstructor
from LION.utils.math import power_method


class PnP(LIONReconstructor):
    def __init__(
        self,
        physics: Geometry | Operator,
        prior_fn: Callable[[torch.Tensor], torch.Tensor],
        default_algorithm: Literal["ADMM", "HQS", "FBS"] = "ADMM",
    ):
        """
        Plug-and-Play Reconstructor using a denoiser as prior.

        Parameters
        ----------
        physics : Geometry or Operator
            The physics of the imaging system.
            This can be either a CT Geometry object or a pre-defined Operator.
            If a Geometry is provided, the corresponding CT operator will be created.
        prior_fn : Callable[[torch.Tensor], torch.Tensor]
            The prior for plug-and-play, for example, a pre-trained denoiser model.
        algorithm : Literal["ADMM", "HQS", "FBS"], optional
            The reconstruction algorithm to use. See the options in the notes below. Default is "ADMM".

        Notes
        -----
        The following algorithms are implemented:
        - "ADMM": Alternating Direction Method of Multipliers
        - "HQS": Half Quadratic Splitting
        - "FBS": Forward-Backward Splitting
        """
        super().__init__(physics)
        self.model = prior_fn

        if default_algorithm == "HQS":
            # Half Quadratic Splitting, as from the paper "Plug-and-Play Image Restoration with Deep Denoiser Prior"
            self.default_algorithm = "HQS"  # Half Quadratic Splitting
        elif (
            default_algorithm == "FBS"
            or default_algorithm == "ForwardBackwardSplitting"
        ):
            self.default_algorithm = "FBS"  # Forward-Backward Splitting
        elif default_algorithm == "ADMM":
            self.default_algorithm = (
                "ADMM"  # Alternating Direction Method of Multipliers
            )
        else:
            raise ValueError(f"Unknown algorithm: {default_algorithm}")

    def reconstruct_sample(
        self,
        sino,
        *,
        prog_bar: bool = False,
        **kwargs,
    ):
        """
        Reconstruct the sinogram using the model and geometry.

        :param sino: Sinogram tensor.
        :return: Reconstructed image tensor.
        """
        if not isinstance(sino, torch.Tensor):
            raise TypeError("Sinogram must be a torch.Tensor")
        if sino.dim() != 3:
            raise ValueError(
                f"Sinogram must be a 3D tensor (batch_size, num_angles, num_detectors), but got {sino.dim()}"
            )
        # Apply the reconstruction algorithm
        if self.default_algorithm == "HQS":
            # Implement the Half Quadratic Splitting algorithm here
            # This is a placeholder for the actual implementation
            recon = self.hqs_algorithm(sino, prog_bar=prog_bar, **kwargs)
        elif self.default_algorithm == "FBS":
            # Implement the Forward-Backward Splitting algorithm here
            # This is a placeholder for the actual implementation
            recon = self.forward_backward_splitting(sino, prog_bar=prog_bar, **kwargs)
        elif self.default_algorithm == "ADMM":
            recon = self.admm_algorithm(sino, prog_bar=prog_bar, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {self.default_algorithm}")

        return recon

    def hqs_algorithm(
        self,
        sino,
        *,
        lambda_=0.23,
        mu=0.1,
        max_iter=100,
        noise_level=None,
        prog_bar: bool = False,
    ):
        """
        Placeholder for the Half Quadratic Splitting algorithm implementation.

        :param sino: Sinogram tensor.
        :param lambda: Regularization parameter.
        :param mu: Step size.
        :param max_iter: Maximum number of iterations.
        :return: Reconstructed image tensor.
        """
        if noise_level is not None:
            print("Warning: ignoring value of mu, estimating from noise_level")
            sigma = noise_level
            mu = lambda_ / noise_level**2
        else:
            sigma = np.sqrt(lambda_ / mu)
            # make sigma the operator norm
            sigma = power_method(self.op)
            noise_level = np.sqrt(lambda_ / mu)

        # initialize the reconstruction
        x = fdk(sino, self.op)  # Use FDK as an initial guess
        z = torch.zeros_like(x)
        iterator = range(max_iter)
        if prog_bar:
            iterator = tqdm(iterator, desc="HQS iterations")
        for i in iterator:
            x = x - 1 / (sigma**2) * self.op.T(self.op(x) - sino) + 2 * mu * (x - z)
            if (
                hasattr(self.model_parameters, "use_noise_level")
                and self.model_parameters.use_noise_level
            ):
                z = self.model(x, noise_level=noise_level)
            else:
                z = self.model(x)
        # This is where the actual HQS algorithm would be implemented
        # For now, we return a dummy tensor
        return torch.zeros_like(sino)

    def forward_backward_splitting(
        self,
        sino,
        step_size=None,
        max_iter=10,
        noise_level=None,
        prog_bar: bool = False,
    ):
        """
        Forward-Backward Splitting algorithm implementation.
        """
        if step_size is None:
            op_norm = power_method(self.op)
            step_size = 1.0 / (op_norm**2)

        x = fdk(sino, self.op)
        iterator = range(max_iter)
        if prog_bar:
            iterator = tqdm(iterator, desc="FBS iterations")
        for i in iterator:
            # TODO: would it not make sense to have an adaptive noise_level?
            if (
                hasattr(self.model.model_parameters, "use_noise_level")
                and self.model.model_parameters.use_noise_level
            ):
                step = x - step_size * self.op.T(self.op(x) - sino)

                step = self.model.normalise(step)
                x = self.model(step.unsqueeze(0), noise_level=noise_level).squeeze(0)
                x = self.model.unnormalise(x)
            else:
                step = x - step_size * self.op.T(self.op(x) - sino)

                step = self.model.normalise(step)
                x = self.model(step.unsqueeze(0)).squeeze(0)
                x = self.model.unnormalise(x)

        return x

    def admm_algorithm(
        self,
        measurement: torch.Tensor,
        eta: float = 1e-2,
        max_iter: int = 10,
        cg_max_iter: int = 100,
        cg_eps: float = 1e-14,
        cg_rel_tol: float = 0.0,
        prog_bar: Callable | None = None,
        cg_prog_bar: Callable | None = None,
    ) -> torch.Tensor:
        # TODO: Explore auto-tuning eta during iterations (residual balancing)
        #       web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

        x = torch.zeros(self.op.domain_shape, device=measurement.device)
        v = torch.zeros(self.op.domain_shape, device=measurement.device)
        u = torch.zeros(self.op.domain_shape, device=measurement.device)

        # Normalize data fidelity term by number of measurements so that eta has consistent meaning
        # regardless of measurement size
        measurement_size = measurement.numel()

        def matmul_closure(x: torch.Tensor) -> torch.Tensor:
            # return self.op.adjoint(self.op(x)) + eta * x
            return self.op.adjoint(self.op(x)) / measurement_size + eta * x

        # AT_y = self.op.adjoint(measurement)
        AT_y = self.op.adjoint(measurement) / measurement_size
        iterator = (
            prog_bar(range(max_iter), desc="ADMM iterations")
            if prog_bar
            else range(max_iter)
        )
        for _ in iterator:
            d = AT_y + eta * (v - u)
            x = conjugate_gradient(
                matmul_closure,
                d,
                x,
                max_iter=cg_max_iter,
                eps=cg_eps,
                rel_tol=cg_rel_tol,
                prog_bar=cg_prog_bar,
            )
            v = self.model(x + u)
            u = u + (x - v)
        return x
