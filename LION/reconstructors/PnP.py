"""Plug-and-Play (PnP) Reconstructor using a prior function."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import torch
from tqdm import tqdm

from LION.classical_algorithms.conjugate_gradient import conjugate_gradient
from LION.classical_algorithms.fdk import fdk
from LION.CTtools.ct_geometry import Geometry
from LION.models.LIONmodel import LIONmodel
from LION.operators import CTProjectionOp, Operator
from LION.reconstructors.LIONreconstructor import LIONReconstructor
from LION.utils.math import power_method


class PnP(LIONReconstructor):
    def __init__(
        self,
        physics: Geometry | Operator,
        prior_fn: Callable[[torch.Tensor], torch.Tensor],
        algorithm: Literal["ADMM", "HQS", "FBS"] = "ADMM",
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
            A LIONmodel prior is called on a batch of one image (with its
            normalisation, and the noise level if it uses one). Any other callable
            is called on the image as it is (the operator's domain shape) and
            must return the same shape.
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

        if algorithm == "HQS":
            # Half Quadratic Splitting, as from the paper "Plug-and-Play Image Restoration with Deep Denoiser Prior"
            self.algorithm = "HQS"  # Half Quadratic Splitting
        elif algorithm == "FBS" or algorithm == "ForwardBackwardSplitting":
            self.algorithm = "FBS"  # Forward-Backward Splitting
        elif algorithm == "ADMM":
            self.algorithm = "ADMM"  # Alternating Direction Method of Multipliers
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def reconstruct_sample(
        self,
        sino,
        *,
        prog_bar: bool = False,
        **kwargs,
    ):
        """
        Reconstruct one measurement (e.g. a (C, angles, detectors) sinogram)
        with the selected algorithm. Extra keyword arguments are passed to it.

        :param sino: Measurement tensor, in the range of the operator.
        :return: Reconstructed image tensor, in the domain of the operator.
        """
        if not isinstance(sino, torch.Tensor):
            raise TypeError("Sinogram must be a torch.Tensor")
        if self.algorithm == "HQS":
            return self.hqs_algorithm(sino, prog_bar=prog_bar, **kwargs)
        if self.algorithm == "FBS":
            return self.forward_backward_splitting(sino, prog_bar=prog_bar, **kwargs)
        if self.algorithm == "ADMM":
            return self.admm_algorithm(sino, prog_bar=prog_bar, **kwargs)
        raise ValueError(f"Unknown algorithm: {self.algorithm}")

    # ------------------------------------------------------------------ helpers
    def _uses_noise_level(self) -> bool:
        params = getattr(self.model, "model_parameters", None)
        return bool(getattr(params, "use_noise_level", False))

    def _apply_prior(self, x: torch.Tensor, noise_level=None) -> torch.Tensor:
        """Apply the prior to an image ``x`` with the operator's domain shape.

        LIONmodel priors get a batch of one (with the model's normalisation and,
        if the model takes one, the noise level). Other callables get ``x`` as it
        is and must return the same shape.
        """
        if not isinstance(self.model, LIONmodel):
            return self.model(x)
        x_in = self.model.normalise(x.unsqueeze(0))
        if self._uses_noise_level():
            out = self.model(x_in, noise_level=noise_level)
        else:
            out = self.model(x_in)
        return self.model.unnormalise(out).squeeze(0)

    def _initial_guess(self, sino: torch.Tensor) -> torch.Tensor:
        """FDK for CT operators, zeros otherwise."""
        if isinstance(self.op, CTProjectionOp):
            return fdk(sino, self.op)
        return torch.zeros(self.op.domain_shape, device=sino.device, dtype=sino.dtype)

    def _solve_data_step(self, sino, z, weight, x0, cg_max_iter, cg_tol):
        """argmin_x ||A x - y||^2 / 2 + weight ||x - z||^2 / 2, by conjugate gradient."""

        def normal_operator(v: torch.Tensor) -> torch.Tensor:
            return self.op.adjoint(self.op(v)) + weight * v

        rhs = self.op.adjoint(sino) + weight * z
        return conjugate_gradient(
            normal_operator, rhs, x0, max_iter=cg_max_iter, tol=cg_tol
        )

    # --------------------------------------------------------------- algorithms
    def hqs_algorithm(
        self,
        sino,
        *,
        lambda_=0.23,
        mu=0.1,
        max_iter=100,
        noise_level=None,
        cg_max_iter=100,
        cg_tol=1e-7,
        prog_bar: bool = False,
    ):
        """
        Half Quadratic Splitting, as in "Plug-and-Play Image Restoration with Deep
        Denoiser Prior" (Zhang et al.). Alternates

            x = argmin ||A x - y||^2 / 2 + mu ||x - z||^2 / 2   (conjugate gradient)
            z = prior(x)

        :param sino: Measurement tensor.
        :param lambda_: Regularisation parameter.
        :param mu: Weight of the quadratic coupling term.
        :param max_iter: Number of HQS iterations.
        :param noise_level: If given, mu = lambda_ / noise_level**2; otherwise the
            noise level passed to the prior is sqrt(lambda_ / mu).
        :return: Reconstructed image tensor.
        """
        if noise_level is not None:
            mu = lambda_ / noise_level**2
        else:
            noise_level = float(np.sqrt(lambda_ / mu))

        x = self._initial_guess(sino)
        z = x.clone()
        iterator = range(max_iter)
        if prog_bar:
            iterator = tqdm(iterator, desc="HQS iterations")
        for _ in iterator:
            x = self._solve_data_step(sino, z, mu, x, cg_max_iter, cg_tol)
            z = self._apply_prior(x, noise_level)
        return z

    def forward_backward_splitting(
        self,
        sino,
        step_size=None,
        max_iter=10,
        noise_level=None,
        prog_bar: bool = False,
    ):
        """
        Forward-Backward Splitting: x = prior(x - step_size * A^T (A x - y)).

        :param step_size: Gradient step; 1 / ||A||^2 if None.
        """
        if step_size is None:
            op_norm = power_method(self.op, device=sino.device)
            step_size = 1.0 / (op_norm**2)

        x = self._initial_guess(sino)
        iterator = range(max_iter)
        if prog_bar:
            iterator = tqdm(iterator, desc="FBS iterations")
        for _ in iterator:
            # TODO: would it not make sense to have an adaptive noise_level?
            step = x - step_size * self.op.adjoint(self.op(x) - sino)
            x = self._apply_prior(step, noise_level)
        return x

    def admm_algorithm(
        self,
        measurement: torch.Tensor,
        eta: float = 1e-4,
        max_iter: int = 10,
        cg_max_iter: int = 100,
        cg_tol: float = 1e-7,
        noise_level=None,
        prog_bar: bool = False,
    ) -> torch.Tensor:
        """
        ADMM with the prior as the proximal step:

            x = argmin ||A x - y||^2 / 2 + eta ||x - (v - u)||^2 / 2   (CG)
            v = prior(x + u)
            u = u + x - v
        """
        x = torch.zeros(self.op.domain_shape, device=measurement.device)
        v = torch.zeros(self.op.domain_shape, device=measurement.device)
        u = torch.zeros(self.op.domain_shape, device=measurement.device)

        iterator = range(max_iter)
        if prog_bar:
            iterator = tqdm(iterator, desc="ADMM iterations")
        for _ in iterator:
            x = self._solve_data_step(measurement, v - u, eta, x, cg_max_iter, cg_tol)
            v = self._apply_prior(x + u, noise_level)
            u = u + (x - v)
        return x
