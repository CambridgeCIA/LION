"""Plug-and-Play (PnP) Reconstructor using a prior function."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from LION.classical_algorithms.conjugate_gradient import conjugate_gradient
from LION.classical_algorithms.fdk import fdk
from LION.models.LIONmodel import LIONmodel
from LION.operators import CTProjectionOp, Operator
from LION.reconstructors.LIONreconstructor import LIONReconstructor
from LION.utils.math import power_method
from LION.utils.parameter import LIONParameter

# The settings of each algorithm, and their defaults.
ALGORITHMS = {
    "ADMM": dict(eta=1e-4, max_iter=10, cg_max_iter=100, cg_tol=1e-7, noise_level=None),
    "HQS": dict(
        lambda_=0.23,
        mu=0.1,
        max_iter=100,
        noise_level=None,
        cg_max_iter=100,
        cg_tol=1e-7,
    ),
    "FBS": dict(step_size=None, max_iter=10, noise_level=None),
}


class PnP(LIONReconstructor):
    """Plug-and-Play reconstruction with a denoiser (``model``) as the prior.

    ``PnP(operator, model, parameters)``, with ``parameters`` from
    ``PnP.default_parameters(algorithm)``:

    - ``algorithm``: "ADMM" (Alternating Direction Method of Multipliers),
      "HQS" (Half Quadratic Splitting) or "FBS" (Forward-Backward Splitting);
    - that algorithm's settings (see ``admm_algorithm``, ``hqs_algorithm``,
      ``forward_backward_splitting``). A setting missing from the parameters
      has its default.

    ``model``: the denoiser. A LIONmodel prior is called on a batch of one
    image (with its normalisation, and the noise level if it uses one). Any
    other callable is called on the image as it is (the operator's domain
    shape) and must return the same shape.
    """

    def __init__(self, operator, model=None, parameters=None):
        super().__init__(operator, model, parameters)
        algorithm = getattr(self.parameters, "algorithm", None)
        if algorithm not in ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {algorithm!r} (one of {', '.join(ALGORITHMS)})"
            )
        self._op_norm = None  # ||A||, computed once when first needed (FBS)

    @staticmethod
    def default_parameters(algorithm="ADMM") -> LIONParameter:
        """``algorithm`` and its settings, with their defaults."""
        if algorithm not in ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {algorithm!r} (one of {', '.join(ALGORITHMS)})"
            )
        return LIONParameter(algorithm=algorithm, **ALGORITHMS[algorithm])

    def setting(self, name):
        """A setting of the algorithm: from the parameters, or its default."""
        return getattr(
            self.parameters, name, ALGORITHMS[self.parameters.algorithm][name]
        )

    def reconstruct_sample(self, sino, x0=None, prog_bar: bool = False):
        """Reconstruct one measurement (e.g. a (C, angles, detectors)
        sinogram) with the algorithm and settings in ``self.parameters``."""
        if not isinstance(sino, torch.Tensor):
            raise TypeError("Sinogram must be a torch.Tensor")
        algorithm = self.parameters.algorithm
        settings = {k: self.setting(k) for k in ALGORITHMS[algorithm]}
        if algorithm == "HQS":
            return self.hqs_algorithm(sino, x0=x0, prog_bar=prog_bar, **settings)
        if algorithm == "FBS":
            return self.forward_backward_splitting(
                sino, x0=x0, prog_bar=prog_bar, **settings
            )
        return self.admm_algorithm(sino, x0=x0, prog_bar=prog_bar, **settings)

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
        return self.model.unnormalise(out, x.unsqueeze(0)).squeeze(0)

    def initial_guess(self, sino: torch.Tensor) -> torch.Tensor:
        """FDK for CT operators, zeros otherwise (the default start of HQS/FBS)."""
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
        x0=None,
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
        :param x0: Starting image; ``initial_guess(sino)`` (FDK for CT) if None.
        :return: Reconstructed image tensor.
        """
        if noise_level is not None:
            mu = lambda_ / noise_level**2
        else:
            noise_level = float(np.sqrt(lambda_ / mu))

        x = self.initial_guess(sino) if x0 is None else x0
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
        x0=None,
        prog_bar: bool = False,
    ):
        """
        Forward-Backward Splitting: x = prior(x - step_size * A^T (A x - y)).

        :param step_size: Gradient step; 1 / ||A||^2 if None (||A|| is computed
            once per reconstructor, on the first call).
        :param x0: Starting image; ``initial_guess(sino)`` (FDK for CT) if None.
        """
        if step_size is None:
            if self._op_norm is None:
                self._op_norm = power_method(self.op, device=sino.device)
            step_size = 1.0 / (self._op_norm**2)

        x = self.initial_guess(sino) if x0 is None else x0
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
        x0=None,
        prog_bar: bool = False,
    ) -> torch.Tensor:
        """
        ADMM with the prior as the proximal step:

            x = argmin ||A x - y||^2 / 2 + eta ||x - (v - u)||^2 / 2   (CG)
            v = prior(x + u)
            u = u + x - v

        :param x0: Starting image for x and v; zeros if None. For example
            ``x0=pnp.initial_guess(measurement)`` starts from FDK (CT).
        """
        if x0 is None:
            x = torch.zeros(self.op.domain_shape, device=measurement.device)
        else:
            x = x0
        v = x.clone()
        u = torch.zeros(self.op.domain_shape, device=measurement.device)

        iterator = range(max_iter)
        if prog_bar:
            iterator = tqdm(iterator, desc="ADMM iterations")
        for _ in iterator:
            x = self._solve_data_step(measurement, v - u, eta, x, cg_max_iter, cg_tol)
            v = self._apply_prior(x + u, noise_level)
            u = u + (x - v)
        return x
