"""Plug-and-Play Reconstructor using a denoiser as prior."""

from typing import Callable, Literal, Union

import numpy as np
import torch

from LION.CTtools.ct_geometry import Geometry
from LION.classical_algorithms import conjugate_gradient, fdk
from LION.operators import Operator
from LION.reconstructors import LIONReconstructor
from LION.utils.math import power_method


class PnPReconstructor(LIONReconstructor):

    def __init__(
        self,
        physics: Union[Geometry, Operator],
        denoiser: Callable[[torch.Tensor], torch.Tensor],
        algorithm: Literal["ADMM", "HQS", "FBS"] = "ADMM",
    ):
        """
        Plug-and-Play Reconstructor using a denoiser as prior.

        Parameters
        ----------
        physics : Geometry or Operator
            The forward operator or information required to create a forward operator.
            If a Geometry is provided, the corresponding CT operator will be created.
        denoiser : Callable[[torch.Tensor], torch.Tensor]
            A denoising function that takes a torch.Tensor and returns a denoised torch.Tensor.
            This is most likely a pre-trained model.
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
        self.denoiser = denoiser

        algorithms = {
            # Half Quadratic Splitting, as from the paper "Plug-and-Play Image Restoration with Deep Denoiser Prior"
            "HQS": self.half_quadratic_splitting_algorithm,
            #
            # Forward-Backward Splitting
            "FBS": self.forward_backward_splitting_algorithm,
            #
            # Alternating direction method of multipliers (ADMM)
            "ADMM": self.admm_algorithm,
        }

        if algorithm not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self.algorithm_fn = algorithms[algorithm]

    def reconstruct_sample(self, sino, **kwargs):
        """
        Reconstruct the sinogram using the model and geometry.

        :param sino: Sinogram tensor.
        :return: Reconstructed image tensor.
        """
        if not isinstance(sino, torch.Tensor):
            raise TypeError("Sinogram must be a torch.Tensor")
        if sino.dim() != 3:
            raise ValueError(
                "Sinogram must be a 3D tensor (batch_size, num_angles, num_detectors), but got {}".format(
                    sino.dim()
                )
            )
        # Apply the reconstruction algorithm
        recon = self.algorithm_fn(sino, **kwargs)
        return recon

    def half_quadratic_splitting_algorithm(
        self, sino, lambda_=0.23, mu=0.1, max_iter=100, noise_level=None
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
        for i in range(max_iter):
            x = x - 1 / (sigma**2) * self.op.adjoint(self.op(x) - sino) + 2 * mu * (x - z)
            if (
                hasattr(self.model_parameters, "use_noise_level")
                and self.model_parameters.use_noise_level
            ):
                z = self.denoiser(x, noise_level=noise_level)
            else:
                z = self.denoiser(x)
        # This is where the actual HQS algorithm would be implemented
        # For now, we return a dummy tensor
        return torch.zeros_like(sino)

    def forward_backward_splitting_algorithm(
        self, sino, step_size=None, max_iter=10, noise_level=None
    ):
        """
        Forward-Backward Splitting algorithm implementation.
        """

        if step_size is None:
            op_norm = power_method(self.op)
            step_size = 1.0 / (op_norm**2)

        x = fdk(sino, self.op)
        with torch.no_grad():
            for i in range(max_iter):
                # TODO: would it not make sense to have an adaptive noise_level?
                if (
                    # hasattr(self.denoiser.model_parameters, "use_noise_level")
                    # and self.denoiser.model_parameters.use_noise_level
                    False
                ):
                    step = x - step_size * self.op.adjoint(self.op(x) - sino)

                    step = self.denoiser.normalise(step)
                    x = self.denoiser(
                        step.unsqueeze(0), noise_level=noise_level
                    ).squeeze(0)
                    x = self.denoiser.unnormalise(x)
                else:
                    step = x - step_size * self.op.adjoint(self.op(x) - sino)

                    # step = self.denoiser.normalise(step)
                    x = self.denoiser(step.unsqueeze(0)).squeeze(0)
                    # x = self.denoiser.unnormalise(x)

        return x

    def admm_algorithm(
        self,
        measurement: torch.Tensor,
        eta: float = 1e-4,
        max_iter: int = 10,
        cg_max_iter: int = 100,
        cg_tol: float = 1e-7
    ) -> torch.Tensor:
        x = torch.zeros(self.op.domain_shape, device=measurement.device)
        v = torch.zeros(self.op.domain_shape, device=measurement.device)
        u = torch.zeros(self.op.domain_shape, device=measurement.device)

        def matmul_closure(x: torch.Tensor) -> torch.Tensor:
            return self.op.adjoint(self.op(x)) + eta * x

        AT_y = self.op.adjoint(measurement)
        for _ in range(max_iter):
            d = AT_y + eta * (v - u)
            x = conjugate_gradient(matmul_closure, d, x, max_iter=cg_max_iter, tol=cg_tol)
            v = self.denoiser(x + u)
            u = u + (x - v)
        return x
