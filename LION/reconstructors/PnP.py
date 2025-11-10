import torch
import numpy as np
from tqdm import tqdm

from LION.reconstructors.LIONreconstructor import LIONReconstructor
from LION.classical_algorithms.conjugate_gradient import conjugate_gradient
from LION.classical_algorithms.fdk import fdk
from LION.utils.math import power_method


class PnP(LIONReconstructor):
    def __init__(self, geometry, model, algorithm):
        super().__init__(geometry)
        self.model = model

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
        if self.algorithm == "HQS":
            # Implement the Half Quadratic Splitting algorithm here
            # This is a placeholder for the actual implementation
            recon = self.hqs_algorithm(sino, prog_bar=prog_bar, **kwargs)
        elif self.algorithm == "FBS":
            # Implement the Forward-Backward Splitting algorithm here
            # This is a placeholder for the actual implementation
            recon = self.forward_backward_splitting(sino, prog_bar=prog_bar, **kwargs)
        elif self.algorithm == "ADMM":
            recon = self.admm_algorithm(sino, prog_bar=prog_bar, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

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
        eta: float = 1e-4,
        max_iter: int = 10,
        cg_max_iter: int = 100,
        cg_tol: float = 1e-7,
        prog_bar: bool = False,
    ) -> torch.Tensor:
        x = torch.zeros(self.op.domain_shape, device=measurement.device)
        v = torch.zeros(self.op.domain_shape, device=measurement.device)
        u = torch.zeros(self.op.domain_shape, device=measurement.device)

        def matmul_closure(x: torch.Tensor) -> torch.Tensor:
            return self.op.T(self.op(x)) + eta * x

        AT_y = self.op.T(measurement)
        iterator = range(max_iter)
        if prog_bar:
            iterator = tqdm(iterator, desc="ADMM iterations")
        for _ in iterator:
            d = AT_y + eta * (v - u)
            x = conjugate_gradient(
                matmul_closure, d, x, max_iter=cg_max_iter, tol=cg_tol
            )
            v = self.model(x + u)
            u = u + (x - v)
        return x
