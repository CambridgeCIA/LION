from typing import Callable, Optional
import warnings
import numpy as np
from LION.CTtools.ct_geometry import Geometry
from LION.classical_algorithms.fdk import fdk
from LION.models.LIONmodel import LIONmodel
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from LION.optimizers.LIONsolver import LIONsolver, warn_if_not_mse
from LION.utils.parameter import LIONParameter
from LION.utils.tomosipo_import import import_tomosipo
import random
import torchvision.transforms as TT


# Parameters of the noise model of LION's E2I until Oct 2026
# (Poisson + Gaussian in raw counts, with ct_utils.sinogram_add_noise)
_OLD_NOISE_PARAMETERS = ("I0", "sigma", "sigma_blur", "noise_model")


def _robust_std(values: torch.Tensor) -> torch.Tensor:
    """Standard deviation of Gaussian values from their median absolute
    deviation, insensitive to outliers (here: edges of the object)."""
    deviation = (values - values.median()).abs().median()
    return 1.4826 * deviation


def estimate_noise_std(sinos: torch.Tensor, noise_kernel=None) -> float:
    """Estimate the standard deviation of additive Gaussian noise in sinograms.

    The noise is ``noise_kernel * w`` along the detector, with ``w`` iid
    Gaussian of the returned standard deviation (``noise_kernel=None``: iid
    noise). The estimate uses second differences along the detector,
    ``y[j-1] - 2 y[j] + y[j+1]``, where a smooth signal cancels: for that noise
    their standard deviation is ``std * ||second difference of noise_kernel||``.
    A robust standard deviation (from the median absolute deviation) ignores
    the large differences at edges of the object. Triplets with a 0, e.g.
    post-log values clipped at 0 in air, are left out.

    The noise of CT data depends on the signal (it is larger behind more
    attenuation), so this is a typical level over the sinograms. The E2I
    paper measures it in a background region instead; give the solver's
    ``noise_std`` to use such a value.

    Args:
        sinos: Sinograms (..., angles, detectors).
        noise_kernel: 1D blur of the noise along the detector, or None.

    Returns:
        The estimated standard deviation (before the blur).
    """
    left, centre, right = sinos[..., :-2], sinos[..., 1:-1], sinos[..., 2:]
    keep = (left != 0) & (centre != 0) & (right != 0)
    if keep.sum() < 100:
        raise ValueError(
            "Too few nonzero detector pixels to estimate the noise; "
            "set the solver's noise_std."
        )
    differences = (left - 2 * centre + right)[keep].flatten().float()
    if differences.numel() > 10_000_000:  # the median needs no more than this
        step = differences.numel() // 10_000_000 + 1
        differences = differences[::step]
    kernel = torch.ones(1) if noise_kernel is None else torch.as_tensor(noise_kernel)
    kernel = kernel.flatten().double()
    # second differences of the kernel, with zeros outside it
    padded = torch.cat([kernel.new_zeros(2), kernel, kernel.new_zeros(2)])
    gain = torch.linalg.vector_norm(padded[:-2] - 2 * padded[1:-1] + padded[2:])
    return float(_robust_std(differences) / gain)


class Equivariance2InverseSolver(LIONsolver):
    """Equivariance2Inverse (E2I).

    The training loss is the mean squared error, as in the method's
    formulation: a projection-wise cross-validation (data consistency) term
    plus ``equivariance_strength`` (lambda in the paper) times the equivariance
    term. ``loss_fn`` is only used as the default validation and testing
    function.

    The equivariance term rotates the reconstruction by a random angle,
    projects it and adds new Gaussian noise to the (post-log) sinogram, as in
    the paper and the authors' code (github.com/D1rk123/equivariance2inverse):
    iid with standard deviation ``noise_std`` or, with ``noise_kernel``, blurred
    along the detector with that kernel (the detector's point response).
    ``noise_std=None`` estimates it from the first training batch
    (``estimate_noise_std``) and stores it in the solver parameters.
    """

    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn,
        solver_params: Optional[LIONParameter] = None,
        geometry: Geometry = None,
        verbose: bool = True,
        device: torch.device = None,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            loss_fn,
            geometry,
            verbose,
            device,
            solver_params=solver_params,
        )
        warn_if_not_mse(loss_fn, "Equivariance2InverseSolver")
        old = [k for k in _OLD_NOISE_PARAMETERS if hasattr(self.solver_params, k)]
        if old:
            warnings.warn(
                f"Equivariance2InverseSolver no longer uses {old}: it adds Gaussian "
                "noise to the post-log sinograms, as in the paper (noise_std, "
                "noise_kernel)."
            )
        # parameters saved until Oct 2026 do not have these
        for key, value in vars(self.default_parameters()).items():
            if not hasattr(self.solver_params, key):
                setattr(self.solver_params, key, value)
        to_autograd = import_tomosipo(
            "tomosipo.torch_support", "Equivariance2InverseSolver"
        ).to_autograd

        self.operator = self.op  # the base solver already built it
        self.model.geometry = self.geometry
        self.model.operator = self.operator
        self.projector = to_autograd(self.operator, num_extra_dims=1)
        self.recon_fn = self.solver_params.recon_fn

    @staticmethod
    def default_parameters() -> LIONParameter:
        params = LIONParameter()
        params.recon_fn = fdk
        # lambda in the paper, chosen per dataset there
        params.equivariance_strength = 1.0
        # Standard deviation of the Gaussian noise added to the rotated
        # (post-log) sinograms, before noise_kernel. None: estimated from the
        # first training batch (estimate_noise_std).
        params.noise_std = None
        # 1D kernel (along the detector) that blurs that noise, e.g. a measured
        # point response function. None: iid noise (the paper's default).
        params.noise_kernel = None
        return params

    def _noise(self, sinos: torch.Tensor) -> torch.Tensor:
        """New Gaussian noise for the sinograms (N_BG in the paper)."""
        noise = torch.randn_like(sinos) * self.solver_params.noise_std
        kernel = self.solver_params.noise_kernel
        if kernel is not None:
            kernel = torch.as_tensor(kernel, dtype=sinos.dtype, device=sinos.device)
            kernel = kernel.flatten()
            # as the authors' Conv2d(padding="same", padding_mode="reflect")
            left = (kernel.numel() - 1) // 2
            lines = noise.reshape(-1, 1, noise.shape[-1])
            lines = F.pad(lines, (left, kernel.numel() - 1 - left), mode="reflect")
            noise = F.conv1d(lines, kernel.view(1, 1, -1)).reshape(noise.shape)
        return noise

    def mini_batch_step(self, sinos, targets):
        if self.solver_params.noise_std is None:
            self.solver_params.noise_std = estimate_noise_std(
                sinos, self.solver_params.noise_kernel
            )
            if self.verbose:
                print(
                    "Equivariance2InverseSolver: noise_std estimated from the "
                    f"data: {self.solver_params.noise_std:.4g}"
                )
        # masking
        NP = sinos.shape[2]
        YJ_num = torch.randint(0, NP, (1,)).item()
        YJ = sinos[:, :, YJ_num, :]

        YJc = sinos.clone()
        YJc[:, :, YJ_num, :] = 0

        weight = NP / (NP - 1)
        RJc = self.recon_fn(YJc * weight, self.model.operator)
        output_recon_1 = self.model(RJc)

        output_sino_1 = self.projector(output_recon_1)
        AJ = output_sino_1[:, :, YJ_num, :]
        batch_loss = ((AJ - YJ) ** 2).mean()

        angle = random.uniform(0, 360)
        rotated_output_recon_1 = TT.functional.rotate(
            output_recon_1, angle, interpolation=TT.InterpolationMode.BILINEAR
        )
        rotated_sinogram = self.projector(rotated_output_recon_1)
        with torch.no_grad():
            noise = self._noise(rotated_sinogram)
        rotated_sinogram_noisy = rotated_sinogram + noise

        rotated_noisy_image = self.recon_fn(rotated_sinogram_noisy, self.model.operator)
        output_recon_2 = self.model(rotated_noisy_image)
        equivariance = ((output_recon_2 - rotated_output_recon_1) ** 2).mean()
        batch_loss += self.solver_params.equivariance_strength * equivariance
        return batch_loss

    def _forward(self, data):
        # validation and testing evaluate the reconstruction
        return self.reconstruct(data)

    def reconstruct(self, sinos):
        input_recon = self.recon_fn(sinos, self.operator)
        output_recon = self.model(input_recon)
        return output_recon
