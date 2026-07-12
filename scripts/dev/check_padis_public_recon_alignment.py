"""Focused checks for LION PaDIS public-repo reconstruction compatibility."""

from __future__ import annotations

import pathlib
import random
import sys

import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
PADIS_REPO = PROJECT_ROOT / "PaDIS_lion_recon"
if str(PROJECT_ROOT / "LION") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "LION"))
if str(PADIS_REPO) not in sys.path:
    sys.path.insert(0, str(PADIS_REPO))

from denoise_padding import denoisedFromPatches, getIndices  # noqa: E402
from inverse_nodist import dps as public_dps  # noqa: E402
from LION.operators import Operator  # noqa: E402
from LION.reconstructors import PaDIS  # noqa: E402
from LION.utils.parameter import LIONParameter  # noqa: E402


class PublicStyleDenoiser(torch.nn.Module):
    """Analytic denoiser exposing the public PaDIS model interface."""

    def forward(self, x, sigma, x_pos=None, class_labels=None):
        del class_labels
        sigma_view = torch.as_tensor(sigma, device=x.device, dtype=x.dtype).reshape(
            1, 1, 1, 1
        )
        position_term = 0.0
        if x_pos is not None:
            position_term = 0.05 * x_pos[:, 0:1] - 0.03 * x_pos[:, 1:2]
        return x + position_term + 0.01 * sigma_view

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class RawEDMModel(torch.nn.Module):
    """Wrap an analytic denoiser with raw EDM preconditioning semantics."""

    def __init__(self, *, pad_width=2, patch_size=5):
        super().__init__()
        self.model_parameters = LIONParameter()
        self.model_parameters.input_position_channels = 2
        self.model_parameters.prior_mode = "patch"
        self.model_parameters.pad_width = int(pad_width)
        self.model_parameters.largest_patch_size = int(patch_size)

    def forward(self, model_input, c_noise):
        sigma = torch.exp(4.0 * c_noise).to(
            device=model_input.device, dtype=model_input.dtype
        )
        sigma_view = sigma.reshape(model_input.shape[0], 1, 1, 1)
        sigma_data = torch.as_tensor(
            0.5, device=model_input.device, dtype=model_input.dtype
        )
        c_skip = sigma_data.square() / (sigma_view.square() + sigma_data.square())
        c_out = (
            sigma_view * sigma_data / (sigma_view.square() + sigma_data.square()).sqrt()
        )
        c_in = 1 / (sigma_data.square() + sigma_view.square()).sqrt()

        image = model_input[:, 0:1] / c_in
        position = model_input[:, 1:3]
        denoised = (
            image
            + 0.05 * position[:, 0:1]
            - 0.03 * position[:, 1:2]
            + 0.01 * sigma_view
        )
        return (denoised - c_skip * image) / c_out


class IdentityOperator(Operator):
    """Small identity forward model used to isolate sampler mechanics."""

    def __init__(self, shape):
        super().__init__(device=torch.device("cpu"))
        self._shape = tuple(shape)

    def __call__(self, x, out=None):
        result = self.forward(x)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def forward(self, x):
        return x

    def A(self, x):
        return self.forward(x)

    def adjoint(self, y):
        return y

    def AT(self, y):
        return self.adjoint(y)

    def inverse(self, y):
        return y

    def Adagger(self, y):
        return self.inverse(y)

    @property
    def domain_shape(self):
        return self._shape

    @property
    def range_shape(self):
        return self._shape


def make_lion_reconstructor(*, image_size=8, pad_width=2, patch_size=5):
    model = RawEDMModel(pad_width=pad_width, patch_size=patch_size)
    params = PaDIS.padis_repo_ct_parameters(model)
    params.num_steps = 2
    params.inner_steps = 10
    params.sigma_min = 0.5
    params.sigma_max = 1.0
    params.pad_width = int(pad_width)
    params.patch_size = int(patch_size)
    params.clip_output = False
    params.trace_interval = 1
    params.data_consistency_scale = 1.0
    return PaDIS(
        IdentityOperator((1, image_size, image_size)), model, parameters=params
    )


def full_position_grid(height, width, device):
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=0).unsqueeze(0)


def assert_close(name, got, expected, atol=2e-5, rtol=2e-5):
    if not torch.allclose(got, expected, atol=atol, rtol=rtol):
        diff = torch.max(torch.abs(got - expected)).item()
        raise AssertionError(f"{name} mismatch: max abs diff {diff:g}")


def test_noise_schedule():
    reconstructor = make_lion_reconstructor()
    params = reconstructor.parameters
    schedule = reconstructor._noise_schedule(params, torch.device("cpu"))
    step_indices = torch.arange(params.num_steps, dtype=torch.float64)
    expected = (
        params.sigma_max ** (1 / params.rho)
        + step_indices
        / (params.num_steps - 1)
        * (params.sigma_min ** (1 / params.rho) - params.sigma_max ** (1 / params.rho))
    ) ** params.rho
    expected = torch.cat([expected, torch.zeros(1, dtype=expected.dtype)])
    assert_close("noise schedule", schedule, expected, atol=0, rtol=0)


def test_patch_layout_and_denoising():
    reconstructor = make_lion_reconstructor()
    params = reconstructor.parameters
    device = torch.device("cpu")
    x = torch.linspace(-0.2, 1.2, 12 * 12, device=device).reshape(1, 1, 12, 12)
    sigma = torch.tensor([0.75], device=device)
    patches = 8 // 5 + 1
    spaced = torch.arange(patches, device=device).cpu().numpy() * 5

    random.seed(1234)
    public_indices = getIndices(spaced, patches, 2, 5)
    random.seed(1234)
    lion_layout = reconstructor._patch_layout((8, 8), params, device)
    if [list(index) for index in lion_layout.indices] != public_indices:
        raise AssertionError("patch layout mismatch")

    positions = full_position_grid(12, 12, device)
    public = denoisedFromPatches(
        PublicStyleDenoiser(),
        x,
        sigma.squeeze(0),
        positions,
        None,
        public_indices,
        t_goal=0,
    )
    lion = reconstructor._denoise_patches(x, sigma, lion_layout, params)
    assert_close("patch denoising assembly", lion, public, atol=3e-5, rtol=3e-5)


def test_dps_loop_identity_operator():
    image_size = 256
    pad_width = 2
    patch_size = 129
    measurement = torch.linspace(0.0, 1.0, image_size * image_size).reshape(
        1, image_size, image_size
    )
    latents_pos = full_position_grid(
        image_size + 2 * pad_width,
        image_size + 2 * pad_width,
        measurement.device,
    )

    random.seed(7)
    torch.manual_seed(11)
    latents = torch.randn([1, 1, image_size, image_size])
    public = public_dps(
        PublicStyleDenoiser(),
        latents,
        latents_pos,
        IdentityOperator((1, image_size, image_size)),
        noisy=measurement,
        clean=measurement.numpy(),
        num_steps=2,
        sigma_min=0.5,
        sigma_max=1.0,
        rho=7,
        zeta=0.3,
        pad=pad_width,
        psize=patch_size,
        intermediate_interval=0,
    )
    public = public[
        :,
        pad_width : pad_width + image_size,
        pad_width : pad_width + image_size,
    ].detach()

    random.seed(7)
    torch.manual_seed(11)
    lion = make_lion_reconstructor(
        image_size=image_size,
        pad_width=pad_width,
        patch_size=patch_size,
    ).reconstruct_sample(measurement)
    assert_close("DPS loop", lion, public, atol=5e-4, rtol=5e-4)


def main():
    """Run deterministic public/LION sampler-alignment assertions."""
    test_noise_schedule()
    test_patch_layout_and_denoising()
    test_dps_loop_identity_operator()
    print("PaDIS public reconstruction compatibility checks passed.")


if __name__ == "__main__":
    main()
