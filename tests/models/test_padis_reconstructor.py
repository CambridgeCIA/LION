import torch
from torch import nn

from LION.operators import Operator
from LION.reconstructors import PaDIS
from LION.utils.parameter import LIONParameter


class IdentityOp(Operator):
    @property
    def domain_shape(self):
        return (1, 8, 8)

    @property
    def range_shape(self):
        return (1, 8, 8)

    def __call__(self, x, out=None):
        return x

    def forward(self, x):
        return x

    def adjoint(self, y):
        return y

    def inverse(self, y):
        return y


class ScaledIdentityOp(IdentityOp):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x, out=None):
        return self.scale * x

    def forward(self, x):
        return self.scale * x

    def adjoint(self, y):
        return self.scale * y

    def inverse(self, y):
        return y / self.scale


class ZeroPatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_parameters = LIONParameter()
        self.model_parameters.pad_width = 2
        self.model_parameters.largest_patch_size = 4

    def forward(self, x, time_cond):
        del time_cond
        return torch.zeros(
            x.shape[0], 1, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype
        )


def _sampler_params(model):
    params = PaDIS.default_parameters(model)
    params.num_steps = 1
    params.inner_steps = 1
    params.patch_size = 4
    params.pad_width = 2
    params.sigma_min = 0.01
    params.sigma_max = 0.02
    params.patch_batch_size = 2
    return params


def test_padis_dps_langevin_reconstructor_smoke():
    torch.manual_seed(0)
    model = ZeroPatchModel()
    reconstructor = PaDIS(
        IdentityOp(), model, _sampler_params(model), algorithm="dps_langevin"
    )
    measurement = torch.rand(1, 8, 8)
    recon = reconstructor.reconstruct_sample(measurement)
    assert recon.shape == measurement.shape
    assert torch.isfinite(recon).all()


def test_padis_dps_alias_maps_to_dps_langevin():
    model = ZeroPatchModel()
    reconstructor = PaDIS(IdentityOp(), model, _sampler_params(model), algorithm="dps")
    assert reconstructor.algorithm == "dps_langevin"


def test_padis_langevin_reconstructor_smoke():
    torch.manual_seed(0)
    model = ZeroPatchModel()
    reconstructor = PaDIS(
        IdentityOp(), model, _sampler_params(model), algorithm="langevin"
    )
    measurement = torch.rand(1, 8, 8)
    recon = reconstructor.reconstruct_sample(measurement)
    assert recon.shape == measurement.shape
    assert torch.isfinite(recon).all()


def test_padis_patch_denoising_zeroes_padding_border():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="dps_langevin")
    x = torch.ones(1, 1, 12, 12)
    layout = reconstructor._patch_layout((8, 8), params, x.device)
    denoised = reconstructor._denoise_patches(x, torch.tensor([0.02]), layout, params)
    pad = params.pad_width
    assert torch.count_nonzero(denoised[:, :, :pad]) == 0
    assert torch.count_nonzero(denoised[:, :, -pad:]) == 0
    assert torch.count_nonzero(denoised[:, :, :, :pad]) == 0
    assert torch.count_nonzero(denoised[:, :, :, -pad:]) == 0


def test_padis_data_gradient_normalization_uses_measurement_operator_norm():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.measurement_scale = 3.0
    params.operator_norm = 5.0
    params.data_consistency_normalization = "operator_norm"
    reconstructor = PaDIS(ScaledIdentityOp(5.0), model, params)

    gradient = torch.full((1, 1, 12, 12), 30.0)
    scaled, normalizer = reconstructor._normalise_data_gradient(gradient, params)

    assert normalizer == 15.0
    assert torch.allclose(scaled, torch.full_like(gradient, 2.0))


def test_padis_data_gradient_normalization_can_be_disabled():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.operator_norm = 5.0
    params.data_consistency_normalization = "none"
    reconstructor = PaDIS(ScaledIdentityOp(5.0), model, params)

    gradient = torch.full((1, 1, 12, 12), 30.0)
    scaled, normalizer = reconstructor._normalise_data_gradient(gradient, params)

    assert normalizer == 1.0
    assert torch.equal(scaled, gradient)
