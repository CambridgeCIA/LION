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


class ZeroWholeImageModel(ZeroPatchModel):
    def __init__(self):
        super().__init__()
        self.model_parameters.pad_width = 0
        self.model_parameters.largest_patch_size = 8
        self.model_parameters.prior_mode = "whole_image"
        self.model_parameters.input_position_channels = 2


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


def _whole_image_sampler_params(model):
    params = _sampler_params(model)
    params.prior_mode = "whole_image"
    params.patch_size = 8
    params.pad_width = 0
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


def test_whole_image_diffusion_reconstructor_smoke():
    torch.manual_seed(0)
    model = ZeroWholeImageModel()
    reconstructor = PaDIS(
        IdentityOp(),
        model,
        _whole_image_sampler_params(model),
        algorithm="dps_langevin",
    )
    measurement = torch.rand(1, 8, 8)
    recon = reconstructor.reconstruct_sample(measurement)
    assert recon.shape == measurement.shape
    assert torch.isfinite(recon).all()


def test_padis_dps_alias_maps_to_dps_langevin():
    model = ZeroPatchModel()
    reconstructor = PaDIS(IdentityOp(), model, _sampler_params(model), algorithm="dps")
    assert reconstructor.algorithm == "dps_langevin"


def test_padis_paper_ct_sampling_preset():
    model = ZeroPatchModel()
    params = PaDIS.paper_ct_parameters(model)
    assert params.num_steps == 100
    assert params.inner_steps == 10
    assert params.sigma_min == 0.003
    assert params.sigma_max == 10.0
    assert params.initial_reconstruction == "fdk"
    assert params.clip_initial is True


def test_padis_default_sampling_uses_unscaled_data_step_like_original_repo():
    model = ZeroPatchModel()
    params = PaDIS.default_parameters(model)
    assert params.data_consistency_normalization == "none"


def test_padis_data_consistency_scale_schedule():
    model = ZeroPatchModel()
    params = PaDIS.default_parameters(model)
    params.data_consistency_scale = 4.0
    params.data_consistency_scale_schedule = "edm"
    params.data_consistency_scale_power = 1.0
    params.sigma_data = 0.5
    reconstructor = PaDIS(IdentityOp(), model, params)
    high_sigma = reconstructor._scheduled_data_consistency_scale(
        params, torch.tensor(10.0), torch.device("cpu")
    )
    low_sigma = reconstructor._scheduled_data_consistency_scale(
        params, torch.tensor(0.003), torch.device("cpu")
    )
    assert high_sigma < low_sigma
    assert low_sigma <= params.data_consistency_scale


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


def test_padis_predictor_corrector_reconstructor_smoke():
    torch.manual_seed(0)
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.num_steps = 2
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="pc")
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
    scaled, normalizer, scale = reconstructor._normalise_data_gradient(gradient, params)

    assert normalizer == 15.0
    assert scale == 1.0
    assert torch.allclose(scaled, torch.full_like(gradient, 2.0))


def test_padis_data_gradient_normalization_can_be_disabled():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.operator_norm = 5.0
    params.data_consistency_normalization = "none"
    reconstructor = PaDIS(ScaledIdentityOp(5.0), model, params)

    gradient = torch.full((1, 1, 12, 12), 30.0)
    scaled, normalizer, scale = reconstructor._normalise_data_gradient(gradient, params)

    assert normalizer == 1.0
    assert scale == 1.0
    assert torch.equal(scaled, gradient)
