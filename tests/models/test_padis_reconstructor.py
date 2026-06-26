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
    assert params.sigma_min == 0.002
    assert params.sigma_max == 10.0
    assert params.noise_schedule == "geometric"
    assert params.initial_reconstruction == "noise"
    assert params.clip_initial is False
    assert params.clip_output is False
    assert params.dps_epsilon == 1.0
    assert params.data_consistency_gradient == "paper_squared_residual"
    assert params.adjoint_data_step_schedule == "paper"

    params_8_view = PaDIS.paper_ct_parameters(model, views=8)
    assert params_8_view.sigma_min == 0.003


def test_padis_public_repo_ct_sampling_preset():
    model = ZeroPatchModel()
    params = PaDIS.padis_repo_ct_parameters(model)
    assert params.num_steps == 100
    assert params.inner_steps == 10
    assert params.sigma_min == 0.003
    assert params.sigma_max == 10.0
    assert params.noise_schedule == "edm"
    assert params.initial_reconstruction == "fdk"
    assert params.clip_initial is True
    assert params.clip_output is True
    assert params.dps_epsilon == 0.5
    assert params.data_consistency_gradient == "norm"
    assert params.adjoint_data_step_schedule == "public_repo"


def test_padis_default_sampling_uses_unscaled_data_step_like_original_repo():
    model = ZeroPatchModel()
    params = PaDIS.default_parameters(model)
    assert params.noise_schedule == "edm"
    assert params.data_consistency_normalization == "none"
    assert params.data_consistency_gradient == "norm"


def test_padis_noise_schedule_modes():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="dps_langevin")

    params.num_steps = 4
    params.sigma_max = 8.0
    params.sigma_min = 1.0
    params.noise_schedule = "geometric"
    geometric = reconstructor._noise_schedule(params, torch.device("cpu"))[:-1]
    ratios = geometric[:-1] / geometric[1:]
    assert torch.allclose(ratios, torch.full_like(ratios, ratios[0]))
    assert torch.allclose(geometric[[0, -1]], torch.tensor([8.0, 1.0]))

    params.noise_schedule = "edm"
    params.rho = 1.0
    edm = reconstructor._noise_schedule(params, torch.device("cpu"))[:-1]
    expected = torch.linspace(8.0, 1.0, 4)
    assert torch.allclose(edm, expected)


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


def test_padis_patch_layout_uses_supplied_generator():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.pad_width = 24
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="dps_langevin")

    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)

    layout_a = reconstructor._patch_layout(
        (8, 8), params, torch.device("cpu"), generator_a
    )
    layout_b = reconstructor._patch_layout(
        (8, 8), params, torch.device("cpu"), generator_b
    )

    assert layout_a.indices == layout_b.indices


def test_padis_paper_squared_residual_gradient_matches_formula():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.pad_width = 0
    params.patch_size = 8
    params.zeta = 0.3
    params.data_consistency_gradient = "paper_squared_residual"
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="dps_langevin")

    x = torch.zeros(1, 1, 8, 8, requires_grad=True)
    denoised = x
    measurement = torch.ones(1, 8, 8)
    (
        gradient,
        raw_gradient,
        residual,
        _,
        _,
        step_size,
    ) = reconstructor._dps_data_gradient(
        measurement, x, denoised, params, sigma=torch.tensor(0.02)
    )

    residual_norm = torch.linalg.norm(residual).detach()
    assert torch.allclose(raw_gradient, torch.full_like(x, -2.0))
    assert torch.allclose(gradient, raw_gradient)
    assert step_size == params.zeta / float(residual_norm)


def test_padis_norm_gradient_matches_public_repo_formula():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.pad_width = 0
    params.patch_size = 8
    params.zeta = 0.3
    params.data_consistency_gradient = "norm"
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="dps_langevin")

    x = torch.zeros(1, 1, 8, 8, requires_grad=True)
    measurement = torch.ones(1, 8, 8)
    (
        gradient,
        raw_gradient,
        residual,
        _,
        _,
        step_size,
    ) = reconstructor._dps_data_gradient(
        measurement, x, x, params, sigma=torch.tensor(0.02)
    )

    residual_norm = torch.linalg.norm(residual).detach()
    expected = torch.full_like(x, -1.0 / float(residual_norm))
    assert torch.allclose(raw_gradient, expected)
    assert torch.allclose(gradient, raw_gradient)
    assert step_size == params.zeta


def test_padis_scaled_identity_paper_gradient_matches_closed_form():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.pad_width = 0
    params.patch_size = 8
    params.zeta = 0.3
    params.data_consistency_gradient = "paper_squared_residual"
    reconstructor = PaDIS(
        ScaledIdentityOp(2.0), model, params, algorithm="dps_langevin"
    )

    x = torch.zeros(1, 1, 8, 8, requires_grad=True)
    measurement = torch.ones(1, 8, 8)
    (
        gradient,
        raw_gradient,
        residual,
        _,
        _,
        step_size,
    ) = reconstructor._dps_data_gradient(
        measurement, x, x, params, sigma=torch.tensor(0.02)
    )

    residual_norm = torch.linalg.norm(residual).detach()
    assert torch.allclose(raw_gradient, torch.full_like(x, -4.0))
    assert torch.allclose(gradient, raw_gradient)
    assert step_size == params.zeta / float(residual_norm)


def test_padis_adjoint_correction_matches_scaled_identity_closed_form():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.pad_width = 0
    params.data_consistency_normalization = "none"
    reconstructor = PaDIS(ScaledIdentityOp(3.0), model, params, algorithm="langevin")

    x = torch.zeros(1, 1, 8, 8)
    residual = torch.ones(1, 8, 8)
    (
        updated,
        correction,
        raw_correction,
        normalizer,
        data_scale,
    ) = reconstructor._apply_adjoint_correction(
        x, residual, torch.tensor(0.5), params, torch.tensor(0.02)
    )

    expected_correction = torch.full_like(x, 3.0)
    assert normalizer == 1.0
    assert data_scale == 1.0
    assert torch.allclose(raw_correction, expected_correction)
    assert torch.allclose(correction, expected_correction)
    assert torch.allclose(updated, torch.full_like(x, 1.5))


def test_padis_adjoint_data_step_schedule_matches_paper_and_public_repo():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.zeta = 0.3
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="langevin")
    residual = torch.ones(1, 8, 8)
    sigma = torch.tensor(0.02)
    base_step = params.zeta / torch.linalg.norm(residual)

    params.adjoint_data_step_schedule = "paper"
    paper_step = reconstructor._adjoint_data_step_size(
        residual, sigma, params, public_repo_multiplier=True
    )
    assert torch.allclose(paper_step, base_step)

    params.adjoint_data_step_schedule = "public_repo"
    public_step = reconstructor._adjoint_data_step_size(
        residual, sigma, params, public_repo_multiplier=True
    )
    assert torch.allclose(public_step, base_step * 4.0)


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
