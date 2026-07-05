from types import SimpleNamespace
import importlib

import torch
from torch import nn

from LION.operators import Operator
from LION.CTtools.ct_geometry import Geometry
from LION.models.CNNs.drunet import DRUNet
from LION.reconstructors import PaDIS
from LION.utils.parameter import LIONParameter
from scripts.paper_scripts.PaDIS.PaDIS_LIDC_reconstruction import (
    PnPDenoiser,
    fdk_baseline,
    load_pnp_denoiser,
    pnp_reconstruction,
)


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


class GradModePatchModel(ZeroPatchModel):
    def __init__(self):
        super().__init__()
        self.grad_enabled_calls = []

    def forward(self, x, time_cond):
        self.grad_enabled_calls.append(torch.is_grad_enabled())
        return super().forward(x, time_cond)


class ZeroWholeImageModel(ZeroPatchModel):
    def __init__(self):
        super().__init__()
        self.model_parameters.pad_width = 0
        self.model_parameters.largest_patch_size = 8
        self.model_parameters.prior_mode = "whole_image"
        self.model_parameters.input_position_channels = 2


class AffineDenoiserModel(nn.Module):
    def __init__(self, *, use_noise_level=False):
        super().__init__()
        self.model_parameters = LIONParameter()
        self.model_parameters.use_noise_level = use_noise_level
        self.last_noise_level = None

    def normalise(self, x):
        return 2.0 * x

    def unnormalise(self, x, target=None):
        del target
        return 0.5 * x

    def forward(self, x, noise_level=None):
        self.last_noise_level = noise_level
        if noise_level is None:
            return x + 2.0
        return x + float(noise_level)


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


def _prior_free_params():
    params = LIONParameter()
    params.measurement_scale = 2.0
    params.measurement_offset = 0.1
    params.clip_initial = True
    params.initial_fdk_padded = True
    params.initial_fdk_filter_type = None
    params.initial_fdk_frequency_scaling = 1.0
    params.initial_fdk_batch_size = 1
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


def test_padis_dps_disable_data_consistency_denoises_without_grad():
    torch.manual_seed(0)
    model = GradModePatchModel()
    params = _sampler_params(model)
    params.disable_data_consistency = True
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="dps_langevin")
    measurement = torch.rand(1, 8, 8)

    recon = reconstructor.reconstruct_sample(measurement)

    assert recon.shape == measurement.shape
    assert model.grad_enabled_calls
    assert not any(model.grad_enabled_calls)


def test_fdk_baseline_uses_inverse_and_measurement_domain_conversion():
    model = ZeroPatchModel()
    params = _prior_free_params()
    reconstructor = PaDIS(IdentityOp(), model, _sampler_params(model))
    measurement = torch.full((1, 8, 8), 0.5)

    recon = fdk_baseline(measurement, reconstructor, params)

    assert torch.allclose(recon, torch.full_like(recon, 0.2))


def test_pnp_denoiser_wraps_normalisation_and_optional_noise_level():
    image = torch.ones(1, 8, 8)

    denoiser = PnPDenoiser(AffineDenoiserModel(), noise_level=None)
    denoised = denoiser(image)
    assert denoised.shape == image.shape
    assert torch.allclose(denoised, torch.full_like(image, 2.0))

    noise_model = AffineDenoiserModel(use_noise_level=True)
    noise_denoiser = PnPDenoiser(noise_model, noise_level=0.25)
    denoised_with_noise = noise_denoiser(image)
    assert noise_model.last_noise_level == 0.25
    assert torch.allclose(denoised_with_noise, torch.full_like(image, 1.125))


def test_load_pnp_denoiser_accepts_string_path(tmp_path):
    params = DRUNet.default_parameters()
    params.int_channels = 8
    params.n_blocks = 1
    model = DRUNet(params)
    checkpoint = tmp_path / "pnp_lidc_drunet.pt"
    model.save(
        checkpoint,
        training=LIONParameter(),
        dataset=LIONParameter(),
        geometry=Geometry.default_parameters(image_scaling=0.5),
    )

    denoiser = load_pnp_denoiser(str(checkpoint), torch.device("cpu"), noise_level=None)
    with torch.no_grad():
        output = denoiser(torch.zeros(1, 8, 8))

    assert output.shape == (1, 8, 8)
    assert torch.isfinite(output).all()


def test_pnp_reconstruction_runs_with_lion_operator_without_ct_backend():
    params = _prior_free_params()
    params.measurement_scale = 1.0
    params.measurement_offset = 0.0
    denoiser = PnPDenoiser(nn.Identity())
    args = SimpleNamespace(
        pnp_eta=0.0,
        pnp_iterations=1,
        pnp_cg_iterations=5,
        pnp_cg_tolerance=1e-8,
        pnp_clip=True,
        prog_bar=False,
    )
    measurement = torch.full((1, 8, 8), 0.25)

    recon = pnp_reconstruction(measurement, IdentityOp(), denoiser, params, args)

    assert recon.shape == measurement.shape
    assert torch.allclose(recon, measurement)


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
    assert params.data_consistency_scale == 0.0405
    assert params.adjoint_data_consistency_scale == 0.1022
    assert params.pc_corrector_denoise_sigma == "current"
    assert params.pc_reuse_predictor_layout is True


def test_padis_default_sampling_uses_unscaled_data_step_like_original_repo():
    model = ZeroPatchModel()
    params = PaDIS.default_parameters(model)
    assert params.noise_schedule == "edm"
    assert params.data_consistency_normalization == "none"
    assert params.data_consistency_gradient == "norm"


def test_padis_lion_physics_sampling_uses_lipschitz_scaled_least_squares():
    model = ZeroPatchModel()
    params = PaDIS.lion_physics_ct_parameters(model, views=20)
    assert params.noise_schedule == "geometric"
    assert params.sigma_min == 0.002
    assert params.sigma_max == 10.0
    assert params.initial_reconstruction == "fdk"
    assert params.initial_fdk_frequency_scaling == 0.2
    assert params.zeta == 3.0
    assert params.data_consistency_gradient == "least_squares"
    assert params.data_consistency_normalization == "operator_lipschitz"
    assert params.data_consistency_scale == 1.0
    assert params.adjoint_data_consistency_scale is None
    assert params.adjoint_data_step_schedule == "paper"
    assert params.pc_snr == 0.04


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


def test_padis_adjoint_data_consistency_scale_can_differ_from_dps_scale():
    model = ZeroPatchModel()
    params = PaDIS.default_parameters(model)
    params.data_consistency_scale = 0.04
    params.adjoint_data_consistency_scale = 0.1
    reconstructor = PaDIS(IdentityOp(), model, params)

    dps_scale = reconstructor._scheduled_data_consistency_scale(
        params, torch.tensor(0.02), torch.device("cpu")
    )
    adjoint_scale = reconstructor._scheduled_adjoint_data_consistency_scale(
        params, torch.tensor(0.02), torch.device("cpu")
    )

    assert dps_scale == 0.04
    assert adjoint_scale == 0.1


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


def test_padis_ddnm_pseudoinverse_is_not_noise_initial_state():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.initial_reconstruction = "noise"
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="langevin")
    measurement = torch.full((1, 8, 8), 0.25)

    initial = reconstructor._initial_reconstruction(measurement, params)
    pseudoinverse = reconstructor._pseudoinverse_reconstruction(measurement, params)

    assert torch.count_nonzero(initial) == 0
    assert torch.allclose(pseudoinverse, measurement)


def test_padis_ddnm_pseudoinverse_clip_can_be_overridden_per_term():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.ddnm_pseudoinverse_clip = True
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="langevin")
    measurement = torch.tensor([[[-0.5, 0.25], [1.25, 2.0]]])

    clipped = reconstructor._pseudoinverse_reconstruction(measurement, params)
    unclipped = reconstructor._pseudoinverse_reconstruction(
        measurement,
        params,
        clip=False,
    )

    assert torch.allclose(clipped, measurement.clamp(0.0, 1.0))
    assert torch.allclose(unclipped, measurement)


def test_padis_ddnm_corrected_clip_clamps_score_target(monkeypatch):
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.pad_width = 0
    params.patch_size = 8
    params.sigma_min = 1.0
    params.sigma_max = 1.0
    params.langevin_ddnm = True
    params.disable_langevin_noise = True
    params.clip_output = False
    params.ddnm_corrected_clip = True
    params.trace_interval = 1
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="langevin")

    def denoise_prior(x, sigma, params, image_shape, generator=None):
        del sigma, params, image_shape, generator
        return torch.full_like(x, 2.0)

    def pseudoinverse(measurement, params, *, clip=None):
        del params, clip
        if torch.allclose(measurement, torch.full_like(measurement, 0.25)):
            return torch.full_like(measurement, 0.25)
        return torch.full_like(measurement, 3.0)

    monkeypatch.setattr(reconstructor, "_denoise_prior", denoise_prior)
    monkeypatch.setattr(reconstructor, "_pseudoinverse_reconstruction", pseudoinverse)
    measurement = torch.full((1, 8, 8), 0.25)

    reconstructor.reconstruct_sample(measurement)

    assert reconstructor.last_trace
    assert reconstructor.last_trace[0]["projected_min"] == 0.0
    assert reconstructor.last_trace[0]["projected_max"] == 0.0


def test_padis_langevin_honours_stop_after_outer_steps(monkeypatch):
    torch.manual_seed(0)
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.num_steps = 4
    params.inner_steps = 1
    params.stop_after_outer_steps = 1
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="langevin")
    original_denoise = reconstructor._denoise_prior
    denoise_calls = 0

    def counting_denoise(*args, **kwargs):
        nonlocal denoise_calls
        denoise_calls += 1
        return original_denoise(*args, **kwargs)

    monkeypatch.setattr(reconstructor, "_denoise_prior", counting_denoise)
    measurement = torch.rand(1, 8, 8)

    recon = reconstructor.reconstruct_sample(measurement)

    assert recon.shape == measurement.shape
    assert denoise_calls == 1


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


def test_padis_predictor_corrector_uses_paper_linear_step_size_by_default():
    noise = torch.tensor([3.0, 4.0])
    score = torch.tensor([6.0, 8.0])

    step_size = PaDIS._pc_corrector_step_size(noise, score, snr=0.2)

    assert torch.allclose(step_size, torch.tensor(0.2))


def test_padis_predictor_corrector_keeps_score_sde_squared_step_size_option():
    noise = torch.tensor([3.0, 4.0])
    score = torch.tensor([6.0, 8.0])

    step_size = PaDIS._pc_corrector_step_size(
        noise, score, snr=0.2, rule="score_sde_squared"
    )

    assert torch.allclose(step_size, torch.tensor(0.02))


def test_padis_predictor_corrector_denoises_corrector_at_next_sigma(monkeypatch):
    torch.manual_seed(0)
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.num_steps = 2
    params.sigma_max = 0.08
    params.sigma_min = 0.02
    params.noise_schedule = "geometric"
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="pc")
    original_denoise = reconstructor._denoise_prior
    sigmas = []

    def recording_denoise(x, sigma, params, image_shape, generator=None):
        sigmas.append(float(sigma.item()))
        return original_denoise(x, sigma, params, image_shape, generator)

    monkeypatch.setattr(reconstructor, "_denoise_prior", recording_denoise)
    measurement = torch.rand(1, 8, 8)

    reconstructor.reconstruct_sample(measurement)

    assert torch.allclose(torch.tensor(sigmas), torch.tensor([0.08, 0.02]))


def test_padis_predictor_corrector_public_compat_denoises_corrector_at_current_sigma(
    monkeypatch,
):
    torch.manual_seed(0)
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.num_steps = 2
    params.sigma_max = 0.08
    params.sigma_min = 0.02
    params.noise_schedule = "geometric"
    params.pc_corrector_denoise_sigma = "current"
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="pc")
    original_denoise = reconstructor._denoise_prior
    sigmas = []

    def recording_denoise(x, sigma, params, image_shape, generator=None):
        sigmas.append(float(sigma.item()))
        return original_denoise(x, sigma, params, image_shape, generator)

    monkeypatch.setattr(reconstructor, "_denoise_prior", recording_denoise)
    measurement = torch.rand(1, 8, 8)

    reconstructor.reconstruct_sample(measurement)

    assert torch.allclose(torch.tensor(sigmas), torch.tensor([0.08, 0.08]))


def test_padis_predictor_corrector_can_reuse_predictor_patch_layout(monkeypatch):
    torch.manual_seed(0)
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.num_steps = 2
    params.sigma_max = 0.08
    params.sigma_min = 0.02
    params.noise_schedule = "geometric"
    params.pc_reuse_predictor_layout = True
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="pc")
    original_patch_layout = reconstructor._patch_layout
    layouts = []

    def recording_patch_layout(*args, **kwargs):
        layout = original_patch_layout(*args, **kwargs)
        layouts.append(layout)
        return layout

    monkeypatch.setattr(reconstructor, "_patch_layout", recording_patch_layout)
    measurement = torch.rand(1, 8, 8)

    reconstructor.reconstruct_sample(measurement)

    assert len(layouts) == 1


def test_padis_predictor_corrector_honours_stop_after_outer_steps(monkeypatch):
    torch.manual_seed(0)
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.num_steps = 4
    params.stop_after_outer_steps = 1
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="pc")
    original_denoise = reconstructor._denoise_prior
    denoise_calls = 0

    def counting_denoise(*args, **kwargs):
        nonlocal denoise_calls
        denoise_calls += 1
        return original_denoise(*args, **kwargs)

    monkeypatch.setattr(reconstructor, "_denoise_prior", counting_denoise)
    measurement = torch.rand(1, 8, 8)

    recon = reconstructor.reconstruct_sample(measurement)

    assert recon.shape == measurement.shape
    assert denoise_calls == 2


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


def test_fixed_overlap_patch_layout_uses_paper_overlap():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.patch_size = 4
    params.pad_width = 2
    params.patch_overlap = 1
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="dps_langevin")

    layout = reconstructor._fixed_overlap_patch_layout((8, 8), params)

    assert layout.indices == [
        (2, 6, 2, 6),
        (2, 6, 5, 9),
        (2, 6, 8, 12),
        (5, 9, 2, 6),
        (5, 9, 5, 9),
        (5, 9, 8, 12),
        (8, 12, 2, 6),
        (8, 12, 5, 9),
        (8, 12, 8, 12),
    ]


def test_fixed_overlap_patch_layout_covers_full_central_image():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.patch_size = 4
    params.pad_width = 2
    params.patch_overlap = 1
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="dps_langevin")

    layout = reconstructor._fixed_overlap_patch_layout((8, 8), params)
    coverage = torch.zeros(1, 1, 12, 12)
    for top, bottom, left, right in layout.indices:
        coverage[:, :, top:bottom, left:right] += 1

    pad = params.pad_width
    central = coverage[:, :, pad : pad + 8, pad : pad + 8]
    assert torch.all(central > 0)
    assert torch.count_nonzero(coverage[:, :, :pad]) == 0
    assert torch.count_nonzero(coverage[:, :, :, :pad]) == 0


def test_fixed_overlap_public_helper_layouts_match_public_repo_starts():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.patch_size = 56
    params.pad_width = 24
    params.patch_overlap = 8
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="dps_langevin")

    params.fixed_overlap_layout = "public_overlap"
    overlap_layout = reconstructor._fixed_overlap_patch_layout((256, 256), params)
    overlap_starts = sorted({top for top, _, _, _ in overlap_layout.indices})
    assert overlap_starts == [24, 72, 120, 168, 216, 248]

    params.fixed_overlap_layout = "public_tile"
    tile_layout = reconstructor._fixed_overlap_patch_layout((256, 256), params)
    tile_starts = sorted({top for top, _, _, _ in tile_layout.indices})
    assert tile_starts == [4, 52, 100, 148, 196, 244]


def test_fixed_overlap_patch_average_and_stitch_smoke():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.patch_size = 4
    params.pad_width = 2
    params.patch_overlap = 1
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="dps_langevin")
    x = torch.ones(1, 1, 12, 12)
    sigma = torch.tensor([0.02])
    layout = reconstructor._fixed_overlap_patch_layout((8, 8), params)

    average = reconstructor._denoise_fixed_overlap_patches(
        x, sigma, layout, params, assembly="fixed_average"
    )
    stitch = reconstructor._denoise_fixed_overlap_patches(
        x, sigma, layout, params, assembly="fixed_stitch"
    )

    assert average.shape == x.shape
    assert stitch.shape == x.shape
    assert torch.isfinite(average).all()
    assert torch.isfinite(stitch).all()
    pad = params.pad_width
    assert torch.count_nonzero(average[:, :, :pad]) == 0
    assert torch.count_nonzero(stitch[:, :, :pad]) == 0


def test_fixed_overlap_patch_average_and_stitch_overlap_semantics(monkeypatch):
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.patch_size = 4
    params.pad_width = 2
    params.patch_overlap = 1
    params.patch_batch_size = None
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="dps_langevin")
    x = torch.zeros(1, 1, 12, 12)
    sigma = torch.tensor([0.02])
    layout = reconstructor._fixed_overlap_patch_layout((8, 8), params)

    def indexed_denoise(image_batch, position_batch, sigma, params, **kwargs):
        del position_batch, sigma, params, kwargs
        values = torch.arange(
            1,
            image_batch.shape[0] + 1,
            device=image_batch.device,
            dtype=image_batch.dtype,
        ).reshape(-1, 1, 1, 1)
        return values.expand_as(image_batch)

    monkeypatch.setattr(reconstructor, "_edm_denoise_batch", indexed_denoise)

    average = reconstructor._denoise_fixed_overlap_patches(
        x, sigma, layout, params, assembly="fixed_average"
    )
    stitch = reconstructor._denoise_fixed_overlap_patches(
        x, sigma, layout, params, assembly="fixed_stitch"
    )

    assert average[0, 0, 2, 2] == 1
    assert stitch[0, 0, 2, 2] == 1
    assert average[0, 0, 5, 5] == 3
    assert stitch[0, 0, 5, 5] == 5
    assert torch.count_nonzero(average[:, :, : params.pad_width]) == 0
    assert torch.count_nonzero(stitch[:, :, : params.pad_width]) == 0


def test_fixed_overlap_patch_denoising_respects_patch_batch_size(monkeypatch):
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.patch_size = 4
    params.pad_width = 2
    params.patch_overlap = 1
    params.patch_batch_size = 2
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="dps_langevin")
    x = torch.zeros(1, 1, 12, 12)
    sigma = torch.tensor([0.02])
    layout = reconstructor._fixed_overlap_patch_layout((8, 8), params)
    chunk_sizes = []

    def chunked_denoise(image_batch, position_batch, sigma, params, **kwargs):
        del position_batch, sigma, params, kwargs
        chunk_sizes.append(image_batch.shape[0])
        return torch.ones_like(image_batch)

    monkeypatch.setattr(reconstructor, "_edm_denoise_batch", chunked_denoise)

    output = reconstructor._denoise_fixed_overlap_patches(
        x, sigma, layout, params, assembly="fixed_average"
    )

    assert output.shape == x.shape
    assert chunk_sizes
    assert max(chunk_sizes) <= params.patch_batch_size
    assert sum(chunk_sizes) == len(layout.indices)


def test_regular_patch_denoising_respects_patch_batch_size(monkeypatch):
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.patch_size = 4
    params.pad_width = 2
    params.patch_batch_size = 2
    params.patch_checkpoint_denoiser = True
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="dps_langevin")
    x = torch.zeros(1, 1, 12, 12)
    sigma = torch.tensor([0.02])
    generator = torch.Generator().manual_seed(1)
    layout = reconstructor._patch_layout((8, 8), params, torch.device("cpu"), generator)
    chunk_sizes = []
    checkpoint_flags = []

    def chunked_denoise(image_batch, position_batch, sigma, params, **kwargs):
        del position_batch, sigma, params
        chunk_sizes.append(image_batch.shape[0])
        checkpoint_flags.append(kwargs.get("use_checkpoint"))
        return torch.ones_like(image_batch)

    monkeypatch.setattr(reconstructor, "_edm_denoise_batch", chunked_denoise)

    output = reconstructor._denoise_patches(x, sigma, layout, params)

    assert output.shape == x.shape
    assert chunk_sizes
    assert max(chunk_sizes) <= params.patch_batch_size
    assert sum(chunk_sizes) == len(layout.indices)
    assert all(checkpoint_flags)


def test_regular_patch_denoising_honours_legacy_checkpoint_alias(monkeypatch):
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.patch_size = 4
    params.pad_width = 2
    params.patch_batch_size = 2
    params.fixed_overlap_checkpoint_denoiser = True
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="dps_langevin")
    x = torch.zeros(1, 1, 12, 12)
    sigma = torch.tensor([0.02])
    generator = torch.Generator().manual_seed(1)
    layout = reconstructor._patch_layout((8, 8), params, torch.device("cpu"), generator)
    checkpoint_flags = []

    def chunked_denoise(image_batch, position_batch, sigma, params, **kwargs):
        del position_batch, sigma, params
        checkpoint_flags.append(kwargs.get("use_checkpoint"))
        return torch.ones_like(image_batch)

    monkeypatch.setattr(reconstructor, "_edm_denoise_batch", chunked_denoise)

    output = reconstructor._denoise_patches(x, sigma, layout, params)

    assert output.shape == x.shape
    assert checkpoint_flags
    assert all(checkpoint_flags)


def test_fixed_overlap_patch_denoising_can_checkpoint_batches(monkeypatch):
    padis_module = importlib.import_module("LION.reconstructors.PaDIS")
    checkpoint_calls = []

    def fake_checkpoint(function, *args, use_reentrant=None):
        checkpoint_calls.append({"num_args": len(args), "use_reentrant": use_reentrant})
        return function(*args)

    monkeypatch.setattr(padis_module, "activation_checkpoint", fake_checkpoint)

    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.patch_size = 4
    params.pad_width = 2
    params.patch_overlap = 1
    params.patch_batch_size = 2
    params.fixed_overlap_checkpoint_denoiser = True
    reconstructor = PaDIS(IdentityOp(), model, params, algorithm="dps_langevin")
    x = torch.ones(1, 1, 12, 12, requires_grad=True)
    sigma = torch.tensor([0.02])
    layout = reconstructor._fixed_overlap_patch_layout((8, 8), params)

    output = reconstructor._denoise_fixed_overlap_patches(
        x, sigma, layout, params, assembly="fixed_average"
    )

    assert output.shape == x.shape
    assert checkpoint_calls
    assert all(call["use_reentrant"] is False for call in checkpoint_calls)


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


def test_padis_least_squares_gradient_uses_lipschitz_step_form():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.pad_width = 0
    params.patch_size = 8
    params.zeta = 0.3
    params.data_consistency_gradient = "least_squares"
    params.data_consistency_normalization = "operator_lipschitz"
    params.operator_norm = 2.0
    reconstructor = PaDIS(
        ScaledIdentityOp(2.0), model, params, algorithm="dps_langevin"
    )

    x = torch.zeros(1, 1, 8, 8, requires_grad=True)
    measurement = torch.ones(1, 8, 8)
    (
        gradient,
        raw_gradient,
        _residual,
        normalizer,
        data_scale,
        step_size,
    ) = reconstructor._dps_data_gradient(
        measurement, x, x, params, sigma=torch.tensor(0.02)
    )

    assert torch.allclose(raw_gradient, torch.full_like(x, -2.0))
    assert normalizer == 4.0
    assert data_scale == 1.0
    assert torch.allclose(gradient, torch.full_like(x, -0.5))
    assert step_size == params.zeta


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
    params.adjoint_data_consistency_scale = 0.25
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

    expected_raw_correction = torch.full_like(x, 3.0)
    expected_correction = torch.full_like(x, 0.75)
    assert normalizer == 1.0
    assert data_scale == 0.25
    assert torch.allclose(raw_correction, expected_raw_correction)
    assert torch.allclose(correction, expected_correction)
    assert torch.allclose(updated, torch.full_like(x, 0.375))


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


def test_padis_least_squares_adjoint_step_uses_lipschitz_form():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.pad_width = 0
    params.zeta = 3.0
    params.data_consistency_gradient = "least_squares"
    params.data_consistency_normalization = "operator_lipschitz"
    params.operator_norm = 2.0
    reconstructor = PaDIS(ScaledIdentityOp(2.0), model, params, algorithm="langevin")

    x = torch.zeros(1, 1, 8, 8)
    residual = torch.ones(1, 8, 8)
    sigma = torch.tensor(0.02)
    step = reconstructor._adjoint_data_step_size(
        residual, sigma, params, public_repo_multiplier=True
    )
    (
        updated,
        correction,
        raw_correction,
        normalizer,
        data_scale,
    ) = reconstructor._apply_adjoint_correction(x, residual, step, params, sigma)

    assert torch.allclose(step, torch.tensor(3.0))
    assert normalizer == 4.0
    assert data_scale == 1.0
    assert torch.allclose(raw_correction, torch.full_like(x, 2.0))
    assert torch.allclose(correction, torch.full_like(x, 0.5))
    assert torch.allclose(updated, torch.full_like(x, 1.5))


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


def test_padis_data_gradient_normalization_uses_measurement_lipschitz_constant():
    model = ZeroPatchModel()
    params = _sampler_params(model)
    params.measurement_scale = 3.0
    params.measurement_offset = 7.0
    params.operator_norm = 5.0
    params.data_consistency_normalization = "operator_lipschitz"
    reconstructor = PaDIS(ScaledIdentityOp(5.0), model, params)

    gradient = torch.full((1, 1, 12, 12), 450.0)
    scaled, normalizer, scale = reconstructor._normalise_data_gradient(gradient, params)

    assert normalizer == 225.0
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
