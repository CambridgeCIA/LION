import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from LION.CTtools.ct_geometry import Geometry
from LION.operators import Operator
from LION.utils.parameter import LIONParameter
from scripts.paper_scripts.PaDIS import PaDIS_LIDC_reconstruction as recon_script
from scripts.paper_scripts.PaDIS.PaDIS_LIDC_reconstruction import (
    PnPDenoiser,
    run_reconstruction_variant,
    validate_public_repo_method,
)


class IdentityOp(Operator):
    @property
    def domain_shape(self):
        return (1, 8, 8)

    @property
    def range_shape(self):
        return (1, 8, 8)

    def __call__(self, x, out=None):
        del out
        return x

    def forward(self, x):
        return x

    def adjoint(self, y):
        return y

    def inverse(self, y):
        return y


class TinyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        target = torch.linspace(0.0, 1.0, 64, dtype=torch.float32).reshape(1, 8, 8)
        return None, target


def _args(method: str):
    return SimpleNamespace(
        seed=0,
        method=method,
        algorithm="dps_langevin",
        pnp_checkpoint="unused.pt",
        pnp_noise_level=None,
        pnp_eta=0.0,
        pnp_iterations=1,
        pnp_cg_iterations=5,
        pnp_cg_tolerance=1e-8,
        tv_lambda=0.001,
        tv_iterations=1,
        tv_lipschitz=None,
        tv_non_negativity=False,
        measurement_source="normal",
        noise="none",
        split="test",
        start_index=0,
        max_samples=1,
        prog_bar=False,
        trace_interval=0,
        trace_images=False,
        save_previews=False,
        body_threshold=0.02,
        nonair_threshold=1e-4,
        body_bbox_padding=1,
        data_range=1.0,
        error_vmax=0.1,
        preview_vmax=1.0,
        public_reference_reconstructions=None,
        diagnose_ddnm_pseudoinverse=False,
        experiment="none",
        implementation="paper",
        geometry="lion",
    )


def _params():
    params = LIONParameter()
    params.measurement_scale = 1.0
    params.measurement_offset = 0.0
    params.clip_initial = True
    params.initial_fdk_padded = False
    params.initial_fdk_filter_type = None
    params.initial_fdk_frequency_scaling = 1.0
    params.initial_fdk_batch_size = 1
    params.prior_mode = "patch"
    return params


def _model_params():
    params = LIONParameter()
    params.largest_patch_size = 56
    return params


def _run_driver(tmp_path, method):
    return run_reconstruction_variant(
        args=_args(method),
        dataset=TinyDataset(),
        checkpoint_path=tmp_path / "unused.pt",
        geometry=IdentityOp(),
        reconstruction_geometry=IdentityOp(),
        experiment=None,
        model=None,
        model_params=_model_params(),
        base_params=_params(),
        variant_name="run",
        variant_overrides={},
        output_folder=tmp_path / method,
        device=torch.device("cpu"),
        from_experiment=False,
        experiment_measurement_source="normal",
        reference_reconstructions=None,
    )


def test_baseline_driver_writes_metrics_and_tensors(tmp_path):
    summary = _run_driver(tmp_path, "baseline")

    with open(tmp_path / "baseline" / "metrics.json") as f:
        payload = json.load(f)
    assert len(payload["metrics"]) == 1
    assert payload["method_settings"] == {"baseline": "fdk"}
    assert summary["mean_psnr"] == float("inf")
    assert (tmp_path / "baseline" / "metrics.json").is_file()
    assert (tmp_path / "baseline" / "reconstructions.pt").is_file()


def test_admm_tv_driver_uses_lion_tv_path(monkeypatch, tmp_path):
    calls = []

    def fake_tv_min(sinogram, op, **kwargs):
        calls.append((sinogram.clone(), op, kwargs))
        return sinogram.clone()

    monkeypatch.setattr(recon_script, "tv_min", fake_tv_min)

    summary = _run_driver(tmp_path, "admm_tv")

    assert summary["mean_psnr"] == float("inf")
    with open(tmp_path / "admm_tv" / "metrics.json") as f:
        payload = json.load(f)
    assert payload["method_settings"]["tv_lambda"] == 0.001
    assert payload["method_settings"]["tv_iterations"] == 1
    assert calls
    assert calls[0][0].shape == (1, 1, 8, 8)
    assert calls[0][2]["lam"] == 0.001


def test_pnp_admm_driver_uses_lion_pnp_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        recon_script,
        "load_pnp_denoiser",
        lambda checkpoint, device, noise_level: PnPDenoiser(nn.Identity()),
    )

    summary = _run_driver(tmp_path, "pnp_admm")

    assert summary["mean_psnr"] == float("inf")
    with open(tmp_path / "pnp_admm" / "metrics.json") as f:
        metrics_payload = json.load(f)
    assert metrics_payload["method_settings"]["pnp_checkpoint"] == "unused.pt"
    assert metrics_payload["method_settings"]["pnp_iterations"] == 1
    assert metrics_payload["method_settings"]["pnp_eta"] == 0.0
    payload = torch.load(tmp_path / "pnp_admm" / "reconstructions.pt")
    assert payload["reconstructions"].shape == (1, 1, 8, 8)


def test_driver_can_record_ddnm_pseudoinverse_diagnostics(tmp_path):
    args = _args("baseline")
    args.diagnose_ddnm_pseudoinverse = True
    summary = run_reconstruction_variant(
        args=args,
        dataset=TinyDataset(),
        checkpoint_path=tmp_path / "unused.pt",
        geometry=IdentityOp(),
        reconstruction_geometry=IdentityOp(),
        experiment=None,
        model=None,
        model_params=_model_params(),
        base_params=_params(),
        variant_name="run",
        variant_overrides={},
        output_folder=tmp_path / "ddnm_diagnostic",
        device=torch.device("cpu"),
        from_experiment=False,
        experiment_measurement_source="normal",
        reference_reconstructions=None,
    )

    with open(tmp_path / "ddnm_diagnostic" / "metrics.json") as f:
        payload = json.load(f)
    item = payload["metrics"][0]
    assert item["ddnm_pseudoinverse_diagnostic"]["formula"].startswith("A^dagger y + x")
    assert math.isinf(item["ddnm_perfect_denoiser_corrected_psnr"])
    assert math.isinf(summary["mean_ddnm_perfect_corrected_psnr"])


def test_reconstruction_cli_validation_rejects_public_repo_for_methods_without_public_analogue():
    with pytest.raises(ValueError, match="no runnable public-repo equivalent"):
        validate_public_repo_method("public_repo", "baseline")

    validate_public_repo_method("public_repo", "padis_dps")
    validate_public_repo_method("public_repo", "patch_average")
    validate_public_repo_method("public_repo", "patch_stitch")


def test_public_repo_helper_initialization_is_opt_in_for_helper_methods():
    parser = recon_script.build_arg_parser()

    dps_args = parser.parse_args(
        [
            "--method",
            "padis_dps",
            "--implementation",
            "public_repo",
            "--public-repo-helper-initialization",
        ]
    )
    dps_params = recon_script.build_sampler_params(
        dps_args, model=None, measurement_source="normal"
    )
    assert dps_params.initial_reconstruction == "fdk"
    assert dps_params.clip_initial is True

    pc_args = parser.parse_args(
        [
            "--method",
            "predictor_corrector",
            "--implementation",
            "public_repo",
        ]
    )
    pc_params = recon_script.build_sampler_params(
        pc_args, model=None, measurement_source="normal"
    )
    assert pc_params.initial_reconstruction == "fdk"

    helper_args = parser.parse_args(
        [
            "--method",
            "predictor_corrector",
            "--implementation",
            "public_repo",
            "--public-repo-helper-initialization",
        ]
    )
    helper_params = recon_script.build_sampler_params(
        helper_args, model=None, measurement_source="normal"
    )
    assert helper_params.initial_reconstruction == "noise"
    assert helper_params.noise_initialization == "central_then_pad"
    assert helper_params.clip_initial is False
    assert helper_params.pc_corrector_denoise_sigma == "current"

    langevin_helper_args = parser.parse_args(
        [
            "--method",
            "langevin",
            "--implementation",
            "public_repo",
            "--public-repo-helper-initialization",
        ]
    )
    langevin_helper_params = recon_script.build_sampler_params(
        langevin_helper_args, model=None, measurement_source="normal"
    )
    assert langevin_helper_params.initial_reconstruction == "noise"
    assert langevin_helper_params.noise_initialization == "padded"
    assert langevin_helper_params.clip_initial is False


def test_checkpoint_metadata_fallback_infers_whole_image_preset(tmp_path):
    training = LIONParameter()
    training.paper_preset = "padis-paper-whole-ct-256"
    geometry = Geometry.default_parameters(image_scaling=0.5)
    checkpoint = tmp_path / "whole_image_lidc_256.pt"
    torch.save(
        {"model_state_dict": {}, "training_params": training, "geometry": geometry},
        checkpoint,
    )

    with pytest.warns(UserWarning, match="inferred 'padis-paper-whole-ct-256'"):
        model_params, loaded_geometry = recon_script.load_checkpoint_metadata(
            checkpoint,
            image_scaling=1.0,
            disable_position_channels=False,
        )

    assert model_params.prior_mode == "whole_image"
    assert model_params.largest_patch_size == 256
    assert model_params.pad_width == 0
    assert loaded_geometry.image_scaling == 0.5


def test_checkpoint_metadata_fallback_infers_patch_ablation_without_position(tmp_path):
    training = LIONParameter()
    training.paper_preset = "padis-paper-ct-p96-no-position"
    checkpoint = tmp_path / "padis_lidc_256.pt"
    torch.save({"model_state_dict": {}, "training_params": training}, checkpoint)

    with pytest.warns(UserWarning, match="inferred 'padis-paper-ct-p96-no-position'"):
        model_params, _ = recon_script.load_checkpoint_metadata(
            checkpoint,
            image_scaling=0.5,
            disable_position_channels=False,
        )

    assert model_params.largest_patch_size == 96
    assert model_params.pad_width == 32
    assert model_params.input_position_channels == 0
