import json

from scripts.paper_scripts.PaDIS.PaDIS_verify_reconstruction_matrix import (
    build_arg_parser,
    check_records,
    find_records,
    parse_method_thresholds,
)


def _write_metrics(
    root,
    *,
    method="baseline",
    algorithm="dps_langevin",
    prior_mode="patch",
    experiment="ct_20",
    implementation="paper",
    geometry="lion",
    checkpoint="/tmp/checkpoint.pt",
    psnr=32.0,
    ssim=0.82,
    mae=0.015,
    sampler=None,
    method_settings=None,
):
    folder = root / method / experiment / implementation / geometry
    folder.mkdir(parents=True)
    path = folder / "metrics.json"
    payload = {
        "checkpoint": checkpoint,
        "method": method,
        "algorithm": algorithm,
        "prior_mode": prior_mode,
        "experiment": experiment,
        "implementation": implementation,
        "geometry_tag": geometry,
        "method_settings": method_settings or {},
        "sampler": sampler
        or {
            "num_steps": 100,
            "inner_steps": 10,
            "sigma_min": 0.002,
            "sigma_max": 10.0,
            "noise_schedule": "geometric",
            "initial_reconstruction": "noise",
            "data_consistency_gradient": "paper_squared_residual",
            "adjoint_data_step_schedule": "paper",
            "prior_mode": prior_mode,
        },
        "metrics": [
            {
                "mse": 0.001,
                "psnr": psnr,
                "ssim": ssim,
                "mae": mae,
                "fdk_psnr": 28.0,
                "recon_relative_sinogram_residual": 0.1,
            }
        ],
    }
    path.write_text(json.dumps(payload))
    return path


def _args(tmp_path, *extra):
    parser = build_arg_parser()
    return parser.parse_args(["--root", str(tmp_path), *extra])


def test_parse_method_thresholds_requires_method_value_form():
    assert parse_method_thresholds(["baseline=28", "padis_dps=31.5"]) == {
        "baseline": 28.0,
        "padis_dps": 31.5,
    }


def test_verifier_finds_records_and_accepts_required_method_and_experiment(tmp_path):
    _write_metrics(tmp_path, method="baseline", experiment="ct_20")
    _write_metrics(tmp_path, method="admm_tv", experiment="ct_20")
    _write_metrics(tmp_path, method="padis_dps", experiment="ct_8")
    args = _args(
        tmp_path,
        "--methods",
        "baseline,admm_tv",
        "--experiments",
        "ct_20",
        "--require-methods",
        "baseline,admm_tv",
        "--require-experiments",
        "ct_20",
        "--min-mean-psnr",
        "30",
        "--min-method-mean-ssim",
        "baseline=0.8",
    )

    records = find_records(args)
    failures = check_records(args, records)

    assert [record["summary"]["method"] for record in records] == [
        "admm_tv",
        "baseline",
    ]
    assert failures == []


def test_verifier_reports_missing_required_method(tmp_path):
    _write_metrics(tmp_path, method="baseline", experiment="ct_20")
    args = _args(tmp_path, "--require-methods", "baseline,padis_dps")

    failures = check_records(args, find_records(args))

    assert "Missing required method: padis_dps" in failures


def test_verifier_reports_method_specific_threshold_failure(tmp_path):
    _write_metrics(tmp_path, method="baseline", experiment="ct_20", psnr=25.0)
    args = _args(tmp_path, "--min-method-mean-psnr", "baseline=30")

    failures = check_records(args, find_records(args))

    assert any(
        "baseline ct_20 paper lion: mean_psnr=25.0 < 30.0" in failure
        for failure in failures
    )


def test_verifier_method_fdk_gate_checks_only_selected_methods(tmp_path):
    _write_metrics(tmp_path, method="baseline", experiment="ct_20", psnr=28.0)
    _write_metrics(tmp_path, method="padis_dps", experiment="ct_20", psnr=27.0)
    args = _args(
        tmp_path,
        "--require-method-mean-better-than-fdk",
        "padis_dps",
    )

    failures = check_records(args, find_records(args))

    assert len(failures) == 1
    assert (
        "padis_dps ct_20 paper lion: mean_psnr=27.0 <= mean_fdk_psnr=28.0"
        in failures[0]
    )
    assert "baseline" not in failures[0]


def test_verifier_method_each_sample_fdk_gate_uses_min_margin(tmp_path):
    _write_metrics(tmp_path, method="padis_dps", experiment="ct_20", psnr=27.0)
    args = _args(
        tmp_path,
        "--require-method-each-better-than-fdk",
        "padis_dps",
    )

    failures = check_records(args, find_records(args))

    assert any(
        "at least one sample did not beat FDK" in failure for failure in failures
    )


def test_verifier_global_fdk_gate_applies_to_baseline_too(tmp_path):
    _write_metrics(tmp_path, method="baseline", experiment="ct_20", psnr=28.0)
    args = _args(tmp_path, "--require-mean-better-than-fdk")

    failures = check_records(args, find_records(args))

    assert any("baseline ct_20 paper lion" in failure for failure in failures)


def test_verifier_reports_expected_record_count_mismatch(tmp_path):
    _write_metrics(tmp_path, method="baseline", experiment="ct_20")
    args = _args(tmp_path, "--expected-records", "2")

    failures = check_records(args, find_records(args))

    assert "Expected 2 matching metrics.json files, found 1." in failures


def test_verifier_reports_expected_sample_count_mismatch(tmp_path):
    _write_metrics(tmp_path, method="baseline", experiment="ct_20")
    args = _args(tmp_path, "--expected-samples", "2")

    failures = check_records(args, find_records(args))

    assert any("expected 2 samples, found 1." in failure for failure in failures)


def test_verifier_reports_expected_job_manifest_mismatch(tmp_path):
    _write_metrics(tmp_path, method="baseline", experiment="ct_20")
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "method": "baseline",
                    "algorithm": "dps_langevin",
                    "prior_mode": "patch",
                    "experiment": "ct_20",
                    "implementation": "paper",
                    "geometry": "lion",
                    "checkpoint": "/tmp/checkpoint.pt",
                },
                {
                    "method": "admm_tv",
                    "algorithm": "dps_langevin",
                    "prior_mode": "patch",
                    "experiment": "ct_20",
                    "implementation": "paper",
                    "geometry": "lion",
                    "checkpoint": "/tmp/checkpoint.pt",
                },
            ]
        )
    )
    args = _args(tmp_path, "--expected-jobs-json", str(manifest))

    failures = check_records(args, find_records(args))

    assert any(
        "Missing expected reconstruction record" in failure for failure in failures
    )
    assert any("method=admm_tv" in failure for failure in failures)


def test_verifier_manifest_checkpoint_identity_resolves_symlinks(tmp_path):
    real_checkpoint = tmp_path / "real_checkpoint.pt"
    real_checkpoint.write_bytes(b"checkpoint")
    linked_checkpoint = tmp_path / "linked_checkpoint.pt"
    linked_checkpoint.symlink_to(real_checkpoint)
    _write_metrics(
        tmp_path,
        method="baseline",
        experiment="ct_20",
        checkpoint=str(real_checkpoint),
    )
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "method": "baseline",
                    "algorithm": "dps_langevin",
                    "prior_mode": "patch",
                    "experiment": "ct_20",
                    "implementation": "paper",
                    "geometry": "lion",
                    "checkpoint": str(linked_checkpoint),
                },
            ]
        )
    )
    args = _args(tmp_path, "--expected-jobs-json", str(manifest))

    failures = check_records(args, find_records(args))

    assert failures == []


def test_verifier_reports_unexpected_job_not_in_manifest(tmp_path):
    _write_metrics(tmp_path, method="baseline", experiment="ct_20")
    manifest = tmp_path / "jobs.json"
    manifest.write_text(json.dumps([]))
    args = _args(tmp_path, "--expected-jobs-json", str(manifest))

    failures = check_records(args, find_records(args))

    assert any("Unexpected reconstruction record" in failure for failure in failures)
    assert any("method=baseline" in failure for failure in failures)


def test_verifier_manifest_identity_includes_algorithm(tmp_path):
    _write_metrics(tmp_path, method="langevin", algorithm="dps_langevin")
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "method": "langevin",
                    "algorithm": "langevin",
                    "prior_mode": "patch",
                    "experiment": "ct_20",
                    "implementation": "paper",
                    "geometry": "lion",
                    "checkpoint": "/tmp/checkpoint.pt",
                }
            ]
        )
    )
    args = _args(tmp_path, "--expected-jobs-json", str(manifest))

    failures = check_records(args, find_records(args))

    assert any(
        "Missing expected reconstruction record" in failure for failure in failures
    )
    assert any("algorithm=langevin" in failure for failure in failures)
    assert any("Unexpected reconstruction record" in failure for failure in failures)
    assert any("algorithm=dps_langevin" in failure for failure in failures)


def test_verifier_manifest_identity_includes_prior_mode(tmp_path):
    _write_metrics(tmp_path, method="whole_image_diffusion", prior_mode="patch")
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "method": "whole_image_diffusion",
                    "algorithm": "dps_langevin",
                    "prior_mode": "whole_image",
                    "experiment": "ct_20",
                    "implementation": "paper",
                    "geometry": "lion",
                    "checkpoint": "/tmp/checkpoint.pt",
                }
            ]
        )
    )
    args = _args(tmp_path, "--expected-jobs-json", str(manifest))

    failures = check_records(args, find_records(args))

    assert any(
        "Missing expected reconstruction record" in failure for failure in failures
    )
    assert any("prior_mode=whole_image" in failure for failure in failures)
    assert any("Unexpected reconstruction record" in failure for failure in failures)
    assert any("prior_mode=patch" in failure for failure in failures)


def test_verifier_checks_expected_sampler_settings_from_manifest(tmp_path):
    _write_metrics(
        tmp_path,
        method="padis_dps",
        sampler={
            "num_steps": 100,
            "inner_steps": 10,
            "sigma_min": 0.002,
            "sigma_max": 10.0,
            "noise_schedule": "edm",
            "initial_reconstruction": "noise",
            "data_consistency_gradient": "paper_squared_residual",
            "adjoint_data_step_schedule": "paper",
            "prior_mode": "patch",
        },
    )
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "method": "padis_dps",
                    "algorithm": "dps_langevin",
                    "prior_mode": "patch",
                    "experiment": "ct_20",
                    "implementation": "paper",
                    "geometry": "lion",
                    "checkpoint": "/tmp/checkpoint.pt",
                    "expected_sampler": {
                        "num_steps": 100,
                        "noise_schedule": "geometric",
                        "sigma_min": 0.002,
                        "prior_mode": "patch",
                    },
                }
            ]
        )
    )
    args = _args(tmp_path, "--expected-jobs-json", str(manifest))

    failures = check_records(args, find_records(args))

    assert any(
        "sampler noise_schedule='edm' does not match expected 'geometric'" in failure
        for failure in failures
    )


def test_verifier_checks_expected_ve_ddnm_layout_from_manifest(tmp_path):
    _write_metrics(
        tmp_path,
        method="ve_ddnm",
        algorithm="langevin",
        sampler={
            "num_steps": 100,
            "inner_steps": 10,
            "sigma_min": 0.002,
            "sigma_max": 10.0,
            "noise_schedule": "geometric",
            "initial_reconstruction": "noise",
            "data_consistency_gradient": "paper_squared_residual",
            "adjoint_data_step_schedule": "paper",
            "prior_mode": "patch",
            "langevin_ddnm": True,
            "ddnm_pseudoinverse_clip": True,
            "ddnm_projected_pseudoinverse_clip": True,
            "ve_ddnm_nfe_layout": "public_inner",
        },
    )
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "method": "ve_ddnm",
                    "algorithm": "langevin",
                    "prior_mode": "patch",
                    "experiment": "ct_20",
                    "implementation": "paper",
                    "geometry": "lion",
                    "checkpoint": "/tmp/checkpoint.pt",
                    "expected_sampler": {
                        "num_steps": 1000,
                        "inner_steps": 1,
                        "ve_ddnm_nfe_layout": "paper_1000x1",
                    },
                }
            ]
        )
    )
    args = _args(tmp_path, "--expected-jobs-json", str(manifest))

    failures = check_records(args, find_records(args))

    assert any(
        "sampler num_steps=100 does not match expected 1000" in failure
        for failure in failures
    )
    assert any(
        "sampler inner_steps=10 does not match expected 1" in failure
        for failure in failures
    )
    assert any(
        "sampler ve_ddnm_nfe_layout='public_inner' "
        "does not match expected 'paper_1000x1'" in failure
        for failure in failures
    )


def test_verifier_checks_expected_fixed_overlap_checkpoint_from_manifest(tmp_path):
    _write_metrics(
        tmp_path,
        method="patch_average",
        sampler={
            "num_steps": 100,
            "inner_steps": 10,
            "sigma_min": 0.002,
            "sigma_max": 10.0,
            "noise_schedule": "geometric",
            "initial_reconstruction": "noise",
            "data_consistency_gradient": "paper_squared_residual",
            "adjoint_data_step_schedule": "paper",
            "prior_mode": "patch",
            "patch_assembly": "fixed_average",
            "fixed_overlap_checkpoint_denoiser": False,
        },
    )
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "method": "patch_average",
                    "algorithm": "dps_langevin",
                    "prior_mode": "patch",
                    "experiment": "ct_20",
                    "implementation": "paper",
                    "geometry": "lion",
                    "checkpoint": "/tmp/checkpoint.pt",
                    "expected_sampler": {
                        "patch_assembly": "fixed_average",
                        "fixed_overlap_checkpoint_denoiser": True,
                    },
                }
            ]
        )
    )
    args = _args(tmp_path, "--expected-jobs-json", str(manifest))

    failures = check_records(args, find_records(args))

    assert any(
        "sampler fixed_overlap_checkpoint_denoiser=False "
        "does not match expected True" in failure
        for failure in failures
    )


def test_verifier_checks_expected_method_settings_from_manifest(tmp_path):
    _write_metrics(
        tmp_path,
        method="admm_tv",
        method_settings={
            "tv_lambda": 0.01,
            "tv_iterations": 500,
            "tv_lipschitz": None,
            "tv_non_negativity": False,
        },
    )
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "method": "admm_tv",
                    "algorithm": "dps_langevin",
                    "prior_mode": "patch",
                    "experiment": "ct_20",
                    "implementation": "paper",
                    "geometry": "lion",
                    "checkpoint": "/tmp/checkpoint.pt",
                    "expected_method_settings": {
                        "tv_lambda": 0.001,
                        "tv_iterations": 500,
                        "tv_lipschitz": None,
                        "tv_non_negativity": False,
                    },
                }
            ]
        )
    )
    args = _args(tmp_path, "--expected-jobs-json", str(manifest))

    failures = check_records(args, find_records(args))

    assert any(
        "method tv_lambda=0.01 does not match expected 0.001" in failure
        for failure in failures
    )


def test_verifier_checks_expected_pnp_checkpoint_from_manifest(tmp_path):
    _write_metrics(
        tmp_path,
        method="pnp_admm",
        method_settings={
            "pnp_checkpoint": "/tmp/actual_denoiser.pt",
            "pnp_iterations": 10,
            "pnp_eta": 1e-4,
            "pnp_cg_iterations": 100,
            "pnp_cg_tolerance": 1e-7,
            "pnp_noise_level": None,
        },
    )
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "method": "pnp_admm",
                    "algorithm": "dps_langevin",
                    "prior_mode": "patch",
                    "experiment": "ct_20",
                    "implementation": "paper",
                    "geometry": "lion",
                    "checkpoint": "/tmp/checkpoint.pt",
                    "expected_method_settings": {
                        "pnp_checkpoint": "/tmp/expected_denoiser.pt",
                        "pnp_iterations": 10,
                        "pnp_eta": 1e-4,
                        "pnp_cg_iterations": 100,
                        "pnp_cg_tolerance": 1e-7,
                        "pnp_noise_level": None,
                    },
                }
            ]
        )
    )
    args = _args(tmp_path, "--expected-jobs-json", str(manifest))

    failures = check_records(args, find_records(args))

    assert any(
        "method pnp_checkpoint='/tmp/actual_denoiser.pt' "
        "does not match expected '/tmp/expected_denoiser.pt'" in failure
        for failure in failures
    )
