from pathlib import Path
import subprocess
import sys

import pytest

from scripts.paper_scripts.PaDIS.PaDIS_run_reconstruction_matrix import (
    build_arg_parser,
    build_jobs,
    command_for_job,
    input_check_failures,
    job_json,
)


MATRIX_SCRIPT = "scripts/paper_scripts/PaDIS/PaDIS_run_reconstruction_matrix.py"


def _args(tmp_path, *extra):
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--training-root",
            str(tmp_path / "training"),
            "--output-root",
            str(tmp_path / "recon"),
            *extra,
        ]
    )
    args.training_root = args.training_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.pnp_root is None:
        args.pnp_root = args.training_root / "pnp_lidc_drunet"
    return args


def test_method_default_matrix_uses_method_specific_models(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "baseline,whole_image_diffusion,padis_dps",
        "--experiments",
        "ct_20",
    )

    jobs = build_jobs(args)

    assert [job.method.name for job in jobs] == [
        "baseline",
        "whole_image_diffusion",
        "padis_dps",
    ]
    assert [job.model.name for job in jobs] == [
        "patch_lidc_default",
        "whole_lidc_default",
        "patch_lidc_default",
    ]
    assert [job.implementation for job in jobs] == [
        "paper",
        "paper",
        "public_repo",
    ]


def test_method_default_matrix_uses_native_512_model_for_512_experiment(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "baseline,admm_tv,padis_dps",
        "--experiments",
        "ct_512_60",
    )

    jobs = build_jobs(args)

    assert [job.model.name for job in jobs] == [
        "patch_lidc_512",
        "patch_lidc_512",
        "patch_lidc_512",
    ]
    assert [job.experiment for job in jobs] == ["ct_512_60"] * 3


def test_reconstruction_smoke_selector_has_six_expected_jobs(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "baseline,admm_tv,padis_dps,langevin,predictor_corrector,ve_ddnm",
        "--experiments",
        "ct_20",
        "--max-samples",
        "1",
    )

    jobs = build_jobs(args)

    assert len(jobs) == 6
    assert [job.method.name for job in jobs] == [
        "baseline",
        "admm_tv",
        "padis_dps",
        "langevin",
        "predictor_corrector",
        "ve_ddnm",
    ]
    assert {job.model.name for job in jobs} == {"patch_lidc_default"}


def test_method_command_contains_method_and_expected_checkpoint_family(tmp_path):
    args = _args(tmp_path, "--methods", "pnp_admm", "--experiments", "ct_20")
    job = build_jobs(args)[0]

    command = command_for_job(args, job)

    assert command[command.index("--method") + 1] == "pnp_admm"
    assert "--checkpoint" not in command
    assert command[command.index("--pnp-checkpoint") + 1].endswith(
        "pnp_lidc_drunet/pnp_lidc_drunet.pt"
    )


def test_input_check_reports_missing_required_checkpoints_once(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "padis_dps,langevin,pnp_admm",
        "--experiments",
        "ct_20",
        "--allow-off-paper-experiments",
    )

    failures = input_check_failures(args, build_jobs(args))

    assert len(failures) == 2
    assert any("Missing checkpoint for patch_lidc_default" in item for item in failures)
    assert any("Missing PnP denoiser checkpoint" in item for item in failures)


def test_job_manifest_contains_verifier_identity_fields(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "whole_image_diffusion,langevin",
        "--experiments",
        "ct_20",
    )

    jobs = build_jobs(args)
    payloads = [job_json(args, job) for job in jobs]

    whole_payload = payloads[0]
    assert whole_payload["method"] == "whole_image_diffusion"
    assert whole_payload["algorithm"] == "dps_langevin"
    assert whole_payload["prior_mode"] == "whole_image"

    langevin_payload = payloads[1]
    assert langevin_payload["method"] == "langevin"
    assert langevin_payload["algorithm"] == "langevin"
    assert langevin_payload["prior_mode"] == "patch"


def test_job_manifest_contains_expected_sampler_settings(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "padis_dps,predictor_corrector,ve_ddnm,patch_average,patch_stitch",
        "--experiments",
        "ct_20",
    )

    payloads = {job.method.name: job_json(args, job) for job in build_jobs(args)}

    padis_sampler = payloads["padis_dps"]["expected_sampler"]
    assert padis_sampler["noise_schedule"] == "geometric"
    assert padis_sampler["sigma_min"] == 0.002
    assert padis_sampler["sigma_max"] == 10.0
    assert padis_sampler["noise_initialization"] == "padded"
    assert padis_sampler["initial_reconstruction"] == "fdk"
    assert padis_sampler["clip_initial"] is True
    assert padis_sampler["clip_output"] is True
    assert padis_sampler["dps_epsilon"] == 0.5
    assert padis_sampler["data_consistency_gradient"] == "norm"
    assert padis_sampler["adjoint_data_step_schedule"] == "public_repo"
    assert padis_sampler["data_consistency_scale"] == 0.0405
    assert padis_sampler["adjoint_data_consistency_scale"] == 0.1022

    assert payloads["ve_ddnm"]["implementation"] == "lion_quality"
    assert payloads["ve_ddnm"]["expected_sampler"]["langevin_ddnm"] is True
    assert payloads["ve_ddnm"]["expected_sampler"]["num_steps"] == 1000
    assert payloads["ve_ddnm"]["expected_sampler"]["inner_steps"] == 1
    assert (
        payloads["ve_ddnm"]["expected_sampler"]["ve_ddnm_nfe_layout"] == "paper_1000x1"
    )
    assert payloads["ve_ddnm"]["expected_sampler"]["ddnm_pseudoinverse_clip"] is True
    assert (
        payloads["ve_ddnm"]["expected_sampler"]["ddnm_projected_pseudoinverse_clip"]
        is True
    )
    assert payloads["ve_ddnm"]["expected_sampler"]["ddnm_corrected_clip"] is True
    assert payloads["ve_ddnm"]["expected_sampler"]["sampling_epsilon"] == 0.1
    assert payloads["ve_ddnm"]["expected_sampler"]["initial_reconstruction"] == "noise"
    assert payloads["ve_ddnm"]["expected_sampler"]["noise_initialization"] == "padded"
    assert payloads["ve_ddnm"]["expected_sampler"]["clip_initial"] is False
    assert payloads["ve_ddnm"]["expected_sampler"]["clip_output"] is False
    assert payloads["predictor_corrector"]["expected_sampler"]["pc_snr"] == 0.16
    assert (
        payloads["predictor_corrector"]["expected_sampler"]["pc_corrector_step_rule"]
        == "paper_linear"
    )
    assert (
        payloads["predictor_corrector"]["expected_sampler"][
            "pc_corrector_denoise_sigma"
        ]
        == "current"
    )
    assert (
        payloads["predictor_corrector"]["expected_sampler"]["pc_reuse_predictor_layout"]
        is True
    )
    assert (
        payloads["patch_average"]["expected_sampler"]["patch_assembly"]
        == "fixed_average"
    )
    assert payloads["patch_average"]["implementation"] == "public_repo"
    assert (
        payloads["patch_average"]["expected_sampler"]["fixed_overlap_layout"]
        == "public_overlap"
    )
    assert (
        payloads["patch_average"]["expected_sampler"][
            "fixed_overlap_checkpoint_denoiser"
        ]
        is True
    )
    assert (
        payloads["patch_stitch"]["expected_sampler"]["patch_assembly"] == "fixed_stitch"
    )
    assert payloads["patch_stitch"]["implementation"] == "public_repo"
    assert (
        payloads["patch_stitch"]["expected_sampler"]["fixed_overlap_layout"]
        == "public_tile"
    )
    assert (
        payloads["patch_stitch"]["expected_sampler"][
            "fixed_overlap_checkpoint_denoiser"
        ]
        is True
    )


def test_public_repo_manifest_contains_public_sampler_settings(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "padis_dps,predictor_corrector",
        "--experiments",
        "ct_20",
        "--implementations",
        "public_repo",
    )

    payloads = {job.method.name: job_json(args, job) for job in build_jobs(args)}
    payload = payloads["padis_dps"]
    sampler = payload["expected_sampler"]

    assert sampler["noise_schedule"] == "geometric"
    assert sampler["sigma_min"] == 0.002
    assert sampler["noise_initialization"] == "padded"
    assert sampler["initial_reconstruction"] == "fdk"
    assert sampler["clip_initial"] is True
    assert sampler["clip_output"] is True
    assert sampler["dps_epsilon"] == 0.5
    assert sampler["data_consistency_gradient"] == "norm"
    assert sampler["adjoint_data_step_schedule"] == "public_repo"
    assert sampler["data_consistency_scale"] == 0.0405
    assert sampler["adjoint_data_consistency_scale"] == 0.1022
    pc_sampler = payloads["predictor_corrector"]["expected_sampler"]
    assert pc_sampler["pc_corrector_step_rule"] == "paper_linear"
    assert pc_sampler["pc_corrector_denoise_sigma"] == "current"
    assert pc_sampler["pc_reuse_predictor_layout"] is True


def test_job_manifest_contains_expected_method_settings(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "baseline,admm_tv,pnp_admm",
        "--experiments",
        "ct_20",
        "--tv-lambda",
        "0.002",
        "--tv-iterations",
        "250",
        "--pnp-iterations",
        "12",
        "--pnp-eta",
        "0.0002",
        "--pnp-cg-iterations",
        "80",
        "--pnp-cg-tolerance",
        "1e-6",
        "--pnp-noise-level",
        "0.02",
    )

    payloads = {job.method.name: job_json(args, job) for job in build_jobs(args)}

    assert payloads["baseline"]["checkpoint"] == ""
    assert payloads["admm_tv"]["checkpoint"] == ""
    assert payloads["pnp_admm"]["checkpoint"] == ""
    assert payloads["baseline"]["expected_method_settings"] == {"baseline": "fdk"}
    assert payloads["admm_tv"]["expected_method_settings"] == {
        "tv_lambda": 0.002,
        "tv_iterations": 250,
        "tv_lipschitz": None,
        "tv_non_negativity": False,
    }
    assert payloads["pnp_admm"]["expected_method_settings"] == {
        "pnp_checkpoint": str(args.pnp_root / "pnp_lidc_drunet.pt"),
        "pnp_iterations": 12,
        "pnp_eta": 0.0002,
        "pnp_cg_iterations": 80,
        "pnp_cg_tolerance": 1e-6,
        "pnp_noise_level": 0.02,
    }


def test_full_method_default_matrix_has_one_method_set_per_default_experiment(tmp_path):
    args = _args(tmp_path, "--methods", "all")

    jobs = build_jobs(args)

    assert len(jobs) == 26
    assert {job.method.name for job in jobs} == {
        "baseline",
        "admm_tv",
        "pnp_admm",
        "whole_image_diffusion",
        "langevin",
        "predictor_corrector",
        "ve_ddnm",
        "patch_average",
        "patch_stitch",
        "padis_dps",
    }


def test_paper_matrix_uses_method_specific_experiment_sets(tmp_path):
    args = _args(tmp_path, "--methods", "pnp_admm,langevin,padis_dps")

    jobs = build_jobs(args)

    assert [job.experiment for job in jobs if job.method.name == "pnp_admm"] == [
        "ct_20",
        "ct_8",
    ]
    assert [job.experiment for job in jobs if job.method.name == "langevin"] == [
        "ct_20"
    ]
    assert [job.experiment for job in jobs if job.method.name == "padis_dps"] == [
        "ct_20",
        "ct_8",
        "ct_60",
        "ct_fanbeam_180",
        "ct_512_60",
    ]


def test_public_repo_implementation_is_restricted_to_public_sampler_methods(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "baseline",
        "--implementations",
        "public_repo",
        "--experiments",
        "ct_20",
    )

    with pytest.raises(ValueError, match="no runnable public-repo equivalent"):
        build_jobs(args)


def test_public_repo_implementation_allows_public_sampler_methods(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "padis_dps,langevin,predictor_corrector,ve_ddnm,patch_average,patch_stitch",
        "--implementations",
        "public_repo",
        "--experiments",
        "ct_20",
    )

    jobs = build_jobs(args)

    assert len(jobs) == 6
    assert {job.implementation for job in jobs} == {"public_repo"}


def test_lion_quality_implementation_can_be_selected_for_stabilized_rows(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "ve_ddnm",
        "--implementations",
        "lion_quality",
        "--experiments",
        "ct_20",
    )

    jobs = build_jobs(args)
    payload = job_json(args, jobs[0])

    assert len(jobs) == 1
    assert jobs[0].implementation == "lion_quality"
    assert payload["expected_sampler"]["sampling_epsilon"] == 0.1
    assert payload["expected_sampler"]["ddnm_corrected_clip"] is True


def test_method_default_rejects_off_paper_experiment_selection(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "pnp_admm",
        "--experiments",
        "ct_512_60",
    )

    with pytest.raises(ValueError, match="not part of the paper reconstruction matrix"):
        build_jobs(args)


def test_method_default_can_allow_off_paper_experiment_selection(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "pnp_admm",
        "--experiments",
        "ct_512_60",
        "--allow-off-paper-experiments",
    )

    jobs = build_jobs(args)

    assert len(jobs) == 1
    assert jobs[0].method.name == "pnp_admm"
    assert jobs[0].experiment == "ct_512_60"


def test_explicit_model_rejects_off_paper_experiment_selection(tmp_path):
    args = _args(
        tmp_path,
        "--models",
        "patch_lidc_default",
        "--methods",
        "padis_dps",
        "--experiments",
        "ct_512_60",
    )

    with pytest.raises(ValueError, match="model 'patch_lidc_default'"):
        build_jobs(args)


def test_dry_run_lists_full_matrix_without_existing_checkpoints(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            MATRIX_SCRIPT,
            "--training-root",
            str(tmp_path / "missing_training"),
            "--output-root",
            str(tmp_path / "recon"),
            "--methods",
            "all",
            "--models",
            "method_default",
            "--experiments",
            "paper_matrix",
            "--implementations",
            "method_default",
            "--geometries",
            "lion",
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Executing reconstruction job:") == 26
    assert "--method pnp_admm" in result.stdout
