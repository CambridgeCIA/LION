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
    resolve_training_root,
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


def test_training_root_preset_resolves_gcp_final_model_root(tmp_path):
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--training-root-preset",
            "gcp",
            "--run-root",
            str(tmp_path / "runs"),
            "--gcp-run-name",
            "PaDIS-Reproduction-GCP-test",
            "--output-root",
            str(tmp_path / "recon"),
        ]
    )

    assert (
        resolve_training_root(args)
        == (
            tmp_path / "runs" / "final_real_runs" / "PaDIS-Reproduction-GCP-test"
        ).resolve()
    )


def test_training_root_preset_resolves_slurm_final_model_root_from_stamp(tmp_path):
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--training-root-preset",
            "slurm",
            "--run-root",
            str(tmp_path / "runs"),
            "--run-stamp",
            "20260704_120000",
            "--output-root",
            str(tmp_path / "recon"),
        ]
    )

    assert (
        resolve_training_root(args)
        == (
            tmp_path / "runs" / "final_real_runs" / "a100_training_20260704_120000"
        ).resolve()
    )


def test_explicit_training_root_overrides_preset(tmp_path):
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--training-root",
            str(tmp_path / "explicit"),
            "--training-root-preset",
            "gcp",
            "--run-root",
            str(tmp_path / "runs"),
            "--gcp-run-name",
            "ignored",
            "--output-root",
            str(tmp_path / "recon"),
        ]
    )

    assert resolve_training_root(args) == (tmp_path / "explicit").resolve()


def test_method_default_matrix_uses_method_specific_models(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "baseline,whole_image_diffusion,padis_dps",
        "--experiments",
        "ct_20",
    )

    jobs = build_jobs(args)

    assert [(job.method.name, job.implementation) for job in jobs] == [
        ("baseline", "lion_physics"),
        ("whole_image_diffusion", "lion_physics"),
        ("whole_image_diffusion", "paper"),
        ("padis_dps", "lion_physics"),
        ("padis_dps", "public_repo"),
        ("padis_dps", "paper"),
    ]
    assert [job.model.name for job in jobs] == [
        "patch_lidc_default",
        "whole_lidc_default",
        "whole_lidc_default",
        "patch_lidc_default",
        "patch_lidc_default",
        "patch_lidc_default",
    ]
    whole_payload = job_json(args, jobs[1])
    assert whole_payload["checkpoint"].endswith(
        "whole_lidc_default/whole_image_lidc_256_min_val.pt"
    )


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
        "patch_lidc_512",
        "patch_lidc_512",
    ]
    assert [job.experiment for job in jobs] == ["ct_512_60"] * 5
    padis_payload = next(
        job_json(args, job)
        for job in jobs
        if job.method.name == "padis_dps" and job.implementation == "lion_physics"
    )
    sampler = padis_payload["expected_sampler"]
    assert sampler["patch_batch_size"] == 1
    assert sampler["patch_checkpoint_denoiser"] is True
    assert sampler["fixed_overlap_checkpoint_denoiser"] is False


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

    assert len(jobs) == 14
    assert {(job.method.name, job.implementation) for job in jobs} == {
        ("baseline", "lion_physics"),
        ("admm_tv", "lion_physics"),
        ("padis_dps", "lion_physics"),
        ("padis_dps", "public_repo"),
        ("padis_dps", "paper"),
        ("langevin", "lion_physics"),
        ("langevin", "public_repo"),
        ("langevin", "paper"),
        ("predictor_corrector", "lion_physics"),
        ("predictor_corrector", "public_repo"),
        ("predictor_corrector", "paper"),
        ("ve_ddnm", "lion_physics"),
        ("ve_ddnm", "public_repo"),
        ("ve_ddnm", "paper"),
    }
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
    assert command[command.index("--pnp-iterations") + 1] == "60"
    assert command[command.index("--pnp-eta") + 1] == "2e-05"


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

    whole_payload = next(
        payload
        for payload in payloads
        if payload["method"] == "whole_image_diffusion"
        and payload["implementation"] == "lion_physics"
    )
    assert whole_payload["method"] == "whole_image_diffusion"
    assert whole_payload["algorithm"] == "dps_langevin"
    assert whole_payload["prior_mode"] == "whole_image"
    assert whole_payload["implementation"] == "lion_physics"
    assert whole_payload["checkpoint"].endswith(
        "whole_lidc_default/whole_image_lidc_256_min_val.pt"
    )
    assert whole_payload["expected_sampler"]["prior_mode"] == "whole_image"
    assert (
        whole_payload["expected_sampler"]["data_consistency_normalization"]
        == "operator_lipschitz"
    )

    langevin_payload = next(
        payload
        for payload in payloads
        if payload["method"] == "langevin"
        and payload["implementation"] == "lion_physics"
    )
    assert langevin_payload["method"] == "langevin"
    assert langevin_payload["algorithm"] == "langevin"
    assert langevin_payload["prior_mode"] == "patch"


def test_whole_image_lion_physics_fanbeam_uses_stabilized_epsilon(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "whole_image_diffusion",
        "--experiments",
        "ct_fanbeam_180",
        "--implementations",
        "lion_physics",
    )

    payload = job_json(args, build_jobs(args)[0])
    sampler = payload["expected_sampler"]

    assert payload["implementation"] == "lion_physics"
    assert payload["experiment"] == "ct_fanbeam_180"
    assert sampler["prior_mode"] == "whole_image"
    assert sampler["dps_epsilon"] == 0.5
    assert sampler["data_consistency_gradient"] == "least_squares"
    assert sampler["data_consistency_normalization"] == "operator_lipschitz"


def test_job_manifest_contains_expected_sampler_settings(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "padis_dps,predictor_corrector,ve_ddnm,patch_average,patch_stitch",
        "--experiments",
        "ct_20",
        "--implementations",
        "lion_physics",
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
    assert padis_sampler["zeta"] == 4.5
    assert padis_sampler["dps_epsilon"] == 0.5
    assert padis_sampler["data_consistency_gradient"] == "least_squares"
    assert padis_sampler["adjoint_data_step_schedule"] == "paper"
    assert padis_sampler["data_consistency_normalization"] == "operator_lipschitz"
    assert padis_sampler["data_consistency_scale"] == 1.0
    assert padis_sampler["adjoint_data_consistency_scale"] is None

    assert payloads["ve_ddnm"]["implementation"] == "lion_physics"
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
    assert payloads["predictor_corrector"]["implementation"] == "lion_physics"
    assert payloads["predictor_corrector"]["expected_sampler"]["pc_snr"] == 0.04
    assert payloads["predictor_corrector"]["expected_sampler"]["zeta"] == 4.25
    assert (
        payloads["predictor_corrector"]["expected_sampler"]["pc_corrector_step_rule"]
        == "paper_linear"
    )
    assert (
        payloads["predictor_corrector"]["expected_sampler"][
            "pc_corrector_denoise_sigma"
        ]
        == "next"
    )
    assert (
        payloads["predictor_corrector"]["expected_sampler"]["pc_reuse_predictor_layout"]
        is False
    )
    assert (
        payloads["patch_average"]["expected_sampler"]["patch_assembly"]
        == "fixed_average"
    )
    assert payloads["patch_average"]["implementation"] == "lion_physics"
    assert payloads["patch_average"]["expected_sampler"]["dps_epsilon"] == 0.5
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
    assert payloads["patch_average"]["expected_sampler"]["patch_batch_size"] == 1
    assert (
        payloads["patch_stitch"]["expected_sampler"]["patch_assembly"] == "fixed_stitch"
    )
    assert payloads["patch_stitch"]["implementation"] == "lion_physics"
    assert payloads["patch_stitch"]["expected_sampler"]["dps_epsilon"] == 0.5
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
    assert payloads["patch_stitch"]["expected_sampler"]["patch_batch_size"] == 1


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
    assert sampler["zeta"] == 0.2
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
        "pnp_clip": True,
    }


def test_full_method_default_matrix_has_requested_core_grid(tmp_path):
    args = _args(tmp_path, "--methods", "all")

    jobs = build_jobs(args)

    assert len(jobs) == 59
    counts = {}
    for job in jobs:
        counts[(job.method.name, job.implementation)] = (
            counts.get((job.method.name, job.implementation), 0) + 1
        )
    assert counts == {
        ("baseline", "lion_physics"): 5,
        ("admm_tv", "lion_physics"): 5,
        ("pnp_admm", "lion_physics"): 2,
        ("whole_image_diffusion", "lion_physics"): 4,
        ("whole_image_diffusion", "paper"): 2,
        ("langevin", "lion_physics"): 3,
        ("langevin", "public_repo"): 2,
        ("langevin", "paper"): 2,
        ("predictor_corrector", "lion_physics"): 3,
        ("predictor_corrector", "public_repo"): 2,
        ("predictor_corrector", "paper"): 2,
        ("ve_ddnm", "lion_physics"): 3,
        ("ve_ddnm", "public_repo"): 2,
        ("ve_ddnm", "paper"): 2,
        ("patch_average", "lion_physics"): 2,
        ("patch_average", "public_repo"): 2,
        ("patch_stitch", "lion_physics"): 2,
        ("patch_stitch", "public_repo"): 2,
        ("padis_dps", "lion_physics"): 5,
        ("padis_dps", "public_repo"): 5,
        ("padis_dps", "paper"): 2,
    }
    whole_image_sampling_payloads = [
        job_json(args, job)
        for job in jobs
        if job.model.name == "whole_lidc_default"
        and job.method.name in {"langevin", "predictor_corrector", "ve_ddnm"}
    ]
    assert {payload["display_label"] for payload in whole_image_sampling_payloads} == {
        "Whole image - Langevin",
        "Whole image - Predictor-corrector",
        "Whole image - VE-DDNM",
    }
    assert {payload["experiment"] for payload in whole_image_sampling_payloads} == {
        "ct_20"
    }
    assert {payload["prior_mode"] for payload in whole_image_sampling_payloads} == {
        "whole_image"
    }
    patch_langevin_payload = next(
        job_json(args, job)
        for job in jobs
        if job.model.name == "patch_lidc_default"
        and job.method.name == "langevin"
        and job.implementation == "lion_physics"
        and job.experiment == "ct_20"
    )
    assert patch_langevin_payload["display_label"] == "Patch - Langevin"


def test_full_lion_physics_matrix_uses_lipschitz_scaled_data_updates(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "all",
        "--experiments",
        "paper_matrix",
        "--implementations",
        "lion_physics",
    )

    payloads = [job_json(args, job) for job in build_jobs(args)]
    diffusion_methods = {
        "whole_image_diffusion",
        "langevin",
        "predictor_corrector",
        "ve_ddnm",
        "patch_average",
        "patch_stitch",
        "padis_dps",
    }
    diffusion_payloads = [
        payload for payload in payloads if payload["method"] in diffusion_methods
    ]

    assert len(payloads) == 34
    assert len(diffusion_payloads) == 22
    for payload in diffusion_payloads:
        sampler = payload["expected_sampler"]
        assert payload["implementation"] == "lion_physics"
        assert sampler["noise_schedule"] == "geometric"
        assert sampler["sigma_max"] == 10.0
        assert sampler["data_consistency_gradient"] == "least_squares"
        assert sampler["data_consistency_normalization"] == "operator_lipschitz"
        assert sampler["data_consistency_scale"] == 1.0
        assert sampler["adjoint_data_consistency_scale"] is None
        assert sampler.get("data_consistency_scale") != 0.0405
        assert sampler.get("adjoint_data_consistency_scale") != 0.1022
        assert "public_repo" not in sampler.values()

    sigma_by_experiment = {
        payload["experiment"]: payload["expected_sampler"]["sigma_min"]
        for payload in diffusion_payloads
    }
    assert sigma_by_experiment["ct_8"] == 0.003
    assert sigma_by_experiment["ct_20"] == 0.002
    assert sigma_by_experiment["ct_60"] == 0.002
    assert sigma_by_experiment["ct_fanbeam_180"] == 0.002
    assert sigma_by_experiment["ct_512_60"] == 0.002


def test_paper_matrix_uses_method_specific_experiment_sets(tmp_path):
    args = _args(tmp_path, "--methods", "pnp_admm,langevin,padis_dps")

    jobs = build_jobs(args)

    assert sorted(
        {job.experiment for job in jobs if job.method.name == "pnp_admm"}
    ) == [
        "ct_20",
        "ct_8",
    ]
    assert sorted(
        {job.experiment for job in jobs if job.method.name == "langevin"}
    ) == [
        "ct_20",
        "ct_8",
    ]
    assert sorted(
        {job.experiment for job in jobs if job.method.name == "padis_dps"}
    ) == [
        "ct_20",
        "ct_512_60",
        "ct_60",
        "ct_8",
        "ct_fanbeam_180",
    ]


def test_method_default_matrix_excludes_whole_image_512_rows(tmp_path):
    args = _args(tmp_path, "--methods", "whole_image_diffusion,padis_dps")

    jobs = build_jobs(args)

    assert {
        job.experiment for job in jobs if job.method.name == "whole_image_diffusion"
    } == {
        "ct_20",
        "ct_60",
        "ct_8",
        "ct_fanbeam_180",
    }
    assert "ct_512_60" in {
        job.experiment for job in jobs if job.method.name == "padis_dps"
    }


def test_paper_matrix_includes_8_view_rows_for_all_table1_methods(tmp_path):
    args = _args(tmp_path, "--methods", "all")

    jobs = build_jobs(args)
    methods_with_ct8 = {job.method.name for job in jobs if job.experiment == "ct_8"}

    assert methods_with_ct8 == {
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


def test_trained_ablation_matrix_appends_trained_checkpoint_families(tmp_path):
    args = _args(tmp_path, "--methods", "all", "--ablations", "all")

    jobs = build_jobs(args)
    ablation_jobs = [job for job in jobs if job.matrix_group != "main"]

    assert len(jobs) == 101
    assert {
        (
            job.matrix_group,
            job.method.name,
            job.model.name,
            job.experiment,
            job.implementation,
        )
        for job in ablation_jobs
        if job.matrix_group.startswith("patch_size_")
    } >= {
        (
            "patch_size_p8",
            "padis_dps",
            "patch_lidc_p8_default",
            "ct_20",
            "lion_physics",
        ),
        ("patch_size_p8", "padis_dps", "patch_lidc_p8_default", "ct_20", "public_repo"),
        ("patch_size_p56", "padis_dps", "patch_lidc_default", "ct_20", "lion_physics"),
        (
            "patch_size_p96",
            "padis_dps",
            "patch_lidc_p96_default",
            "ct_20",
            "public_repo",
        ),
    }
    assert (
        len([job for job in ablation_jobs if job.matrix_group.startswith("schedule_")])
        == 16
    )
    assert (
        len(
            [job for job in ablation_jobs if job.matrix_group.startswith("patch_size_")]
        )
        == 10
    )
    assert (
        len(
            [
                job
                for job in ablation_jobs
                if job.matrix_group.startswith("dataset_size_")
            ]
        )
        == 8
    )
    assert (
        len([job for job in ablation_jobs if job.matrix_group.startswith("position_")])
        == 8
    )
    assert {
        job.model.name
        for job in jobs
        if job.method.name not in {"baseline", "admm_tv", "pnp_admm"}
    } == {
        "patch_lidc_default",
        "patch_lidc_full",
        "patch_lidc_p8_default",
        "patch_lidc_p16_default",
        "patch_lidc_p32_default",
        "patch_lidc_p96_default",
        "patch_lidc_no_pos_default",
        "whole_lidc_default",
        "whole_lidc_full",
        "patch_lidc_512",
    }


def test_schedule_init_grid_covers_256_experiments_for_lion_and_public(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "padis_dps",
        "--ablations",
        "schedule_init",
    )

    jobs = [job for job in build_jobs(args) if job.matrix_group != "main"]

    assert len(jobs) == 16
    assert {(job.experiment, job.implementation, job.matrix_group) for job in jobs} == {
        (experiment, implementation, f"schedule_{schedule}_{init}_init")
        for experiment in (
            "ct_20",
            "ct_8",
        )
        for implementation in ("lion_physics", "public_repo")
        for schedule in ("geometric", "edm")
        for init in ("fdk", "noise")
    }
    assert "ct_512_60" not in {job.experiment for job in jobs}


def test_patch_and_dataset_ablation_selector_excludes_position_row(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "all",
        "--ablations",
        "patch_size,dataset_size",
    )

    jobs = build_jobs(args)
    ablation_groups = {job.matrix_group for job in jobs if job.matrix_group != "main"}

    assert len(jobs) == 77
    assert "position_no_encoding_noise_init" not in ablation_groups
    assert "position_no_encoding_fdk_init" not in ablation_groups
    assert "position_with_encoding_noise_init" not in ablation_groups
    assert "position_with_encoding_fdk_init" not in ablation_groups


def test_position_ablation_includes_noise_and_fdk_initialization_rows(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "padis_dps",
        "--ablations",
        "position_encoding",
    )

    jobs = build_jobs(args)
    ablation_jobs = {
        (job.matrix_group, job.implementation): job
        for job in jobs
        if job.matrix_group != "main"
    }

    assert set(ablation_jobs) == {
        ("position_no_encoding_noise_init", "lion_physics"),
        ("position_no_encoding_noise_init", "public_repo"),
        ("position_no_encoding_fdk_init", "lion_physics"),
        ("position_no_encoding_fdk_init", "public_repo"),
        ("position_with_encoding_noise_init", "lion_physics"),
        ("position_with_encoding_noise_init", "public_repo"),
        ("position_with_encoding_fdk_init", "lion_physics"),
        ("position_with_encoding_fdk_init", "public_repo"),
    }

    noise_payload = job_json(
        args, ablation_jobs[("position_no_encoding_noise_init", "lion_physics")]
    )
    noise_sampler = noise_payload["expected_sampler"]
    assert "--no-position-channels" in noise_payload["command"]
    assert noise_sampler["initial_reconstruction"] == "noise"
    assert noise_sampler["clip_initial"] is False
    assert noise_sampler["clip_output"] is False
    assert "--initial-reconstruction" in noise_payload["command"]
    assert (
        noise_payload["command"][
            noise_payload["command"].index("--initial-reconstruction") + 1
        ]
        == "noise"
    )
    noise_output_folder = noise_payload["command"][
        noise_payload["command"].index("--output-folder") + 1
    ]
    assert noise_output_folder.endswith(
        "padis_dps/patch_lidc_no_pos_default/lion_physics/lion/ct_20/position_no_encoding_noise_init"
    )

    init_payload = job_json(
        args, ablation_jobs[("position_no_encoding_fdk_init", "lion_physics")]
    )
    init_sampler = init_payload["expected_sampler"]
    assert "--no-position-channels" in init_payload["command"]
    assert init_sampler["initial_reconstruction"] == "fdk"
    assert init_sampler["clip_initial"] is True
    assert init_sampler["clip_output"] is True
    init_output_folder = init_payload["command"][
        init_payload["command"].index("--output-folder") + 1
    ]
    assert init_output_folder.endswith(
        "padis_dps/patch_lidc_no_pos_default/lion_physics/lion/ct_20/position_no_encoding_fdk_init"
    )

    with_pos_noise = job_json(
        args, ablation_jobs[("position_with_encoding_noise_init", "lion_physics")]
    )
    assert "--no-position-channels" not in with_pos_noise["command"]
    assert with_pos_noise["expected_sampler"]["initial_reconstruction"] == "noise"


def test_ablations_do_not_expand_explicit_model_selection(tmp_path):
    args = _args(
        tmp_path,
        "--models",
        "patch_lidc_p8_default",
        "--methods",
        "padis_dps",
        "--ablations",
        "all",
    )

    jobs = build_jobs(args)

    assert len(jobs) == 3
    assert {job.model.name for job in jobs} == {"patch_lidc_p8_default"}
    assert {job.implementation for job in jobs} == {
        "lion_physics",
        "public_repo",
        "paper",
    }
    assert {job.matrix_group for job in jobs} == {"main"}


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


def test_lion_physics_implementation_uses_operator_normalized_settings(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "padis_dps,langevin,predictor_corrector,ve_ddnm,patch_average,patch_stitch",
        "--implementations",
        "lion_physics",
        "--experiments",
        "ct_20",
    )

    payloads = {job.method.name: job_json(args, job) for job in build_jobs(args)}
    sampler = payloads["padis_dps"]["expected_sampler"]

    assert sampler["noise_schedule"] == "geometric"
    assert sampler["sigma_min"] == 0.002
    assert sampler["sigma_max"] == 10.0
    assert sampler["initial_reconstruction"] == "fdk"
    assert sampler["initial_fdk_filter_type"] == "hann"
    assert sampler["initial_fdk_frequency_scaling"] == 0.2
    assert sampler["zeta"] == 4.5
    assert sampler["dps_epsilon"] == 0.5
    assert sampler["data_consistency_gradient"] == "least_squares"
    assert sampler["adjoint_data_step_schedule"] == "paper"
    assert sampler["data_consistency_normalization"] == "operator_lipschitz"
    assert sampler["data_consistency_scale"] == 1.0
    assert sampler["adjoint_data_consistency_scale"] is None
    assert sampler["pc_snr"] == 0.04
    assert "public_repo" not in sampler.values()

    assert (
        payloads["predictor_corrector"]["expected_sampler"][
            "pc_corrector_denoise_sigma"
        ]
        == "next"
    )
    assert payloads["predictor_corrector"]["expected_sampler"]["zeta"] == 4.25
    assert payloads["predictor_corrector"]["expected_sampler"]["pc_snr"] == 0.04
    assert (
        payloads["predictor_corrector"]["expected_sampler"]["pc_reuse_predictor_layout"]
        is False
    )
    assert payloads["langevin"]["expected_sampler"]["zeta"] == 4.0
    assert payloads["langevin"]["expected_sampler"]["sampling_epsilon"] == 0.5
    assert payloads["ve_ddnm"]["expected_sampler"]["ve_ddnm_nfe_layout"] == (
        "paper_1000x1"
    )
    assert payloads["ve_ddnm"]["expected_sampler"]["ddnm_corrected_clip"] is True
    assert payloads["patch_average"]["expected_sampler"]["dps_epsilon"] == 0.5
    assert (
        payloads["patch_average"]["expected_sampler"]["fixed_overlap_layout"]
        == "public_overlap"
    )
    assert payloads["patch_stitch"]["expected_sampler"]["dps_epsilon"] == 0.5
    assert (
        payloads["patch_stitch"]["expected_sampler"]["fixed_overlap_layout"]
        == "public_tile"
    )


def test_method_default_rejects_off_paper_experiment_selection(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "whole_image_diffusion",
        "--experiments",
        "ct_512_60",
    )

    with pytest.raises(ValueError, match="not part of the paper reconstruction matrix"):
        build_jobs(args)


def test_method_default_can_allow_off_paper_experiment_selection(tmp_path):
    args = _args(
        tmp_path,
        "--methods",
        "whole_image_diffusion",
        "--experiments",
        "ct_512_60",
        "--allow-off-paper-experiments",
    )

    jobs = build_jobs(args)

    assert len(jobs) == 2
    assert {job.implementation for job in jobs} == {"lion_physics", "paper"}
    assert {job.method.name for job in jobs} == {"whole_image_diffusion"}
    assert {job.experiment for job in jobs} == {"ct_512_60"}


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
    assert result.stdout.count("Executing reconstruction job:") == 59
    assert "--method pnp_admm" in result.stdout
