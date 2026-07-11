import json
import os
from pathlib import Path
import subprocess
from collections import Counter


LION_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = LION_ROOT / "scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_pipeline.sh"


def _install_fake_sbatch(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch.log"
    counter_path = tmp_path / "sbatch.counter"
    sbatch_path = bin_dir / "sbatch"
    sbatch_path.write_text(
        f"""#!/bin/bash
set -euo pipefail
count=100
if [ -f {counter_path!s} ]; then
        count="$(cat {counter_path!s})"
fi
count=$((count + 1))
printf '%s\\n' "$count" > {counter_path!s}
printf '%s | PADIS_RECON_EXPECTED_RECORDS=%s PADIS_RECON_EXPECTED_SAMPLES=%s PADIS_RECON_EXPECTED_JOBS_JSON=%s\\n' "$*" "${{PADIS_RECON_EXPECTED_RECORDS:-}}" "${{PADIS_RECON_EXPECTED_SAMPLES:-}}" "${{PADIS_RECON_EXPECTED_JOBS_JSON:-}}" >> {log_path!s}
printf 'job%s\\n' "$count"
"""
    )
    sbatch_path.chmod(0o755)
    return bin_dir, log_path


def _run_pipeline(tmp_path, **extra_env):
    bin_dir, log_path = _install_fake_sbatch(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PADIS_RUN_ROOT": str(tmp_path / "runs"),
            "PADIS_RUN_STAMP": "pytest",
            "PADIS_WANDB_MODE": "disabled",
            **extra_env,
        }
    )
    result = subprocess.run(
        ["bash", str(PIPELINE)],
        cwd=LION_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log_text = log_path.read_text() if log_path.is_file() else ""
    return result, log_text


def test_pipeline_guard_blocks_pnp_reconstruction_without_denoiser(tmp_path):
    result, sbatch_log = _run_pipeline(
        tmp_path,
        PADIS_SUBMIT_RECONSTRUCTION="1",
        PADIS_SUBMIT_PNP_TRAINING="0",
        PADIS_RECON_METHODS="all",
    )

    assert result.returncode == 1
    assert "matrix containing pnp_admm" in result.stderr
    assert sbatch_log == ""


def test_pipeline_guard_blocks_spaced_pnp_reconstruction_without_denoiser(tmp_path):
    result, sbatch_log = _run_pipeline(
        tmp_path,
        PADIS_SUBMIT_RECONSTRUCTION="1",
        PADIS_SUBMIT_PNP_TRAINING="0",
        PADIS_RECON_METHODS="baseline, pnp_admm",
        PADIS_RECON_EXPERIMENTS="ct_20",
    )

    assert result.returncode == 1
    assert "matrix containing pnp_admm" in result.stderr
    assert sbatch_log == ""


def test_pipeline_preflight_rejects_off_paper_matrix_before_submission(tmp_path):
    result, sbatch_log = _run_pipeline(
        tmp_path,
        PADIS_SUBMIT_RECONSTRUCTION="1",
        PADIS_SUBMIT_PNP_TRAINING="0",
        PADIS_RECON_METHODS="whole_image_diffusion",
        PADIS_RECON_EXPERIMENTS="ct_512_60",
    )

    assert result.returncode == 1
    assert "not part of the paper reconstruction matrix" in result.stderr
    assert sbatch_log == ""


def test_pipeline_can_submit_explicitly_allowed_off_paper_matrix(tmp_path):
    result, sbatch_log = _run_pipeline(
        tmp_path,
        PADIS_SUBMIT_RECONSTRUCTION="1",
        PADIS_SUBMIT_PNP_TRAINING="0",
        PADIS_RECON_METHODS="whole_image_diffusion",
        PADIS_RECON_EXPERIMENTS="ct_512_60",
        PADIS_RECON_ALLOW_OFF_PAPER_EXPERIMENTS="1",
        PADIS_RECON_VERIFY="0",
    )

    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_reconstruction_array.sh" in sbatch_log
    assert "--array 0-1%10" in sbatch_log

    manifest = (
        tmp_path
        / "runs/final_real_runs/a100_reconstruction_pytest/reconstruction_matrix_jobs.json"
    )
    jobs = json.loads(manifest.read_text())
    assert len(jobs) == 2
    assert {job["implementation"] for job in jobs} == {"lion_physics", "paper"}
    assert {job["method"] for job in jobs} == {"whole_image_diffusion"}
    assert {job["experiment"] for job in jobs} == {"ct_512_60"}


def test_pipeline_can_submit_reconstruction_when_pnp_row_is_excluded(tmp_path):
    result, sbatch_log = _run_pipeline(
        tmp_path,
        PADIS_SUBMIT_RECONSTRUCTION="1",
        PADIS_SUBMIT_PNP_TRAINING="0",
        PADIS_RECON_METHODS="baseline,admm_tv",
        PADIS_RECON_EXPERIMENTS="ct_20",
        PADIS_RECON_MAX_SAMPLES="1",
        PADIS_RECON_VERIFY="1",
    )

    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_pnp_training.sh" not in sbatch_log
    assert "slurm_PaDIS_A100_reconstruction_array.sh" in sbatch_log
    assert "slurm_PaDIS_A100_reconstruction_verify.sh" in sbatch_log
    assert "slurm_PaDIS_A100_finalise.sh" in sbatch_log
    assert "--array 0-1%10" in sbatch_log
    assert (
        "slurm_PaDIS_A100_reconstruction_verify.sh | "
        "PADIS_RECON_EXPECTED_RECORDS=2 PADIS_RECON_EXPECTED_SAMPLES=1 "
        "PADIS_RECON_EXPECTED_JOBS_JSON="
    ) in sbatch_log
    assert (
        tmp_path
        / "runs/final_real_runs/a100_reconstruction_pytest/reconstruction_matrix_jobs.json"
    ).is_file()


def test_pipeline_reconstruction_manifest_records_extra_args(tmp_path):
    result, sbatch_log = _run_pipeline(
        tmp_path,
        PADIS_SUBMIT_RECONSTRUCTION="1",
        PADIS_SUBMIT_PNP_TRAINING="0",
        PADIS_RECON_METHODS="baseline",
        PADIS_RECON_EXPERIMENTS="ct_20",
        PADIS_RECON_MAX_SAMPLES="1",
        PADIS_RECON_VERIFY="0",
        PADIS_RECON_EXTRA_ARGS="--ddnm-corrected-clip",
    )

    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_reconstruction_array.sh" in sbatch_log

    manifest = (
        tmp_path
        / "runs/final_real_runs/a100_reconstruction_pytest/reconstruction_matrix_jobs.json"
    )
    jobs = json.loads(manifest.read_text())
    assert len(jobs) == 1
    assert "--ddnm-corrected-clip" in jobs[0]["command"]


def test_pipeline_can_use_existing_pnp_checkpoint_when_training_is_disabled(tmp_path):
    checkpoint = tmp_path / "existing_pnp.pt"
    checkpoint.write_text("placeholder")

    result, sbatch_log = _run_pipeline(
        tmp_path,
        PADIS_SUBMIT_RECONSTRUCTION="1",
        PADIS_SUBMIT_PNP_TRAINING="0",
        PADIS_RECON_METHODS="pnp_admm",
        PADIS_RECON_EXPERIMENTS="ct_20",
        PADIS_PNP_CHECKPOINT=str(checkpoint),
        PADIS_PNP_NOISE_LEVEL="0.02",
        PADIS_RECON_VERIFY="0",
    )

    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_pnp_training.sh" not in sbatch_log
    assert "slurm_PaDIS_A100_reconstruction_array.sh" in sbatch_log
    assert "slurm_PaDIS_A100_reconstruction_verify.sh" not in sbatch_log
    assert "--array 0-0%10" in sbatch_log
    manifest = (
        tmp_path
        / "runs/final_real_runs/a100_reconstruction_pytest/reconstruction_matrix_jobs.json"
    )
    jobs = json.loads(manifest.read_text())
    command = jobs[0]["command"]
    assert command[command.index("--pnp-checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--pnp-noise-level") + 1] == "0.02"


def test_pipeline_reconstruction_waits_for_submitted_pnp_training(tmp_path):
    result, sbatch_log = _run_pipeline(
        tmp_path,
        PADIS_SUBMIT_RECONSTRUCTION="1",
        PADIS_SUBMIT_PNP_TRAINING="1",
        PADIS_RECON_METHODS="pnp_admm",
        PADIS_RECON_EXPERIMENTS="ct_20",
        PADIS_RECON_VERIFY="0",
    )

    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_pnp_training.sh" in sbatch_log
    assert "slurm_PaDIS_A100_reconstruction_array.sh" in sbatch_log
    assert "--dependency afterok:job104:job105" in sbatch_log


def test_pipeline_full_default_reconstruction_matrix_waits_for_training_and_pnp(
    tmp_path,
):
    result, sbatch_log = _run_pipeline(
        tmp_path,
        PADIS_SUBMIT_RECONSTRUCTION="1",
        PADIS_SUBMIT_PNP_TRAINING="1",
    )

    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_training_array.sh" in sbatch_log
    assert "slurm_PaDIS_A100_pnp_training.sh" in sbatch_log
    assert "slurm_PaDIS_A100_reconstruction_array.sh" in sbatch_log
    assert "slurm_PaDIS_A100_reconstruction_verify.sh" in sbatch_log
    assert "--array 0-108%10" in sbatch_log
    assert "--dependency afterok:job104:job105" in sbatch_log
    assert "--dependency afterok:job107" in sbatch_log
    assert (
        "slurm_PaDIS_A100_reconstruction_verify.sh | "
        "PADIS_RECON_EXPECTED_RECORDS=109 PADIS_RECON_EXPECTED_SAMPLES=25 "
        "PADIS_RECON_EXPECTED_JOBS_JSON="
    ) in sbatch_log

    manifest = (
        tmp_path
        / "runs/final_real_runs/a100_reconstruction_pytest/reconstruction_matrix_jobs.json"
    )
    jobs = json.loads(manifest.read_text())
    assert len(jobs) == 109
    assert Counter(job["method"] for job in jobs) == {
        "baseline": 5,
        "admm_tv": 5,
        "pnp_admm": 4,
        "whole_image_diffusion": 10,
        "langevin": 7,
        "predictor_corrector": 7,
        "ve_ddnm": 13,
        "patch_average": 4,
        "patch_stitch": 4,
        "padis_dps": 50,
    }
    assert {job["geometry"] for job in jobs} == {"lion"}
    assert {
        job["display_label"]
        for job in jobs
        if job["model"] == "whole_lidc_default"
        and job["method"] in {"langevin", "predictor_corrector", "ve_ddnm"}
    } == {
        "Whole image - Langevin",
        "Whole image - Predictor-corrector",
        "Whole image - VE-DDNM",
    }
    pnp_jobs = [job for job in jobs if job["method"] == "pnp_admm"]
    assert all("--checkpoint" not in job["command"] for job in pnp_jobs)
    expected_pnp_checkpoint = (
        tmp_path
        / "runs/final_real_runs/a100_training_pytest/pnp_lidc_drunet/pnp_lidc_drunet.pt"
    )
    standard_pnp_jobs = [job for job in pnp_jobs if job["matrix_group"] == "main"]
    assert all(
        job["command"][job["command"].index("--pnp-checkpoint") + 1]
        == str(expected_pnp_checkpoint)
        for job in standard_pnp_jobs
    )


def test_pipeline_uses_custom_pnp_training_checkpoint_for_reconstruction(tmp_path):
    pnp_root = tmp_path / "custom_pnp"
    expected_checkpoint = pnp_root / "denoiser_run/final_denoiser.pt"

    result, sbatch_log = _run_pipeline(
        tmp_path,
        PADIS_SUBMIT_RECONSTRUCTION="1",
        PADIS_SUBMIT_PNP_TRAINING="1",
        PADIS_RECON_METHODS="pnp_admm",
        PADIS_RECON_EXPERIMENTS="ct_20",
        PADIS_RECON_VERIFY="0",
        PADIS_PNP_OUTPUT_ROOT=str(pnp_root),
        PADIS_PNP_RUN_NAME="denoiser_run",
        PADIS_PNP_FINAL_NAME="final_denoiser.pt",
    )

    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_pnp_training.sh" in sbatch_log
    assert "slurm_PaDIS_A100_reconstruction_array.sh" in sbatch_log
    assert f"PnP denoiser: {expected_checkpoint}" in result.stdout

    manifest = (
        tmp_path
        / "runs/final_real_runs/a100_reconstruction_pytest/reconstruction_matrix_jobs.json"
    )
    jobs = json.loads(manifest.read_text())
    command = jobs[0]["command"]
    assert command[command.index("--pnp-checkpoint") + 1] == str(expected_checkpoint)
