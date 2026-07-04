import json
import os
from pathlib import Path
import subprocess
from collections import Counter

from scripts.paper_scripts.PaDIS.PaDIS_run_reconstruction_matrix import MODEL_TASKS


LION_ROOT = Path(__file__).resolve().parents[2]
SUBMIT_RECONSTRUCTION = (
    LION_ROOT / "scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_reconstruction.sh"
)
SMOKE_RECONSTRUCTION = (
    LION_ROOT
    / "scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_reconstruction_smoke.sh"
)
RECONSTRUCTION_ARRAY = (
    LION_ROOT
    / "scripts/paper_scripts/PaDIS/slurm/slurm_PaDIS_A100_reconstruction_array.sh"
)
RECONSTRUCTION_VERIFY = (
    LION_ROOT
    / "scripts/paper_scripts/PaDIS/slurm/slurm_PaDIS_A100_reconstruction_verify.sh"
)


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
printf '%s | PADIS_RECON_EXPECTED_RECORDS=%s PADIS_RECON_EXPECTED_SAMPLES=%s PADIS_RECON_EXPECTED_JOBS_JSON=%s PADIS_RECON_VERIFY_MIN_METHOD_MEAN_PSNR=%s PADIS_RECON_VERIFY_REQUIRE_METHOD_MEAN_BETTER_THAN_FDK=%s PADIS_RECON_VERIFY_REQUIRE_METHOD_EACH_BETTER_THAN_FDK=%s\\n' "$*" "${{PADIS_RECON_EXPECTED_RECORDS:-}}" "${{PADIS_RECON_EXPECTED_SAMPLES:-}}" "${{PADIS_RECON_EXPECTED_JOBS_JSON:-}}" "${{PADIS_RECON_VERIFY_MIN_METHOD_MEAN_PSNR:-}}" "${{PADIS_RECON_VERIFY_REQUIRE_METHOD_MEAN_BETTER_THAN_FDK:-}}" "${{PADIS_RECON_VERIFY_REQUIRE_METHOD_EACH_BETTER_THAN_FDK:-}}" >> {log_path!s}
printf 'job%s\\n' "$count"
"""
    )
    sbatch_path.chmod(0o755)
    return bin_dir, log_path


def _run_submitter(tmp_path, script=SUBMIT_RECONSTRUCTION, **extra_env):
    bin_dir, log_path = _install_fake_sbatch(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PADIS_RUN_ROOT": str(tmp_path / "runs"),
            "PADIS_RUN_STAMP": "pytest",
            "PADIS_TRAIN_ROOT": str(tmp_path / "training"),
            "PADIS_RECON_ROOT": str(tmp_path / "recon"),
            "PADIS_WANDB_MODE": "disabled",
            **extra_env,
        }
    )
    result = subprocess.run(
        ["bash", str(script)],
        cwd=LION_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log_text = log_path.read_text() if log_path.is_file() else ""
    return result, log_text


def test_reconstruction_verify_script_supports_explicit_mean_gate_env_names():
    text = RECONSTRUCTION_VERIFY.read_text()

    assert (
        "PADIS_RECON_VERIFY_MIN_METHOD_MEAN_PSNR"
        ":-${PADIS_RECON_VERIFY_MIN_METHOD_PSNR:-}"
    ) in text
    assert (
        "PADIS_RECON_VERIFY_MIN_METHOD_MEAN_SSIM"
        ":-${PADIS_RECON_VERIFY_MIN_METHOD_SSIM:-}"
    ) in text
    assert (
        "PADIS_RECON_VERIFY_MAX_METHOD_MEAN_MAE"
        ":-${PADIS_RECON_VERIFY_MAX_METHOD_MAE:-}"
    ) in text


def test_reconstruction_smoke_defaults_cover_validated_quality_rows(tmp_path):
    checkpoint = tmp_path / "training/patch_lidc_default/padis_lidc_256.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("placeholder")

    result, sbatch_log = _run_submitter(tmp_path, script=SMOKE_RECONSTRUCTION)

    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_reconstruction_array.sh" in sbatch_log
    assert "slurm_PaDIS_A100_reconstruction_verify.sh" in sbatch_log
    assert "--array 0-5%3" in sbatch_log
    assert "PADIS_RECON_EXPECTED_RECORDS=6" in sbatch_log
    assert "PADIS_RECON_EXPECTED_SAMPLES=1" in sbatch_log
    assert (
        "PADIS_RECON_VERIFY_REQUIRE_METHOD_MEAN_BETTER_THAN_FDK="
        "admm_tv,padis_dps,langevin,predictor_corrector,ve_ddnm"
    ) in sbatch_log
    assert (
        "PADIS_RECON_VERIFY_REQUIRE_METHOD_EACH_BETTER_THAN_FDK="
        "admm_tv,padis_dps,langevin,predictor_corrector,ve_ddnm"
    ) in sbatch_log
    assert (
        "PADIS_RECON_VERIFY_MIN_METHOD_MEAN_PSNR="
        "admm_tv=28 padis_dps=33 langevin=32 predictor_corrector=29 ve_ddnm=32"
    ) in sbatch_log

    manifest = tmp_path / "recon/reconstruction_matrix_jobs.json"
    jobs = json.loads(manifest.read_text())
    assert [job["method"] for job in jobs] == [
        "baseline",
        "admm_tv",
        "padis_dps",
        "langevin",
        "predictor_corrector",
        "ve_ddnm",
    ]
    assert {job["implementation"] for job in jobs} == {"lion_physics"}


def test_standalone_reconstruction_submitter_writes_manifest_and_verifier_env(
    tmp_path,
):
    result, sbatch_log = _run_submitter(
        tmp_path,
        PADIS_RECON_METHODS="baseline,admm_tv",
        PADIS_RECON_EXPERIMENTS="ct_20",
        PADIS_RECON_MAX_SAMPLES="1",
        PADIS_RECON_VERIFY="1",
        PADIS_RECON_GEOMETRIES="lion",
        PADIS_RECON_IMPLEMENTATIONS="method_default",
    )

    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_reconstruction_array.sh" in sbatch_log
    assert "slurm_PaDIS_A100_reconstruction_verify.sh" in sbatch_log
    assert "--array 0-1%10" in sbatch_log
    assert (
        "slurm_PaDIS_A100_reconstruction_verify.sh | "
        "PADIS_RECON_EXPECTED_RECORDS=2 PADIS_RECON_EXPECTED_SAMPLES=1 "
        "PADIS_RECON_EXPECTED_JOBS_JSON="
    ) in sbatch_log

    manifest = tmp_path / "recon/reconstruction_matrix_jobs.json"
    assert manifest.is_file()
    jobs = json.loads(manifest.read_text())
    assert [job["method"] for job in jobs] == ["baseline", "admm_tv"]
    assert {job["experiment"] for job in jobs} == {"ct_20"}


def test_standalone_reconstruction_submitter_full_default_matrix_with_checkpoints(
    tmp_path,
):
    training_root = tmp_path / "training"
    for model_task in MODEL_TASKS:
        checkpoint = training_root / model_task.name / model_task.checkpoint_name
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text("placeholder")
    pnp_checkpoint = training_root / "pnp_lidc_drunet/pnp_lidc_drunet.pt"
    pnp_checkpoint.parent.mkdir(parents=True)
    pnp_checkpoint.write_text("placeholder")

    result, sbatch_log = _run_submitter(tmp_path, PADIS_RECON_VERIFY="1")

    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_reconstruction_array.sh" in sbatch_log
    assert "slurm_PaDIS_A100_reconstruction_verify.sh" in sbatch_log
    assert "--array 0-100%10" in sbatch_log
    assert "PADIS_RECON_EXPECTED_RECORDS=101" in sbatch_log
    assert "PADIS_RECON_EXPECTED_SAMPLES=25" in sbatch_log

    manifest = tmp_path / "recon/reconstruction_matrix_jobs.json"
    jobs = json.loads(manifest.read_text())
    assert len(jobs) == 101
    assert Counter(job["method"] for job in jobs) == {
        "baseline": 5,
        "admm_tv": 5,
        "pnp_admm": 2,
        "whole_image_diffusion": 10,
        "langevin": 7,
        "predictor_corrector": 7,
        "ve_ddnm": 7,
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
    assert {job["implementation"] for job in jobs} == {
        "lion_physics",
        "public_repo",
        "paper",
    }
    no_prior_jobs = [
        job for job in jobs if job["method"] in {"baseline", "admm_tv", "pnp_admm"}
    ]
    assert all(job["checkpoint"] == "" for job in no_prior_jobs)
    assert all("--checkpoint" not in job["command"] for job in no_prior_jobs)
    pnp_jobs = [job for job in jobs if job["method"] == "pnp_admm"]
    assert len(pnp_jobs) == 2
    assert all(
        job["command"][job["command"].index("--pnp-checkpoint") + 1]
        == str(pnp_checkpoint)
        for job in pnp_jobs
    )


def test_reconstruction_array_passes_dash_prefixed_extra_args_with_equals():
    text = RECONSTRUCTION_ARRAY.read_text()

    assert 'CMD+=("--reconstruction-arg=$item")' in text


def test_reconstruction_array_defaults_to_fragmentation_resistant_cuda_allocator():
    text = RECONSTRUCTION_ARRAY.read_text()

    assert (
        'PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"'
        in text
    )
    assert "export PYTORCH_CUDA_ALLOC_CONF" in text


def test_standalone_reconstruction_submitter_records_extra_reconstruction_args(
    tmp_path,
):
    result, sbatch_log = _run_submitter(
        tmp_path,
        PADIS_RECON_METHODS="baseline",
        PADIS_RECON_EXPERIMENTS="ct_20",
        PADIS_RECON_MAX_SAMPLES="1",
        PADIS_RECON_VERIFY="0",
        PADIS_RECON_EXTRA_ARGS="--ddnm-corrected-clip",
    )

    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_reconstruction_array.sh" in sbatch_log

    manifest = tmp_path / "recon/reconstruction_matrix_jobs.json"
    jobs = json.loads(manifest.read_text())
    assert len(jobs) == 1
    assert {job["implementation"] for job in jobs} == {"lion_physics"}
    assert "--ddnm-corrected-clip" in jobs[0]["command"]


def test_standalone_reconstruction_submitter_can_skip_verifier(tmp_path):
    result, sbatch_log = _run_submitter(
        tmp_path,
        PADIS_RECON_METHODS="baseline",
        PADIS_RECON_EXPERIMENTS="ct_20",
        PADIS_RECON_MAX_SAMPLES="1",
        PADIS_RECON_VERIFY="0",
    )

    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_reconstruction_array.sh" in sbatch_log
    assert "slurm_PaDIS_A100_reconstruction_verify.sh" not in sbatch_log
    assert "--array 0-0%10" in sbatch_log


def test_standalone_reconstruction_submitter_blocks_pnp_without_denoiser(tmp_path):
    result, sbatch_log = _run_submitter(
        tmp_path,
        PADIS_RECON_METHODS="pnp_admm",
        PADIS_RECON_EXPERIMENTS="ct_20",
        PADIS_RECON_VERIFY="0",
    )

    assert result.returncode == 1
    assert "matrix containing pnp_admm" in result.stderr
    assert "slurm_PaDIS_A100_reconstruction_array.sh" not in sbatch_log


def test_standalone_reconstruction_submitter_blocks_missing_diffusion_checkpoint(
    tmp_path,
):
    result, sbatch_log = _run_submitter(
        tmp_path,
        PADIS_RECON_METHODS="padis_dps",
        PADIS_RECON_EXPERIMENTS="ct_20",
        PADIS_RECON_VERIFY="0",
    )

    assert result.returncode == 1
    assert "Missing checkpoint for patch_lidc_default" in result.stderr
    assert "slurm_PaDIS_A100_reconstruction_array.sh" not in sbatch_log


def test_standalone_reconstruction_submitter_blocks_spaced_pnp_selection(tmp_path):
    result, sbatch_log = _run_submitter(
        tmp_path,
        PADIS_RECON_METHODS="baseline, pnp_admm",
        PADIS_RECON_EXPERIMENTS="ct_20",
        PADIS_RECON_VERIFY="0",
    )

    assert result.returncode == 1
    assert "matrix containing pnp_admm" in result.stderr
    assert "slurm_PaDIS_A100_reconstruction_array.sh" not in sbatch_log


def test_standalone_reconstruction_submitter_records_pnp_overrides(tmp_path):
    checkpoint = tmp_path / "existing_pnp.pt"
    checkpoint.write_text("placeholder")

    result, sbatch_log = _run_submitter(
        tmp_path,
        PADIS_RECON_METHODS="pnp_admm",
        PADIS_RECON_EXPERIMENTS="ct_20",
        PADIS_RECON_MAX_SAMPLES="1",
        PADIS_RECON_VERIFY="0",
        PADIS_PNP_CHECKPOINT=str(checkpoint),
        PADIS_PNP_NOISE_LEVEL="0.02",
    )

    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_reconstruction_array.sh" in sbatch_log

    manifest = tmp_path / "recon/reconstruction_matrix_jobs.json"
    jobs = json.loads(manifest.read_text())
    assert len(jobs) == 1
    command = jobs[0]["command"]
    assert command[command.index("--pnp-checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--pnp-noise-level") + 1] == "0.02"


def test_standalone_reconstruction_submitter_derives_custom_pnp_checkpoint(tmp_path):
    pnp_output_root = tmp_path / "custom_pnp"
    checkpoint = pnp_output_root / "denoiser_run/final_denoiser.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("placeholder")

    result, sbatch_log = _run_submitter(
        tmp_path,
        PADIS_RECON_METHODS="pnp_admm",
        PADIS_RECON_EXPERIMENTS="ct_20",
        PADIS_RECON_MAX_SAMPLES="1",
        PADIS_RECON_VERIFY="0",
        PADIS_PNP_OUTPUT_ROOT=str(pnp_output_root),
        PADIS_PNP_RUN_NAME="denoiser_run",
        PADIS_PNP_FINAL_NAME="final_denoiser.pt",
    )

    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_reconstruction_array.sh" in sbatch_log

    manifest = tmp_path / "recon/reconstruction_matrix_jobs.json"
    jobs = json.loads(manifest.read_text())
    command = jobs[0]["command"]
    assert command[command.index("--pnp-checkpoint") + 1] == str(checkpoint)


def test_standalone_reconstruction_submitter_rejects_off_paper_matrix(tmp_path):
    result, sbatch_log = _run_submitter(
        tmp_path,
        PADIS_RECON_METHODS="whole_image_diffusion",
        PADIS_RECON_EXPERIMENTS="ct_512_60",
        PADIS_RECON_VERIFY="0",
    )

    assert result.returncode == 1
    assert "not part of the paper reconstruction matrix" in result.stderr
    assert "slurm_PaDIS_A100_reconstruction_array.sh" not in sbatch_log


def test_standalone_reconstruction_submitter_can_allow_off_paper_matrix(tmp_path):
    checkpoint = (
        tmp_path / "training/whole_lidc_default/whole_image_lidc_256_min_val.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("placeholder")

    result, sbatch_log = _run_submitter(
        tmp_path,
        PADIS_RECON_METHODS="whole_image_diffusion",
        PADIS_RECON_EXPERIMENTS="ct_512_60",
        PADIS_RECON_ALLOW_OFF_PAPER_EXPERIMENTS="1",
        PADIS_RECON_VERIFY="0",
    )

    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_reconstruction_array.sh" in sbatch_log
    assert "--array 0-1%10" in sbatch_log

    manifest = tmp_path / "recon/reconstruction_matrix_jobs.json"
    jobs = json.loads(manifest.read_text())
    assert len(jobs) == 2
    assert {job["implementation"] for job in jobs} == {"lion_physics", "paper"}
    assert {job["method"] for job in jobs} == {"whole_image_diffusion"}
    assert {job["experiment"] for job in jobs} == {"ct_512_60"}


def test_reconstruction_array_derives_custom_pnp_checkpoint_before_command():
    script = RECONSTRUCTION_ARRAY.read_text()

    assert "PADIS_PNP_OUTPUT_ROOT" in script
    assert "PADIS_PNP_RUN_NAME" in script
    assert "PADIS_PNP_FINAL_NAME" in script
    assert 'PADIS_PNP_CHECKPOINT="$PADIS_PNP_ROOT/$PADIS_PNP_FINAL_NAME"' in script
    assert 'CMD+=(--pnp-checkpoint "$PADIS_PNP_CHECKPOINT")' in script
    assert script.index(
        'PADIS_PNP_CHECKPOINT="$PADIS_PNP_ROOT/$PADIS_PNP_FINAL_NAME"'
    ) < script.index("CMD=(")
