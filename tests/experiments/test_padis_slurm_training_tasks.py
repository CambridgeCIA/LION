import os
from pathlib import Path
import subprocess

from scripts.paper_scripts.PaDIS.PaDIS_run_reconstruction_matrix import MODEL_TASKS


LION_ROOT = Path(__file__).resolve().parents[2]
COMMON = LION_ROOT / "scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh"
SUBMIT_ALL_TRAINING = (
    LION_ROOT / "scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_all_training.sh"
)
TRAINING_ARRAY = (
    LION_ROOT / "scripts/paper_scripts/PaDIS/slurm/slurm_PaDIS_A100_training_array.sh"
)
LEGACY_LIDC_256 = (
    LION_ROOT / "scripts/paper_scripts/PaDIS/slurm/slurm_PaDIS_LIDC_256.sh"
)
PNP_TRAINING = (
    LION_ROOT / "scripts/paper_scripts/PaDIS/slurm/slurm_PaDIS_A100_pnp_training.sh"
)


def _read_training_tasks():
    script = f"""
set -euo pipefail
source {COMMON!s}
padis_init_training_tasks
if [ "${{#PADIS_TASK_NAMES[@]}}" -ne "${{#PADIS_TASK_ENGINES[@]}}" ]; then
        exit 2
fi
if [ "${{#PADIS_TASK_NAMES[@]}}" -ne "${{#PADIS_TASK_BATCH_SIZES[@]}}" ]; then
        exit 3
fi
if [ "${{#PADIS_TASK_NAMES[@]}}" -ne "${{#PADIS_TASK_ARGUMENTS[@]}}" ]; then
        exit 4
fi
printf 'COUNT\\t%s\\n' "$(padis_training_task_count)"
for i in "${{!PADIS_TASK_NAMES[@]}}"; do
        printf '%s\\t%s\\t%s\\t%s\\n' "${{PADIS_TASK_NAMES[$i]}}" "${{PADIS_TASK_ENGINES[$i]}}" "${{PADIS_TASK_BATCH_SIZES[$i]}}" "${{PADIS_TASK_ARGUMENTS[$i]}}"
done
"""
    result = subprocess.run(
        ["bash", "-lc", script],
        cwd=LION_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    rows = result.stdout.strip().splitlines()
    count = int(rows[0].split("\t")[1])
    tasks = {}
    for row in rows[1:]:
        name, engine, batch_size, arguments = row.split("\t", 3)
        tasks[name] = {
            "engine": engine,
            "batch_size": int(batch_size),
            "arguments": arguments,
        }
    return count, tasks


def test_a100_training_tasks_cover_reconstruction_model_families():
    count, tasks = _read_training_tasks()
    model_names = {task.name for task in MODEL_TASKS}

    assert count == len(tasks)
    assert set(tasks) == model_names

    for name, task in tasks.items():
        assert f"--run-name {name}" in task["arguments"]
        assert task["batch_size"] > 0


def test_a100_training_tasks_use_expected_engines_for_special_priors():
    _, tasks = _read_training_tasks()

    assert tasks["patch_lidc_512"]["engine"] == "lidc512"
    assert "--max-slices-per-patient 4" in tasks["patch_lidc_512"]["arguments"]

    assert tasks["whole_lidc_default"]["engine"] == "lidc256"
    assert "--prior-mode whole-image" in tasks["whole_lidc_default"]["arguments"]
    assert "--max-slices-per-patient 4" in tasks["whole_lidc_default"]["arguments"]

    assert tasks["whole_lidc_full"]["engine"] == "lidc256"
    assert "--prior-mode whole-image" in tasks["whole_lidc_full"]["arguments"]
    assert "--full-lidc" in tasks["whole_lidc_full"]["arguments"]

    assert tasks["patch_lidc_no_pos_default"]["engine"] == "lidc256"
    assert "--no-position-channels" in tasks["patch_lidc_no_pos_default"]["arguments"]


def test_a100_training_tasks_produce_matrix_checkpoint_filenames():
    _, tasks = _read_training_tasks()

    for model_task in MODEL_TASKS:
        task = tasks[model_task.name]
        arguments = task["arguments"]
        if task["engine"] == "lidc512":
            produced_checkpoint = "padis_lidc_512.pt"
        elif "--prior-mode whole-image" in arguments:
            produced_checkpoint = "whole_image_lidc_256_min_val.pt"
        else:
            produced_checkpoint = "padis_lidc_256.pt"

        assert produced_checkpoint == model_task.checkpoint_name


def test_pnp_training_default_final_checkpoint_matches_reconstruction_matrix():
    script = PNP_TRAINING.read_text()

    assert 'PADIS_PNP_RUN_NAME="${PADIS_PNP_RUN_NAME:-pnp_lidc_drunet}"' in script
    assert (
        'PADIS_PNP_FINAL_NAME="${PADIS_PNP_FINAL_NAME:-pnp_lidc_drunet.pt}"' in script
    )
    assert '--final-name "$PADIS_PNP_FINAL_NAME"' in script


def test_slurm_training_defaults_to_lion_dev_with_padis_dev_fallback():
    common = COMMON.read_text()
    legacy_lidc_256 = LEGACY_LIDC_256.read_text()

    for script in (common, legacy_lidc_256):
        assert 'LION_MAMBA_ENV="${LION_MAMBA_ENV:-lion-dev}"' in script
        assert (
            'LION_MAMBA_ENV_FALLBACKS="${LION_MAMBA_ENV_FALLBACKS:-padis-dev}"'
            in script
        )


def test_a100_training_array_keeps_wandb_artifacts_enabled_by_default():
    script = TRAINING_ARRAY.read_text()

    assert 'PADIS_WANDB_MODE="${PADIS_WANDB_MODE:-online}"' in script
    assert 'PADIS_NO_WANDB="${PADIS_NO_WANDB:-0}"' in script
    assert 'NO_WANDB_ARTIFACT="${PADIS_NO_WANDB_ARTIFACT:-0}"' in script
    assert 'if [ "$NO_WANDB_ARTIFACT" = "1" ]; then' in script
    assert "wandb_args+=(--no-wandb-artifact)" in script
    assert script.count('CMD+=("${wandb_args[@]}")') >= 2


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
printf '%s | PADIS_TRAIN_ROOT=%s PADIS_PNP_OUTPUT_ROOT=%s PADIS_PNP_RUN_NAME=%s\\n' "$*" "${{PADIS_TRAIN_ROOT:-}}" "${{PADIS_PNP_OUTPUT_ROOT:-}}" "${{PADIS_PNP_RUN_NAME:-}}" >> {log_path!s}
printf 'job%s\\n' "$count"
"""
    )
    sbatch_path.chmod(0o755)
    return bin_dir, log_path


def _run_all_training_submitter(tmp_path, **extra_env):
    bin_dir, log_path = _install_fake_sbatch(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PADIS_RUN_ROOT": str(tmp_path / "runs"),
        "PADIS_RUN_STAMP": "pytest",
        "PADIS_WANDB_MODE": "disabled",
        **extra_env,
    }
    result = subprocess.run(
        ["bash", str(SUBMIT_ALL_TRAINING)],
        cwd=LION_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log_text = log_path.read_text() if log_path.is_file() else ""
    return result, log_text


def test_all_training_submitter_launches_diffusion_array_and_pnp_denoiser(tmp_path):
    result, sbatch_log = _run_all_training_submitter(tmp_path)

    expected_train_root = tmp_path / "runs/final_real_runs/a100_training_pytest"
    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_training_array.sh" in sbatch_log
    assert "slurm_PaDIS_A100_pnp_training.sh" in sbatch_log
    assert "--array 0-9%10" in sbatch_log
    assert f"PADIS_TRAIN_ROOT={expected_train_root}" in sbatch_log
    assert f"PADIS_PNP_OUTPUT_ROOT={expected_train_root}" in sbatch_log
    assert "PADIS_PNP_RUN_NAME=pnp_lidc_drunet" in sbatch_log
    assert "PaDIS diffusion training array: job101" in result.stdout
    assert "PnP denoiser training: job102" in result.stdout
    assert (
        f"Expected PnP checkpoint: {expected_train_root}/pnp_lidc_drunet/"
        "pnp_lidc_drunet.pt"
    ) in result.stdout


def test_all_training_submitter_is_executable():
    assert SUBMIT_ALL_TRAINING.stat().st_mode & 0o111


def test_all_training_submitter_can_skip_pnp_training(tmp_path):
    result, sbatch_log = _run_all_training_submitter(
        tmp_path,
        PADIS_SUBMIT_PNP_TRAINING="0",
    )

    assert result.returncode == 0, result.stderr
    assert "slurm_PaDIS_A100_training_array.sh" in sbatch_log
    assert "slurm_PaDIS_A100_pnp_training.sh" not in sbatch_log
    assert "PnP denoiser training: skipped" in result.stdout
