import os
from pathlib import Path
import subprocess


LION_ROOT = Path(__file__).resolve().parents[2]
SUBMIT_PNP = (
    LION_ROOT
    / "scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm/submit_PaDIS_A100_pnp_training.sh"
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
printf '%s | PADIS_TRAIN_ROOT=%s PADIS_PNP_OUTPUT_ROOT=%s PADIS_PNP_RUN_NAME=%s\\n' "$*" "${{PADIS_TRAIN_ROOT:-}}" "${{PADIS_PNP_OUTPUT_ROOT:-}}" "${{PADIS_PNP_RUN_NAME:-}}" >> {log_path!s}
printf 'job%s\\n' "$count"
"""
    )
    sbatch_path.chmod(0o755)
    return bin_dir, log_path


def _run_submitter(tmp_path, **extra_env):
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
        ["bash", str(SUBMIT_PNP)],
        cwd=LION_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log_text = log_path.read_text() if log_path.is_file() else ""
    return result, log_text


def test_standalone_pnp_submitter_uses_training_root_defaults(tmp_path):
    result, sbatch_log = _run_submitter(tmp_path)

    assert result.returncode == 0, result.stderr
    expected_train_root = tmp_path / "runs/final_real_runs/a100_training_pytest"
    assert "slurm_PaDIS_A100_pnp_training.sh" in sbatch_log
    assert f"PADIS_TRAIN_ROOT={expected_train_root}" in sbatch_log
    assert f"PADIS_PNP_OUTPUT_ROOT={expected_train_root}" in sbatch_log
    assert "PADIS_PNP_RUN_NAME=pnp_lidc_drunet" in sbatch_log
    assert (
        f"Expected checkpoint: {expected_train_root}/pnp_lidc_drunet/"
        "pnp_lidc_drunet.pt"
    ) in result.stdout


def test_standalone_pnp_submitter_preserves_explicit_output_root_and_run_name(tmp_path):
    output_root = tmp_path / "custom_pnp"

    result, sbatch_log = _run_submitter(
        tmp_path,
        PADIS_PNP_OUTPUT_ROOT=str(output_root),
        PADIS_PNP_RUN_NAME="custom_run",
        PADIS_PNP_FINAL_NAME="custom_final.pt",
    )

    assert result.returncode == 0, result.stderr
    assert f"PADIS_PNP_OUTPUT_ROOT={output_root}" in sbatch_log
    assert "PADIS_PNP_RUN_NAME=custom_run" in sbatch_log
    assert f"Expected checkpoint: {output_root}/custom_run/custom_final.pt" in (
        result.stdout
    )


def test_pnp_slurm_job_forwards_training_cli_options():
    script = (
        LION_ROOT
        / "scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm/slurm_PaDIS_A100_pnp_training.sh"
    ).read_text()

    expected_flags = {
        "--output-root",
        "--run-name",
        "--batch-size",
        "--epochs",
        "--learning-rate",
        "--beta1",
        "--beta2",
        "--noise-min",
        "--noise-max",
        "--image-scaling",
        "--max-slices-per-patient",
        "--int-channels",
        "--n-blocks",
        "--patches-per-image",
        "--validation-every",
        "--checkpoint-every",
        "--seed",
        "--device",
        "--num-workers",
        "--final-name",
        "--checkpoint-pattern",
        "--validation-name",
        "--full-lidc",
        "--max-train-samples",
        "--max-validation-samples",
        "--use-noise-level",
        "--patch-size",
        "--data-folder",
    }

    for flag in expected_flags:
        assert flag in script
