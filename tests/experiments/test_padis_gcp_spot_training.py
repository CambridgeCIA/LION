import os
from pathlib import Path
import subprocess


LION_ROOT = Path(__file__).resolve().parents[2]
GCP_RUNNER = (
    LION_ROOT / "scripts/paper_scripts/PaDIS/gcp/run_PaDIS_GCP_spot_training.sh"
)


def _run_gcp_dry_run(tmp_path, task_order):
    data_root = tmp_path / "Datasets"
    run_root = data_root / "experiments/PaDIS"
    env = {
        **os.environ,
        "LION_DATA_PATH": str(data_root),
        "LION_EXPERIMENTS_PATH": str(data_root / "experiments"),
        "PADIS_RUN_ROOT": str(run_root),
        "PADIS_GCP_SKIP_ENV_ACTIVATE": "1",
        "PADIS_GCP_DRY_RUN": "1",
        "PADIS_GCP_STAGE_CACHES": "0",
        "PADIS_GCP_TASK_ORDER": task_order,
        "PADIS_GCP_GPU_IDS": "0,1",
        "PADIS_NO_WANDB": "1",
        "PADIS_WANDB_MODE": "disabled",
    }
    result = subprocess.run(
        ["bash", str(GCP_RUNNER)],
        cwd=LION_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    train_root = run_root / "final_real_runs/gcp_spot_training"
    return result, train_root


def _command_text(train_root, task_name):
    command_path = train_root / ".gcp_spot_dry_run/logs" / f"{task_name}.command.txt"
    return command_path.read_text()


def test_gcp_spot_runner_is_executable():
    assert GCP_RUNNER.stat().st_mode & 0o111


def test_gcp_spot_runner_dry_run_builds_expected_training_commands(tmp_path):
    result, train_root = _run_gcp_dry_run(
        tmp_path,
        "whole_lidc_default,patch_lidc_default,pnp_lidc_drunet",
    )

    assert result.returncode == 0, result.stderr

    whole_command = _command_text(train_root, "whole_lidc_default")
    assert "PaDIS_LIDC_256.py" in whole_command
    assert "--prior-mode whole-image" in whole_command
    assert "--max-train-seconds 64800" in whole_command
    assert "--checkpoint-interval-seconds 600" in whole_command
    assert "--keep-final-periodic-checkpoints 1" in whole_command
    assert "--cache-folder /mnt/ram-disk/lion_lidc_cache_256" in whole_command

    patch_command = _command_text(train_root, "patch_lidc_default")
    assert "PaDIS_LIDC_256.py" in patch_command
    assert "--max-train-seconds 21600" in patch_command
    assert "--checkpoint-interval-seconds 600" in patch_command
    assert "--keep-final-periodic-checkpoints 1" in patch_command

    pnp_command = _command_text(train_root, "pnp_lidc_drunet")
    assert "PaDIS_LIDC_PnP_denoiser.py" in pnp_command
    assert "--max-train-seconds" not in pnp_command
    assert "--final-full-name pnp_lidc_drunet_full.pt" in pnp_command
    assert "--checkpoint-interval-seconds 600" in pnp_command


def test_gcp_spot_runner_records_final_and_full_checkpoints(tmp_path):
    result, train_root = _run_gcp_dry_run(
        tmp_path,
        "patch_lidc_512,pnp_lidc_drunet",
    )

    assert result.returncode == 0, result.stderr

    patch_done = train_root / ".gcp_spot_dry_run/done/patch_lidc_512.done"
    assert "padis_lidc_512.pt" in patch_done.read_text()
    assert "padis_lidc_512_full.pt" in patch_done.read_text()

    pnp_done = train_root / ".gcp_spot_dry_run/done/pnp_lidc_drunet.done"
    assert "pnp_lidc_drunet.pt" in pnp_done.read_text()
    assert "pnp_lidc_drunet_full.pt" in pnp_done.read_text()
