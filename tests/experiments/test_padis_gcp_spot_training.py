import os
from pathlib import Path
import subprocess


LION_ROOT = Path(__file__).resolve().parents[2]
GCP_RUNNER = (
    LION_ROOT / "scripts/paper_scripts/PaDIS/gcp/run_PaDIS_GCP_spot_training.sh"
)
GCP_STARTUP = LION_ROOT / "scripts/paper_scripts/PaDIS/gcp/padis_gcp_spot_startup.sh"
GCP_METADATA_STARTUP = (
    LION_ROOT / "scripts/paper_scripts/PaDIS/gcp/padis_gcp_spot_metadata_startup.sh"
)
GCP_SHUTDOWN = LION_ROOT / "scripts/paper_scripts/PaDIS/gcp/padis_gcp_spot_shutdown.sh"
DEFAULT_GCP_RUN_NAME = "PaDIS-Reproduction-GCP"


def _run_gcp_dry_run(tmp_path, task_order, extra_env=None, runtime_seconds=None):
    data_root = tmp_path / "Datasets"
    run_root = data_root / "experiments/PaDIS"
    run_name = (extra_env or {}).get("PADIS_GCP_RUN_NAME", DEFAULT_GCP_RUN_NAME)
    train_root = run_root / "final_real_runs" / run_name
    if runtime_seconds:
        runtime_dir = train_root / ".gcp_spot_dry_run/runtime"
        runtime_dir.mkdir(parents=True)
        for task_name, seconds in runtime_seconds.items():
            (runtime_dir / f"{task_name}.seconds").write_text(f"{seconds}\n")
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
    if extra_env is not None:
        env.update(extra_env)
    result = subprocess.run(
        ["bash", str(GCP_RUNNER)],
        cwd=LION_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, train_root


def _command_text(train_root, task_name):
    command_path = train_root / ".gcp_spot_dry_run/logs" / f"{task_name}.command.txt"
    return command_path.read_text()


def test_gcp_spot_runner_is_executable():
    assert GCP_RUNNER.stat().st_mode & 0o111
    assert GCP_STARTUP.stat().st_mode & 0o111
    assert GCP_METADATA_STARTUP.stat().st_mode & 0o111
    assert GCP_SHUTDOWN.stat().st_mode & 0o111


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
    assert "--checkpoint-interval-seconds 300" in whole_command
    assert "--keep-final-periodic-checkpoints 1" in whole_command
    assert "--cache-folder /mnt/ram-disk/lion_lidc_cache_256" in whole_command

    patch_command = _command_text(train_root, "patch_lidc_default")
    assert "PaDIS_LIDC_256.py" in patch_command
    assert "--max-train-seconds 21600" in patch_command
    assert "--checkpoint-interval-seconds 300" in patch_command
    assert "--keep-final-periodic-checkpoints 1" in patch_command

    pnp_command = _command_text(train_root, "pnp_lidc_drunet")
    assert "PaDIS_LIDC_PnP_denoiser.py" in pnp_command
    assert "--max-train-seconds" not in pnp_command
    assert "--final-full-name pnp_lidc_drunet_full.pt" in pnp_command
    assert "--checkpoint-interval-seconds 300" in pnp_command


def test_gcp_spot_runner_default_run_name_and_wandb_prefix(tmp_path):
    result, train_root = _run_gcp_dry_run(
        tmp_path,
        "patch_lidc_default",
        extra_env={
            "PADIS_NO_WANDB": "0",
            "PADIS_WANDB_MODE": "online",
        },
    )

    assert result.returncode == 0, result.stderr
    assert train_root.name == DEFAULT_GCP_RUN_NAME
    patch_command = _command_text(train_root, "patch_lidc_default")
    assert "--wandb-name PaDIS-Reproduction-GCP_patch_lidc_default" in patch_command


def test_gcp_spot_runner_defaults_to_one_gpu(tmp_path):
    result, train_root = _run_gcp_dry_run(
        tmp_path,
        "patch_lidc_default,patch_lidc_512",
    )

    assert result.returncode == 0, result.stderr
    manifest = train_root / ".gcp_spot_dry_run/manifest.txt"
    assert "gpu_ids=0\n" in manifest.read_text()


def test_gcp_spot_runner_subtracts_previous_runtime_from_wall_budget(tmp_path):
    result, train_root = _run_gcp_dry_run(
        tmp_path,
        "patch_lidc_default,whole_lidc_default",
        extra_env={
            "PADIS_GCP_PATCH_TRAIN_TIME": "120",
            "PADIS_GCP_WHOLE_TRAIN_TIME": "300",
        },
        runtime_seconds={
            "patch_lidc_default": 45,
            "whole_lidc_default": 260,
        },
    )

    assert result.returncode == 0, result.stderr

    patch_command = _command_text(train_root, "patch_lidc_default")
    whole_command = _command_text(train_root, "whole_lidc_default")
    assert "--max-train-seconds 75" in patch_command
    assert "--max-train-seconds 40" in whole_command


def test_gcp_spot_runner_uses_finalize_window_when_budget_is_exhausted(tmp_path):
    result, train_root = _run_gcp_dry_run(
        tmp_path,
        "patch_lidc_default",
        extra_env={
            "PADIS_GCP_PATCH_TRAIN_TIME": "120",
            "PADIS_GCP_FINALIZE_SECONDS": "17",
        },
        runtime_seconds={"patch_lidc_default": 130},
    )

    assert result.returncode == 0, result.stderr

    patch_command = _command_text(train_root, "patch_lidc_default")
    assert "--max-train-seconds 17" in patch_command


def test_gcp_shutdown_hook_refreshes_active_runtime_ledger(tmp_path):
    state_dir = tmp_path / "state"
    running_dir = state_dir / "running"
    runtime_dir = state_dir / "runtime"
    running_dir.mkdir(parents=True)
    runtime_dir.mkdir()
    start_epoch = int(subprocess.check_output(["date", "+%s"], text=True).strip()) - 8
    (running_dir / "patch_lidc_default.running").write_text(
        "\n".join(
            [
                "task=patch_lidc_default",
                "gpu=0",
                f"start_epoch={start_epoch}",
                "start_elapsed=45",
            ]
        )
        + "\n"
    )
    env = {
        **os.environ,
        "PADIS_GCP_STATE_DIR": str(state_dir),
        "PADIS_GCP_SHUTDOWN_DRY_RUN": "1",
        "PADIS_GCP_SHUTDOWN_GRACE_SECONDS": "1",
    }

    result = subprocess.run(
        ["bash", str(GCP_SHUTDOWN)],
        cwd=LION_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    elapsed = int((runtime_dir / "patch_lidc_default.seconds").read_text())
    assert elapsed >= 53


def test_gcp_startup_hook_dry_run_prepares_env_and_runner_command(tmp_path):
    data_mount = tmp_path / "mnt_data"
    lion_root = data_mount / "LION"
    runner_dir = lion_root / "scripts/paper_scripts/PaDIS/gcp"
    runner_dir.mkdir(parents=True)
    runner_path = runner_dir / "run_PaDIS_GCP_spot_training.sh"
    runner_path.symlink_to(GCP_RUNNER)
    run_root = data_mount / "Datasets/experiments/PaDIS"
    env = {
        **os.environ,
        "PADIS_DATA_MOUNT": str(data_mount),
        "LION_ROOT": str(lion_root),
        "LION_DATA_PATH": str(data_mount / "Datasets"),
        "LION_EXPERIMENTS_PATH": str(data_mount / "Datasets/experiments"),
        "PADIS_RUN_ROOT": str(run_root),
        "PADIS_RAM_DISK": str(tmp_path / "ram-disk"),
        "PADIS_GCP_STARTUP_DRY_RUN": "1",
        "PADIS_GCP_REQUIRE_NVIDIA_SMI": "0",
        "PADIS_GCP_GPU_IDS": "0,1,2,3",
        "PADIS_GCP_TASK_ORDER": "patch_lidc_default",
    }

    result = subprocess.run(
        ["bash", str(GCP_STARTUP)],
        cwd=LION_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    train_root = run_root / "final_real_runs" / DEFAULT_GCP_RUN_NAME
    env_file = train_root / ".gcp_spot/startup_env.sh"
    assert env_file.is_file()
    env_text = env_file.read_text()
    assert f"LION_ROOT={lion_root}" in env_text
    assert f"PADIS_TRAIN_ROOT={train_root}" in env_text
    assert "PADIS_GCP_GPU_IDS=0\\,1\\,2\\,3" in env_text
    assert "PADIS_GCP_TASK_ORDER=patch_lidc_default" in env_text
    mem_total_kb = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemTotal:")
    )
    expected_ramdisk_size = min(
        mem_total_kb * 1024 // 2,
        100 * 1024 * 1024 * 1024,
    )
    assert f"size {expected_ramdisk_size}" in result.stdout
    assert "Dry-run runner command" in result.stdout
    assert str(runner_path) in result.stdout


def test_gcp_metadata_startup_bootstrap_delegates_after_data_mount(tmp_path):
    data_mount = tmp_path / "mnt_data"
    lion_root = data_mount / "LION"
    startup_dir = lion_root / "scripts/paper_scripts/PaDIS/gcp"
    startup_dir.mkdir(parents=True)
    startup_hook = startup_dir / "padis_gcp_spot_startup.sh"
    startup_hook.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'delegated LION_ROOT=%s\\n' \"$LION_ROOT\"\n"
        "printf 'delegated DATA=%s\\n' \"$PADIS_DATA_MOUNT\"\n"
    )
    startup_hook.chmod(0o755)
    env = {
        **os.environ,
        "PADIS_DATA_MOUNT": str(data_mount),
        "LION_ROOT": str(lion_root),
        "PADIS_GCP_STARTUP_DRY_RUN": "1",
    }

    result = subprocess.run(
        ["bash", str(GCP_METADATA_STARTUP)],
        cwd=LION_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "metadata startup bootstrap starting" in result.stdout
    assert "delegated LION_ROOT=" in result.stdout
    assert str(lion_root) in result.stdout


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
