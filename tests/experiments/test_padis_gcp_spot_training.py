import json
import os
from pathlib import Path
import subprocess


LION_ROOT = Path(__file__).resolve().parents[2]
GCP_RUNNER = (
    LION_ROOT
    / "scripts/paper_scripts/PaDIS-Reproduction/platforms/gcp/run_PaDIS_GCP_spot_training.sh"
)
GCP_MANUAL_RECONSTRUCTION = (
    LION_ROOT
    / "scripts/paper_scripts/PaDIS-Reproduction/platforms/gcp/run_PaDIS_GCP_manual_reconstruction.sh"
)
GCP_STARTUP = (
    LION_ROOT
    / "scripts/paper_scripts/PaDIS-Reproduction/platforms/gcp/padis_gcp_spot_startup.sh"
)
GCP_METADATA_STARTUP = (
    LION_ROOT
    / "scripts/paper_scripts/PaDIS-Reproduction/platforms/gcp/padis_gcp_spot_metadata_startup.sh"
)
GCP_SHUTDOWN = (
    LION_ROOT
    / "scripts/paper_scripts/PaDIS-Reproduction/platforms/gcp/padis_gcp_spot_shutdown.sh"
)
DEFAULT_GCP_RUN_NAME = "PaDIS-Reproduction-GCP"


def _run_gcp_dry_run(
    tmp_path,
    task_order,
    extra_env=None,
    runtime_seconds=None,
    done_tasks=None,
):
    data_root = tmp_path / "Datasets"
    run_root = data_root / "experiments/PaDIS"
    run_name = (extra_env or {}).get("PADIS_GCP_RUN_NAME", DEFAULT_GCP_RUN_NAME)
    train_root = run_root / "final_real_runs" / run_name
    if runtime_seconds:
        runtime_dir = train_root / ".gcp_spot_dry_run/runtime"
        runtime_dir.mkdir(parents=True)
        for task_name, seconds in runtime_seconds.items():
            (runtime_dir / f"{task_name}.seconds").write_text(f"{seconds}\n")
    if done_tasks:
        done_dir = train_root / ".gcp_spot_dry_run/done"
        done_dir.mkdir(parents=True)
        for task_name in done_tasks:
            (done_dir / f"{task_name}.done").write_text(
                f"task={task_name}\nphase=base\n"
            )
    env = {
        **os.environ,
        "LION_DATA_PATH": str(data_root),
        "LION_EXPERIMENTS_PATH": str(data_root / "experiments"),
        "PADIS_RUN_ROOT": str(run_root),
        "PADIS_GCP_SKIP_ENV_ACTIVATE": "1",
        "PADIS_GCP_DRY_RUN": "1",
        "PADIS_GCP_STAGE_CACHES": "0",
        "PADIS_GCP_RECONSTRUCTION_PHASE": "0",
        "PADIS_GCP_GPU_IDS": "0,1",
        "PADIS_NO_WANDB": "1",
        "PADIS_WANDB_MODE": "disabled",
    }
    if task_order is not None:
        env["PADIS_GCP_TASK_ORDER"] = task_order
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


def _command_text(train_root, task_name, phase=None):
    key = task_name if phase in (None, "base") else f"{task_name}.{phase}"
    command_path = train_root / ".gcp_spot_dry_run/logs" / f"{key}.command.txt"
    return command_path.read_text()


def _command_path(train_root, task_name, phase=None):
    key = task_name if phase in (None, "base") else f"{task_name}.{phase}"
    return train_root / ".gcp_spot_dry_run/logs" / f"{key}.command.txt"


def test_gcp_spot_runner_is_executable():
    assert GCP_RUNNER.stat().st_mode & 0o111
    assert GCP_MANUAL_RECONSTRUCTION.stat().st_mode & 0o111
    assert GCP_STARTUP.stat().st_mode & 0o111
    assert GCP_METADATA_STARTUP.stat().st_mode & 0o111
    assert GCP_SHUTDOWN.stat().st_mode & 0o111


def test_gcp_manual_reconstruction_dry_run_uses_bucket_mount_paths(tmp_path):
    data_mount = tmp_path / "mnt_data"
    env = {
        **os.environ,
        "PADIS_MANUAL_RECON_DRY_RUN": "1",
        "PADIS_MANUAL_RECON_SKIP_ENV_ACTIVATE": "1",
        "PADIS_MOUNT_BUCKET": "0",
        "PADIS_DATA_MOUNT": str(data_mount),
        "LION_DATA_PATH": str(data_mount / "Datasets"),
        "LION_EXPERIMENTS_PATH": str(data_mount / "Datasets/experiments"),
        "PADIS_RUN_ROOT": str(data_mount / "Datasets/experiments/PaDIS"),
        "PADIS_RECON_METHODS": "baseline",
        "PADIS_RECON_EXPERIMENTS": "ct_20",
        "PADIS_RECON_ABLATIONS": "none",
        "PADIS_RECON_MAX_SAMPLES": "1",
        "PADIS_RECON_SAVE_PREVIEWS": "0",
        "PADIS_RECON_TASKS_PER_GPU": "1",
        "PADIS_RECON_TRAIN_MISSING_CHECKPOINTS": "0",
        "PADIS_GENERATION_PHASE": "0",
    }

    result = subprocess.run(
        ["bash", str(GCP_MANUAL_RECONSTRUCTION)],
        cwd=LION_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    recon_root = (
        data_mount
        / "Datasets/experiments/PaDIS/final_real_runs"
        / f"{DEFAULT_GCP_RUN_NAME}_reconstruction"
    )
    state_dir = recon_root / ".manual_gcp_reconstruction"
    manifest_text = (state_dir / "manifest.txt").read_text()
    assert "gcs_bucket=padis-bucket\n" in manifest_text
    assert f"data_mount={data_mount}\n" in manifest_text
    assert "mount_bucket=0\n" in manifest_text
    assert "reconstruction_checkpoint_policy=min_intense_val\n" in manifest_text
    assert "reconstruction_job_order=gcp_spot\n" in manifest_text
    assert "reconstruction_hparam_defaults=json\n" in manifest_text

    jobs = json.loads((recon_root / "reconstruction_matrix_jobs.json").read_text())
    assert len(jobs) == 1
    assert jobs[0]["method"] == "baseline"
    assert jobs[0]["experiment"] == "ct_20"

    command = (
        state_dir / "logs/reconstruction_000000.reconstruction.command.txt"
    ).read_text()
    assert "--training-root" in command
    assert str(data_mount / "Datasets/experiments/PaDIS/final_real_runs") in command
    assert "--checkpoint-policy min_intense_val" in command
    assert "--hparam-defaults json" in command
    assert "--job-order gcp_spot" in command

    done_marker = state_dir / "done/reconstruction_000000.reconstruction.done"
    assert done_marker.is_file()
    assert (state_dir / "last_sync.txt").is_file()


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


def test_gcp_spot_runner_default_order_runs_p96_immediately_after_pnp(tmp_path):
    result, train_root = _run_gcp_dry_run(tmp_path, None)

    assert result.returncode == 0, result.stderr

    expected_base_order = (
        "whole_lidc_full whole_lidc_default pnp_lidc_drunet "
        "pnp_lidc_drunet_noise_cond "
        "patch_lidc_p96_default patch_lidc_full patch_lidc_512 "
        "patch_lidc_default patch_lidc_p32_default patch_lidc_p16_default "
        "patch_lidc_p8_default patch_lidc_no_pos_default"
    )
    expected_validation_order = (
        "whole_lidc_full whole_lidc_default patch_lidc_p96_default "
        "patch_lidc_full patch_lidc_512 patch_lidc_default "
        "patch_lidc_p32_default patch_lidc_p16_default "
        "patch_lidc_p8_default patch_lidc_no_pos_default"
    )
    manifest = train_root / ".gcp_spot_dry_run/manifest.txt"
    assert f"tasks={expected_base_order}\n" in manifest.read_text()
    assert f"Task order: {expected_base_order}" in result.stdout
    assert (
        "Starting validation-heavy continuation phase for tasks: "
        f"{expected_validation_order}"
    ) in result.stdout


def test_gcp_spot_runner_adds_validation_heavy_continuation_phase(tmp_path):
    result, train_root = _run_gcp_dry_run(
        tmp_path,
        "whole_lidc_default,patch_lidc_default,patch_lidc_512,pnp_lidc_drunet",
    )

    assert result.returncode == 0, result.stderr

    patch_command = _command_text(
        train_root, "patch_lidc_default", phase="validation_heavy"
    )
    assert "--max-train-seconds 21600" in patch_command
    assert "--validation-interval-patches 20000" in patch_command
    assert "--validation-max-patches 4000" in patch_command
    assert "--validation-name padis_lidc_256_min_intense_val.pt" in patch_command
    assert "--validation-summary-key min_intense_validation_loss" in patch_command
    assert (
        "--validation-checkpoint-summary-key min_intense_validation_checkpoint"
        in patch_command
    )
    assert "--validation-repeat-until-max-patches" in patch_command

    whole_command = _command_text(
        train_root, "whole_lidc_default", phase="validation_heavy"
    )
    assert "--max-train-seconds 21600" in whole_command
    assert "--validation-interval-patches 2500" in whole_command
    assert "--validation-max-patches 328" in whole_command
    assert "--validation-name whole_image_lidc_256_min_intense_val.pt" in whole_command
    assert "--validation-summary-key min_intense_validation_loss" in whole_command
    assert (
        "--validation-checkpoint-summary-key min_intense_validation_checkpoint"
        in whole_command
    )
    assert "--validation-repeat-until-max-patches" in whole_command

    patch_512_command = _command_text(
        train_root, "patch_lidc_512", phase="validation_heavy"
    )
    assert "--validation-name padis_lidc_512_min_intense_val.pt" in patch_512_command
    assert "--validation-summary-key min_intense_validation_loss" in patch_512_command
    assert (
        "--validation-checkpoint-summary-key min_intense_validation_checkpoint"
        in patch_512_command
    )
    manifest = train_root / ".gcp_spot_dry_run/manifest.txt"
    manifest_text = manifest.read_text()
    assert "whole_validation_heavy_interval_images=2500\n" in manifest_text
    assert "whole_validation_heavy_max_images=328\n" in manifest_text

    assert not _command_path(
        train_root, "pnp_lidc_drunet", phase="validation_heavy"
    ).exists()


def test_gcp_spot_runner_validation_phase_resumes_base_done_tasks(tmp_path):
    result, train_root = _run_gcp_dry_run(
        tmp_path,
        "patch_lidc_default",
        done_tasks=["patch_lidc_default"],
    )

    assert result.returncode == 0, result.stderr
    assert not _command_path(train_root, "patch_lidc_default").exists()
    patch_command = _command_text(
        train_root, "patch_lidc_default", phase="validation_heavy"
    )
    assert "--validation-interval-patches 20000" in patch_command
    validation_done = (
        train_root / ".gcp_spot_dry_run/done/patch_lidc_default.validation_heavy.done"
    )
    assert validation_done.is_file()
    validation_done_text = validation_done.read_text()
    assert "padis_lidc_256_min_intense_val.pt" in validation_done_text
    assert "padis_lidc_256_min_intense_val_full.pt" in validation_done_text


def test_gcp_spot_runner_validation_phase_uses_separate_runtime_ledger(tmp_path):
    result, train_root = _run_gcp_dry_run(
        tmp_path,
        "patch_lidc_default",
        done_tasks=["patch_lidc_default"],
        runtime_seconds={
            "patch_lidc_default": 21600,
            "patch_lidc_default.validation_heavy": 7200,
        },
    )

    assert result.returncode == 0, result.stderr
    assert not _command_path(train_root, "patch_lidc_default").exists()
    patch_command = _command_text(
        train_root, "patch_lidc_default", phase="validation_heavy"
    )
    assert "--max-train-seconds 14400" in patch_command
    assert "--validation-interval-patches 20000" in patch_command


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


def test_gcp_spot_runner_uses_128_batch_for_p96_by_default(tmp_path):
    result, train_root = _run_gcp_dry_run(
        tmp_path,
        "patch_lidc_p96_default",
    )

    assert result.returncode == 0, result.stderr
    command = _command_text(train_root, "patch_lidc_p96_default")
    assert "--patch-size-preset 96" in command
    assert "--batch-size 128" in command


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


def test_gcp_spot_runner_uses_finalise_window_when_budget_is_exhausted(tmp_path):
    result, train_root = _run_gcp_dry_run(
        tmp_path,
        "patch_lidc_default",
        extra_env={
            "PADIS_GCP_PATCH_TRAIN_TIME": "120",
            "PADIS_GCP_FINALISE_SECONDS": "17",
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
    runner_dir = lion_root / "scripts/paper_scripts/PaDIS-Reproduction/platforms/gcp"
    runner_dir.mkdir(parents=True)
    runner_path = runner_dir / "run_PaDIS_GCP_spot_training.sh"
    runner_path.symlink_to(GCP_RUNNER)
    netrc_path = data_mount / ".netrc"
    netrc_path.write_text("machine api.wandb.ai\n  login user\n  password test-key\n")
    netrc_path.chmod(0o600)
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
    assert f"PADIS_WANDB_NETRC={netrc_path}" in env_text
    assert f"NETRC={netrc_path}" in env_text
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
    assert "Dry-run: would configure W&B netrc" in result.stdout
    assert "Dry-run runner command" in result.stdout
    assert str(runner_path) in result.stdout


def test_gcp_metadata_startup_bootstrap_delegates_after_data_mount(tmp_path):
    data_mount = tmp_path / "mnt_data"
    lion_root = data_mount / "LION"
    startup_dir = lion_root / "scripts/paper_scripts/PaDIS-Reproduction/platforms/gcp"
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


def test_gcp_spot_runner_runs_resumable_reconstruction_phase(tmp_path):
    result, train_root = _run_gcp_dry_run(
        tmp_path,
        "patch_lidc_default",
        extra_env={
            "PADIS_GCP_RECONSTRUCTION_PHASE": "1",
            "PADIS_GCP_VALIDATION_HEAVY_PHASE": "0",
            "PADIS_RECON_METHODS": "baseline,padis_dps,patch_average",
            "PADIS_RECON_EXPERIMENTS": "ct_20,ct_512_60",
            "PADIS_RECON_ALLOW_OFF_PAPER_EXPERIMENTS": "1",
            "PADIS_RECON_MAX_SAMPLES": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    manifest = train_root / ".gcp_spot_dry_run/manifest.txt"
    manifest_text = manifest.read_text()
    assert "reconstruction_enabled=1\n" in manifest_text
    assert "reconstruction_tasks_per_gpu_requested=auto\n" in manifest_text
    assert "reconstruction_tasks_per_gpu=1\n" in manifest_text
    assert "reconstruction_checkpoint_policy=min_intense_val\n" in manifest_text
    assert "reconstruction_job_order=gcp_spot\n" in manifest_text
    assert "reconstruction_hparam_defaults=json\n" in manifest_text
    assert "reconstruction_hparam_defaults_json=" in manifest_text

    jobs_path = (
        tmp_path
        / "Datasets/experiments/PaDIS/final_real_runs"
        / f"{DEFAULT_GCP_RUN_NAME}_reconstruction/reconstruction_matrix_jobs.json"
    )
    jobs = json.loads(jobs_path.read_text())
    first_512 = next(
        index for index, job in enumerate(jobs) if job["experiment"] == "ct_512_60"
    )
    first_fixed = next(
        index
        for index, job in enumerate(jobs)
        if job["method"] in {"patch_stitch", "patch_average"}
    )
    assert first_fixed < first_512
    assert all(
        job["experiment"] != "ct_512_60"
        and job["method"] not in {"patch_stitch", "patch_average"}
        for job in jobs[:first_fixed]
    )
    assert all(
        job["experiment"] != "ct_512_60"
        and job["method"] in {"patch_stitch", "patch_average"}
        for job in jobs[first_fixed:first_512]
    )
    assert all(job["experiment"] == "ct_512_60" for job in jobs[first_512:])

    command = (
        train_root
        / ".gcp_spot_dry_run/logs/reconstruction_000000.reconstruction.command.txt"
    ).read_text()
    assert "--checkpoint-policy min_intense_val" in command
    assert "--hparam-defaults json" in command
    assert "--hparam-defaults-json" in command
    assert "--job-order gcp_spot" in command
    assert "pnp_lidc_drunet_min_val.pt" in command
    done_marker = (
        train_root / ".gcp_spot_dry_run/done/reconstruction_000000.reconstruction.done"
    )
    assert done_marker.is_file()


def test_gcp_spot_runner_allows_multiple_reconstruction_slots_per_gpu(tmp_path):
    result, train_root = _run_gcp_dry_run(
        tmp_path,
        "patch_lidc_default",
        extra_env={
            "PADIS_GCP_RECONSTRUCTION_PHASE": "1",
            "PADIS_GCP_VALIDATION_HEAVY_PHASE": "0",
            "PADIS_GCP_RECON_TASKS_PER_GPU": "2",
            "PADIS_RECON_METHODS": "baseline,padis_dps",
            "PADIS_RECON_EXPERIMENTS": "ct_20",
            "PADIS_RECON_MAX_SAMPLES": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    manifest = train_root / ".gcp_spot_dry_run/manifest.txt"
    manifest_text = manifest.read_text()
    assert "reconstruction_tasks_per_gpu_requested=2\n" in manifest_text
    assert "reconstruction_tasks_per_gpu=2\n" in manifest_text
    assert "with 2 worker slot(s) per GPU" in result.stdout

    commands = sorted(
        (train_root / ".gcp_spot_dry_run/logs").glob(
            "reconstruction_*.reconstruction.command.txt"
        )
    )
    assert len(commands) >= 2
    done_markers = sorted(
        (train_root / ".gcp_spot_dry_run/done").glob(
            "reconstruction_*.reconstruction.done"
        )
    )
    assert len(done_markers) >= 2
    assert all("slot=" in marker.read_text() for marker in done_markers[:2])


def test_gcp_spot_runner_auto_uses_three_reconstruction_slots_on_large_gpu(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/bash\n"
        'if [[ "$*" == *"--query-gpu=memory.total"* ]]; then\n'
        "  printf '97887\\n'\n"
        "else\n"
        "  printf 'NVIDIA RTX PRO 6000 Blackwell\\n'\n"
        "fi\n"
    )
    nvidia_smi.chmod(0o755)

    result, train_root = _run_gcp_dry_run(
        tmp_path,
        "patch_lidc_default",
        extra_env={
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PADIS_GCP_RECONSTRUCTION_PHASE": "1",
            "PADIS_GCP_VALIDATION_HEAVY_PHASE": "0",
            "PADIS_RECON_METHODS": "baseline,padis_dps",
            "PADIS_RECON_EXPERIMENTS": "ct_20",
            "PADIS_RECON_MAX_SAMPLES": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    manifest = train_root / ".gcp_spot_dry_run/manifest.txt"
    manifest_text = manifest.read_text()
    assert "reconstruction_tasks_per_gpu_requested=auto\n" in manifest_text
    assert "reconstruction_tasks_per_gpu=3\n" in manifest_text
    assert "with 3 worker slot(s) per GPU" in result.stdout
