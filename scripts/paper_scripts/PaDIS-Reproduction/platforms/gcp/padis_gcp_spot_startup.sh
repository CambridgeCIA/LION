#!/usr/bin/env bash
#
# GCP startup hook for the PaDIS spot runner.
#
# Configure this as the VM startup script for a stateful MIG. It verifies the
# retained /mnt/data disk, prepares the tmpfs cache, and starts the PaDIS runner
# as the data-owner user so training artifacts are not root-owned.

set -euo pipefail

log() {
        printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
        echo "$*" >&2
        exit 1
}

is_root() {
        [ "$(id -u)" -eq 0 ]
}

run_or_echo() {
        if [ "$PADIS_GCP_STARTUP_DRY_RUN" = "1" ]; then
                printf 'DRY_RUN:'
                printf ' %q' "$@"
                printf '\n'
        else
                "$@"
        fi
}

wait_for_path() {
        local path="$1"
        local timeout="$2"
        local deadline
        deadline=$((SECONDS + timeout))
        while [ "$SECONDS" -lt "$deadline" ]; do
                [ -e "$path" ] && return 0
                sleep 2
        done
        [ -e "$path" ]
}

resolve_training_user() {
        local owner
        if [ -n "${PADIS_GCP_RUN_AS_USER:-}" ]; then
                printf '%s\n' "$PADIS_GCP_RUN_AS_USER"
                return
        fi
        if [ -e "$LION_ROOT" ]; then
                owner="$(stat -c '%U' "$LION_ROOT" 2>/dev/null || true)"
                if [ -n "$owner" ] && [ "$owner" != "UNKNOWN" ]; then
                        printf '%s\n' "$owner"
                        return
                fi
        fi
        if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
                printf '%s\n' "$SUDO_USER"
                return
        fi
        for owner in thomas ubuntu; do
                if id "$owner" >/dev/null 2>&1; then
                        printf '%s\n' "$owner"
                        return
                fi
        done
        printf 'root\n'
}

resolve_user_home() {
        local user="$1"
        local home_dir
        home_dir=""
        if command -v getent >/dev/null 2>&1; then
                home_dir="$(getent passwd "$user" 2>/dev/null | cut -d: -f6 || true)"
        fi
        if [ -n "$home_dir" ]; then
                printf '%s\n' "$home_dir"
        elif [ "$user" = "root" ]; then
                printf '/root\n'
        else
                printf '/home/%s\n' "$user"
        fi
}

ensure_data_mount() {
        mkdir -p "$PADIS_DATA_MOUNT"
        if mountpoint -q "$PADIS_DATA_MOUNT"; then
                log "$PADIS_DATA_MOUNT is already mounted."
                return
        fi
        if [ "$PADIS_GCP_STARTUP_DRY_RUN" = "1" ] && [ -d "$LION_ROOT" ]; then
                log "Dry-run: accepting existing $LION_ROOT without mounted $PADIS_DATA_MOUNT."
                return
        fi
        if [ -n "${PADIS_DATA_DEVICE:-}" ]; then
                log "Waiting for data device $PADIS_DATA_DEVICE."
                wait_for_path "$PADIS_DATA_DEVICE" "$PADIS_GCP_DEVICE_WAIT_SECONDS" \
                        || die "Data device did not appear: $PADIS_DATA_DEVICE"
                log "Mounting $PADIS_DATA_DEVICE at $PADIS_DATA_MOUNT."
                run_or_echo mount "$PADIS_DATA_DEVICE" "$PADIS_DATA_MOUNT"
                return
        fi
        if findmnt --fstab --target "$PADIS_DATA_MOUNT" >/dev/null 2>&1; then
                log "Mounting $PADIS_DATA_MOUNT from /etc/fstab."
                run_or_echo mount "$PADIS_DATA_MOUNT"
                return
        fi
        die "$PADIS_DATA_MOUNT is not mounted. Configure a stateful disk, /etc/fstab, or PADIS_DATA_DEVICE."
}

ensure_lion_root() {
        if [ ! -x "$LION_ROOT/scripts/paper_scripts/PaDIS-Reproduction/platforms/gcp/run_PaDIS_GCP_spot_training.sh" ]; then
                die "Cannot find executable PaDIS GCP runner under $LION_ROOT."
        fi
}

ensure_ramdisk() {
        local fs_type
        mkdir -p "$PADIS_RAM_DISK"
        resolve_ramdisk_size
        fs_type="$(findmnt -n -o FSTYPE --target "$PADIS_RAM_DISK" 2>/dev/null || true)"
        case "$fs_type" in
                tmpfs|ramfs)
                        log "$PADIS_RAM_DISK is already mounted as $fs_type."
                        return
                        ;;
        esac
        if [ "$PADIS_GCP_STARTUP_DRY_RUN" = "1" ]; then
                log "Dry-run: would mount tmpfs at $PADIS_RAM_DISK with size $PADIS_RAM_DISK_SIZE."
                return
        fi
        if [ -n "$fs_type" ] && [ "$fs_type" != "tmpfs" ] && [ "$fs_type" != "ramfs" ]; then
                die "$PADIS_RAM_DISK is backed by $fs_type, not tmpfs/ramfs."
        fi
        log "Mounting tmpfs at $PADIS_RAM_DISK with size $PADIS_RAM_DISK_SIZE."
        run_or_echo mount -t tmpfs -o "size=$PADIS_RAM_DISK_SIZE" tmpfs "$PADIS_RAM_DISK"
}

resolve_ramdisk_size() {
        local mem_total_kb half_mem_bytes cap_bytes chosen_bytes
        if [ -n "$PADIS_RAM_DISK_SIZE" ]; then
                return
        fi
        mem_total_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
        if ! [[ "$mem_total_kb" =~ ^[0-9]+$ ]] || [ "$mem_total_kb" -le 0 ]; then
                PADIS_RAM_DISK_SIZE="100g"
                return
        fi
        half_mem_bytes=$((mem_total_kb * 1024 / 2))
        cap_bytes=$((100 * 1024 * 1024 * 1024))
        if [ "$half_mem_bytes" -lt "$cap_bytes" ]; then
                chosen_bytes="$half_mem_bytes"
        else
                chosen_bytes="$cap_bytes"
        fi
        PADIS_RAM_DISK_SIZE="${chosen_bytes}"
}

wait_for_gpu() {
        local deadline
        if [ "$PADIS_GCP_REQUIRE_NVIDIA_SMI" != "1" ]; then
                return
        fi
        deadline=$((SECONDS + PADIS_GCP_GPU_WAIT_SECONDS))
        while [ "$SECONDS" -lt "$deadline" ]; do
                if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
                        log "nvidia-smi is available."
                        return
                fi
                sleep 5
        done
        die "nvidia-smi did not become available within ${PADIS_GCP_GPU_WAIT_SECONDS}s."
}

ensure_wandb_netrc() {
        local user_home home_netrc current_target
        if [ ! -f "$PADIS_WANDB_NETRC" ]; then
                log "W&B netrc not found at $PADIS_WANDB_NETRC; leaving W&B auth to the environment."
                return
        fi

        NETRC="$PADIS_WANDB_NETRC"
        export NETRC

        if [ "$PADIS_GCP_STARTUP_DRY_RUN" = "1" ]; then
                log "Dry-run: would configure W&B netrc from $PADIS_WANDB_NETRC for $PADIS_GCP_RUN_AS_USER."
                return
        fi

        chmod 600 "$PADIS_WANDB_NETRC" 2>/dev/null || true
        chown "$PADIS_GCP_RUN_AS_USER":"$PADIS_GCP_RUN_AS_USER" "$PADIS_WANDB_NETRC" 2>/dev/null || true

        user_home="$(resolve_user_home "$PADIS_GCP_RUN_AS_USER")"
        mkdir -p "$user_home"
        home_netrc="$user_home/.netrc"
        current_target="$(readlink "$home_netrc" 2>/dev/null || true)"
        if [ -e "$home_netrc" ] || [ -L "$home_netrc" ]; then
                if [ "$current_target" != "$PADIS_WANDB_NETRC" ]; then
                        log "Leaving existing $home_netrc in place; exported NETRC=$NETRC for W&B."
                        return
                fi
        else
                ln -s "$PADIS_WANDB_NETRC" "$home_netrc"
                chown -h "$PADIS_GCP_RUN_AS_USER":"$PADIS_GCP_RUN_AS_USER" "$home_netrc" 2>/dev/null || true
        fi
        log "Configured W&B netrc at $NETRC."
}

write_shell_assignment() {
        local name="$1"
        case "$name" in
                ''|*[!A-Za-z0-9_]*)
                        return
                        ;;
        esac
        [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || return
        printf '%s=%q\n' "$name" "${!name}"
}

write_env_file() {
        mkdir -p "$(dirname "$PADIS_GCP_ENV_FILE")"
        {
                local name
                for name in $(compgen -v); do
                        case "$name" in
                                LION_*|PADIS_*|CONDA_EXE|CONDA_ENVS_PATH|CUDA_VISIBLE_DEVICES|MPLCONFIGDIR|NETRC|OMP_NUM_THREADS|PYTHONUNBUFFERED|WANDB_DIR)
                                        write_shell_assignment "$name"
                                        ;;
                        esac
                done
        } > "$PADIS_GCP_ENV_FILE"
        chown "$PADIS_GCP_RUN_AS_USER":"$PADIS_GCP_RUN_AS_USER" "$PADIS_GCP_ENV_FILE" 2>/dev/null || true
}

runner_already_active() {
        pgrep -u "$PADIS_GCP_RUN_AS_USER" -f "$RUNNER_PATH" >/dev/null 2>&1
}

start_runner() {
        local runner_cmd
        mkdir -p "$PADIS_TRAIN_ROOT/.gcp_spot/logs"
        chown -R "$PADIS_GCP_RUN_AS_USER":"$PADIS_GCP_RUN_AS_USER" "$PADIS_TRAIN_ROOT" 2>/dev/null || true
        runner_cmd="set -euo pipefail; set -a; source '$PADIS_GCP_ENV_FILE'; set +a; cd '$LION_ROOT'; nohup '$RUNNER_PATH' >> '$PADIS_GCP_RUNNER_LOG' 2>&1 &"
        if runner_already_active; then
                log "PaDIS GCP runner is already active for user $PADIS_GCP_RUN_AS_USER."
                return
        fi
        if [ "$PADIS_GCP_STARTUP_DRY_RUN" = "1" ]; then
                log "Dry-run runner command as $PADIS_GCP_RUN_AS_USER: $runner_cmd"
                return
        fi
        if is_root && [ "$PADIS_GCP_RUN_AS_USER" != "root" ]; then
                log "Starting runner as $PADIS_GCP_RUN_AS_USER. Log: $PADIS_GCP_RUNNER_LOG"
                if command -v runuser >/dev/null 2>&1; then
                        runuser -u "$PADIS_GCP_RUN_AS_USER" -- bash -lc "$runner_cmd"
                elif command -v sudo >/dev/null 2>&1; then
                        sudo -H -u "$PADIS_GCP_RUN_AS_USER" bash -lc "$runner_cmd"
                else
                        su -s /bin/bash -c "$runner_cmd" "$PADIS_GCP_RUN_AS_USER"
                fi
        else
                log "Starting runner as $(id -un). Log: $PADIS_GCP_RUNNER_LOG"
                bash -lc "$runner_cmd"
        fi
}

PADIS_DATA_MOUNT="${PADIS_DATA_MOUNT:-/mnt/data}"
LION_ROOT="${LION_ROOT:-$PADIS_DATA_MOUNT/LION}"
LION_DATA_PATH="${LION_DATA_PATH:-$PADIS_DATA_MOUNT/Datasets}"
LION_EXPERIMENTS_PATH="${LION_EXPERIMENTS_PATH:-$LION_DATA_PATH/experiments}"
PADIS_GCP_CONDA_ROOT="${PADIS_GCP_CONDA_ROOT:-$PADIS_DATA_MOUNT/conda}"
CONDA_EXE="${CONDA_EXE:-$PADIS_GCP_CONDA_ROOT/miniconda3/bin/conda}"
CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$PADIS_GCP_CONDA_ROOT/envs}"
LION_CONDA_ENV="${LION_CONDA_ENV:-$CONDA_ENVS_PATH/lion}"
PADIS_RUN_ROOT="${PADIS_RUN_ROOT:-$LION_EXPERIMENTS_PATH/PaDIS}"
PADIS_GCP_RUN_NAME="${PADIS_GCP_RUN_NAME:-PaDIS-Reproduction-GCP}"
PADIS_TRAIN_ROOT="${PADIS_TRAIN_ROOT:-$PADIS_RUN_ROOT/final_real_runs/$PADIS_GCP_RUN_NAME}"
PADIS_RAM_DISK="${PADIS_RAM_DISK:-/mnt/ram-disk}"
PADIS_RAM_DISK_SIZE="${PADIS_RAM_DISK_SIZE:-}"
PADIS_WANDB_PROJECT="${PADIS_WANDB_PROJECT:-PaDIS-Reproduction}"
PADIS_WANDB_MODE="${PADIS_WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-$PADIS_TRAIN_ROOT/wandb}"
PADIS_WANDB_NETRC="${PADIS_WANDB_NETRC:-$PADIS_DATA_MOUNT/.netrc}"
PADIS_GCP_STARTUP_DRY_RUN="${PADIS_GCP_STARTUP_DRY_RUN:-0}"
PADIS_GCP_START_RUNNER="${PADIS_GCP_START_RUNNER:-1}"
PADIS_GCP_REQUIRE_NVIDIA_SMI="${PADIS_GCP_REQUIRE_NVIDIA_SMI:-1}"
PADIS_GCP_GPU_WAIT_SECONDS="${PADIS_GCP_GPU_WAIT_SECONDS:-180}"
PADIS_GCP_DEVICE_WAIT_SECONDS="${PADIS_GCP_DEVICE_WAIT_SECONDS:-120}"
PADIS_GCP_ENV_FILE="${PADIS_GCP_ENV_FILE:-$PADIS_TRAIN_ROOT/.gcp_spot/startup_env.sh}"
PADIS_GCP_RUNNER_LOG="${PADIS_GCP_RUNNER_LOG:-$PADIS_TRAIN_ROOT/.gcp_spot/logs/startup_runner.log}"
RUNNER_PATH="$LION_ROOT/scripts/paper_scripts/PaDIS-Reproduction/platforms/gcp/run_PaDIS_GCP_spot_training.sh"

log "PaDIS GCP startup hook starting."
ensure_data_mount
ensure_lion_root
ensure_ramdisk
wait_for_gpu
PADIS_GCP_RUN_AS_USER="$(resolve_training_user)"
export LION_ROOT LION_DATA_PATH LION_EXPERIMENTS_PATH PADIS_RUN_ROOT
export CONDA_EXE CONDA_ENVS_PATH LION_CONDA_ENV
export PADIS_GCP_RUN_NAME PADIS_TRAIN_ROOT PADIS_RAM_DISK
export PADIS_WANDB_PROJECT PADIS_WANDB_MODE WANDB_DIR PADIS_WANDB_NETRC PADIS_GCP_RUN_AS_USER
ensure_wandb_netrc
write_env_file
if [ "$PADIS_GCP_START_RUNNER" = "1" ]; then
        start_runner
else
        log "Skipping runner start because PADIS_GCP_START_RUNNER=$PADIS_GCP_START_RUNNER."
fi
log "PaDIS GCP startup hook finished."
