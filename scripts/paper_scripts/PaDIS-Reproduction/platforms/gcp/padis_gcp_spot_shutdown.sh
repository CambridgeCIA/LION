#!/usr/bin/env bash
#
# GCP shutdown hook for the PaDIS spot runner.
#
# Configure this as the VM shutdown script so preemption updates the runtime
# ledger and asks active training processes to stop cleanly before the instance
# is terminated.

set -euo pipefail

log() {
        printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

read_marker_value() {
        local marker="$1"
        local key="$2"
        local line
        line="$(grep -m 1 "^${key}=" "$marker" 2>/dev/null || true)"
        if [ -n "$line" ]; then
                printf '%s\n' "${line#*=}"
        fi
}

write_task_elapsed_seconds() {
        local task_name="$1"
        local seconds="$2"
        local path="$RUNTIME_DIR/$task_name.seconds"
        mkdir -p "$RUNTIME_DIR"
        printf '%s\n' "$seconds" > "$path.tmp.$$"
        mv "$path.tmp.$$" "$path"
}

refresh_active_runtimes() {
        local marker task start_elapsed start_epoch now total
        now="$(date +%s)"
        for marker in "$RUNNING_DIR"/*.running; do
                [ -e "$marker" ] || continue
                task="$(read_marker_value "$marker" task)"
                if [ -z "$task" ]; then
                        task="${marker##*/}"
                        task="${task%.running}"
                fi
                start_elapsed="$(read_marker_value "$marker" start_elapsed)"
                start_epoch="$(read_marker_value "$marker" start_epoch)"
                if [[ "$start_elapsed" =~ ^[0-9]+$ && "$start_epoch" =~ ^[0-9]+$ ]]; then
                        total=$((start_elapsed + now - start_epoch))
                        if [ "$total" -lt "$start_elapsed" ]; then
                                total="$start_elapsed"
                        fi
                        write_task_elapsed_seconds "$task" "$total"
                        log "Updated runtime for $task to ${total}s."
                fi
        done
}

marker_pids() {
        local marker child_pid monitor_pid
        for marker in "$RUNNING_DIR"/*.running; do
                [ -e "$marker" ] || continue
                child_pid="$(read_marker_value "$marker" child_pid)"
                if [[ "$child_pid" =~ ^[0-9]+$ ]]; then
                        printf '%s\n' "$child_pid"
                fi
                monitor_pid="$(read_marker_value "$marker" monitor_pid)"
                if [[ "$monitor_pid" =~ ^[0-9]+$ ]]; then
                        printf '%s\n' "$monitor_pid"
                fi
        done
}

runner_pids() {
        local pid
        pgrep -f "${PADIS_GCP_RUNNER_PATTERN:-run_PaDIS_GCP_spot_training.sh}" 2>/dev/null \
                | while read -r pid; do
                        if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
                                printf '%s\n' "$pid"
                        fi
                done
}

signal_pids() {
        local label="$1"
        local signal="$2"
        shift
        shift
        local pid
        for pid in "$@"; do
                if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
                        continue
                fi
                if ! kill -0 "$pid" >/dev/null 2>&1; then
                        continue
                fi
                if [ "$PADIS_GCP_SHUTDOWN_DRY_RUN" = "1" ]; then
                        log "Would send $signal to $label pid $pid."
                else
                        log "Sending $signal to $label pid $pid."
                        kill "-$signal" "$pid" >/dev/null 2>&1 || true
                fi
        done
}

wait_for_pids() {
        local deadline pid any_alive
        deadline=$((SECONDS + PADIS_GCP_SHUTDOWN_GRACE_SECONDS))
        while [ "$SECONDS" -lt "$deadline" ]; do
                any_alive=0
                for pid in "$@"; do
                        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" >/dev/null 2>&1; then
                                any_alive=1
                                break
                        fi
                done
                [ "$any_alive" -eq 0 ] && return 0
                sleep 1
        done
        return 1
}

LION_DATA_PATH="${LION_DATA_PATH:-/mnt/data/Datasets}"
LION_EXPERIMENTS_PATH="${LION_EXPERIMENTS_PATH:-$LION_DATA_PATH/experiments}"
PADIS_RUN_ROOT="${PADIS_RUN_ROOT:-$LION_EXPERIMENTS_PATH/PaDIS}"
PADIS_GCP_RUN_NAME="${PADIS_GCP_RUN_NAME:-PaDIS-Reproduction-GCP}"
PADIS_TRAIN_ROOT="${PADIS_TRAIN_ROOT:-$PADIS_RUN_ROOT/final_real_runs/$PADIS_GCP_RUN_NAME}"
PADIS_GCP_DRY_RUN="${PADIS_GCP_DRY_RUN:-0}"
PADIS_GCP_SHUTDOWN_DRY_RUN="${PADIS_GCP_SHUTDOWN_DRY_RUN:-0}"
PADIS_GCP_SHUTDOWN_GRACE_SECONDS="${PADIS_GCP_SHUTDOWN_GRACE_SECONDS:-20}"
PADIS_GCP_SHUTDOWN_FORCE_KILL="${PADIS_GCP_SHUTDOWN_FORCE_KILL:-0}"

if [ -n "${PADIS_GCP_STATE_DIR:-}" ]; then
        STATE_DIR="$PADIS_GCP_STATE_DIR"
elif [ "$PADIS_GCP_DRY_RUN" = "1" ]; then
        STATE_DIR="$PADIS_TRAIN_ROOT/.gcp_spot_dry_run"
else
        STATE_DIR="$PADIS_TRAIN_ROOT/.gcp_spot"
fi
RUNNING_DIR="$STATE_DIR/running"
RUNTIME_DIR="$STATE_DIR/runtime"

mkdir -p "$RUNNING_DIR" "$RUNTIME_DIR"

log "PaDIS GCP shutdown hook starting for $PADIS_TRAIN_ROOT."
refresh_active_runtimes

mapfile -t active_pids < <(marker_pids | sort -u)
mapfile -t active_runner_pids < <(runner_pids | sort -u)

if [ "${#active_pids[@]}" -gt 0 ]; then
        signal_pids "training" TERM "${active_pids[@]}"
fi
if [ "${#active_runner_pids[@]}" -gt 0 ]; then
        signal_pids "runner" TERM "${active_runner_pids[@]}"
fi

if [ "$PADIS_GCP_SHUTDOWN_DRY_RUN" != "1" ]; then
        if ! wait_for_pids "${active_pids[@]}" "${active_runner_pids[@]}"; then
                log "Grace period expired with some PaDIS processes still alive."
                if [ "$PADIS_GCP_SHUTDOWN_FORCE_KILL" = "1" ]; then
                        signal_pids "remaining process" KILL "${active_pids[@]}" "${active_runner_pids[@]}"
                fi
        fi
fi

refresh_active_runtimes
log "PaDIS GCP shutdown hook finished."
