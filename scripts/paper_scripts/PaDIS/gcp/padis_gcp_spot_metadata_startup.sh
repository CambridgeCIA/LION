#!/usr/bin/env bash
#
# Minimal metadata/GCS startup script for PaDIS stateful GCP MIGs.
#
# This script must be available before /mnt/data is mounted. Put this script in
# instance-template metadata, a GCS startup-script-url, or the boot image. It
# mounts the stateful data disk, then delegates to the full startup hook stored
# in the retained LION checkout.

set -euo pipefail

log() {
        printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
        echo "$*" >&2
        exit 1
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

ensure_data_mount() {
        mkdir -p "$PADIS_DATA_MOUNT"
        if mountpoint -q "$PADIS_DATA_MOUNT"; then
                log "$PADIS_DATA_MOUNT is already mounted."
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
        if [ "$PADIS_GCP_STARTUP_DRY_RUN" = "1" ] && [ -x "$PADIS_GCP_STARTUP_HOOK" ]; then
                log "Dry-run: accepting existing $PADIS_GCP_STARTUP_HOOK without mounted $PADIS_DATA_MOUNT."
                return
        fi
        die "$PADIS_DATA_MOUNT is not mounted. Configure /etc/fstab or PADIS_DATA_DEVICE."
}

PADIS_DATA_MOUNT="${PADIS_DATA_MOUNT:-/mnt/data}"
LION_ROOT="${LION_ROOT:-$PADIS_DATA_MOUNT/LION}"
PADIS_GCP_STARTUP_HOOK="${PADIS_GCP_STARTUP_HOOK:-$LION_ROOT/scripts/paper_scripts/PaDIS/gcp/padis_gcp_spot_startup.sh}"
PADIS_GCP_STARTUP_DRY_RUN="${PADIS_GCP_STARTUP_DRY_RUN:-0}"
PADIS_GCP_DEVICE_WAIT_SECONDS="${PADIS_GCP_DEVICE_WAIT_SECONDS:-120}"

log "PaDIS GCP metadata startup bootstrap starting."
ensure_data_mount
if [ ! -x "$PADIS_GCP_STARTUP_HOOK" ]; then
        die "Cannot execute PaDIS startup hook: $PADIS_GCP_STARTUP_HOOK"
fi
log "Delegating to $PADIS_GCP_STARTUP_HOOK."
exec "$PADIS_GCP_STARTUP_HOOK"
