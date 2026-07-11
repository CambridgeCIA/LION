#!/bin/bash
#
# Run the PaDIS A100 gate checks inside an existing interactive Slurm
# allocation. This wraps slurm_PaDIS_A100_checks.sh so the checks stay identical
# to the batch version while using an interactive-specific run stamp.
#
# Example:
#   salloc -A MPHIL-DIS-SL2-GPU -p ampere --gres=gpu:1 --cpus-per-task=32 --time=00:30:00
#   scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm/run_PaDIS_A100_checks_interactive.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm/padis_a100_common.sh
. "$SCRIPT_DIR/padis_a100_common.sh"

LION_ROOT="$(padis_lion_root)"
PADIS_RUN_ROOT="$(padis_default_run_root)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
        echo "Warning: SLURM_JOB_ID is not set; this does not look like an interactive Slurm allocation." >&2
        echo "The checks require CUDA and are normally run from an A100 allocation." >&2
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "Warning: CUDA_VISIBLE_DEVICES is not set before environment activation." >&2
fi

PADIS_RUN_STAMP="${PADIS_RUN_STAMP:-interactive_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
PADIS_CHECK_DIR="${PADIS_CHECK_DIR:-$PADIS_RUN_ROOT/debug_runs/a100_checks_$PADIS_RUN_STAMP}"

export LION_ROOT PADIS_RUN_ROOT PADIS_RUN_STAMP PADIS_CHECK_DIR
export PADIS_SLURM_DIR="$SCRIPT_DIR"

exec "$SCRIPT_DIR/slurm_PaDIS_A100_checks.sh"
