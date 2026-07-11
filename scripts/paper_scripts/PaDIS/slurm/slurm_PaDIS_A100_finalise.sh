#!/bin/bash
#SBATCH -J PaDIS_finalise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -p ampere
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/padis_a100_common.sh"
padis_setup_modules
padis_activate_environment
LION_ROOT="$(padis_lion_root)"
export LION_ROOT PADIS_TIMING_MODE=slurm
export PADIS_TIMING_LOG_ROOT="${PADIS_TIMING_LOG_ROOT:-$(padis_default_run_root)/debug_runs/slurm_logs}"
exec bash "$LION_ROOT/scripts/paper_scripts/PaDIS/PaDIS_finalise_pipeline.sh"
