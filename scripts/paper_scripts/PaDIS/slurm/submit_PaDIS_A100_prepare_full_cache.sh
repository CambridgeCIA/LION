#!/bin/bash
#
# Submit the one-time PaDIS LIDC cache build. The job writes reusable tensor
# caches under RDS and, by default, zstd-compressed .pt.zst archives.
#
# Useful overrides:
#   PADIS_SLURM_ACCOUNT=MPHIL-DIS-SL2-GPU
#   PADIS_CACHE_SLURM_ACCOUNT=MPHIL-DIS-SL2-CPU
#   PADIS_CACHE_PARTITION=icelake
#   PADIS_CACHE_CPUS_PER_TASK=8
#   PADIS_CACHE_MEM=128G
#   PADIS_CACHE_SLURM_TIME=08:00:00
#   PADIS_CACHE_PREP_VARIANTS=256-default,256-full,512-default
#   PADIS_CACHE_ROOT=/path/to/LIDC-IDRI-cache
#   PADIS_WRITE_CACHE_ARCHIVE=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh
. "$SCRIPT_DIR/padis_a100_common.sh"

LION_ROOT="$(padis_lion_root)"
cd "$LION_ROOT"

account="${PADIS_SLURM_ACCOUNT:-MPHIL-DIS-SL2-GPU}"
cache_account="${PADIS_CACHE_SLURM_ACCOUNT:-MPHIL-DIS-SL2-CPU}"
cache_partition="${PADIS_CACHE_PARTITION:-icelake}"
cache_cpus="${PADIS_CACHE_CPUS_PER_TASK:-8}"
cache_mem="${PADIS_CACHE_MEM:-128G}"
cache_time="${PADIS_CACHE_SLURM_TIME:-08:00:00}"
cache_variants="${PADIS_CACHE_PREP_VARIANTS:-256-default,256-full,512-default}"
run_root="$(padis_default_run_root)"
run_stamp="${PADIS_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
data_root="${LION_DATA_PATH:-/home/tjh200/rds/hpc-work/Datasets}"
cache_root="${PADIS_CACHE_ROOT:-$data_root/processed/LIDC-IDRI-cache}"
cache_256_archive="${PADIS_256_CACHE_ARCHIVE_FOLDER:-${PADIS_CACHE_ARCHIVE_FOLDER:-$cache_root/padis_256/archives}}"
cache_512_archive="${PADIS_512_CACHE_ARCHIVE_FOLDER:-$cache_root/padis_512/archives}"

mkdir -p "$run_root/debug_runs/slurm_logs" "$cache_root"

export PADIS_RUN_STAMP="$run_stamp"
export PADIS_SLURM_DIR="$SCRIPT_DIR"
export LION_ROOT
export PADIS_CACHE_ROOT="$cache_root"
export PADIS_CACHE_PREP_VARIANTS="$cache_variants"

cache_job="$(
        sbatch \
                --parsable \
                -A "$cache_account" \
                -p "$cache_partition" \
                --cpus-per-task "$cache_cpus" \
                --mem "$cache_mem" \
                --time "$cache_time" \
                --export=ALL \
                --output "$run_root/debug_runs/slurm_logs/%x-%j.out" \
                "$SCRIPT_DIR/slurm_PaDIS_A100_prepare_full_cache.sh"
)"

cat <<EOF
Submitted PaDIS LIDC cache build: $cache_job

Run root: $run_root
Run stamp: $run_stamp
Variants: $cache_variants
Cache root: $cache_root
256 archive folder: $cache_256_archive
512 archive folder: $cache_512_archive
Slurm log: $run_root/debug_runs/slurm_logs/PaDIS_cache_lidc-$cache_job.out

Monitor:
  squeue -j $cache_job

The training/pilot/profile scripts use these archives by default after this job completes.
EOF
