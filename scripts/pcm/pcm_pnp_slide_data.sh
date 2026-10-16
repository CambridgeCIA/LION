#!/usr/bin/env bash
set -euo pipefail

# export PCM_DATA_DIR="${PCM_DATA_DIR:-/store/LION/ttv22/pcm/data/photocurrent_data}"
# # export PCM_DATA_DIR="${PCM_DATA_DIR:-/Volumes/T7 Touch/store/LION/ttv22/pcm/data/photocurrent_data}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
# export PCM_DEVICE="${PCM_DEVICE:-mps}"
export PCM_DEVICE="${PCM_DEVICE:-cuda}"
export PCM_OUTPUT_ROOT="${PCM_OUTPUT_ROOT:-$PWD/pcm_slide_results}"

args=(--output-root "$PCM_OUTPUT_ROOT" --device "$PCM_DEVICE")
if [[ -n "${PCM_DATA_DIR:-}" ]]; then
    args+=(--data-root "$PCM_DATA_DIR")
fi


# python scripts/pcm/pcm_pnp_slide_data.py \
#     --data-root "$PCM_DATA_DIR" \
#     --output-root "${PCM_OUTPUT_ROOT:-pcm_slide_results}" \
#     --device "${PCM_DEVICE:-auto}" \
#     "$@"


# python scripts/pcm/pcm_pnp_slide_data.py "${args[@]}" "$@"


# python scripts/pcm/pcm_pnp_slide_data.py "${args[@]}" \
#   --preset synthetic_png_dataset_box_plots \
#   --category CIGS \
#   --sample-id 288 \
#   "$@"

python scripts/pcm/pcm_pnp_slide_data.py "${args[@]}" \
  --preset cigs_example_256x256 \
  --output-name cia_pnp_slide_cigs_example_256x256 \
  --scalar-tv-lambda 0.001 \
  "$@"
