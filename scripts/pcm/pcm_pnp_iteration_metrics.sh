#!/usr/bin/env bash
set -e

export MPLBACKEND="${MPLBACKEND:-agg}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Multiple method IDs can be listed after --method-ids.
# Multiple integers can be listed after --noise-realizations.
# Plot modes: individual, methods-per-scenario, scenarios-per-method.
# Snapshot position: top, bottom, left, or right.
python scripts/pcm/pcm_pnp_iteration_metrics.py \
    --method-ids pnp_admm_drunet \
    --sample-name cigs_example_256x256 \
    --preset cigs_example_256x256 \
    --sampling.sampling-ratios 0.1 \
    --sampling.coarse-j-values 6 \
    --output-name pnp_iteration_demo_cigs_256x256 \
    --include-no-noise \
    --noise-realizations 0 \
    --plot.snapshot-iterations 1 5 10 \
    --plot.snapshot-every 5 \
    --plot.show-best \
    --plot.best-metric psnr \
    --plot.show-last \
    --plot.show-early-stop \
    --plot.show-backtracking-stop \
    --plot.metrics psnr ssim haarpsi mse \
    --plot.x-axes iteration time \
    --plot.modes individual \
    --plot.snapshot-position top \
    --plot.dpi 300 \
    --runtime.device cuda \
    --runtime.num-trials 1 \
    --runtime.num-trials-skip 0 \
    --sampling.randomising-scheme multilevel \
    --log.save-metrics \
    --log.save-plots

# Larger example:
#   --sample-name e27_pos1_average_map
#   --preset e27_pos1_average_map_simulated_noise
#   --sampling.sampling-ratios 0.2
#   --sampling.coarse-j-values 8
