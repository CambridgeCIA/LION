export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# ITERS="256"
ITERS="1024"

python scripts/pcm/pcm_scalar_tv_sweep.py \
  --category CIGS \
  --sample-id 288 \
  --device cuda \
  --lambdas 0.0001 0.0005 0.001 0.005 0.01 0.02 0.1 \
  --iters "$ITERS" \
  --metric-every 4 \
  --output-root "pcm_tv_sweep_$ITERS" \


# python scripts/pcm/pcm_scalar_tv_sweep.py \
#   --preset cigs_example_256x256 \
#   --device cuda \
#   --lambdas 0.0001 0.0005 0.001 0.005 0.01 0.02 0.1 \
#   --iters "$ITERS" \
#   --metric-every 4 \
#   --output-root "pcm_tv_sweep_cigs_example_$ITERS"