export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

python -m scripts.pcm.photocurrent_mapping \
  --preset cigs_example_256x256 \
  --runtime.num-trials 10 \
  --sampling.randomising-scheme 'uniform' \
  --sampling.sampling-ratios 0.02 0.025 0.03 0.035 0.04 0.045 0.05 \
  --pnp.denoiser-name 'dwtdnet' \
  --pnp.iters 50 \
  --spgl1.no-enabled \
  --runtime.device cpu
