CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
python scripts/pcm/photocurrent_mapping.py \
    --preset silicon_1_measurement_768x1024_real_noise \
    --sampling.sampling-ratios 0.05 0.1 0.2 0.5 1.0 \
    --sampling.coarse-j-values 6 7 8 9 10 \
    --runtime.num-trials 1
    # tv:none \
    # spgl1:none
    # --runtime.no-skip-exceptions \
    # pnp-admm:none
