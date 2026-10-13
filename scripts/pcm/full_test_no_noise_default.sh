CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
python scripts/pcm/photocurrent_mapping.py \
    --preset silicon_1_measurement_768x1024 \
    --sampling.sampling-ratios 0.01 0.02 0.03 0.04 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --sampling.coarse_j_values 1 2 3 4 5 6 7 8 9 10 \
    noise:none \
    spgl1:none \
    pnp_admm:pnp_admm_config --pnp_admm.iters 30
