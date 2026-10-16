CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
python scripts/pcm/photocurrent_mapping.py \
    --preset cigs_example_256x256 \
    --sampling.sampling-ratios 0.2 \
    --sampling.coarse-j-values 7 \
    spgl1:none \
    pnp_admm:pnp_admm_config \
    --pnp_admm.drunet_sigma 0.05 \
    --pnp_admm.eta 0.1 \
    --pnp_admm.iters 50
    # --gaussian_noise_std_ratio 0.1 \
    # --pnp_admm.drunet_sigma 0.1 \
    # --pnp_admm.eta 1.0 \



