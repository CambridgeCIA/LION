CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
python scripts/pcm/photocurrent_mapping.py \
    --preset silicon_1_image_768x1024_no_noise \
    --sampling.sampling-ratios 0.05 0.1 0.2 0.5 1.0 \
    --sampling.coarse-j-values 6 7 8 9 10 \
    --runtime.num-trials 1 \
    pcm-noise:none \
    pnp_admm:pnp_admm_config \
    --pnp_admm.denoiser-name "drunet" \
    --pnp_admm.drunet_sigma 0.05 \
    --pnp_admm.eta 0.01 \
    --pnp_admm.iters 50

    # tv:none \
    # pnp-admm:none
    # --runtime.no-skip-exceptions
