CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
python scripts/pcm/pcm_test_single_sample.py \
    --preset cigs_example_256x256_scale_noise_default_no_dark \
    --sampling.sampling-ratios 0.2 0.5 1.0 \
    --sampling.coarse-j-values 6 7 8 \
    --runtime.num-trials 1 \
    spgl1:none \
    pnp_admm:pnp_admm_config \
    --pnp_admm.denoiser-name "drunet"
    # tv:none \
    # pnp-admm:none
    # --runtime.no-skip-exceptions
