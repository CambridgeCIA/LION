CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
python scripts/pcm/photocurrent_mapping.py \
    --preset cigs_example_256x256_scale \
    spgl1:none \
    pnp_admm:pnp_admm_config \
    --pnp_admm.eta 1.0 \
    --pnp_admm.drunet_sigma 0.05 \
    --pnp_admm.iters 100
    # pcm_noise:none \
