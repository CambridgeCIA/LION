# CUDA_DEVICE_ORDER=PCI_BUS_ID \
# CUDA_VISIBLE_DEVICES=1 \
# python scripts/pcm/photocurrent_mapping.py \
#     --preset silicon_1b_Ap100us_measurement_768x1024 \
#     spgl1:none \
#     pnp_admm:pnp_admm_config --pnp_admm.lagged_iters 1 --pnp_admm.lagged_damping 1 --pnp_admm.data_weight 1 \
#     --pnp_admm.iters 30 --pnp_admm.eta 0.01 \
#     pcm-noise:none


# CUDA_DEVICE_ORDER=PCI_BUS_ID \
# CUDA_VISIBLE_DEVICES=1 \
# python scripts/pcm/photocurrent_mapping.py \
#     --preset silicon_1_measurement_768x1024 \
#     --gaussian_noise_std 1e-8 \
#     spgl1:none \
#     pnp_admm:pnp_admm_config --pnp_admm.lagged_iters 1 --pnp_admm.lagged_damping 1 --pnp_admm.data_weight 1 \
#     --pnp_admm.iters 30 \
#     precision-builder:none \
#     pcm-noise:none


CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
python scripts/pcm/photocurrent_mapping.py \
    --preset silicon_1_image_768x1024 \
    --sampling.sampling-ratios 0.2 \
    pcm-noise:none \
    precision-builder:none \
    spgl1:none \
    pnp_admm:pnp_admm_config \
    --pnp_admm.iters 30
    # --gaussian_noise_std 5e-6 \


# CUDA_DEVICE_ORDER=PCI_BUS_ID \
# CUDA_VISIBLE_DEVICES=1 \
# python scripts/pcm/photocurrent_mapping.py \
#     --preset silicon_1_measurement_768x1024 \
#     --sampling.sampling-ratios 0.2 \
#     spgl1:none \
#     pnp_admm:pnp_admm_config \
#     --pnp_admm.iters 30
#     # --gaussian_noise_std 5e-6 \
