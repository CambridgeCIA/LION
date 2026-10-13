export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

# python scripts/pcm/pcm_test_single_sample.py \
#     --log.save-plots \
#     --preset "e27_pos1_real_measurement"
#     # --sampling.coarse-j-values 6 \
#     # --sampling.sampling-ratios 0.2 0.1 \
#     # --runtime.num-trials 3 \
#     # --preset "e27_pos1_average_map_simulated_noise"
#     # --preset "cigs_example_256x256"
#     # --runtime.num-trials 1 \
#     # --sampling.sampling-ratios 0.2 \
#     # --sampling.coarse-j-values 7 \
#     # pnp-admm:pnp-admm-config \
#     # --pnp-admm.denoiser-name "drunet" \
#     # --pnp-admm.drunet-sigma 0.05 \
#     # --pnp-admm.eta 0.01 \
#     # --pnp-admm.iters 50 \
#     # spgl1:none

python scripts/pcm/pcm_test_single_sample.py \
    --log.save-plots \
    --preset "e27_pos1_average_map_simulated_noise" \
    noise:none
    # --preset "cigs_example_256x256"