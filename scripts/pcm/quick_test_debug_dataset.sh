export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export CUDA_VISIBLE_DEVICES="0"
export CUDA_VISIBLE_DEVICES="1"
# export CUDA_VISIBLE_DEVICES="-1"

python scripts/pcm/pcm_test_dataset.py \
    --data.samples.perm-range 991 1000 \
    --preset "synthetic_png_dataset_box_plots" \
    noise:none
    # --data.samples.perm-range 1000 1000 \
    # --data.categories "CIGS" \
    # --data.samples.perm-range 991 1000 \
    # --sampling.sampling-ratios 1.0 \
    # --sampling.coarse-j-values 7 \
    # --noise.n_repeat_acquisitions 6 \
    # --preset "synthetic_png_dataset_grid"
    # --data.samples.ids 623 \
    # --data.samples.ids 623 762 199 681 254 288 \
    # --preset "synthetic_png_dataset_30_trials"
    # --data.categories "CIGS" "Organic" \
    # --sampling.sampling-ratios 0.2 0.5 \
    # --sampling.coarse-j-values 1 6 \
    # --runtime.num-trials 2 \
    # --data.samples.ids 1 2 \
    # --data.samples.perm-range 901 902 \
    # pnp-admm:pnp-admm-config \
    # --pnp-admm.denoiser-name "drunet" \
    # --pnp-admm.drunet-sigma 0.05 \
    # --pnp-admm.eta 0.01 \
    # --pnp-admm.iters 50 \
    # spgl1:none

python scripts/pcm/pcm_test_dataset.py \
    --data.samples.perm-range 991 1000 \
    --preset "synthetic_png_dataset_box_plots"
