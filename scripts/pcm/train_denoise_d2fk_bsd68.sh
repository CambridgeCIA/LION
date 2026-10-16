export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="1"

python "LION/pcm/train_DWTDenoiseNet.py" \
    --model "dwtdnet" \
    --model_size "tiny" \
    --save_dir "/store/LION/ttv22/pnp/denoiser/20260909_210018_dwt_denoise_df2k_sigma005_dwtdnet_tiny_test_example" \
    --batch_size 128
    # --gs \
    # --save_dir "/store/LION/ttv22/pnp/denoiser/20260909_211519_df2k_bsd68_sigma_0_05_dwtdnet_tiny_gs_test_example" \
    # --model_size "small" \
    # --batch_size 24
    # --model_size "regular" \
    # --batch_size 12 
    # --gs_deepinv \
    # --model "drunet" \
    # --model_size "small" \