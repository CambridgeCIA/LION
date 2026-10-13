# Set MPLBACKEND to "agg" to avoid potential issues with matplotlib when running on a server without display.
# Set CUDA_DEVICE_ORDER to "PCI_BUS_ID" for consistent ordering with nvidia-smi.
export MPLBACKEND="agg"
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="1"

python scripts/pcm/train/pcm_self_supervised_sapm_train.py \
    --config config/pcm/pcm_tv_self_supervised.yaml
