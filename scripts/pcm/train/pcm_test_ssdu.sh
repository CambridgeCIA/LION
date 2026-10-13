export MPLBACKEND="agg"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export CUDA_VISIBLE_DEVICES="0"
export CUDA_VISIBLE_DEVICES="-1"

ROOT="/store/LION/ttv22/pcm/results/pcm/pcm_self_supervised"
# EXPERIMENT="20260920_165908_TV_self_supervised"
EXPERIMENT="20260920_183316_TV_self_supervised"

python scripts/pcm/train/pcm_supervised_sapm_test.py \
    --epochs best_val_loss_s0.20_cJ6.pt \
    --root $ROOT --experiment $EXPERIMENT
    # --epochs step_000500 \
    # --sample-ids 611
