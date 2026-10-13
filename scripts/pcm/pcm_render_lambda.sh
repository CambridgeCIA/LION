export CUDA_VISIBLE_DEVICES="-1"

python scripts/pcm/pcm_render_lambda.py \
    --tensor-path "/store/LION/ttv22/pcm/results/pcm/pcm_zero_shot/20260917_012919_TV_zero_shot/CIGS_s0.20_cJ6_sample_0005/recons/best_val_ssdu_loss/lambda_1_v.pt" \
    --cmap magma \
    --vmin 0.0 \
    --vmax 0.5 \
    --image-formats png pdf
    