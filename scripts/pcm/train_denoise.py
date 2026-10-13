import deepinv as dinv
import torch

from LION.models.fully_learned.DWTDenoiseNet import DWTDenoiseNet

device = dinv.utils.get_device()

model = DWTDenoiseNet(
    img_size=128,
    patch_size=1,
    embed_dim=32,
    depths=(3, 3),
    num_heads=(2, 2),
    window_size=8,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.03,
    norm_layer=torch.nn.LayerNorm,
    ape=False,
    patch_norm=True,
    use_checkpoint=False,
    resi_connection="1conv",
    clamp_output=True,
).to(device=device)

physics = dinv.physics.Denoising(
    noise_model=dinv.physics.UniformGaussianNoise(
        sigma_min=0.0,
        sigma_max=50 / 255,
    )
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
)

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    optimizer=optimizer,
    train_dataloader=train_loader,
    eval_dataloader=val_loader,
    losses=dinv.loss.SupLoss(),
    epochs=100,
    device=device,
)

trainer.train()
