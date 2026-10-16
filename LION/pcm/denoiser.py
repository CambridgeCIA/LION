from __future__ import annotations

import deepinv
import torch
from deepinv.models.utils import get_weights_url, load_state_dict_from_url

from LION.models.fully_learned.DWTDenoiseNet import load_dwtdnet
from LION.reconstructors.BasePnP import DenoiserParameter
from LION.reconstructors.GSPnP import GSDenoiser
from LION.reconstructors.PnPADMM import ADMMDenoiser


def load_dwtdnet_checkpoint(
    denoiser: torch.nn.Module, denoiser_param: DenoiserParameter
) -> None:
    assert denoiser_param.weights_path is not None
    ckpt = torch.load(denoiser_param.weights_path, map_location="cpu")
    denoiser.load_state_dict(ckpt["model"])
    # # "epoch": epoch,
    # # "model": model.state_dict(),
    # # "optimizer": optimizer.state_dict(),
    # # "scheduler": scheduler.state_dict() if scheduler is not None else None,
    # # "best_val_psnr": best_val_psnr,
    # # "gpu_id": gpu_id,
    # print(
    #     f'Denoiser {denoiser_param.name}. Epoch {ckpt["epoch"]} checkpoint loaded. Best val PSNR: {ckpt["best_val_psnr"]:.2f} dB.'
    # )


def load_deepinv_gs_drunet():
    # denoiser = deepinv.models.GSDRUNet(
    #     in_channels=1, out_channels=1, device=torch.device("cpu"), pretrained="download"
    # )
    gs_denoiser = deepinv.models.GSDRUNet(
        in_channels=1, out_channels=1, device=torch.device("cpu"), pretrained=None
    )
    file_name = "GSDRUNet_grayscale_torch.ckpt"
    url = get_weights_url(model_name="gradientstep", file_name=file_name)
    # ckpt = load_state_dict_from_url(url, map_location=lambda storage, loc: storage, file_name=file_name)
    ckpt = load_state_dict_from_url(url, map_location="cpu", file_name=file_name)
    model_keys = gs_denoiser.state_dict().keys()
    ckpt = {key: value for key, value in ckpt.items() if key in model_keys}
    gs_denoiser.load_state_dict(ckpt, strict=True)
    return gs_denoiser


def build_grayscale_denoiser(
    denoiser_param: DenoiserParameter, device: torch.device
) -> torch.nn.Module:
    """Build the configured deepinv denoiser."""
    with torch.device("cpu"):
        if denoiser_param.name == "drunet":
            denoiser = deepinv.models.DRUNet(
                pretrained="download", in_channels=1, out_channels=1, device="cpu"
            )
        elif denoiser_param.name == "drunet_admm":
            backend = deepinv.models.DRUNet(
                pretrained="download", in_channels=1, out_channels=1, device="cpu"
            )
            denoiser = ADMMDenoiser(
                network=backend, b=denoiser_param.b, eps=denoiser_param.eps
            )
        elif denoiser_param.name == "gs_drunet":
            # print("Loading deepinv GSDRUNet model's backend only")
            gs_denoiser = load_deepinv_gs_drunet()
            denoiser = GSDenoiser(
                gs_denoiser.student_grad.model
            )  # Get the deepinv backend model only
        elif denoiser_param.name == "gs_drunet_deepinv":
            # print("Loading full deepinv GSDRUNet model")
            gs_denoiser = load_deepinv_gs_drunet()
            denoiser = gs_denoiser  # Use the full deepinv model
        elif denoiser_param.name == "gs_dwtdnet":
            denoiser = GSDenoiser(
                load_dwtdnet(model_size=denoiser_param.size, img_size=128, device="cpu")
            )
            load_dwtdnet_checkpoint(denoiser, denoiser_param)
        elif denoiser_param.name == "dwtdnet":

            # with torch.device("cpu"):
            # denoiser = DWTDenoiseNet(
            #     img_size=128,
            #     patch_size=1,
            #     embed_dim=32,
            #     depths=(3, 3),
            #     num_heads=(2, 2),
            #     window_size=8,
            #     mlp_ratio=4.0,
            #     qkv_bias=True,
            #     drop_rate=0.0,
            #     attn_drop_rate=0.0,
            #     drop_path_rate=0.03,
            #     norm_layer=torch.nn.LayerNorm,
            #     ape=False,
            #     patch_norm=True,
            #     use_checkpoint=False,
            #     resi_connection="1conv",
            #     clamp_output=True,
            # )
            # # ---------- model ----------
            # if model_size == "regular":
            #     model = DWTDenoiseNet(
            #         img_size=patch_size,
            #         patch_size=1,
            #         embed_dim=96,
            #         depths=(6, 6, 6, 6),
            #         num_heads=(6, 6, 6, 6),
            #         window_size=8,
            #         mlp_ratio=4.0,
            #         qkv_bias=True,
            #         drop_rate=0.0,
            #         attn_drop_rate=0.0,
            #         drop_path_rate=0.1,
            #         norm_layer=nn.LayerNorm,
            #         ape=False,
            #         patch_norm=True,
            #         use_checkpoint=False,
            #         resi_connection="1conv",
            #         clamp_output=True,
            #     ).to(device)
            # elif model_size == "small":
            #     model = DWTDenoiseNet(
            #         img_size=patch_size,
            #         patch_size=1,
            #         embed_dim=48,
            #         depths=(4, 4, 4),
            #         num_heads=(3, 3, 3),
            #         window_size=8,
            #         mlp_ratio=4.0,
            #         qkv_bias=True,
            #         drop_rate=0.0,
            #         attn_drop_rate=0.0,
            #         drop_path_rate=0.05,
            #         norm_layer=nn.LayerNorm,
            #         ape=False,
            #         patch_norm=True,
            #         use_checkpoint=False,
            #         resi_connection="1conv",
            #         clamp_output=True,
            #     ).to(device)
            # elif model_size == "tiny":
            #     model = DWTDenoiseNet(
            #         img_size=patch_size,
            #         patch_size=1,
            #         embed_dim=32,
            #         depths=(3, 3),
            #         num_heads=(2, 2),
            #         window_size=8,
            #         mlp_ratio=4.0,
            #         qkv_bias=True,
            #         drop_rate=0.0,
            #         attn_drop_rate=0.0,
            #         drop_path_rate=0.03,
            #         norm_layer=nn.LayerNorm,
            #         ape=False,
            #         patch_norm=True,
            #         use_checkpoint=False,
            #         resi_connection="1conv",
            #         clamp_output=True,
            #     ).to(device)
            # else:
            #     raise ValueError(f"Unknown model size: {model_size}")
            denoiser = load_dwtdnet(
                model_size=denoiser_param.size, img_size=128, device=device
            )

            # weights_path = (
            #     # "/raid/homespace/piecestola/space/troy/checkpoints_no_weights/dwt_denoise_df2k_sigma005_tiny/best.pt"
            # )
            load_dwtdnet_checkpoint(denoiser, denoiser_param)
        elif denoiser_param.name == "dwtdnet_admm":
            backend = load_dwtdnet(
                model_size=denoiser_param.size, img_size=128, device=device
            )
            load_dwtdnet_checkpoint(backend, denoiser_param)
            denoiser = ADMMDenoiser(
                network=backend, b=denoiser_param.b, eps=denoiser_param.eps
            )
        else:
            raise ValueError(f"Unknown denoiser_name '{denoiser_param.name}'.")

    denoiser = denoiser.to(device=device)
    denoiser.eval()
    return denoiser
