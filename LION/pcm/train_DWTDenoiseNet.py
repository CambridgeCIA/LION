from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_wavelets import DWTForward
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from LION.experiments.train_dwtdnet_experiments import DWTDNetTrainParameter
from LION.models.CNNs.drunet import DRUNet
from LION.models.fully_learned.DWTDenoiseNet import load_dwtdnet
from LION.reconstructors.GSPnP import GSDenoiser
from LION.utils.reproduce import seed_from_parts

# Import model

# ============================================================
# Utils
# ============================================================


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(x: Tensor, y: Tensor, data_range: float = 1.0) -> float:
    mse = torch.mean((x - y) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((data_range**2) / mse)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_grayscale_image(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def select_freest_gpu(exclude: set[int] = {2}, allowed: set[int] | None = None) -> int:
    """Selects the GPU with the most free memory among allowed ones.
    Requires nvidia-smi.
    """
    if allowed is None:
        allowed = {0, 1, 3, 4, 5, 6, 7}

    visible = sorted(list(allowed - exclude))
    if len(visible) == 0:
        raise RuntimeError("Nessuna GPU disponibile dopo l'esclusione.")

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, encoding="utf-8")
        rows = out.strip().splitlines()

        candidates: list[tuple[int, int]] = []
        for row in rows:
            idx_str, mem_str = [x.strip() for x in row.split(",")]
            idx = int(idx_str)
            mem_free = int(mem_str)
            if idx in visible:
                candidates.append((idx, mem_free))

        if not candidates:
            raise RuntimeError("Nessuna GPU candidata trovata da nvidia-smi.")

        best_gpu = max(candidates, key=lambda x: x[1])[0]
        return best_gpu

    except Exception as e:
        print(f"[Warning] GPU auto-selection fallita ({e}). Uso cuda:0 se disponibile.")
        return 0


# ============================================================
# Loss
# ============================================================


class DWTLoss(nn.Module):
    """Loss = MSE(out, target) + lambda_wavelet * L1(DWT(out), DWT(target))"""

    def __init__(self, lambda_wavelet: float = 0.1) -> None:
        super().__init__()
        self.lambda_wavelet = lambda_wavelet
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.dwt = DWTForward(J=1, wave="haar", mode="symmetric")

    @staticmethod
    def _flatten_dwt(ll: Tensor, yh: list[Tensor]) -> Tensor:
        parts = [ll]
        for y in yh:
            parts.extend([y[:, :, 0, :, :], y[:, :, 1, :, :], y[:, :, 2, :, :]])
        return torch.cat(parts, dim=1)

    def forward(self, out: Tensor, target: Tensor) -> Tensor:
        mse_term = self.mse(out, target)

        ll_out, yh_out = self.dwt(out)
        ll_tgt, yh_tgt = self.dwt(target)

        dwt_out = self._flatten_dwt(ll_out, yh_out)
        dwt_tgt = self._flatten_dwt(ll_tgt, yh_tgt)

        wavelet_term = self.l1(dwt_out, dwt_tgt)
        return mse_term + self.lambda_wavelet * wavelet_term


class WeightedDWTLoss(nn.Module):
    """Loss = w_mse * MSE(out, tgt) + w_ll * L1(LL) + w_lh * L1(LH) + w_hl * L1(HL) + w_hh * L1(HH)"""

    def __init__(
        self,
        w_mse: float = 1.0,
        w_ll: float = 0.1,  # Low frequencies (approximation / global structure)
        w_lh: float = 0.5,  # High horizontal frequencies
        w_hl: float = 0.5,  # High vertical frequencies
        w_hh: float = 0.8,  # High diagonal frequencies (fine details/noise)
    ) -> None:
        super().__init__()
        self.w_mse = w_mse
        self.w_ll = w_ll
        self.w_lh = w_lh
        self.w_hl = w_hl
        self.w_hh = w_hh

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        # Level J=1.
        self.dwt = DWTForward(J=1, wave="haar", mode="symmetric")

    def forward(self, out: Tensor, target: Tensor) -> Tensor:
        # 1. Standard spatial term
        mse_term = self.w_mse * self.mse(out, target)

        # 2. Wavelet components extraction
        ll_out, yh_out = self.dwt(out)
        ll_tgt, yh_tgt = self.dwt(target)

        # In pytorch_wavelets (J=1), yh_out is a single-element list.
        # Shape of yh_out[0]: [Batch, Channels, 3, H/2, W/2]
        # Indices 0, 1, 2 correspond to LH, HL, and HH respectively.
        lh_out = yh_out[0][:, :, 0, :, :]
        hl_out = yh_out[0][:, :, 1, :, :]
        hh_out = yh_out[0][:, :, 2, :, :]

        lh_tgt = yh_tgt[0][:, :, 0, :, :]
        hl_tgt = yh_tgt[0][:, :, 1, :, :]
        hh_tgt = yh_tgt[0][:, :, 2, :, :]

        # 3. Weighted L1 per band
        loss_ll = self.w_ll * self.l1(ll_out, ll_tgt)
        loss_lh = self.w_lh * self.l1(lh_out, lh_tgt)
        loss_hl = self.w_hl * self.l1(hl_out, hl_tgt)
        loss_hh = self.w_hh * self.l1(hh_out, hh_tgt)

        # 4. Total sum
        return mse_term + loss_ll + loss_lh + loss_hl + loss_hh


class WeightedDWTLoss2(nn.Module):
    """Dynamic Wavelet-based loss.
    Weights change based on epoch to prevent large models from collapsing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()
        self.dwt = DWTForward(J=1, wave="haar", mode="symmetric")

        # Current weights (updated in forward)
        self.cur_w = {"ll": 0.1, "lh": 0.5, "hl": 0.5, "hh": 0.8}

    def forward(self, out: Tensor, target: Tensor, epoch: int) -> Tensor:
        # --- CURRICULUM LOGIC ---
        if epoch <= 2:
            # Phase 1: Full focus on structure (LL) to prevent black output
            self.cur_w = {"ll": 2.0, "lh": 0.05, "hl": 0.05, "hh": 0.05}
        else:
            # Phase 2: Focus on high-frequency denoising
            self.cur_w = {"ll": 0.2, "lh": 0.5, "hl": 0.5, "hh": 1.0}

        # Extract Wavelet components
        ll_out, yh_out = self.dwt(out)
        ll_tgt, yh_tgt = self.dwt(target)

        lh_out, hl_out, hh_out = (
            yh_out[0][:, :, 0],
            yh_out[0][:, :, 1],
            yh_out[0][:, :, 2],
        )
        lh_tgt, hl_tgt, hh_tgt = (
            yh_tgt[0][:, :, 0],
            yh_tgt[0][:, :, 1],
            yh_tgt[0][:, :, 2],
        )

        # Weighted loss calculation
        loss_ll = self.cur_w["ll"] * self.l1(ll_out, ll_tgt)
        loss_lh = self.cur_w["lh"] * self.l1(lh_out, lh_tgt)
        loss_hl = self.cur_w["hl"] * self.l1(hl_out, hl_tgt)
        loss_hh = self.cur_w["hh"] * self.l1(hh_out, hh_tgt)

        return loss_ll + loss_lh + loss_hl + loss_hh


# ============================================================
# Dataset
# ============================================================


class DF2KGrayTrainDataset(Dataset):
    """DF2K grayscale training dataset.
    - convert to grayscale
    - random crop
    - random flip/rotation
    - add Gaussian noise
    """

    def __init__(
        self,
        root: str | Path,
        sigma: float,
        patch_size: int = 128,
        repeats: int = 16,
        base_seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.patch_size = patch_size
        self.sigma = sigma
        self.repeats = repeats
        self.gen = torch.Generator().manual_seed(base_seed)

        self.files = self._gather_files(self.root)
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {self.root}")

    def _gather_files(self, root: Path) -> list[Path]:
        files: list[Path] = []
        for p in root.rglob("*"):
            if p.is_file() and is_image_file(p):
                files.append(p)
        files.sort()
        return files

    def __len__(self) -> int:
        return len(self.files) * self.repeats

    def _random_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        ps = self.patch_size

        if h < ps or w < ps:
            new_h = max(h, ps)
            new_w = max(w, ps)
            img = TF.resize(img, [new_h, new_w], antialias=True)
            w, h = img.size

        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        return TF.crop(img, top, left, ps, ps)

    def _augment(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = TF.hflip(img)
        if random.random() < 0.5:
            img = TF.vflip(img)
        k = random.randint(0, 3)
        if k:
            img = TF.rotate(img, angle=90 * k)
        return img

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img_path = self.files[idx % len(self.files)]
        img = load_grayscale_image(img_path)
        img = self._random_crop(img)
        img = self._augment(img)

        clean = TF.to_tensor(img)  # (1,H,W) in [0,1]
        noise = self.sigma * torch.randn_like(clean, generator=self.gen)
        # noisy = torch.clamp(clean + noise, 0.0, 1.0)  # TODO: Maybe don't clamp?
        noisy = clean + noise  # TODO: Maybe don't clamp?

        return noisy, clean


class BSD68GrayValDataset(Dataset):
    def __init__(self, root: str | Path, sigma: float) -> None:
        super().__init__()
        self.root = Path(root)
        self.sigma = sigma

        self.files = sorted(
            [p for p in self.root.iterdir() if p.is_file() and is_image_file(p)]
        )
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, str]:
        img_path = self.files[idx]
        img = load_grayscale_image(img_path)
        clean = TF.to_tensor(img)

        gen = torch.Generator()
        gen.manual_seed(1234 + idx)
        noise = self.sigma * torch.randn_like(clean, generator=gen)
        # noisy = torch.clamp(clean + noise, 0.0, 1.0)  # TODO: Maybe don't clamp?
        noisy = clean + noise  # TODO: Maybe don't clamp?

        return noisy, clean, img_path.name


# ============================================================
# Train / Val
# ============================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    sigma: float,
    print_freq: int = 100,
) -> float:
    model.train()
    running_loss = 0.0

    for step, (noisy, clean) in enumerate(loader, start=1):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        out = model(noisy, sigma)
        # loss = criterion(out, clean)
        loss = criterion(out, clean, epoch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % print_freq == 0 or step == len(loader):
            print(
                f"[Train] Epoch {epoch:03d} "
                f"Step {step:04d}/{len(loader):04d} "
                f"Loss {running_loss / step:.6f}"
            )

    return running_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    sigma: float,
    images_dir: Path | None = None,
) -> tuple[float, float]:
    model.eval()

    if images_dir is not None:
        images_dir.mkdir(parents=True, exist_ok=True)
        csv_path = images_dir / "psnr.csv"
        with open(csv_path, "w") as f:
            f.write("image_name,psnr\n")

    total_loss = 0.0
    total_psnr = 0.0

    for noisy, clean, _name in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        out = model(noisy, sigma)

        example_noisy = noisy[0].cpu().squeeze(0).numpy()
        example_clean = clean[0].cpu().squeeze(0).numpy()
        example_recon = out[0].cpu().squeeze(0).numpy()
        if images_dir is not None:
            save_dir = images_dir / _name[0]
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.imsave(
                save_dir / "noisy.png", example_noisy, cmap="gray", vmin=0, vmax=1
            )
            plt.imsave(
                save_dir / "clean.png", example_clean, cmap="gray", vmin=0, vmax=1
            )
            plt.imsave(
                save_dir / "recon.png", example_recon, cmap="gray", vmin=0, vmax=1
            )
            example_psnr = psnr(out[0], clean[0], data_range=1.0)
            with open(csv_path, "a") as f:
                f.write(f"{_name[0]},{example_psnr:.4f}\n")

        loss = criterion(out, clean, epoch)

        total_loss += loss.item()
        # total_psnr += psnr(out.clamp(0, 1), clean, data_range=1.0)
        total_psnr += psnr(out, clean, data_range=1.0)

    return total_loss / len(loader), total_psnr / len(loader)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_size",
        type=str,
        # default="regular",
        default="tiny",
        choices=["regular", "small", "tiny"],
        help="Model size: regular | small | tiny",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        # default=4,
        default=12,
        help="Batch size",
    )

    parser.add_argument("--gs", action="store_true", help="Use gradient step")
    parser.add_argument(
        "--gs_deepinv", action="store_true", help="Use deepinv gradient step"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dwtdnet",
        choices=["dwtdnet", "drunet"],
        help="Model type",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Directory to save checkpoints"
    )

    args = parser.parse_args()
    return args


# ============================================================
# Main
# ============================================================


def main() -> None:
    set_seed(42)

    args = parse()

    model_size = args.model_size
    batch_size = args.batch_size

    # ---------- select GPU ----------
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    if visible is None:
        gpu_id = select_freest_gpu(exclude={2}, allowed={0, 1, 3, 4, 5, 6, 7})
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Auto-selected GPU {gpu_id}")
    else:
        gpu_id = int(visible.split(",")[0])
        print(f"Using CUDA_VISIBLE_DEVICES={visible} from environment")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- hyperparameters ----------
    if args.model == "dwtdnet":
        assert model_size is not None
        if args.gs:
            config = {
                "tiny": DWTDNetTrainParameter.gs_dwtdnet_tiny(),
                "small": DWTDNetTrainParameter.gs_dwtdnet_small(),
                "regular": DWTDNetTrainParameter.gs_dwtdnet_regular(),
            }[model_size]
        else:
            if model_size == "tiny":
                config = DWTDNetTrainParameter.dwtdnet_tiny()
            else:
                raise ValueError(
                    f"Unsupported model size for non-gradient-step: {model_size}"
                )
    elif args.model == "drunet":
        config = DWTDNetTrainParameter.gs_drunet()
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # sigma = 0.05
    sigma = config.sigma
    lambda_wavelet = 0.1

    patch_size = 128
    train_repeats = 16
    model_size = args.model_size
    batch_size = args.batch_size
    num_workers = 4

    # epochs = 10
    # lr = 1e-4
    # betas = (0.9, 0.99)
    # eps = 1e-8
    # weight_decay = 0.0
    # optim_config = AdamParameter(name="Adam", learning_rate=1e-4, betas=[0.9, 0.99], eps=1e-8, weight_decay=0.0)
    # scheduler_config = CosineAnnealingLRParameter(T_max=epochs, eta_min=1e-6)

    # ---------- path ----------
    # train_dir = Path("/raid/homespace/piecestola/space/DWTSRNet/DF2K/DF2K_train_HR")
    train_dir = Path("/store/LION/datasets/Download_df2k/datasets/DIV2K_train_HR")
    # val_dir = Path("/raid/homespace/piecestola/space/troy/BSD68/original")
    # val_dir = Path("/store/LION/datasets/CBSD68-dataset/CBSD68/original")
    val_dir = Path("/store/LION/datasets/deepinv/CBSD68/0")
    # save_dir = Path(f"checkpoints/dwt_denoise_df2k_sigma005_{model_size}")
    cur_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.model == "dwtdnet":
        suffix = f"dwtdnet_{model_size}"
        if args.gs:
            suffix += "_gs"
    elif args.model == "drunet":
        suffix = "drunet"
        if args.gs:
            if args.gs_deepinv:
                suffix += "_gs_deepinv"
            else:
                suffix += "_gs"
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    # root_dir = Path("/store/LION/ttv22/dwt/checkpoints/")
    data_name = "df2k_bsd68"
    root_dir = Path("/store/LION/ttv22/pnp/denoiser")
    # save_dir = root_dir / (f"{cur_datetime}_{data_name}_sigma{f'{sigma}'.replace('.', '_')}_{suffix}")
    if isinstance(sigma, float):
        sigma_str = f"sigma_{sigma:.2f}".replace(".", "_")
    else:
        # assert isinstance(sigma, list) and len(sigma) == 2, f"Got {sigma}"
        # sigma_lo = sigma[0]
        # sigma_hi = sigma[1]
        # sigma_str = f"sigmas_{sigma_lo:.2f}_to_{sigma_hi:.2f}".replace(".", "_")
        raise ValueError(f"Expected sigma to be a float, got {sigma}")
    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
    else:
        save_dir = root_dir / (f"{cur_datetime}_{data_name}_{sigma_str}_{suffix}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------- dataset ----------
    train_set = DF2KGrayTrainDataset(
        root=train_dir,
        sigma=sigma,
        patch_size=patch_size,
        repeats=train_repeats,
    )
    val_set = BSD68GrayValDataset(
        root=val_dir,
        sigma=sigma,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

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

    class CustomMSELoss(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mse = nn.MSELoss()

        def forward(self, out: Tensor, target: Tensor, epoch: int) -> Tensor:
            return self.mse(out, target)

    if args.model == "dwtdnet":
        model = load_dwtdnet(model_size=model_size, img_size=patch_size, device=device)
        # criterion = DWTLoss(lambda_wavelet=lambda_wavelet).to(device)
        # criterion = WeightedDWTLoss(lambda_wavelet,0.1,1.0,1.0,1.0).to(device)
        criterion = WeightedDWTLoss2().to(device)
        if args.gs:
            model = GSDenoiser(model).to(device=device)
    elif args.model == "drunet":
        if not args.gs or not args.gs_deepinv:
            from deepinv.models import DRUNet

            model = DRUNet(
                in_channels=1, out_channels=1, pretrained=None, device=device
            )
            if args.gs:
                model = GSDenoiser(model).to(device=device)
        else:
            from deepinv.models import GSDRUNet

            model = GSDRUNet(
                in_channels=1, out_channels=1, pretrained=None, device=device
            )
        criterion = CustomMSELoss()
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    print("Using loss function:", criterion)
    print("Model summary:")
    print(model)

    optim_config = config.optim
    if optim_config.name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optim_config.learning_rate,
            betas=(optim_config.betas[0], optim_config.betas[1]),
            eps=optim_config.eps,
            weight_decay=optim_config.weight_decay,
        )
    elif optim_config.name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optim_config.learning_rate,
            betas=(optim_config.betas[0], optim_config.betas[1]),
            eps=optim_config.eps,
            weight_decay=optim_config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer name: {optim_config.name}")

    epochs = config.epochs
    scheduler_config = config.scheduler
    if scheduler_config is None:
        scheduler = None
    else:
        T_max = scheduler_config.T_max if scheduler_config.T_max is not None else epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=scheduler_config.eta_min
        )

    # ---------- resume ----------
    latest_ckpt = save_dir / "latest.pt"
    # start_epoch = 1
    start_epoch = config.start_epoch
    best_val_psnr = -1.0
    train_csv = save_dir / "train_log.csv"
    if not train_csv.exists():
        with Path.open(train_csv, "w") as f:
            f.write("epoch,train_loss,val_loss,val_psnr,lr\n")
    val_csv = save_dir / "val_log.csv"
    if not val_csv.exists():
        with Path.open(val_csv, "w") as f:
            f.write("epoch,val_loss,val_psnr\n")

    if latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"]) if scheduler is not None else None
        start_epoch = ckpt["epoch"] + 1
        best_val_psnr = ckpt["best_val_psnr"]
        print(f"Resumed from {latest_ckpt} at epoch {ckpt['epoch']}")

    config.start_epoch = start_epoch
    config.save(save_dir / f"{cur_datetime}_train_config.json")

    images_dir = save_dir / "images"

    val_loss, val_psnr = validate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        epoch=start_epoch - 1,
        sigma=sigma,
        images_dir=images_dir / f"epoch_{start_epoch-1:03d}",
    )

    # ---------- training ----------
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            sigma=sigma,
            print_freq=100,
        )

        val_loss, val_psnr = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            sigma=sigma,
            images_dir=images_dir / f"epoch_{epoch:03d}",
        )

        scheduler.step() if scheduler is not None else None
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_psnr={val_psnr:.3f} dB "
            f"lr={lr_now:.12e}"
        )

        with Path.open(train_csv, "a") as f:
            f.write(f"{epoch},{train_loss},{val_loss},{val_psnr},{lr_now}\n")

        with Path.open(val_csv, "a") as f:
            f.write(f"{epoch},{val_loss},{val_psnr}\n")

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_val_psnr": best_val_psnr,
                "gpu_id": gpu_id,
            },
            latest_ckpt,
        )

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                    if scheduler is not None
                    else None,
                    "best_val_psnr": best_val_psnr,
                    "gpu_id": gpu_id,
                },
                save_dir / "best.pt",
            )
            print(f"Saved new best checkpoint with val_psnr={best_val_psnr:.3f} dB")

    print(f"Training finished. Best validation PSNR: {best_val_psnr:.3f} dB")


if __name__ == "__main__":
    main()
