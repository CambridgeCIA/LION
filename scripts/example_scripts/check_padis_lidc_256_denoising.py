"""Visual sanity check for the PaDIS LIDC 256 denoiser checkpoint."""

from __future__ import annotations

import argparse
import os
import pathlib
import warnings

_CACHE_ROOT = pathlib.Path("/tmp") / "lion_matplotlib_cache"
(_CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import torch

from LION.CTtools.ct_geometry import Geometry
from LION.losses.PaDIS import sample_image_patch_with_position_channels, zero_pad_images
from LION.models.diffusion import NCSNpp
from LION.utils.parameter import LIONParameter


DEFAULT_CHECKPOINT = pathlib.Path(
    "Data/experiments/PaDIS/LIDC_256/"
    "padis_lidc_256_reproduction_CSD3/padis_lidc_256.pt"
)


def torch_load(path: pathlib.Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def resolve_checkpoint_path(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((pathlib.Path.cwd() / path, project_root() / path))

    if path.parts[:1] == ("Data",):
        data_root = os.environ.get("LION_DATA_PATH")
        if data_root is not None:
            candidates.append(
                pathlib.Path(data_root).expanduser() / pathlib.Path(*path.parts[1:])
            )

    experiments_root = os.environ.get("LION_EXPERIMENTS_PATH")
    if experiments_root is not None and path == DEFAULT_CHECKPOINT:
        candidates.append(
            pathlib.Path(experiments_root).expanduser()
            / "PaDIS/LIDC_256/padis_lidc_256_reproduction_CSD3/padis_lidc_256.pt"
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Checkpoint not found. Tried:\n  {tried}")


def load_model(checkpoint_path: pathlib.Path, device: torch.device, use_ema: bool):
    json_path = checkpoint_path.with_suffix(".json")
    if json_path.is_file():
        options = LIONParameter()
        options.load(json_path)
        if getattr(options, "model_name", "NCSNpp") != "NCSNpp":
            warnings.warn(
                f"{json_path} says model_name={options.model_name!r}; trying NCSNpp anyway."
            )
        model_params = options.model_parameters
        geometry = Geometry.init_from_parameter(options.geometry)
    else:
        warnings.warn(
            f"No sidecar JSON found at {json_path}; using PaDIS LIDC 256 defaults."
        )
        model_params = NCSNpp.default_parameters("padis-paper-ct-256")
        geometry = Geometry.default_parameters(image_scaling=0.5)

    model = NCSNpp(model_params, geometry).to(device)
    checkpoint = torch_load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    ema_state = (
        checkpoint.get("ema_state_dict") if isinstance(checkpoint, dict) else None
    )
    ema_path = checkpoint_path.with_suffix(".ema.pt")
    if ema_state is None and ema_path.is_file():
        ema_checkpoint = torch_load(ema_path, map_location=device)
        ema_state = ema_checkpoint.get("ema_state_dict")

    if use_ema and ema_state is not None:
        state_dict = dict(state_dict)
        state_dict.update(ema_state)
        print("Loaded EMA weights for denoising check.")

    model.load_state_dict(state_dict)
    model.eval()
    return model, model_params, geometry


def make_synthetic_images(
    num_examples: int, image_size: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    coords = torch.linspace(-1.0, 1.0, image_size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    images = []

    for _ in range(num_examples):
        image = torch.zeros_like(xx)
        for _ in range(6):
            cx, cy = torch.rand(2, generator=generator) * 1.6 - 0.8
            sx, sy = torch.rand(2, generator=generator) * 0.18 + 0.06
            amp = torch.rand(1, generator=generator).item() * 0.7 + 0.3
            blob = torch.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)
            image = image + amp * blob
        image = image / image.amax().clamp_min(1e-6)
        images.append(image.unsqueeze(0))

    return torch.stack(images, dim=0).float()


def load_lidc_images(args, geometry) -> torch.Tensor:
    from LION.data_loaders.LIDC_IDRI import LIDC_IDRI

    data_params = LIDC_IDRI.default_parameters(geometry=geometry, task="image_prior")
    data_params.device = torch.device("cpu")
    if args.data_folder is not None:
        data_params.folder = args.data_folder

    dataset = LIDC_IDRI(
        args.split, parameters=data_params, geometry_parameters=geometry
    )
    if len(dataset) == 0:
        raise ValueError(f"LIDC {args.split!r} split is empty.")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[: args.num_examples]
    images = []
    for index in indices.tolist():
        _, image = dataset[index]
        images.append(image.float().cpu())
    return torch.stack(images, dim=0)


def load_example_images(args, geometry) -> tuple[torch.Tensor, str]:
    if args.source in ("auto", "lidc"):
        try:
            return load_lidc_images(args, geometry), "LIDC-IDRI"
        except Exception as exc:
            if args.source == "lidc":
                raise
            print(f"Could not load LIDC examples ({exc}); using synthetic images.")

    image_size = int(geometry.image_shape[-1])
    return make_synthetic_images(args.num_examples, image_size, args.seed), "synthetic"


def make_denoising_inputs(
    clean_images: torch.Tensor,
    model_params,
    patch_size: int,
    sigma: float,
    seed: int,
    device: torch.device,
):
    torch.manual_seed(seed)
    pad_width = int(getattr(model_params, "pad_width", 24))
    clean_padded = zero_pad_images(clean_images.float(), pad_width)
    clean_patch, position_patch = sample_image_patch_with_position_channels(
        clean_padded, patch_size
    )
    clean_patch = clean_patch.to(device)
    position_patch = position_patch.to(device)

    sigma_tensor = torch.full((clean_patch.shape[0],), float(sigma), device=device)
    sigma_view = sigma_tensor.reshape(clean_patch.shape[0], 1, 1, 1)
    noisy_patch = clean_patch + sigma_view * torch.randn_like(clean_patch)
    return clean_patch, noisy_patch, position_patch, sigma_tensor


@torch.inference_mode()
def denoise_patches(model, clean_patch, noisy_patch, position_patch, sigma_tensor):
    sigma_data = torch.as_tensor(
        0.5, device=noisy_patch.device, dtype=noisy_patch.dtype
    )
    sigma_view = sigma_tensor.reshape(noisy_patch.shape[0], 1, 1, 1)
    c_skip = sigma_data.square() / (sigma_view.square() + sigma_data.square())
    c_out = sigma_view * sigma_data / (sigma_view.square() + sigma_data.square()).sqrt()
    c_in = 1 / (sigma_data.square() + sigma_view.square()).sqrt()
    c_noise = sigma_tensor.log() / 4

    model_input = torch.cat((c_in * noisy_patch, position_patch), dim=1)
    model_output = model(model_input, c_noise)
    denoised = c_skip * noisy_patch + c_out * model_output

    noisy_mse = (noisy_patch - clean_patch).square().flatten(1).mean(dim=1)
    denoised_mse = (denoised - clean_patch).square().flatten(1).mean(dim=1)
    return denoised, noisy_mse, denoised_mse


def save_visual_grid(
    output_path: pathlib.Path,
    clean_patch: torch.Tensor,
    noisy_patch: torch.Tensor,
    denoised: torch.Tensor,
    noisy_mse: torch.Tensor,
    denoised_mse: torch.Tensor,
    *,
    source_name: str,
    sigma: float,
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = clean_patch.detach().cpu().clamp(0, 1)
    noisy = noisy_patch.detach().cpu().clamp(0, 1)
    denoised_cpu = denoised.detach().cpu().clamp(0, 1)
    error = (denoised.detach().cpu() - clean_patch.detach().cpu()).squeeze(1)
    error_limit = max(float(error.abs().amax()), 1e-6)

    rows = clean.shape[0]
    fig, axes = plt.subplots(rows, 4, figsize=(11, 2.8 * rows), squeeze=False)
    fig.suptitle(
        f"PaDIS LIDC 256 denoising check ({source_name}, sigma={sigma:g})", y=0.995
    )
    columns = ("Clean", "Noisy", "Denoised", "Denoised - clean")

    for row in range(rows):
        images = (
            clean[row, 0],
            noisy[row, 0],
            denoised_cpu[row, 0],
            error[row],
        )
        for col, image in enumerate(images):
            ax = axes[row][col]
            if col == 3:
                ax.imshow(image, cmap="bwr", vmin=-error_limit, vmax=error_limit)
            else:
                ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            ax.set_axis_off()
            title = columns[col]
            if col == 1:
                title += f"\nMSE {noisy_mse[row].item():.4g}"
            elif col == 2:
                title += f"\nMSE {denoised_mse[row].item():.4g}"
            ax.set_title(title)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"Saved visual denoising check to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-folder", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--source", choices=("auto", "lidc", "synthetic"), default="auto"
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=56)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--raw-weights", action="store_true", help="Do not prefer EMA weights."
    )
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive.")
    if args.patch_size <= 0 or args.patch_size % 8 != 0:
        raise ValueError("--patch-size must be positive and divisible by 8.")
    if args.sigma <= 0:
        raise ValueError("--sigma must be positive.")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type == "cpu":
        print("CUDA was requested but is not available; using CPU.")

    model, model_params, geometry = load_model(
        checkpoint_path, device, use_ema=not args.raw_weights
    )
    clean_images, source_name = load_example_images(args, geometry)
    clean_patch, noisy_patch, position_patch, sigma_tensor = make_denoising_inputs(
        clean_images,
        model_params,
        args.patch_size,
        args.sigma,
        args.seed,
        device,
    )
    denoised, noisy_mse, denoised_mse = denoise_patches(
        model, clean_patch, noisy_patch, position_patch, sigma_tensor
    )

    output_path = args.output
    if output_path is None:
        output_path = checkpoint_path.parent / "padis_lidc_256_denoising_check.png"
    save_visual_grid(
        output_path,
        clean_patch,
        noisy_patch,
        denoised,
        noisy_mse,
        denoised_mse,
        source_name=source_name,
        sigma=args.sigma,
        show=args.show,
    )

    mean_noisy = noisy_mse.mean().item()
    mean_denoised = denoised_mse.mean().item()
    print(f"Mean noisy MSE:    {mean_noisy:.6g}")
    print(f"Mean denoised MSE: {mean_denoised:.6g}")
    if mean_denoised < mean_noisy:
        print("Denoising check passed: denoised MSE is lower than noisy MSE.")
    else:
        print("Denoising check warning: denoised MSE is not lower than noisy MSE.")


if __name__ == "__main__":
    main()
