"""Generate unconditional samples from a trained PaDIS LIDC prior."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib

_CACHE_ROOT = pathlib.Path("/tmp") / "lion_matplotlib_cache"
(_CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import torch

from LION.reconstructors import PaDIS
from LION.utils.parameter import LIONParameter
from LION.utils.paths import LION_EXPERIMENTS_PATH
from scripts.paper_scripts.PaDIS.PaDIS_LIDC_reconstruction import (
    DEFAULT_CHECKPOINT,
    load_model,
    resolve_checkpoint_path,
    set_run_seed,
)


def jsonable(value):
    if isinstance(value, LIONParameter):
        return {
            key: jsonable(item)
            for key, item in value.__dict__.items()
            if not key.startswith("_")
        }
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_generation_params(args, model) -> LIONParameter:
    params = PaDIS.default_parameters(model)
    params.num_steps = args.num_steps
    params.inner_steps = args.inner_steps
    params.sigma_min = args.sigma_min
    params.sigma_max = args.sigma_max
    params.rho = args.rho
    params.generation_epsilon = args.generation_epsilon
    params.patch_batch_size = args.patch_batch_size
    params.langevin_noise_scale = args.langevin_noise_scale
    params.clip_output = not args.no_clip_output
    params.clip_denoised = args.clip_denoised
    params.clip_state = args.clip_state
    params.disable_langevin_noise = args.disable_langevin_noise
    params.disable_prior_score = args.disable_prior_score
    if args.prior_mode != "auto":
        params.prior_mode = (
            "whole_image" if args.prior_mode == "whole-image" else "patch"
        )
    if args.patch_size is not None:
        params.patch_size = args.patch_size
    if args.pad_width is not None:
        params.pad_width = args.pad_width
    return params


def image_shape_from_args(args, geometry) -> tuple[int, int, int]:
    channels = int(geometry.image_shape[0])
    height = int(geometry.image_shape[1])
    width = int(geometry.image_shape[2])
    if args.image_size is not None:
        height = width = int(args.image_size)
    return channels, height, width


def save_grid(samples: torch.Tensor, path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    images = samples.detach().cpu().clamp(0.0, 1.0)
    count = images.shape[0]
    cols = min(4, count)
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows))
    axes_flat = np.asarray(axes).reshape(-1).tolist()
    for index, axis in enumerate(axes_flat):
        axis.set_axis_off()
        if index >= count:
            continue
        image = images[index]
        if image.shape[0] > 1:
            image = image.mean(dim=0)
        else:
            image = image[0]
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-folder",
        type=pathlib.Path,
        default=LION_EXPERIMENTS_PATH / "PaDIS" / "LIDC_generation",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--generation-mode",
        choices=("padis", "naive-patch"),
        default="padis",
        help="Use the PaDIS overlapping-patch prior sampler or independent patch generation.",
    )
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=40.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--generation-epsilon", type=float, default=1.0)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--pad-width", type=int, default=None)
    parser.add_argument("--patch-batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--langevin-noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--prior-mode",
        choices=("auto", "patch", "whole-image"),
        default="auto",
    )
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument(
        "--no-position-channels",
        action="store_true",
        help="Construct the PaDIS prior without x/y position inputs. The checkpoint must use the same architecture.",
    )
    parser.add_argument("--no-clip-output", action="store_true")
    parser.add_argument("--clip-denoised", action="store_true")
    parser.add_argument("--clip-state", action="store_true")
    parser.add_argument("--disable-langevin-noise", action="store_true")
    parser.add_argument("--disable-prior-score", action="store_true")
    parser.add_argument("--prog-bar", action="store_true")
    parser.add_argument("--no-save-grid", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive.")
    if args.inner_steps <= 0:
        raise ValueError("--inner-steps must be positive.")

    set_run_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type == "cpu":
        print("CUDA was requested but is not available; using CPU.")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    model, model_params, geometry = load_model(
        checkpoint_path,
        device,
        use_ema=not args.raw_weights,
        disable_position_channels=args.no_position_channels,
    )
    params = build_generation_params(args, model)
    image_shape = image_shape_from_args(args, geometry)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sampler = PaDIS(None, model, params)

    if args.generation_mode == "naive-patch":
        samples = sampler.generate_naive_patch_samples(
            num_samples=args.num_samples,
            image_shape=image_shape,
            prog_bar=args.prog_bar,
            generator=generator,
        )
    else:
        samples = sampler.generate_samples(
            num_samples=args.num_samples,
            image_shape=image_shape,
            prog_bar=args.prog_bar,
            generator=generator,
        )

    args.output_folder.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output_folder / "samples.pt"
    torch.save(
        {
            "samples": samples.detach().cpu(),
            "checkpoint": str(checkpoint_path),
            "generation_mode": args.generation_mode,
            "sampler": jsonable(params),
            "model_parameters": jsonable(model_params),
            "geometry": jsonable(geometry),
            "image_shape": image_shape,
        },
        tensor_path,
    )
    manifest_path = args.output_folder / "generation_manifest.json"
    with open(manifest_path, "w") as file:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "num_samples": args.num_samples,
                "seed": args.seed,
                "generation_mode": args.generation_mode,
                "sampler": jsonable(params),
                "image_shape": image_shape,
            },
            file,
            indent=2,
        )
    if not args.no_save_grid:
        save_grid(samples, args.output_folder / "samples.png")

    print(f"Saved samples to {tensor_path}")
    print(f"Saved generation manifest to {manifest_path}")


if __name__ == "__main__":
    main()
