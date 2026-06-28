"""Train the LIDC DRUNet denoiser used by PaDIS paper PnP-ADMM comparisons."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random

_CACHE_ROOT = pathlib.Path("/tmp") / "lion_matplotlib_cache"
(_CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import LION.experiments.ct_experiments as ct_experiments
from LION.models.CNNs.drunet import DRUNet
from LION.optimizers.GaussianDenoiserSolver import GaussianDenoiserSolver
from LION.utils.parameter import LIONParameter
from LION.utils.paths import LION_EXPERIMENTS_PATH


def set_run_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError("cuda_device_index expects a CUDA device.")
    return 0 if device.index is None else int(device.index)


def jsonable(value):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_experiment(args):
    experiment = ct_experiments.PaDISFanBeam20CTRecon(
        dataset="LIDC-IDRI",
        datafolder=args.data_folder,
        image_scaling=args.image_scaling,
    )
    experiment.param.data_loader_params.device = torch.device("cpu")
    if args.full_lidc:
        experiment.param.data_loader_params.max_num_slices_per_patient = -1
    else:
        experiment.param.data_loader_params.max_num_slices_per_patient = (
            args.max_slices_per_patient
        )
    return experiment


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=LION_EXPERIMENTS_PATH / "PaDIS" / "pnp_lidc_drunet",
    )
    parser.add_argument("--run-name", default="pnp_lidc_drunet")
    parser.add_argument("--data-folder", type=pathlib.Path, default=None)
    parser.add_argument("--image-scaling", type=float, default=0.5)
    parser.add_argument("--full-lidc", action="store_true")
    parser.add_argument("--max-slices-per-patient", type=int, default=4)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional training subset size for smoke/debug runs.",
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help="Optional validation subset size for smoke/debug runs.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.05)
    parser.add_argument("--use-noise-level", action="store_true")
    parser.add_argument("--int-channels", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--patches-per-image", type=int, default=1)
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--final-name", default="pnp_lidc_drunet.pt")
    parser.add_argument("--checkpoint-pattern", default="pnp_lidc_drunet_check_*.pt")
    parser.add_argument("--validation-name", default="pnp_lidc_drunet_min_val.pt")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.beta1 < 0 or args.beta1 >= 1 or args.beta2 < 0 or args.beta2 >= 1:
        raise ValueError("--beta1/--beta2 must satisfy 0 <= beta < 1.")
    if args.noise_min < 0 or args.noise_max < args.noise_min:
        raise ValueError("--noise-min/--noise-max must satisfy 0 <= min <= max.")
    if not args.full_lidc and args.max_slices_per_patient <= 0:
        raise ValueError(
            "--max-slices-per-patient must be positive unless --full-lidc is set."
        )
    if args.int_channels <= 0:
        raise ValueError("--int-channels must be positive.")
    if args.n_blocks <= 0:
        raise ValueError("--n-blocks must be positive.")
    if args.patch_size is not None and args.patch_size <= 0:
        raise ValueError("--patch-size must be positive when set.")
    if args.patches_per_image <= 0:
        raise ValueError("--patches-per-image must be positive.")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        raise ValueError("--max-train-samples must be positive when set.")
    if args.max_validation_samples is not None and args.max_validation_samples <= 0:
        raise ValueError("--max-validation-samples must be positive when set.")
    if args.validation_every <= 0:
        raise ValueError("--validation-every must be positive.")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if not args.final_name:
        raise ValueError("--final-name must not be empty.")


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)

    set_run_seed(args.seed)
    requested_device = args.device
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and device.type == "cpu":
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "cuda":
        torch.cuda.set_device(cuda_device_index(device))

    output_root = args.output_root.expanduser().resolve()
    run_folder = output_root / args.run_name
    run_folder.mkdir(parents=True, exist_ok=True)

    experiment = build_experiment(args)
    training = experiment.get_training_dataset()
    validation = experiment.get_validation_dataset()
    if args.max_train_samples is not None:
        training = Subset(training, range(min(args.max_train_samples, len(training))))
    if args.max_validation_samples is not None:
        validation = Subset(
            validation, range(min(args.max_validation_samples, len(validation)))
        )
    train_loader = DataLoader(
        training,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model_params = DRUNet.default_parameters()
    model_params.use_noise_level = bool(args.use_noise_level)
    model_params.int_channels = int(args.int_channels)
    model_params.n_blocks = int(args.n_blocks)
    model = DRUNet(model_params).to(device)
    model.set_normalisation(normalisation_type="dataset", dataset=train_loader)

    train_params = LIONParameter()
    train_params.epochs = int(args.epochs)
    train_params.learning_rate = float(args.learning_rate)
    train_params.betas = (float(args.beta1), float(args.beta2))
    train_params.loss = "MSELoss"
    train_params.optimiser = "adam"

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=train_params.learning_rate,
        betas=train_params.betas,
    )
    solver = GaussianDenoiserSolver(
        model,
        optimiser,
        torch.nn.MSELoss(),
        geometry=experiment.geometry,
        verbose=True,
        device=device,
        save_folder=run_folder,
        noise_level=torch.tensor([args.noise_min, args.noise_max]),
    )
    if args.patch_size is not None:
        solver.set_patch_strategy(
            n_patches=int(args.patches_per_image),
            patch_size=int(args.patch_size),
        )
    solver.set_saving(run_folder, args.final_name)
    solver.set_training(train_loader)
    solver.set_validation(
        validation_loader,
        validation_freq=int(args.validation_every),
        validation_fname=args.validation_name,
        save_folder=run_folder,
    )
    solver.set_checkpointing(
        args.checkpoint_pattern,
        int(args.checkpoint_every),
        load_checkpoint_if_exists=True,
        save_folder=run_folder,
    )

    config = {key: jsonable(value) for key, value in vars(args).items()}
    config["run_folder"] = str(run_folder)
    config["experiment"] = experiment.param.name
    with open(run_folder / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    solver.train(train_params.epochs)
    solver.model.save(
        run_folder / args.final_name,
        epoch=solver.current_epoch,
        training=solver.metadata,
        loss=solver.train_loss,
        dataset=solver.dataset_param,
        geometry=experiment.geometry,
    )
    print(f"Saved PnP denoiser to {run_folder / args.final_name}")


if __name__ == "__main__":
    main()
