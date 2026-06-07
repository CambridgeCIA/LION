"""Train a PaDIS paper-style patch prior on LIDC-IDRI at 256x256."""

import argparse
import pathlib

import torch
from torch.utils.data import DataLoader

from LION.CTtools.ct_geometry import Geometry
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
from LION.losses.PaDIS import PaDISDenoisingLoss
from LION.models.diffusion import NCSNpp
from LION.optimizers import PaDISSolver
from LION.utils.paths import LION_EXPERIMENTS_PATH


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=pathlib.Path, default=None)
    parser.add_argument(
        "--save-folder",
        type=pathlib.Path,
        default=LION_EXPERIMENTS_PATH.joinpath("PaDIS", "LIDC_256"),
    )
    parser.add_argument("--target-patches", type=int, default=200_000_000)
    parser.add_argument("--validation-interval-patches", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-interval-patches", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-ema", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    geometry = Geometry.default_parameters(image_scaling=0.5)
    data_params = LIDC_IDRI.default_parameters(geometry=geometry, task="image_prior")
    data_params.device = torch.device("cpu")
    if args.data_folder is not None:
        data_params.folder = args.data_folder

    train_dataset = LIDC_IDRI(
        "train", parameters=data_params, geometry_parameters=geometry
    )
    validation_dataset = LIDC_IDRI(
        "validation", parameters=data_params, geometry_parameters=geometry
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_params = NCSNpp.default_parameters("padis-paper-ct-256")
    model = NCSNpp(model_params, geometry)
    loss_fn = PaDISDenoisingLoss(
        sigma_min=model_params.sigma_min, sigma_max=model_params.sigma_max
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    solver_params = PaDISSolver.default_parameters("padis-paper-ct-256")
    solver_params.use_ema = not args.no_ema
    solver = PaDISSolver(
        model,
        optimizer,
        loss_fn,
        geometry=geometry,
        solver_params=solver_params,
        device=device,
        save_folder=args.save_folder,
    )
    args.save_folder.mkdir(parents=True, exist_ok=True)
    solver.set_saving(args.save_folder, "padis_lidc_256")
    solver.set_checkpointing(
        "padis_lidc_256_checkpoint_*.pt",
        checkpoint_freq=10**12,
        load_checkpoint_if_exists=True,
    )
    solver.set_training(train_loader)
    solver.set_validation(validation_loader, validation_freq=10**12)
    solver.train_for_patches(
        args.target_patches,
        validation_interval_patches=args.validation_interval_patches,
        checkpoint_interval_patches=args.checkpoint_interval_patches,
    )


if __name__ == "__main__":
    main()
