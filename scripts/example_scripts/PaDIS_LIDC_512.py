"""Train a PaDIS paper-style patch prior on LIDC-IDRI at native 512x512."""

import argparse
from datetime import datetime
import pathlib
import re
import uuid

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from LION.CTtools.ct_geometry import Geometry
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
from LION.losses.PaDIS import PaDISDenoisingLoss
from LION.models.diffusion import NCSNpp
from LION.optimizers import PaDISSolver
from LION.utils.paths import LION_EXPERIMENTS_PATH


def make_run_folder(save_root, run_name, prefix):
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"
    else:
        run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name.strip()).strip("._")
        if not run_name:
            raise ValueError("--run-name must contain at least one valid character.")

    run_folder = save_root / run_name
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder


def save_loss_plots(solver, save_folder):
    plt.figure()
    train_loss = (
        solver.train_loss[1:] if len(solver.train_loss) > 1 else solver.train_loss
    )
    plt.semilogy(train_loss)
    plt.savefig(save_folder / "loss.png")
    plt.close()

    if solver.validation_loss is not None and len(solver.validation_loss) > 0:
        plt.figure()
        plt.semilogy(solver.validation_loss)
        plt.savefig(save_folder / "validation_loss.png")
        plt.close()


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=pathlib.Path, default=None)
    parser.add_argument(
        "--save-folder",
        type=pathlib.Path,
        default=LION_EXPERIMENTS_PATH.joinpath("PaDIS", "LIDC_512"),
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--target-patches", type=int, default=200_000_000)
    parser.add_argument("--validation-interval-patches", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-interval-patches", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-ema", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    geometry = Geometry.default_parameters(image_scaling=1.0)
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

    model_params = NCSNpp.default_parameters("padis-paper-ct-512")
    model = NCSNpp(model_params, geometry)
    loss_fn = PaDISDenoisingLoss(
        sigma_min=model_params.sigma_min, sigma_max=model_params.sigma_max
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    solver_params = PaDISSolver.default_parameters("padis-paper-ct-512")
    solver_params.use_ema = not args.no_ema
    run_folder = make_run_folder(args.save_folder, args.run_name, "padis_lidc_512")
    print(f"Saving PaDIS run to {run_folder}")
    solver = PaDISSolver(
        model,
        optimizer,
        loss_fn,
        geometry=geometry,
        solver_params=solver_params,
        device=device,
        save_folder=run_folder,
    )
    solver.set_saving(run_folder, "padis_lidc_512")
    solver.set_checkpointing(
        "padis_lidc_512_checkpoint_*.pt",
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
    solver.clean_checkpoints()
    solver.save_final_results()
    save_loss_plots(solver, run_folder)


if __name__ == "__main__":
    main()
