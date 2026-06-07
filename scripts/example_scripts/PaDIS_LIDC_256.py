"""Train a PaDIS paper-style patch prior on LIDC-IDRI at 256x256."""

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


def serializable_config(args, run_folder, preset):
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, pathlib.Path) else value
    config["run_folder"] = str(run_folder)
    config["paper_preset"] = preset
    return config


def init_wandb(args, run_folder, preset):
    if args.no_wandb or args.wandb_project is None:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "WandB logging was requested, but wandb is not installed."
        ) from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or run_folder.name,
        dir=str(run_folder),
        config=serializable_config(args, run_folder, preset),
        mode=args.wandb_mode,
    )


def wandb_log_fn(wandb_run):
    if wandb_run is None:
        return None

    def log_fn(metrics, step):
        wandb_run.log(metrics, step=step)

    return log_fn


def data_loader_kwargs(args, device):
    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def log_wandb_outputs(wandb_run, run_folder):
    if wandb_run is None:
        return
    import wandb

    plots = {}
    for key, filename in (
        ("plots/train_loss", "loss.png"),
        ("plots/validation_loss", "validation_loss.png"),
    ):
        path = run_folder / filename
        if path.is_file():
            plots[key] = wandb.Image(str(path))
    if plots:
        wandb_run.log(plots)

    artifact = wandb.Artifact(run_folder.name, type="padis-run")
    for path in run_folder.glob("*.pt"):
        artifact.add_file(str(path))
    for path in run_folder.glob("*.json"):
        artifact.add_file(str(path))
    for path in run_folder.glob("*.png"):
        artifact.add_file(str(path))
    wandb_run.log_artifact(artifact)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=pathlib.Path, default=None)
    parser.add_argument(
        "--save-folder",
        type=pathlib.Path,
        default=LION_EXPERIMENTS_PATH.joinpath("PaDIS", "LIDC_256"),
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--target-patches", type=int, default=200_000_000)
    parser.add_argument("--validation-interval-patches", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-interval-patches", type=int, default=5_000_000)
    parser.add_argument("--log-interval-patches", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--no-wandb", action="store_true")
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
        shuffle=True,
        **data_loader_kwargs(args, device),
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **data_loader_kwargs(args, device),
    )

    model_params = NCSNpp.default_parameters("padis-paper-ct-256")
    model = NCSNpp(model_params, geometry)
    loss_fn = PaDISDenoisingLoss(
        sigma_min=model_params.sigma_min, sigma_max=model_params.sigma_max
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    solver_params = PaDISSolver.default_parameters("padis-paper-ct-256")
    solver_params.use_ema = not args.no_ema
    run_folder = make_run_folder(args.save_folder, args.run_name, "padis_lidc_256")
    print(f"Saving PaDIS run to {run_folder}")
    wandb_run = init_wandb(args, run_folder, "padis-paper-ct-256")
    solver = PaDISSolver(
        model,
        optimizer,
        loss_fn,
        geometry=geometry,
        solver_params=solver_params,
        device=device,
        save_folder=run_folder,
    )
    solver.set_saving(run_folder, "padis_lidc_256")
    solver.set_checkpointing(
        "padis_lidc_256_checkpoint_*.pt",
        checkpoint_freq=10**12,
        load_checkpoint_if_exists=True,
    )
    solver.set_training(train_loader)
    solver.set_validation(validation_loader, validation_freq=10**12)
    try:
        solver.train_for_patches(
            args.target_patches,
            validation_interval_patches=args.validation_interval_patches,
            checkpoint_interval_patches=args.checkpoint_interval_patches,
            log_interval_patches=args.log_interval_patches,
            log_fn=wandb_log_fn(wandb_run),
        )
        solver.clean_checkpoints()
        solver.save_final_results()
        save_loss_plots(solver, run_folder)
        log_wandb_outputs(wandb_run, run_folder)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
