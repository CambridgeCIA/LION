"""Train a PaDIS paper-style patch prior on LIDC-IDRI at native 512x512."""

import argparse
from datetime import datetime
import json
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


def wandb_id_file(run_folder):
    return run_folder / "wandb_run.json"


def extract_wandb_id(path):
    match = re.search(r"(?:offline-run|run)-\d+_\d+-([A-Za-z0-9]+)$", path.name)
    return match.group(1) if match is not None else None


def discover_wandb_id(run_folder):
    metadata_path = wandb_id_file(run_folder)
    if metadata_path.is_file():
        with open(metadata_path) as f:
            metadata = json.load(f)
        if metadata.get("id"):
            return metadata["id"]

    wandb_folder = run_folder / "wandb"
    latest_run = wandb_folder / "latest-run"
    if latest_run.exists():
        run_id = extract_wandb_id(latest_run.resolve())
        if run_id is not None:
            return run_id

    run_dirs = [
        path
        for path in wandb_folder.glob("*")
        if path.is_dir() and extract_wandb_id(path) is not None
    ]
    if run_dirs:
        newest = max(run_dirs, key=lambda path: path.stat().st_mtime)
        return extract_wandb_id(newest)

    return None


def save_wandb_id(run_folder, wandb_run):
    if wandb_run is None:
        return
    with open(wandb_id_file(run_folder), "w") as f:
        json.dump({"id": wandb_run.id, "name": wandb_run.name}, f, indent=2)


def init_wandb(args, run_folder, preset):
    if args.no_wandb or args.wandb_project is None:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "WandB logging was requested, but wandb is not installed."
        ) from exc

    run_id = args.wandb_id or discover_wandb_id(run_folder)
    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_name or run_folder.name,
        "dir": str(run_folder),
        "config": serializable_config(args, run_folder, preset),
        "mode": args.wandb_mode,
    }
    if run_id is not None:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"
        print(f"Resuming WandB run id {run_id}")

    wandb_run = wandb.init(**init_kwargs)
    save_wandb_id(run_folder, wandb_run)
    return wandb_run


def min_validation_loss(solver):
    if solver.validation_loss is None or len(solver.validation_loss) == 0:
        return None
    return float(min(solver.validation_loss))


def wandb_log_fn(wandb_run):
    if wandb_run is None:
        return None

    def log_fn(metrics, step):
        wandb_run.log(metrics, step=step)

    return log_fn


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
        "--full-lidc",
        action="store_true",
        help="Use every available slice from each selected LIDC-IDRI patient. Ignores --pcg-slices-nodule.",
    )
    parser.add_argument(
        "--max-slices-per-patient",
        type=int,
        default=4,
        help="Maximum slices per patient for subset training. Use -1, or --full-lidc, for every available slice.",
    )
    parser.add_argument(
        "--pcg-slices-nodule",
        type=float,
        default=0.5,
        help="Fraction of selected subset slices containing nodules. Ignored when using all slices.",
    )
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
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.max_slices_per_patient == 0 or args.max_slices_per_patient < -1:
        raise ValueError("--max-slices-per-patient must be positive or -1.")
    if not 0.0 <= args.pcg_slices_nodule <= 1.0:
        raise ValueError("--pcg-slices-nodule must be in [0, 1].")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    geometry = Geometry.default_parameters(image_scaling=1.0)
    data_params = LIDC_IDRI.default_parameters(geometry=geometry, task="image_prior")
    data_params.device = torch.device("cpu")
    if args.data_folder is not None:
        data_params.folder = args.data_folder
    data_params.max_num_slices_per_patient = (
        -1 if args.full_lidc else int(args.max_slices_per_patient)
    )
    data_params.pcg_slices_nodule = float(args.pcg_slices_nodule)
    if data_params.max_num_slices_per_patient == -1:
        print("Using all available LIDC-IDRI slices; pcg_slices_nodule is ignored.")

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
    wandb_run = init_wandb(args, run_folder, "padis-paper-ct-512")
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
    try:
        solver.train_for_patches(
            args.target_patches,
            validation_interval_patches=args.validation_interval_patches,
            checkpoint_interval_patches=args.checkpoint_interval_patches,
            log_fn=wandb_log_fn(wandb_run),
        )
        solver.clean_checkpoints()
        solver.save_final_results()
        save_loss_plots(solver, run_folder)
        if wandb_run is not None:
            wandb_run.summary["min_validation_loss"] = min_validation_loss(solver)
            wandb_run.summary["seen_patches"] = solver.seen_patches
        log_wandb_outputs(wandb_run, run_folder)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
