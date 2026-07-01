"""Train the LIDC DRUNet denoiser used by PaDIS paper PnP-ADMM comparisons."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import signal
import time

_CACHE_ROOT = pathlib.Path("/tmp") / "lion_matplotlib_cache"
(_CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import LION.experiments.ct_experiments as ct_experiments
from LION.models.CNNs.drunet import DRUNet
from LION.optimizers.GaussianDenoiserSolver import GaussianDenoiserSolver
from LION.utils.parameter import LIONParameter
from LION.utils.paths import LION_EXPERIMENTS_PATH


class TerminationRequested(Exception):
    """Raised when the process receives a shutdown signal."""


def set_run_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def install_termination_handler():
    previous_handler = signal.getsignal(signal.SIGTERM)

    def request_stop(signum, _frame):
        signal.signal(signum, signal.SIG_IGN)
        raise TerminationRequested(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, request_stop)
    return previous_handler


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


def save_loss_plots(solver, save_folder: pathlib.Path) -> None:
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


def wandb_id_file(run_folder: pathlib.Path) -> pathlib.Path:
    return run_folder / "wandb_run.json"


def discover_wandb_id(run_folder: pathlib.Path) -> str | None:
    id_path = wandb_id_file(run_folder)
    if id_path.is_file():
        try:
            with open(id_path) as f:
                payload = json.load(f)
            run_id = payload.get("id")
            if run_id:
                return str(run_id)
        except (OSError, json.JSONDecodeError):
            pass

    wandb_dir = run_folder / "wandb"
    if not wandb_dir.is_dir():
        return None
    run_dirs = sorted(
        (path for path in wandb_dir.glob("*run-*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    for run_dir in reversed(run_dirs):
        run_id = run_dir.name.split("-")[-1]
        if run_id:
            return run_id
    return None


def init_wandb(args, run_folder: pathlib.Path, config: dict):
    if args.no_wandb or args.wandb_project is None:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "WandB logging was requested, but wandb is not installed."
        ) from exc

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_name or run_folder.name,
        "config": config,
        "dir": str(run_folder),
        "mode": args.wandb_mode,
        "resume": "allow",
    }
    run_id = args.wandb_id or discover_wandb_id(run_folder)
    if run_id is not None:
        init_kwargs["id"] = run_id
    wandb_run = wandb.init(**init_kwargs)
    wandb_run.define_metric("epoch")
    wandb_run.define_metric("train_loss", step_metric="epoch")
    wandb_run.define_metric("validation_loss", step_metric="epoch")
    with open(wandb_id_file(run_folder), "w") as f:
        json.dump({"id": wandb_run.id, "name": wandb_run.name}, f, indent=2)
    return wandb_run


def log_epoch_to_wandb(wandb_run, solver, epoch_index: int) -> None:
    if wandb_run is None:
        return
    metrics = {
        "epoch": int(epoch_index + 1),
        "train_loss": float(solver.train_loss[epoch_index]),
    }
    if solver.validation_loss is not None:
        value = float(solver.validation_loss[epoch_index])
        if np.isfinite(value) and value != 0.0:
            metrics["validation_loss"] = value
    wandb_run.log(metrics)


def log_wandb_outputs(wandb_run, run_folder: pathlib.Path, log_artifact=True) -> None:
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
    if not log_artifact:
        return

    artifact = wandb.Artifact(run_folder.name, type="padis-run")
    for path in run_folder.glob("*.pt"):
        artifact.add_file(str(path))
    for path in run_folder.glob("*.json"):
        artifact.add_file(str(path))
    for path in run_folder.glob("*.png"):
        artifact.add_file(str(path))
    wandb_run.log_artifact(artifact)


def ensure_loss_capacity(loss: np.ndarray, n_epochs: int) -> np.ndarray:
    if len(loss) >= n_epochs:
        return loss
    return np.append(loss, np.zeros(n_epochs - len(loss)))


def periodic_checkpoint_sidecars(checkpoint_path: pathlib.Path) -> list[pathlib.Path]:
    return [
        checkpoint_path.with_suffix(".pt"),
        checkpoint_path.with_suffix(".json"),
        checkpoint_path.with_suffix(".ema.pt"),
    ]


def full_final_name(final_name: str) -> str:
    final_path = pathlib.Path(final_name)
    suffix = final_path.suffix or ".pt"
    return str(final_path.with_name(f"{final_path.stem}_full{suffix}"))


def prune_periodic_checkpoints(solver, max_periodic_checkpoints: int | None) -> None:
    if max_periodic_checkpoints is None:
        return
    if max_periodic_checkpoints < 0:
        raise ValueError("max_periodic_checkpoints must be non-negative or None.")
    if solver.checkpoint_save_folder is None or solver.checkpoint_fname is None:
        return
    checkpoints = sorted(
        path
        for path in solver.checkpoint_save_folder.glob(solver.checkpoint_fname)
        if not path.name.endswith(".ema.pt")
    )
    stale_checkpoints = (
        checkpoints
        if max_periodic_checkpoints == 0
        else checkpoints[:-max_periodic_checkpoints]
    )
    for checkpoint in stale_checkpoints:
        for path in periodic_checkpoint_sidecars(checkpoint):
            if path.exists():
                path.unlink()


def save_checkpoint_with_retention(
    solver,
    epoch: int,
    max_periodic_checkpoints: int | None,
) -> None:
    solver.save_checkpoint(epoch)
    prune_periodic_checkpoints(solver, max_periodic_checkpoints)


def save_interruption_checkpoint(solver, max_periodic_checkpoints: int | None) -> None:
    if solver.checkpoint_save_folder is None or solver.checkpoint_fname is None:
        return
    if getattr(solver, "train_loss", None) is None:
        return
    save_checkpoint_with_retention(
        solver,
        int(solver.current_epoch),
        max_periodic_checkpoints,
    )
    print("Saved interruption checkpoint after shutdown signal.")


def load_full_final_checkpoint_if_exists(
    solver,
    full_final_path: pathlib.Path,
) -> bool:
    if not full_final_path.with_suffix(".pt").is_file():
        return False

    (
        solver.model,
        solver.optimizer,
        solver.current_epoch,
        solver.train_loss,
        _,
    ) = solver.model.load_checkpoint_if_exists(
        full_final_path,
        solver.model,
        solver.optimizer,
        solver.train_loss,
    )
    print(f"Loaded full final PnP training state from {full_final_path}")
    return True


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
    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--checkpoint-interval-seconds",
        type=float,
        default=None,
        help="Also save resumable checkpoints after this many wall-clock seconds.",
    )
    parser.add_argument(
        "--max-periodic-checkpoints",
        type=int,
        default=5,
        help="Maximum periodic checkpoints to retain. Use -1 to keep all.",
    )
    parser.add_argument(
        "--keep-final-periodic-checkpoints",
        type=int,
        default=None,
        help=(
            "Periodic checkpoints to keep after clean completion. By default the "
            "existing periodic retention setting is left unchanged."
        ),
    )
    parser.add_argument(
        "--max-train-seconds",
        type=float,
        default=None,
        help="Stop training cleanly after this many wall-clock seconds.",
    )
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--final-name", default="pnp_lidc_drunet.pt")
    parser.add_argument(
        "--final-full-name",
        default=None,
        help=(
            "Full final training-state checkpoint containing optimizer state. "
            "Defaults to <final-name> with _full before the suffix."
        ),
    )
    parser.add_argument("--checkpoint-pattern", default="pnp_lidc_drunet_check_*.pt")
    parser.add_argument("--validation-name", default="pnp_lidc_drunet_min_val.pt")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument(
        "--no-wandb-artifact",
        action="store_true",
        help="Log WandB metrics/plots but do not upload saved model artifacts.",
    )
    parser.add_argument("--no-wandb", action="store_true")
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
    if args.checkpoint_interval_seconds is not None and (
        args.checkpoint_interval_seconds <= 0
    ):
        raise ValueError("--checkpoint-interval-seconds must be positive when set.")
    if args.max_periodic_checkpoints == 0 or args.max_periodic_checkpoints < -1:
        raise ValueError("--max-periodic-checkpoints must be positive or -1.")
    if args.keep_final_periodic_checkpoints is not None and (
        args.keep_final_periodic_checkpoints < 0
        or (
            args.max_periodic_checkpoints != -1
            and args.keep_final_periodic_checkpoints > args.max_periodic_checkpoints
        )
    ):
        raise ValueError(
            "--keep-final-periodic-checkpoints must be between 0 and "
            "--max-periodic-checkpoints."
        )
    if args.max_train_seconds is not None and args.max_train_seconds <= 0:
        raise ValueError("--max-train-seconds must be positive when set.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if not args.final_name:
        raise ValueError("--final-name must not be empty.")
    if args.final_full_name is not None and not args.final_full_name:
        raise ValueError("--final-full-name must not be empty when set.")


def train_with_optional_time_limit(
    solver,
    n_epochs: int,
    max_train_seconds: float | None,
    max_periodic_checkpoints: int | None,
    checkpoint_interval_seconds: float | None = None,
    full_final_checkpoint_path: pathlib.Path | None = None,
    wandb_run=None,
) -> None:
    assert n_epochs > 0, "Number of epochs must be a positive integer"
    solver.check_training_ready()

    if solver.check_validation_ready() == 0:
        solver.validation_loss = np.zeros((n_epochs))
    if solver.validation_loader is None:
        solver.validation_loss = None

    if solver.do_load_checkpoint:
        print("Loading checkpoint...")
        solver.current_epoch = solver.load_checkpoint()
        if solver.current_epoch == 0 and full_final_checkpoint_path is not None:
            load_full_final_checkpoint_if_exists(solver, full_final_checkpoint_path)
        solver.train_loss = ensure_loss_capacity(solver.train_loss, n_epochs)
    else:
        solver.train_loss = np.zeros(n_epochs)

    solver.model.train()
    train_start_wall = time.monotonic()
    next_timed_checkpoint = (
        float(checkpoint_interval_seconds)
        if checkpoint_interval_seconds is not None
        else None
    )
    while solver.current_epoch < n_epochs:
        print(f"Training epoch {solver.current_epoch + 1}")
        solver.epoch_step(solver.current_epoch)
        log_epoch_to_wandb(wandb_run, solver, solver.current_epoch)

        elapsed = time.monotonic() - train_start_wall
        checkpoint_due = (solver.current_epoch + 1) % solver.checkpoint_freq == 0 or (
            next_timed_checkpoint is not None and elapsed >= next_timed_checkpoint
        )
        if checkpoint_due:
            save_checkpoint_with_retention(
                solver,
                solver.current_epoch,
                max_periodic_checkpoints,
            )
            if next_timed_checkpoint is not None:
                while elapsed >= next_timed_checkpoint:
                    next_timed_checkpoint += float(checkpoint_interval_seconds)

        solver.current_epoch += 1
        if max_train_seconds is not None and elapsed >= max_train_seconds:
            print(
                "Reached --max-train-seconds "
                f"({max_train_seconds:g}); stopping after epoch "
                f"{solver.current_epoch}."
            )
            break


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
    max_periodic_checkpoints = (
        None
        if args.max_periodic_checkpoints == -1
        else int(args.max_periodic_checkpoints)
    )
    prune_periodic_checkpoints(solver, max_periodic_checkpoints)

    config = {key: jsonable(value) for key, value in vars(args).items()}
    config["run_folder"] = str(run_folder)
    config["experiment"] = experiment.param.name
    config["max_periodic_checkpoints_effective"] = max_periodic_checkpoints
    config["final_full_name_effective"] = args.final_full_name or full_final_name(
        args.final_name
    )
    full_final_checkpoint_path = run_folder / config["final_full_name_effective"]
    with open(run_folder / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    wandb_run = init_wandb(args, run_folder, config)
    previous_sigterm_handler = install_termination_handler()
    try:
        try:
            train_with_optional_time_limit(
                solver,
                train_params.epochs,
                args.max_train_seconds,
                max_periodic_checkpoints,
                checkpoint_interval_seconds=args.checkpoint_interval_seconds,
                full_final_checkpoint_path=full_final_checkpoint_path,
                wandb_run=wandb_run,
            )
        except TerminationRequested as exc:
            print(f"{exc}; saving resumable checkpoint before exit.")
            save_interruption_checkpoint(solver, max_periodic_checkpoints)
            raise SystemExit(143)
        solver.model.save(
            run_folder / args.final_name,
            epoch=solver.current_epoch,
            training=solver.metadata,
            loss=solver.train_loss,
            dataset=solver.dataset_param,
            geometry=experiment.geometry,
        )
        solver.model.save_checkpoint(
            full_final_checkpoint_path,
            solver.current_epoch,
            solver.train_loss,
            solver.optimizer,
            solver.metadata,
            dataset=solver.dataset_param,
            geometry=experiment.geometry,
        )
        if wandb_run is not None:
            wandb_run.summary["epochs_completed"] = int(solver.current_epoch)
            finite_train = solver.train_loss[
                np.isfinite(solver.train_loss) & (solver.train_loss != 0.0)
            ]
            if finite_train.size > 0:
                wandb_run.summary["min_train_loss"] = float(np.min(finite_train))
            if solver.validation_loss is not None:
                finite_val = solver.validation_loss[
                    np.isfinite(solver.validation_loss)
                    & (solver.validation_loss != 0.0)
                ]
                if finite_val.size > 0:
                    wandb_run.summary["min_validation_loss"] = float(np.min(finite_val))
        print(f"Saved PnP denoiser to {run_folder / args.final_name}")
        print(
            "Saved full PnP training state to "
            f"{run_folder / config['final_full_name_effective']}"
        )
        if args.keep_final_periodic_checkpoints is not None:
            prune_periodic_checkpoints(solver, args.keep_final_periodic_checkpoints)
        save_loss_plots(solver, run_folder)
        log_wandb_outputs(
            wandb_run, run_folder, log_artifact=not args.no_wandb_artifact
        )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
