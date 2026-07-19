"""Train a PaDIS paper-style patch prior on LIDC-IDRI at native 512x512."""

import argparse
from datetime import datetime
import getpass
import hashlib
import json
import math
import os
import pathlib
import random
import re
import shutil
import signal
import subprocess
import uuid

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from LION.CTtools.ct_geometry import Geometry
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
from LION.losses.PaDIS import PaDISDenoisingLoss
from LION.models.diffusion import NCSNpp
from LION.optimizers import PaDISSolver
from LION.utils.paths import LION_EXPERIMENTS_PATH


class TerminationRequested(Exception):
    """Raised when the process receives a shutdown signal."""


def make_run_folder(save_root, run_name, prefix):
    """Create run folder."""
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
    """Save loss plots."""
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
    """Convert configuration values into serializable objects."""
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, pathlib.Path) else value
    config["run_folder"] = str(run_folder)
    config["paper_preset"] = preset
    return config


def set_run_seed(seed: int) -> None:
    """Set run seed."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def install_termination_handler():
    """Install termination handler."""
    previous_handler = signal.getsignal(signal.SIGTERM)

    def request_stop(signum, _frame):
        """Request stop."""
        signal.signal(signum, signal.SIG_IGN)
        raise TerminationRequested(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, request_stop)
    return previous_handler


def save_interruption_checkpoint(solver):
    """Save interruption checkpoint."""
    if solver.checkpoint_save_folder is None or solver.checkpoint_fname is None:
        return
    if getattr(solver, "train_loss", None) is None:
        return
    checkpoint_pattern = solver.checkpoint_fname
    checkpoint_re = re.compile(
        "^" + re.escape(checkpoint_pattern).replace(r"\*", r"(\d+)") + "$"
    )
    latest_index = 0
    for path in solver.checkpoint_save_folder.glob(checkpoint_pattern):
        if path.name.endswith(".ema.pt"):
            continue
        match = checkpoint_re.match(path.name)
        if match is not None:
            latest_index = max(latest_index, int(match.group(1)))
    solver.save_checkpoint(latest_index)
    print("Saved interruption checkpoint after shutdown signal.")


def wandb_id_file(run_folder):
    """Return the W&B id file."""
    return run_folder / "wandb_run.json"


def extract_wandb_id(path):
    """Extract wandb id."""
    match = re.search(r"(?:offline-run|run)-\d+_\d+-([A-Za-z0-9]+)$", path.name)
    return match.group(1) if match is not None else None


def discover_wandb_id(run_folder):
    """Discover wandb id."""
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
    """Save wandb id."""
    if wandb_run is None:
        return
    with open(wandb_id_file(run_folder), "w") as f:
        json.dump({"id": wandb_run.id, "name": wandb_run.name}, f, indent=2)


def init_wandb(args, run_folder, preset):
    """Initialise or resume W&B logging and persist its stable run identity."""
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
    wandb_run.define_metric("train/seen_patches")
    wandb_run.define_metric("train/*", step_metric="train/seen_patches")
    wandb_run.define_metric("optimizer/*", step_metric="train/seen_patches")
    wandb_run.define_metric("timing/*", step_metric="train/seen_patches")
    wandb_run.define_metric("validation/seen_patches")
    wandb_run.define_metric("validation/*", step_metric="validation/seen_patches")
    save_wandb_id(run_folder, wandb_run)
    return wandb_run


def min_validation_loss(solver):
    """Return the minimum validation loss."""
    if solver.validation_loss is None or len(solver.validation_loss) == 0:
        return None
    return float(min(solver.validation_loss))


def wandb_log_fn(wandb_run):
    """Return the W&B log fn."""
    if wandb_run is None:
        return None

    def log_fn(metrics, step):
        """Provide the log fn callback used by the enclosing operation."""
        _ = step
        wandb_run.log(metrics)

    return log_fn


def data_loader_kwargs(args, device, batch_size):
    """Handle data loader kwargs for the PaDIS workflow."""
    kwargs = {
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        if not args.no_persistent_workers:
            kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


class CachedImagePriorBatchLoader(DataLoader):
    """Small batch loader for cached PaDIS image-prior tensors."""

    def __init__(self, images, batch_size, shuffle, name):
        """Initialize the instance."""
        if images.ndim != 4:
            raise ValueError(
                f"Expected cached images shaped [N, C, H, W], got {images.shape}."
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.images = images.contiguous()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.name = name
        self._order = None
        self._cursor = 0

    def __len__(self):
        """Return the number of available items."""
        return math.ceil(self.images.shape[0] / self.batch_size)

    def _new_order(self):
        """Handle new order for the PaDIS workflow."""
        n_images = self.images.shape[0]
        if self.shuffle:
            return torch.randperm(n_images)
        return torch.arange(n_images)

    def sample_batch(self, batch_size):
        """Sample batch."""
        indices = []
        remaining = int(batch_size)
        while remaining > 0:
            if self._order is None or self._cursor >= self.images.shape[0]:
                self._order = self._new_order()
                self._cursor = 0
            available = self.images.shape[0] - self._cursor
            take = min(remaining, available)
            indices.append(self._order[self._cursor : self._cursor + take])
            self._cursor += take
            remaining -= take
        return self.images.index_select(0, torch.cat(indices))

    def __iter__(self):
        """Return an iterator over the available items."""
        order = self._new_order()
        for start in range(0, self.images.shape[0], self.batch_size):
            indices = order[start : start + self.batch_size]
            batch = self.images.index_select(0, indices)
            yield batch, batch


def default_cache_folder():
    """Return the default cache folder."""
    ramdisk_root = pathlib.Path("/ramdisks")
    if ramdisk_root.is_dir():
        return ramdisk_root / getpass.getuser() / "lion_lidc_cache_512"
    return pathlib.Path("/tmp") / getpass.getuser() / "lion_lidc_cache_512"


def normalized_slices_to_load(dataset):
    """Return normalized slices to load."""
    return {
        str(patient_id): [int(slice_index) for slice_index in slice_indices]
        for patient_id, slice_indices in dataset.slices_to_load.items()
    }


def dataset_cache_metadata(dataset, mode):
    """Handle dataset cache metadata for the PaDIS workflow."""
    params = dataset.params
    geometry = params.geometry
    return {
        "dataset": "LIDC-IDRI",
        "mode": mode,
        "task": params.task,
        "folder": str(pathlib.Path(params.folder).resolve()),
        "image_shape": [int(value) for value in geometry.image_shape],
        "image_scaling": float(geometry.image_scaling),
        "training_proportion": float(params.training_proportion),
        "validation_proportion": float(params.validation_proportion),
        "max_num_slices_per_patient": int(params.max_num_slices_per_patient),
        "pcg_slices_nodule": float(params.pcg_slices_nodule),
        "annotation": params.annotation,
        "clevel": float(params.clevel),
        "slices_to_load": normalized_slices_to_load(dataset),
    }


def cache_path_for_dataset(dataset, mode, cache_folder):
    """Handle cache path for dataset for the PaDIS workflow."""
    metadata = dataset_cache_metadata(dataset, mode)
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    _, height, width = metadata["image_shape"]
    filename = f"lidc_image_prior_{mode}_{height}x{width}_{digest}.pt"
    return cache_folder / filename, metadata


def stage_cache_from_source(mode, cache_path, source_cache_path):
    """Stage a prepared dataset cache from a source directory."""
    if not source_cache_path.is_file():
        return False
    print(f"Staging {mode} image-prior cache from {source_cache_path} to {cache_path}")
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".stage.{os.getpid()}")
    shutil.copy2(source_cache_path, tmp_path)
    tmp_path.replace(cache_path)
    return True


def stage_cache_from_archive(mode, cache_path, archive_path):
    """Restore a prepared dataset cache from an archive."""
    if not archive_path.is_file():
        return False
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError(
            f"Cannot decompress {archive_path}: zstd executable was not found."
        )
    print(
        f"Decompressing {mode} image-prior cache archive {archive_path} to {cache_path}"
    )
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".stage.{os.getpid()}")
    with tmp_path.open("wb") as output:
        subprocess.run(
            [zstd, "-T0", "-d", "-c", str(archive_path)],
            check=True,
            stdout=output,
        )
    tmp_path.replace(cache_path)
    return True


def write_cache_archive(cache_path, archive_folder, overwrite=False):
    """Write cache archive."""
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError("Cannot write compressed cache archive: zstd was not found.")
    archive_folder.mkdir(parents=True, exist_ok=True)
    archive_path = archive_folder / f"{cache_path.name}.zst"
    if archive_path.is_file() and not overwrite:
        print(f"Compressed image-prior cache archive already exists at {archive_path}")
        return
    tmp_path = archive_path.with_suffix(archive_path.suffix + f".tmp.{os.getpid()}")
    action = "Overwriting" if archive_path.is_file() else "Writing"
    print(f"{action} compressed image-prior cache archive at {archive_path}")
    with tmp_path.open("wb") as output:
        subprocess.run(
            [zstd, "-T0", "-3", "-c", str(cache_path)],
            check=True,
            stdout=output,
        )
    tmp_path.replace(archive_path)


def materialize_image_prior_dataset(
    dataset,
    mode,
    cache_folder,
    rebuild_cache,
    source_cache_folder=None,
    cache_archive_folder=None,
    write_archive=False,
    require_cache_hit=False,
):
    """Materialize image prior dataset."""
    cache_path, metadata = cache_path_for_dataset(dataset, mode, cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)
    if (
        not cache_path.is_file()
        and not rebuild_cache
        and source_cache_folder is not None
    ):
        source_cache_path, _ = cache_path_for_dataset(
            dataset, mode, source_cache_folder
        )
        stage_cache_from_source(mode, cache_path, source_cache_path)
    if (
        not cache_path.is_file()
        and not rebuild_cache
        and cache_archive_folder is not None
    ):
        archive_cache_path, _ = cache_path_for_dataset(
            dataset, mode, cache_archive_folder
        )
        archive_path = cache_archive_folder / f"{archive_cache_path.name}.zst"
        stage_cache_from_archive(mode, cache_path, archive_path)
    if cache_path.is_file() and not rebuild_cache:
        if write_archive and cache_archive_folder is not None:
            write_cache_archive(cache_path, cache_archive_folder)
        print(f"Loading {mode} image-prior cache from {cache_path}")
        cached = torch.load(cache_path, map_location="cpu")
        images = cached["images"] if isinstance(cached, dict) else cached
        return images.float().contiguous()

    if require_cache_hit and not rebuild_cache:
        raise FileNotFoundError(
            f"No prepared {mode} image-prior cache found for {cache_path.name}. "
            "Run scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm/submit_PaDIS_A100_prepare_full_cache.sh "
            "or unset PADIS_REQUIRE_CACHE_HIT to allow rebuilding from raw slices."
        )

    if len(dataset) == 0:
        raise ValueError(f"Cannot cache empty {mode} dataset.")

    print(f"Building {mode} image-prior cache at {cache_path}")
    _, first_target = dataset[0]
    first_target = first_target.float().cpu()
    images = torch.empty(
        (len(dataset), *first_target.shape),
        dtype=torch.float32,
    )
    images[0].copy_(first_target)
    for index in tqdm(range(1, len(dataset)), desc=f"Caching {mode} LIDC"):
        _, target = dataset[index]
        images[index].copy_(target.float().cpu())

    payload = {"metadata": metadata, "images": images.contiguous()}
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    tmp_path.replace(cache_path)
    if write_archive and cache_archive_folder is not None:
        write_cache_archive(
            cache_path,
            cache_archive_folder,
            overwrite=rebuild_cache,
        )
    print(
        f"Cached {mode} images: shape={tuple(images.shape)}, "
        f"size={images.numel() * images.element_size() / 1024**3:.2f} GiB"
    )
    return images


def build_cached_loaders(args, train_dataset, validation_dataset, train_batch_size):
    """Build cached loaders."""
    cache_folder = args.cache_folder or default_cache_folder()
    train_images = materialize_image_prior_dataset(
        train_dataset,
        "train",
        cache_folder,
        args.rebuild_cache,
        args.cache_source_folder,
        args.cache_archive_folder,
        args.write_cache_archive,
        args.require_cache_hit,
    )
    validation_images = materialize_image_prior_dataset(
        validation_dataset,
        "validation",
        cache_folder,
        args.rebuild_cache,
        args.cache_source_folder,
        args.cache_archive_folder,
        args.write_cache_archive,
        args.require_cache_hit,
    )
    train_loader = CachedImagePriorBatchLoader(
        train_images,
        batch_size=train_batch_size,
        shuffle=True,
        name="train",
    )
    validation_loader = CachedImagePriorBatchLoader(
        validation_images,
        batch_size=args.batch_size,
        shuffle=False,
        name="validation",
    )
    return train_loader, validation_loader


def log_wandb_outputs(wandb_run, run_folder, log_artifact=True):
    """Log wandb outputs."""
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


def build_arg_parser():
    """Construct the native-512 PaDIS training command-line parser."""
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
    parser.add_argument("--target-patches", type=int, default=400_000_000)
    parser.add_argument("--validation-interval-patches", type=int, default=200_000)
    parser.add_argument(
        "--validation-max-patches",
        type=int,
        default=1_000,
        help="Maximum validation patches per validation event. Use -1 for full validation.",
    )
    parser.add_argument(
        "--validation-repeat-until-max-patches",
        action="store_true",
        help=(
            "Repeat the selected validation image set until "
            "--validation-max-patches is reached. This increases validation "
            "patch draws without selecting more LIDC slices per patient."
        ),
    )
    parser.add_argument(
        "--validation-name",
        type=str,
        default=None,
        help=(
            "Filename for the best-validation checkpoint. Defaults to "
            "padis_lidc_512_min_val.pt."
        ),
    )
    parser.add_argument(
        "--validation-summary-key",
        type=str,
        default="min_validation_loss",
        help="WandB summary key used for the best validation loss.",
    )
    parser.add_argument(
        "--validation-checkpoint-summary-key",
        type=str,
        default="validation_checkpoint",
        help="WandB summary key used for the best validation checkpoint filename.",
    )
    parser.add_argument("--checkpoint-interval-patches", type=int, default=5_000_000)
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
        help="Maximum periodic checkpoint states to retain. Use -1 to keep all.",
    )
    parser.add_argument(
        "--keep-final-periodic-checkpoints",
        type=int,
        default=0,
        help=(
            "Periodic checkpoints to keep after clean completion. Default 0 "
            "preserves the Slurm behavior of deleting all periodic checkpoints."
        ),
    )
    parser.add_argument("--log-interval-patches", type=int, default=128)
    parser.add_argument(
        "--max-train-seconds",
        type=float,
        default=None,
        help="Stop training cleanly after this many wall-clock seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=33,
        help="Seed Python, NumPy, Torch and DataLoader-derived randomness.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=None,
        help=(
            "Optional per-step microbatch size for gradient accumulation. "
            "Use this when the effective PaDIS batch does not fit in GPU memory."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--no-persistent-workers",
        action="store_true",
        help="Disable persistent DataLoader workers for short local pilot runs.",
    )
    parser.add_argument(
        "--cache-dataset",
        choices=("none", "ramdisk"),
        default="none",
        help="Cache image-prior tensors and use a no-worker batch loader.",
    )
    parser.add_argument(
        "--cache-folder",
        type=pathlib.Path,
        default=None,
        help="Folder for cached tensors. Defaults to /ramdisks/$USER/lion_lidc_cache_512 when available.",
    )
    parser.add_argument(
        "--cache-source-folder",
        type=pathlib.Path,
        default=None,
        help="Optional persistent cache folder to stage matching tensor caches from before rebuilding from raw slices.",
    )
    parser.add_argument(
        "--cache-archive-folder",
        type=pathlib.Path,
        default=None,
        help="Optional folder containing or receiving matching .pt.zst cache archives.",
    )
    parser.add_argument(
        "--write-cache-archive",
        action="store_true",
        help="Write zstd-compressed .pt.zst archives for caches materialized in --cache-folder.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild cached image-prior tensors even if matching cache files exist.",
    )
    parser.add_argument(
        "--require-cache-hit",
        action="store_true",
        help="Fail instead of rebuilding from raw slices when no matching cache/archive exists.",
    )
    parser.add_argument(
        "--prepare-cache-only",
        action="store_true",
        help="Build or stage cached LIDC tensors and exit before model construction/training.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument(
        "--no-position-channels",
        action="store_true",
        help="Train the PaDIS ablation model without x/y position channels.",
    )
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


def main():
    """Train or resume the native-512 PaDIS prior."""
    args = build_arg_parser().parse_args()
    if args.max_slices_per_patient == 0 or args.max_slices_per_patient < -1:
        raise ValueError("--max-slices-per-patient must be positive or -1.")
    if args.validation_max_patches == 0 or args.validation_max_patches < -1:
        raise ValueError("--validation-max-patches must be positive or -1.")
    validation_max_patches = (
        None if args.validation_max_patches == -1 else args.validation_max_patches
    )
    if args.max_periodic_checkpoints == 0 or args.max_periodic_checkpoints < -1:
        raise ValueError("--max-periodic-checkpoints must be positive or -1.")
    max_periodic_checkpoints = (
        None if args.max_periodic_checkpoints == -1 else args.max_periodic_checkpoints
    )
    if args.keep_final_periodic_checkpoints < 0 or (
        args.max_periodic_checkpoints != -1
        and args.keep_final_periodic_checkpoints > args.max_periodic_checkpoints
    ):
        raise ValueError(
            "--keep-final-periodic-checkpoints must be between 0 and "
            "--max-periodic-checkpoints."
        )
    if not 0.0 <= args.pcg_slices_nodule <= 1.0:
        raise ValueError("--pcg-slices-nodule must be in [0, 1].")
    set_run_seed(args.seed)
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
    preset = "padis-paper-ct-512"
    if args.no_position_channels:
        preset = f"{preset}-no-position"
    solver_params = PaDISSolver.default_parameters(preset)
    solver_params.use_ema = not args.no_ema
    solver_params.base_patch_batch_size = args.batch_size
    solver_params.microbatch_size = args.microbatch_size
    max_batch_multiplier = max(solver_params.patch_batch_multipliers.values())
    train_loader_batch_size = args.batch_size
    if args.cache_dataset == "ramdisk":
        train_loader_batch_size *= max_batch_multiplier
        train_loader, validation_loader = build_cached_loaders(
            args, train_dataset, validation_dataset, train_loader_batch_size
        )
    else:
        if args.prepare_cache_only:
            raise ValueError("--prepare-cache-only requires --cache-dataset ramdisk.")
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **data_loader_kwargs(args, device, train_loader_batch_size),
        )
        validation_loader = DataLoader(
            validation_dataset,
            shuffle=False,
            **data_loader_kwargs(args, device, args.batch_size),
        )

    if args.prepare_cache_only:
        print("Prepared cached LIDC image-prior tensors; exiting before training.")
        return

    model_params = NCSNpp.default_parameters(preset)
    model = NCSNpp(model_params, geometry)
    loss_fn = PaDISDenoisingLoss(
        sigma_min=model_params.sigma_min,
        sigma_max=model_params.sigma_max,
        sigma_distribution=solver_params.sigma_distribution,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    run_folder = make_run_folder(args.save_folder, args.run_name, "padis_lidc_512")
    print(f"Saving PaDIS run to {run_folder}")
    wandb_run = init_wandb(args, run_folder, preset)
    previous_sigterm_handler = install_termination_handler()
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
    solver.set_checkpoint_retention(max_periodic_checkpoints)
    solver.set_training(train_loader)
    solver.set_validation(
        validation_loader,
        validation_freq=10**12,
        validation_fname=args.validation_name,
    )
    try:
        try:
            solver.train_for_patches(
                args.target_patches,
                validation_interval_patches=args.validation_interval_patches,
                validation_max_patches=validation_max_patches,
                validation_repeat_until_max_patches=(
                    args.validation_repeat_until_max_patches
                ),
                checkpoint_interval_patches=args.checkpoint_interval_patches,
                checkpoint_interval_seconds=args.checkpoint_interval_seconds,
                log_interval_patches=args.log_interval_patches,
                max_train_seconds=args.max_train_seconds,
                log_fn=wandb_log_fn(wandb_run),
            )
        except TerminationRequested as exc:
            print(f"{exc}; saving resumable checkpoint before exit.")
            save_interruption_checkpoint(solver)
            raise SystemExit(143)
        if args.keep_final_periodic_checkpoints == 0:
            solver.clean_checkpoints()
        else:
            solver.prune_periodic_checkpoints(args.keep_final_periodic_checkpoints)
        solver.save_final_results()
        save_loss_plots(solver, run_folder)
        if wandb_run is not None:
            wandb_run.summary[args.validation_summary_key] = min_validation_loss(solver)
            if solver.validation_fname is not None:
                wandb_run.summary[
                    args.validation_checkpoint_summary_key
                ] = solver.validation_fname
            wandb_run.summary["seen_patches"] = solver.seen_patches
            wandb_run.summary["training_steps"] = len(solver.train_loss)
        log_wandb_outputs(
            wandb_run, run_folder, log_artifact=not args.no_wandb_artifact
        )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
