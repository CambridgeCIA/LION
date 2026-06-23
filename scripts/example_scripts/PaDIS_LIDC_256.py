"""Train PaDIS paper-style patch or whole-image diffusion priors on LIDC-IDRI."""

import argparse
from datetime import datetime
import getpass
import hashlib
import json
import math
import os
import pathlib
import re
import uuid

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from LION.CTtools.ct_geometry import Geometry
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
from LION.losses.PaDIS import PaDISDenoisingLoss
from LION.models.diffusion import NCSNpp
from LION.optimizers import PaDISSolver
from LION.utils.paths import LION_EXPERIMENTS_PATH


class CachedImagePriorBatchLoader(DataLoader):
    """Small batch loader for cached PaDIS image-prior tensors."""

    def __init__(self, images, batch_size, shuffle, name):
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
        return math.ceil(self.images.shape[0] / self.batch_size)

    def _new_order(self):
        n_images = self.images.shape[0]
        if self.shuffle:
            return torch.randperm(n_images)
        return torch.arange(n_images)

    def sample_batch(self, batch_size):
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
        order = self._new_order()
        for start in range(0, self.images.shape[0], self.batch_size):
            indices = order[start : start + self.batch_size]
            batch = self.images.index_select(0, indices)
            yield batch, batch


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


def data_loader_kwargs(args, device, batch_size):
    kwargs = {
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def default_cache_folder():
    ramdisk_root = pathlib.Path("/ramdisks")
    if ramdisk_root.is_dir():
        return ramdisk_root / getpass.getuser() / "lion_lidc_cache"
    return pathlib.Path("/tmp") / getpass.getuser() / "lion_lidc_cache"


def normalized_slices_to_load(dataset):
    return {
        str(patient_id): [int(slice_index) for slice_index in slice_indices]
        for patient_id, slice_indices in dataset.slices_to_load.items()
    }


def dataset_cache_metadata(dataset, mode):
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
    metadata = dataset_cache_metadata(dataset, mode)
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    _, height, width = metadata["image_shape"]
    filename = f"lidc_image_prior_{mode}_{height}x{width}_{digest}.pt"
    return cache_folder / filename, metadata


def materialize_image_prior_dataset(dataset, mode, cache_folder, rebuild_cache):
    cache_path, metadata = cache_path_for_dataset(dataset, mode, cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file() and not rebuild_cache:
        print(f"Loading {mode} image-prior cache from {cache_path}")
        cached = torch.load(cache_path, map_location="cpu")
        images = cached["images"] if isinstance(cached, dict) else cached
        return images.float().contiguous()

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
    print(
        f"Cached {mode} images: shape={tuple(images.shape)}, "
        f"size={images.numel() * images.element_size() / 1024**3:.2f} GiB"
    )
    return images


def build_cached_loaders(args, train_dataset, validation_dataset, train_batch_size):
    cache_folder = args.cache_folder or default_cache_folder()
    train_images = materialize_image_prior_dataset(
        train_dataset, "train", cache_folder, args.rebuild_cache
    )
    validation_images = materialize_image_prior_dataset(
        validation_dataset, "validation", cache_folder, args.rebuild_cache
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
    parser.add_argument(
        "--prior-mode",
        choices=("patch", "whole-image"),
        default="patch",
        help="Train the PaDIS patch prior or the paper's whole-image diffusion baseline.",
    )
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
        default=LION_EXPERIMENTS_PATH.joinpath("PaDIS", "LIDC_256"),
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--target-patches", type=int, default=200_000_000)
    parser.add_argument("--validation-interval-patches", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-interval-patches", type=int, default=5_000_000)
    parser.add_argument("--log-interval-patches", type=int, default=1_000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Base batch size. Defaults to 128 for patch PaDIS and 8 for whole-image training.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
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
        help="Folder for cached tensors. Defaults to /ramdisks/$USER/lion_lidc_cache when available.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild cached image-prior tensors even if matching cache files exist.",
    )
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

    geometry = Geometry.default_parameters(image_scaling=0.5)
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
    preset = (
        "padis-paper-whole-ct-256"
        if args.prior_mode == "whole-image"
        else "padis-paper-ct-256"
    )
    if args.batch_size is None:
        args.batch_size = 8 if args.prior_mode == "whole-image" else 128
    solver_params = PaDISSolver.default_parameters(preset)
    solver_params.use_ema = not args.no_ema
    solver_params.base_patch_batch_size = args.batch_size
    max_batch_multiplier = max(solver_params.patch_batch_multipliers.values())
    train_loader_batch_size = args.batch_size * max_batch_multiplier

    train_dataset = LIDC_IDRI(
        "train", parameters=data_params, geometry_parameters=geometry
    )
    validation_dataset = LIDC_IDRI(
        "validation", parameters=data_params, geometry_parameters=geometry
    )
    if args.cache_dataset == "ramdisk":
        train_loader, validation_loader = build_cached_loaders(
            args, train_dataset, validation_dataset, train_loader_batch_size
        )
    else:
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

    model_params = NCSNpp.default_parameters(preset)
    model = NCSNpp(model_params, geometry)
    loss_fn = PaDISDenoisingLoss(
        sigma_min=model_params.sigma_min, sigma_max=model_params.sigma_max
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    run_prefix = (
        "whole_image_lidc_256" if args.prior_mode == "whole-image" else "padis_lidc_256"
    )
    run_folder = make_run_folder(args.save_folder, args.run_name, run_prefix)
    print(f"Saving {args.prior_mode} diffusion run to {run_folder}")
    wandb_run = init_wandb(args, run_folder, preset)
    solver = PaDISSolver(
        model,
        optimizer,
        loss_fn,
        geometry=geometry,
        solver_params=solver_params,
        device=device,
        save_folder=run_folder,
    )
    solver.set_saving(run_folder, run_prefix)
    solver.set_checkpointing(
        f"{run_prefix}_checkpoint_*.pt",
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
        if wandb_run is not None:
            wandb_run.summary["min_validation_loss"] = min_validation_loss(solver)
            wandb_run.summary["seen_patches"] = solver.seen_patches
        log_wandb_outputs(wandb_run, run_folder)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
