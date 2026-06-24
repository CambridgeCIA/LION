"""Run target-machine preflight checks for PaDIS CT training.

The goal is to catch expensive-run failures before launching long jobs. The
suite checks hardware visibility, LIDC loading, model construction, training
throughput, validation overhead, checkpoint resume mechanics, experiment preset
dry-runs, optional golden equivalence, and optional CLI training smoke runs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import resource
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

LION_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(LION_ROOT) not in sys.path:
    sys.path.insert(0, str(LION_ROOT))

DEV_ROOT = pathlib.Path(__file__).resolve().parent
if str(DEV_ROOT) not in sys.path:
    sys.path.insert(0, str(DEV_ROOT))

from LION.CTtools.ct_geometry import Geometry  # noqa: E402
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI  # noqa: E402
from LION.losses.PaDIS import PaDISDenoisingLoss  # noqa: E402
from LION.models.diffusion import NCSNpp  # noqa: E402
from LION.optimizers import PaDISSolver  # noqa: E402


@dataclass(frozen=True)
class ModeCheck:
    name: str
    image_scaling: float
    patch_sizes: tuple[int, ...]


CORE_MODE_CHECKS = (
    ModeCheck("padis-paper-ct-256", 0.5, (16, 32, 56)),
    ModeCheck("padis-paper-ct-512", 1.0, (16, 32, 64)),
    ModeCheck("padis-paper-whole-ct-256", 0.5, (256,)),
)

ABLATION_MODE_CHECKS = (
    ModeCheck("padis-paper-ct-p8", 0.5, (8,)),
    ModeCheck("padis-paper-ct-p16", 0.5, (8, 16)),
    ModeCheck("padis-paper-ct-p32", 0.5, (8, 16, 32)),
    ModeCheck("padis-paper-ct-p56", 0.5, (16, 32, 56)),
    ModeCheck("padis-paper-ct-p96", 0.5, (32, 64, 96)),
    ModeCheck("padis-paper-ct-256-no-position", 0.5, (16, 32, 56)),
)


class LimitedLoader:
    def __init__(self, loader, max_batches: int | None):
        self.loader = loader
        self.max_batches = max_batches

    def __iter__(self):
        for index, batch in enumerate(self.loader):
            if self.max_batches is not None and index >= self.max_batches:
                break
            yield batch

    def __len__(self):
        if self.max_batches is None:
            return len(self.loader)
        return min(len(self.loader), int(self.max_batches))


def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _jsonable(value):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _device_from_arg(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access CUDA.")
    return device


def _cuda_report(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_capability": torch.cuda.get_device_capability(device),
        "cuda_total_gib": total_bytes / 1024**3,
        "cuda_free_gib": free_bytes / 1024**3,
        "cuda_allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
        "cuda_reserved_gib": torch.cuda.memory_reserved(device) / 1024**3,
    }


def _nvidia_smi_report() -> dict[str, object]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi not on PATH"}
    query = "--query-gpu=name,memory.total,memory.used,memory.free,driver_version"
    command = [
        "nvidia-smi",
        query,
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        return {
            "available": False,
            "reason": "nvidia-smi returned a non-zero exit status",
            "returncode": exc.returncode,
            "stderr": exc.stderr,
            "stdout": exc.stdout,
        }
    gpus = []
    for line in completed.stdout.strip().splitlines():
        if not line.strip():
            continue
        name, total, used, free, driver = [part.strip() for part in line.split(",")]
        gpus.append(
            {
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "memory_free_mib": int(free),
                "driver_version": driver,
            }
        )
    return {"available": True, "gpus": gpus}


def _run_checked(name: str, func: Callable[[], dict[str, object]]) -> dict[str, object]:
    start = time.perf_counter()
    try:
        details = func()
        status = "passed"
    except Exception as exc:  # noqa: BLE001 - report everything in a preflight suite.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        details = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
        status = "failed"
    return {
        "name": name,
        "status": status,
        "seconds": time.perf_counter() - start,
        "details": _jsonable(details),
    }


def _data_params(args: argparse.Namespace, geometry: Geometry):
    params = LIDC_IDRI.default_parameters(geometry=geometry, task="image_prior")
    params.device = torch.device("cpu")
    params.max_num_slices_per_patient = int(args.max_slices_per_patient)
    params.pcg_slices_nodule = float(args.pcg_slices_nodule)
    if args.data_folder is not None:
        params.folder = args.data_folder
    return params


def _dataset(args: argparse.Namespace, geometry: Geometry, mode: str):
    return LIDC_IDRI(
        mode,
        parameters=_data_params(args, geometry),
        geometry_parameters=geometry,
    )


def _loader_kwargs(args: argparse.Namespace, device: torch.device, batch_size: int):
    kwargs = {
        "batch_size": batch_size,
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "shuffle": False,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return kwargs


def _make_solver(
    mode: str,
    geometry: Geometry,
    device: torch.device,
    args: argparse.Namespace,
    *,
    patch_sizes: tuple[int, ...] | None = None,
    patch_size: int | None = None,
) -> PaDISSolver:
    model_params = NCSNpp.default_parameters(mode)
    model = NCSNpp(model_params, geometry)
    solver_params = PaDISSolver.default_parameters(mode)
    if patch_size is not None:
        solver_params.patch_sizes = [int(patch_size)]
        solver_params.patch_probabilities = [1.0]
        solver_params.patch_batch_multipliers = {int(patch_size): 1}
    elif patch_sizes is not None:
        solver_params.patch_sizes = list(patch_sizes)
        solver_params.patch_probabilities = [1.0 / len(patch_sizes)] * len(patch_sizes)
        solver_params.patch_batch_multipliers = {int(size): 1 for size in patch_sizes}
    solver_params.base_patch_batch_size = int(args.base_batch_size)
    solver_params.microbatch_size = args.microbatch_size
    solver_params.lr_rampup_kimg = float(args.lr_rampup_kimg)
    solver_params.use_ema = True
    loss_fn = PaDISDenoisingLoss(
        sigma_min=model_params.sigma_min,
        sigma_max=model_params.sigma_max,
        sigma_distribution=solver_params.sigma_distribution,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    solver = PaDISSolver(
        model,
        optimizer,
        loss_fn,
        geometry=geometry,
        verbose=False,
        device=device,
        solver_params=solver_params,
    )
    solver.seen_patches = int(args.initial_seen_patches)
    return solver


def check_environment(args: argparse.Namespace) -> dict[str, object]:
    device = _device_from_arg(args.device)
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "rss_gib": _rss_gib(),
        "nvidia_smi": _nvidia_smi_report(),
        **_cuda_report(device),
    }


def check_experiment_presets(args: argparse.Namespace) -> dict[str, object]:
    script = LION_ROOT / "scripts" / "example_scripts" / "PaDIS_experiments.py"
    commands = [
        [sys.executable, str(script), "list"],
        [
            sys.executable,
            str(script),
            "run",
            "paper-fan-20",
            "--dry-run",
            "--device",
            args.device,
        ],
        [
            sys.executable,
            str(script),
            "run",
            "train-patch-lidc-quarter",
            "--dry-run",
            "--device",
            args.device,
        ],
    ]
    outputs = []
    for command in commands:
        completed = subprocess.run(
            command, cwd=LION_ROOT, check=True, capture_output=True, text=True
        )
        outputs.append(
            {
                "command": command,
                "stdout_first_lines": completed.stdout.splitlines()[:8],
            }
        )
    return {"commands": outputs}


def check_golden(args: argparse.Namespace) -> dict[str, object]:
    script = DEV_ROOT / "check_padis_repo_equivalence.py"
    if not args.padis_root.is_dir():
        return {"skipped": True, "reason": f"Missing PaDIS root: {args.padis_root}"}
    if not args.golden.is_file():
        command = [
            sys.executable,
            str(script),
            "--padis-root",
            str(args.padis_root),
            "--device",
            "cpu",
            "--write-golden",
            str(args.golden),
        ]
        subprocess.run(
            command, cwd=LION_ROOT, check=True, capture_output=True, text=True
        )
    command = [
        sys.executable,
        str(script),
        "--padis-root",
        str(args.padis_root),
        "--device",
        "cpu",
        "--golden",
        str(args.golden),
    ]
    completed = subprocess.run(
        command, cwd=LION_ROOT, check=True, capture_output=True, text=True
    )
    summary = json.loads(completed.stdout)
    return {
        "golden_all_passed": summary.get("golden_all_passed"),
        "max_patch_roundoff": summary.get("golden_public_patch_denoised_max_abs"),
    }


def check_short_run_reproduction(args: argparse.Namespace) -> dict[str, object]:
    script = DEV_ROOT / "check_padis_short_run_reproduction.py"
    if not args.padis_root.is_dir():
        return {"skipped": True, "reason": f"Missing PaDIS root: {args.padis_root}"}
    command = [
        sys.executable,
        str(script),
        "--padis-root",
        str(args.padis_root),
        "--device",
        args.device,
        "--steps",
        str(args.short_run_steps),
        "--tolerance",
        str(args.short_run_tolerance),
        "--relative-tolerance",
        str(args.short_run_relative_tolerance),
    ]
    completed = subprocess.run(
        command, cwd=LION_ROOT, check=True, capture_output=True, text=True
    )
    summary = json.loads(completed.stdout)
    return {
        "passed": summary.get("passed"),
        "steps": summary.get("steps"),
        "max_abs": summary.get("max_abs"),
        "max_l2_relative": summary.get("max_l2_relative"),
        "mapping": summary.get("mapping"),
    }


def check_dataset(args: argparse.Namespace) -> dict[str, object]:
    details = {}
    for scaling in sorted({mode.image_scaling for mode in _selected_modes(args)}):
        geometry = Geometry.default_parameters(image_scaling=scaling)
        mode_details = {}
        for split in ("train", "validation"):
            start = time.perf_counter()
            dataset = _dataset(args, geometry, split)
            sample_start = time.perf_counter()
            _source, target = dataset[0]
            sample_seconds = time.perf_counter() - sample_start
            mode_details[split] = {
                "length": len(dataset),
                "construct_seconds": time.perf_counter() - start,
                "sample_seconds": sample_seconds,
                "target_shape": tuple(target.shape),
                "target_min": float(target.min()),
                "target_max": float(target.max()),
                "slices_to_load_patients": len(dataset.slices_to_load),
            }
            if target.min() < -1e-5 or target.max() > 1 + 1e-5:
                raise ValueError(f"{split} target outside [0, 1] at scaling {scaling}.")
        details[f"image_scaling_{scaling}"] = mode_details
    return details


def _fetch_target_batch(
    args: argparse.Namespace,
    geometry: Geometry,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    dataset = _dataset(args, geometry, "train")
    loader = DataLoader(dataset, **_loader_kwargs(args, device, batch_size))
    _source, target = next(iter(loader))
    return target.float()


def check_training_mode(
    mode: ModeCheck,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    geometry = Geometry.default_parameters(image_scaling=mode.image_scaling)
    model_params = NCSNpp.default_parameters(mode.name)
    solver_defaults = PaDISSolver.default_parameters(mode.name)
    max_batch = max(
        int(args.base_batch_size)
        * int(solver_defaults.patch_batch_multipliers.get(int(size), 1))
        for size in mode.patch_sizes
    )
    target = _fetch_target_batch(args, geometry, device, max_batch)
    if target.min() < -1e-5 or target.max() > 1 + 1e-5:
        raise ValueError("Training target outside [0, 1].")

    solver = _make_solver(mode.name, geometry, device, args)
    trainable_params = sum(param.numel() for param in solver.model.parameters())
    results = {}
    for patch_size in mode.patch_sizes:
        multiplier = int(
            solver_defaults.patch_batch_multipliers.get(int(patch_size), 1)
        )
        effective_batch = int(args.base_batch_size) * multiplier
        patch_target = target[:effective_batch]
        torch.manual_seed(int(args.seed) + int(patch_size))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        step_times = []
        losses = []
        for _ in range(int(args.training_steps)):
            start = time.perf_counter()
            loss = solver._optimizer_step(patch_target, patch_size=int(patch_size))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(time.perf_counter() - start)
            losses.append(float(loss))
        mean_step = float(np.mean(step_times))
        patches_per_second = effective_batch / max(mean_step, 1e-12)
        patch_result = {
            "effective_batch": effective_batch,
            "microbatch_size": args.microbatch_size,
            "step_seconds": step_times,
            "mean_step_seconds": mean_step,
            "patches_per_second": patches_per_second,
            "losses": losses,
            "seen_patches": solver.seen_patches,
        }
        if device.type == "cuda":
            patch_result["cuda_peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated(device) / 1024**3
            )
            patch_result["cuda_peak_reserved_gib"] = (
                torch.cuda.max_memory_reserved(device) / 1024**3
            )
        results[f"patch_{patch_size}"] = patch_result
    return {
        "mode": mode.name,
        "image_scaling": mode.image_scaling,
        "trainable_params": trainable_params,
        "target_batch_shape": tuple(target.shape),
        "results": results,
        "rss_gib": _rss_gib(),
        **_cuda_report(device),
    }


def check_validation_benchmark(
    mode: ModeCheck,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    geometry = Geometry.default_parameters(image_scaling=mode.image_scaling)
    dataset = _dataset(args, geometry, "validation")
    loader = DataLoader(
        dataset,
        **_loader_kwargs(args, device, int(args.validation_batch_size)),
    )
    limited_loader = LimitedLoader(loader, args.validation_batches)
    solver = _make_solver(
        mode.name, geometry, device, args, patch_sizes=mode.patch_sizes
    )
    solver.set_validation(limited_loader, validation_freq=10**12)
    torch.manual_seed(int(args.seed) + 99)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    loss = solver.validate()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measured_seconds = time.perf_counter() - start
    measured_batches = len(limited_loader)
    full_batches = len(loader)
    seconds_per_batch = measured_seconds / max(measured_batches, 1)
    estimated_full_seconds = seconds_per_batch * full_batches
    return {
        "mode": mode.name,
        "validation_loss": loss,
        "measured_batches": measured_batches,
        "full_batches": full_batches,
        "measured_seconds": measured_seconds,
        "seconds_per_batch": seconds_per_batch,
        "estimated_full_validation_seconds": estimated_full_seconds,
        "full_validation_measured": args.validation_batches is None,
        **_validation_fraction_table(args, estimated_full_seconds),
        **_cuda_report(device),
    }


def _validation_fraction_table(
    args: argparse.Namespace,
    validation_seconds: float,
) -> dict[str, object]:
    training_pps = float(args.expected_patches_per_second)
    if training_pps <= 0:
        return {"validation_fraction_estimates": []}
    rows = []
    for interval in args.validation_interval_candidates:
        train_seconds = float(interval) / training_pps
        fraction = validation_seconds / max(train_seconds + validation_seconds, 1e-12)
        rows.append(
            {
                "validation_interval_patches": int(interval),
                "estimated_train_seconds_between_validations": train_seconds,
                "estimated_validation_seconds": validation_seconds,
                "estimated_validation_fraction": fraction,
            }
        )
    return {"validation_fraction_estimates": rows}


def check_checkpoint_resume(
    args: argparse.Namespace, device: torch.device
) -> dict[str, object]:
    mode = ModeCheck("padis-paper-ct-p8", 0.5, (8,))
    geometry = Geometry.default_parameters(image_scaling=mode.image_scaling)
    target = _fetch_target_batch(args, geometry, device, int(args.base_batch_size))
    folder = args.output_dir / "checkpoint_resume"
    folder.mkdir(parents=True, exist_ok=True)

    solver = _make_solver(mode.name, geometry, device, args, patch_size=8)
    solver.set_saving(folder, "resume_smoke")
    solver.set_checkpointing(
        "resume_smoke_checkpoint_*.pt", load_checkpoint_if_exists=False
    )
    solver._optimizer_step(target, patch_size=8)
    solver.save_checkpoint(0)

    resumed = _make_solver(mode.name, geometry, device, args, patch_size=8)
    resumed.set_saving(folder, "resume_smoke")
    resumed.set_checkpointing(
        "resume_smoke_checkpoint_*.pt", load_checkpoint_if_exists=True
    )
    loaded_epoch = resumed.load_checkpoint()
    resumed._optimizer_step(target, patch_size=8)
    resumed.save_final_results(epoch=loaded_epoch + 1)
    full_path = folder / "resume_smoke_full.pt"
    data = torch.load(full_path, map_location="cpu", weights_only=False)
    return {
        "loaded_epoch": int(loaded_epoch),
        "resumed_seen_patches": int(resumed.seen_patches),
        "full_state_exists": full_path.is_file(),
        "full_state_has_optimizer": "optimizer_state_dict" in data,
        "checkpoint_files": sorted(path.name for path in folder.glob("*.pt")),
    }


def check_cli_smoke(args: argparse.Namespace) -> dict[str, object]:
    run_folder = args.output_dir / "cli_smoke"
    command = [
        sys.executable,
        str(LION_ROOT / "scripts" / "example_scripts" / "PaDIS_LIDC_256.py"),
        "--device",
        args.device,
        "--patch-size-preset",
        "8",
        "--batch-size",
        str(args.base_batch_size),
        "--target-patches",
        str(args.cli_target_patches),
        "--validation-interval-patches",
        str(args.cli_target_patches * 100),
        "--checkpoint-interval-patches",
        str(args.cli_target_patches * 100),
        "--log-interval-patches",
        str(max(1, args.cli_target_patches)),
        "--max-slices-per-patient",
        str(args.max_slices_per_patient),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--save-folder",
        str(run_folder),
        "--run-name",
        "preflight_cli",
        "--no-wandb",
    ]
    if args.microbatch_size is not None:
        command.extend(("--microbatch-size", str(args.microbatch_size)))
    completed = subprocess.run(
        command,
        cwd=LION_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=args.cli_timeout_seconds,
    )
    output = run_folder / "preflight_cli"
    full_path = output / "padis_lidc_256_full.pt"
    data = torch.load(full_path, map_location="cpu", weights_only=False)
    return {
        "command": command,
        "stdout_tail": completed.stdout.splitlines()[-12:],
        "output_files": sorted(path.name for path in output.glob("*")),
        "full_state_seen_patches": data.get("seen_patches"),
        "full_state_has_optimizer": "optimizer_state_dict" in data,
    }


def _selected_modes(args: argparse.Namespace) -> tuple[ModeCheck, ...]:
    modes = list(CORE_MODE_CHECKS)
    if args.mode_set == "all":
        modes.extend(ABLATION_MODE_CHECKS)
    if args.skip_whole_image:
        modes = [mode for mode in modes if "whole" not in mode.name]
    return tuple(modes)


def _derive_expected_pps(training_results: list[dict[str, object]]) -> float:
    values = []
    for result in training_results:
        if result["status"] != "passed":
            continue
        for item in result["details"]["results"].values():
            values.append(float(item["patches_per_second"]))
    if not values:
        return 0.0
    return float(np.median(values))


def run_suite(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_arg(args.device)
    report = {
        "started_at_unix": time.time(),
        "lion_root": str(LION_ROOT),
        "output_dir": str(args.output_dir),
        "arguments": _jsonable(vars(args)),
        "checks": [],
    }

    check_specs: list[tuple[str, Callable[[], dict[str, object]]]] = [
        ("environment", lambda: check_environment(args)),
        ("experiment_preset_dry_runs", lambda: check_experiment_presets(args)),
        ("dataset", lambda: check_dataset(args)),
    ]
    if not args.skip_golden:
        check_specs.append(("golden_equivalence", lambda: check_golden(args)))
    if not args.skip_short_run_reproduction:
        check_specs.append(
            (
                "short_run_reproduction",
                lambda: check_short_run_reproduction(args),
            )
        )

    for name, func in check_specs:
        print(f"Running {name}...")
        result = _run_checked(name, func)
        report["checks"].append(result)
        print(f"  {result['status']} in {result['seconds']:.2f}s")

    training_results = []
    for mode in _selected_modes(args):
        name = f"training_{mode.name}"
        print(f"Running {name}...")
        result = _run_checked(
            name, lambda mode=mode: check_training_mode(mode, args, device)
        )
        report["checks"].append(result)
        training_results.append(result)
        print(f"  {result['status']} in {result['seconds']:.2f}s")

    if args.expected_patches_per_second <= 0:
        args.expected_patches_per_second = _derive_expected_pps(training_results)
    for mode in _selected_modes(args):
        name = f"validation_{mode.name}"
        print(f"Running {name}...")
        result = _run_checked(
            name, lambda mode=mode: check_validation_benchmark(mode, args, device)
        )
        report["checks"].append(result)
        print(f"  {result['status']} in {result['seconds']:.2f}s")

    if not args.skip_checkpoint_resume:
        print("Running checkpoint_resume...")
        result = _run_checked(
            "checkpoint_resume", lambda: check_checkpoint_resume(args, device)
        )
        report["checks"].append(result)
        print(f"  {result['status']} in {result['seconds']:.2f}s")

    if args.run_cli_smoke:
        print("Running cli_smoke...")
        result = _run_checked("cli_smoke", lambda: check_cli_smoke(args))
        report["checks"].append(result)
        print(f"  {result['status']} in {result['seconds']:.2f}s")

    report["finished_at_unix"] = time.time()
    report["passed"] = all(check["status"] == "passed" for check in report["checks"])
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/padis_preflight"),
    )
    parser.add_argument("--data-folder", type=pathlib.Path, default=None)
    parser.add_argument("--mode-set", choices=("core", "all"), default="core")
    parser.add_argument("--skip-whole-image", action="store_true")
    parser.add_argument("--base-batch-size", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--training-steps", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Measure the full validation set instead of extrapolating from a cap.",
    )
    parser.add_argument(
        "--validation-interval-candidates",
        type=int,
        nargs="+",
        default=(100_000, 1_000_000, 5_000_000),
    )
    parser.add_argument("--expected-patches-per-second", type=float, default=0.0)
    parser.add_argument("--max-slices-per-patient", type=int, default=4)
    parser.add_argument("--pcg-slices-nodule", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-rampup-kimg", type=float, default=10_000)
    parser.add_argument("--initial-seen-patches", type=int, default=1_000)
    parser.add_argument("--skip-golden", action="store_true")
    parser.add_argument("--skip-short-run-reproduction", action="store_true")
    parser.add_argument("--short-run-steps", type=int, default=3)
    parser.add_argument("--short-run-tolerance", type=float, default=2e-2)
    parser.add_argument("--short-run-relative-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--padis-root",
        type=pathlib.Path,
        default=LION_ROOT.parent / "PaDIS",
    )
    parser.add_argument(
        "--golden",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/padis_lion_golden.pt"),
    )
    parser.add_argument("--skip-checkpoint-resume", action="store_true")
    parser.add_argument("--run-cli-smoke", action="store_true")
    parser.add_argument("--cli-target-patches", type=int, default=2)
    parser.add_argument("--cli-timeout-seconds", type=float, default=600)
    parser.add_argument("--json", type=pathlib.Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.full_validation:
        args.validation_batches = None
    if args.json is None:
        args.json = args.output_dir / "preflight_report.json"
    report = run_suite(args)
    text = json.dumps(_jsonable(report), indent=2, sort_keys=True)
    print(text)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(text + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
