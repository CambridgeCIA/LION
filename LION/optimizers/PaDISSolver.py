"""LION-native solver for PaDIS patch-prior training."""

from __future__ import annotations

from typing import Callable, Optional
import pathlib
import time

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from LION.CTtools.ct_geometry import Geometry
from LION.models.LIONmodel import LIONmodel
from LION.optimizers.LIONsolver import LIONsolver, SolverParams
from LION.utils.parameter import LIONParameter
from LION.losses.PaDIS import (
    build_position_grid,
    sample_image_patch,
    sample_image_patch_with_position_channels,
    sample_patch_size,
    validate_patch_schedule,
    zero_pad_images,
)


_PATCH_ABLATION_PRESETS = {
    "padis-paper-ct-p8": {
        "patch_sizes": [8],
        "patch_probabilities": [1.0],
        "patch_batch_multipliers": {8: 1},
        "pad_width": 8,
        "largest_patch_size": 8,
    },
    "padis-paper-ct-p16": {
        "patch_sizes": [8, 16],
        "patch_probabilities": [0.3, 0.7],
        "patch_batch_multipliers": {8: 2, 16: 1},
        "pad_width": 16,
        "largest_patch_size": 16,
    },
    "padis-paper-ct-p32": {
        "patch_sizes": [8, 16, 32],
        "patch_probabilities": [0.2, 0.3, 0.5],
        "patch_batch_multipliers": {8: 4, 16: 2, 32: 1},
        "pad_width": 32,
        "largest_patch_size": 32,
    },
    "padis-paper-ct-p56": {
        "patch_sizes": [16, 32, 56],
        "patch_probabilities": [0.2, 0.3, 0.5],
        "patch_batch_multipliers": {16: 4, 32: 2, 56: 1},
        "pad_width": 24,
        "largest_patch_size": 56,
    },
    "padis-paper-ct-p96": {
        "patch_sizes": [32, 64, 96],
        "patch_probabilities": [0.2, 0.3, 0.5],
        "patch_batch_multipliers": {32: 4, 64: 2, 96: 1},
        "pad_width": 32,
        "largest_patch_size": 96,
    },
}


def _split_position_suffix(mode: str) -> tuple[str, bool]:
    suffix = "-no-position"
    if mode.endswith(suffix):
        return mode[: -len(suffix)], True
    return mode, False


class PaDISSolver(LIONsolver):
    """Train a PaDIS patch-based diffusion prior with LION's solver API."""

    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn: Callable,
        geometry: Geometry = None,
        verbose: bool = True,
        device: torch.device = None,
        solver_params: Optional[SolverParams] = None,
        save_folder: Optional[pathlib.Path] = None,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            loss_fn,
            geometry,
            verbose,
            device,
            solver_params=solver_params,
            save_folder=save_folder,
        )
        if not hasattr(self.solver_params, "prior_mode"):
            self.solver_params.prior_mode = "patch"
        if not hasattr(self.solver_params, "use_position_channels"):
            self.solver_params.use_position_channels = True
        if not hasattr(self.solver_params, "sigma_distribution"):
            self.solver_params.sigma_distribution = "edm_lognormal_truncated"
        if not hasattr(self.solver_params, "microbatch_size"):
            self.solver_params.microbatch_size = None
        validate_patch_schedule(
            self.solver_params.patch_sizes, self.solver_params.patch_probabilities
        )
        self._validate_solver_configuration()
        self.ema_state: dict[str, torch.Tensor] | None = None
        self.seen_patches = 0
        if self.solver_params.use_ema:
            self.ema_state = {
                name: param.detach().clone()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }
        if not hasattr(self, "checkpoint_freq"):
            self.checkpoint_freq = 10**12
        self.last_validation_patches = 0
        self.max_periodic_checkpoints: int | None = None
        self.metadata = LIONParameter()
        self.metadata.method = (
            "PaDIS paper whole-image denoising"
            if self.solver_params.prior_mode == "whole_image"
            else "PaDIS paper patch denoising"
        )
        self.metadata.paper_preset = self.solver_params.paper_preset
        self.metadata.prior_mode = self.solver_params.prior_mode
        self.metadata.patch_sizes = self.solver_params.patch_sizes
        self.metadata.patch_probabilities = self.solver_params.patch_probabilities
        self.metadata.pad_width = self.solver_params.pad_width
        self.metadata.sigma_min = self.solver_params.sigma_min
        self.metadata.sigma_max = self.solver_params.sigma_max
        self.metadata.sigma_distribution = self.solver_params.sigma_distribution
        self.metadata.use_position_channels = self.solver_params.use_position_channels
        self.metadata.microbatch_size = self.solver_params.microbatch_size
        self.metadata.ema_half_life_patches = self.solver_params.ema_half_life_patches
        self.metadata.ema_rampup_ratio = self.solver_params.ema_rampup_ratio
        self.metadata.lr_rampup_kimg = self.solver_params.lr_rampup_kimg
        self.metadata.lidc_image_scaling = getattr(self.geometry, "image_scaling", None)
        self.metadata.lidc_resize_policy = (
            "CPU resize and patch extraction before final device transfer"
        )

    @staticmethod
    def default_parameters(mode: str = "padis-paper-ct-256") -> SolverParams:
        base_mode, no_position = _split_position_suffix(mode)
        params = SolverParams()
        params.paper_preset = mode
        params.sigma_min = 0.002
        params.sigma_max = 40.0
        params.sigma_distribution = "edm_lognormal_truncated"
        params.prior_mode = "patch"
        params.use_position_channels = True
        params.use_ema = True
        params.ema_half_life_patches = 500_000
        params.ema_rampup_ratio = 0.05
        params.lr_rampup_kimg = 10_000
        params.enforce_data_range = True
        params.input_mode = "image"
        params.microbatch_size = None
        if base_mode == "padis-paper-ct-256":
            params.patch_sizes = [16, 32, 56]
            params.patch_probabilities = [0.2, 0.3, 0.5]
            params.patch_batch_multipliers = {16: 4, 32: 2, 56: 1}
            params.pad_width = 24
            params.largest_patch_size = 56
        elif base_mode == "padis-paper-ct-512":
            params.patch_sizes = [16, 32, 64]
            params.patch_probabilities = [0.2, 0.3, 0.5]
            params.patch_batch_multipliers = {16: 4, 32: 2, 64: 1}
            params.pad_width = 64
            params.largest_patch_size = 64
        elif base_mode in ("padis-paper-whole-ct-256", "whole-image-ct-256"):
            params.prior_mode = "whole_image"
            params.patch_sizes = [256]
            params.patch_probabilities = [1.0]
            params.patch_batch_multipliers = {256: 1}
            params.pad_width = 0
            params.largest_patch_size = 256
            params.default_batch_size = 8
        elif base_mode in _PATCH_ABLATION_PRESETS:
            for key, value in _PATCH_ABLATION_PRESETS[base_mode].items():
                if isinstance(value, (dict, list)):
                    value = value.copy()
                setattr(params, key, value)
        else:
            raise ValueError(f"Mode {mode} not recognized.")
        if no_position:
            params.use_position_channels = False
        return params

    def _validate_solver_configuration(self) -> None:
        model_params = getattr(self.model, "model_parameters", None)
        expected_position_channels = (
            2 if bool(self.solver_params.use_position_channels) else 0
        )
        model_position_channels = getattr(
            model_params, "input_position_channels", expected_position_channels
        )
        if int(model_position_channels) != expected_position_channels:
            raise ValueError(
                "Model input_position_channels must match "
                "solver_params.use_position_channels."
            )

        largest_patch_size = int(self.solver_params.largest_patch_size)
        max_training_patch_size = max(
            int(size) for size in self.solver_params.patch_sizes
        )
        if largest_patch_size < max_training_patch_size:
            raise ValueError("largest_patch_size must be at least max(patch_sizes).")
        model_largest_patch_size = int(
            getattr(model_params, "largest_patch_size", largest_patch_size)
        )
        if model_largest_patch_size < max_training_patch_size:
            raise ValueError(
                "Model largest_patch_size must support every solver patch size."
            )

        model_prior_mode = getattr(model_params, "prior_mode", "patch")
        if self.solver_params.prior_mode == "whole_image":
            if int(self.solver_params.pad_width) != 0:
                raise ValueError("Whole-image PaDIS training expects pad_width=0.")
            if len(self.solver_params.patch_sizes) != 1:
                raise ValueError("Whole-image PaDIS training expects one patch size.")
            if int(self.solver_params.patch_sizes[0]) != largest_patch_size:
                raise ValueError(
                    "Whole-image PaDIS training patch size must equal largest_patch_size."
                )
            if model_prior_mode != "whole_image":
                raise ValueError(
                    "Whole-image solver parameters require a whole-image model preset."
                )
        elif self.solver_params.prior_mode == "patch":
            if model_prior_mode == "whole_image":
                raise ValueError(
                    "Patch solver parameters require a patch PaDIS model preset."
                )
        else:
            raise ValueError("prior_mode must be 'patch' or 'whole_image'.")

        microbatch_size = getattr(self.solver_params, "microbatch_size", None)
        if microbatch_size is not None and int(microbatch_size) <= 0:
            raise ValueError("microbatch_size must be positive or None.")

    def _sample_training_patch(
        self, images: torch.Tensor, patch_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.solver_params.prior_mode == "whole_image":
            if int(self.solver_params.pad_width) != 0:
                raise ValueError("Whole-image PaDIS training expects pad_width=0.")
            if self.solver_params.use_position_channels:
                positions = build_position_grid(
                    images.shape[0],
                    images.shape[-2],
                    images.shape[-1],
                    device=images.device,
                    dtype=images.dtype,
                )
                return images, positions
            return images, None

        padded = zero_pad_images(images, int(self.solver_params.pad_width))
        if patch_size is None:
            patch_size = sample_patch_size(
                self.solver_params.patch_sizes,
                self.solver_params.patch_probabilities,
                device=padded.device,
            )
        if self.solver_params.use_position_channels:
            return sample_image_patch_with_position_channels(padded, patch_size)
        return sample_image_patch(padded, patch_size), None

    def mini_batch_step(
        self, sino_batch, target_batch, patch_size: int | None = None
    ) -> torch.Tensor:
        del sino_batch
        clean_images = target_batch.float()
        self._check_data_range(clean_images)
        clean_patch, position_patch = self._sample_training_patch(
            clean_images, patch_size
        )
        non_blocking = self.device.type == "cuda"
        clean_patch = clean_patch.to(self.device, non_blocking=non_blocking)
        if position_patch is not None:
            position_patch = position_patch.to(self.device, non_blocking=non_blocking)
        return self.loss_fn(self.model, clean_patch, position_patch)

    def _check_data_range(self, images: torch.Tensor) -> None:
        if not self.solver_params.enforce_data_range:
            return
        if torch.amin(images) < -1e-5 or torch.amax(images) > 1 + 1e-5:
            raise ValueError(
                "PaDIS prior training expects images scaled to [0, 1]. "
                "Use LIDC task='image_prior' or provide an equivalent transform."
            )

    def _update_ema(self, batch_patch_count: int) -> None:
        if self.ema_state is None:
            return
        half_life = float(self.solver_params.ema_half_life_patches)
        if self.solver_params.ema_rampup_ratio is not None:
            half_life = min(
                half_life,
                float(self.seen_patches) * float(self.solver_params.ema_rampup_ratio),
            )
        beta = 0.5 ** (float(batch_patch_count) / max(half_life, 1e-8))
        with torch.no_grad():
            grouped_ema: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
            grouped_params: dict[
                tuple[torch.device, torch.dtype], list[torch.Tensor]
            ] = {}
            for name, param in self.model.named_parameters():
                if name in self.ema_state:
                    ema = self.ema_state[name]
                    key = (ema.device, ema.dtype)
                    grouped_ema.setdefault(key, []).append(ema)
                    grouped_params.setdefault(key, []).append(param.detach())
            try:
                for key, ema_tensors in grouped_ema.items():
                    torch._foreach_mul_(ema_tensors, beta)
                    torch._foreach_add_(
                        ema_tensors, grouped_params[key], alpha=1.0 - beta
                    )
            except RuntimeError:
                for name, param in self.model.named_parameters():
                    if name in self.ema_state:
                        self.ema_state[name].mul_(beta).add_(
                            param.detach(), alpha=1.0 - beta
                        )

    def _apply_ema_weights(self) -> dict[str, torch.Tensor] | None:
        if self.ema_state is None:
            return None
        raw_state = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.ema_state:
                    raw_state[name] = param.detach().clone()
                    param.copy_(
                        self.ema_state[name].to(param.device, dtype=param.dtype)
                    )
        return raw_state

    def _restore_raw_weights(self, raw_state: dict[str, torch.Tensor] | None) -> None:
        if raw_state is None:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in raw_state:
                    param.copy_(raw_state[name].to(param.device, dtype=param.dtype))

    def _choose_patch_size(self) -> int:
        return sample_patch_size(
            self.solver_params.patch_sizes,
            self.solver_params.patch_probabilities,
            device=self.device,
        )

    def _collect_training_targets(
        self,
        data_iter,
        patch_size: int,
        *,
        restart_on_stop: bool,
    ) -> tuple[torch.Tensor | None, int, object]:
        if self.train_loader is None:
            raise ValueError("Training dataloader not set: Please call set_training")
        if len(self.train_loader) == 0:
            raise ValueError("Training dataloader is empty.")

        batch_mul = int(
            self.solver_params.patch_batch_multipliers.get(int(patch_size), 1)
        )
        base_batch_size = getattr(self.solver_params, "base_patch_batch_size", None)
        if base_batch_size is not None:
            wanted = int(base_batch_size) * batch_mul
            sample_batch = getattr(self.train_loader, "sample_batch", None)
            if sample_batch is not None:
                return sample_batch(wanted), 1, data_iter
            while True:
                try:
                    _, target = next(data_iter)
                except StopIteration:
                    if not restart_on_stop:
                        return None, 0, data_iter
                    data_iter = iter(self.train_loader)
                    continue
                if target.shape[0] >= wanted:
                    return target[:wanted], 1, data_iter
                targets = [target]
                consumed = 1
                while sum(batch.shape[0] for batch in targets) < wanted:
                    try:
                        _, target = next(data_iter)
                    except StopIteration:
                        if not restart_on_stop:
                            break
                        data_iter = iter(self.train_loader)
                        continue
                    targets.append(target)
                    consumed += 1
                target = torch.cat(targets, dim=0)
                return target[:wanted], consumed, data_iter

        targets = []
        consumed = 0
        while consumed < batch_mul:
            try:
                _, target = next(data_iter)
            except StopIteration:
                if not restart_on_stop:
                    break
                data_iter = iter(self.train_loader)
                continue
            targets.append(target)
            consumed += 1
        if not targets:
            return None, consumed, data_iter
        return torch.cat(targets, dim=0), consumed, data_iter

    def _training_microbatches(self, target: torch.Tensor):
        microbatch_size = getattr(self.solver_params, "microbatch_size", None)
        if microbatch_size is None:
            yield target
            return
        microbatch_size = int(microbatch_size)
        if microbatch_size >= int(target.shape[0]):
            yield target
            return
        for start in range(0, int(target.shape[0]), microbatch_size):
            yield target[start : start + microbatch_size]

    def _optimizer_step(self, target: torch.Tensor, patch_size: int) -> float:
        self.optimizer.zero_grad()
        total_images = int(target.shape[0])
        if total_images <= 0:
            raise ValueError("Cannot optimize an empty PaDIS target batch.")
        total_loss = 0.0
        for microbatch in self._training_microbatches(target):
            batch_loss = self.mini_batch_step(None, microbatch, patch_size=patch_size)
            weight = float(microbatch.shape[0]) / float(total_images)
            (batch_loss * weight).backward()
            total_loss += float(batch_loss.item()) * weight
        if self.solver_params.lr_rampup_kimg is not None:
            lr_scale = min(
                float(self.seen_patches)
                / max(float(self.solver_params.lr_rampup_kimg) * 1000, 1e-8),
                1.0,
            )
            for group in self.optimizer.param_groups:
                group.setdefault("base_lr", group["lr"])
                group["lr"] = group["base_lr"] * lr_scale
        for param in self.model.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad
                )
        self.optimizer.step()
        self._update_ema(int(target.shape[0]))
        self.seen_patches += int(target.shape[0])
        return total_loss

    def train_step(self):
        if self.train_loader is None:
            raise ValueError("Training dataloader not set: Please call set_training")
        self.model.train()
        epoch_loss = 0.0
        step_count = 0
        data_iter = iter(self.train_loader)
        progress = tqdm(total=len(self.train_loader))
        while True:
            patch_size = self._choose_patch_size()
            target, consumed, data_iter = self._collect_training_targets(
                data_iter, patch_size, restart_on_stop=False
            )
            if target is None:
                break
            epoch_loss += self._optimizer_step(target, patch_size)
            step_count += 1
            progress.update(consumed)
        progress.close()
        return epoch_loss / max(step_count, 1)

    def _existing_min_validation_loss(self) -> float | None:
        if self.validation_fname is None or self.validation_save_folder is None:
            return None
        validation_path = self.validation_save_folder.joinpath(self.validation_fname)
        if not validation_path.with_suffix(".pt").is_file():
            return None
        data = torch.load(
            validation_path.with_suffix(".pt"),
            map_location=self.device,
            weights_only=False,
        )
        loss = data.get("loss")
        if loss is None:
            return None
        if isinstance(loss, np.ndarray):
            if len(loss) == 0:
                return None
            loss = loss[-1]
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu().item()
        return float(loss)

    def train_for_patches(
        self,
        target_patches: int,
        *,
        validation_interval_patches: int | None = None,
        validation_max_patches: int | None = None,
        checkpoint_interval_patches: int | None = None,
        log_interval_patches: int | None = None,
        max_train_seconds: float | None = None,
        log_fn: Callable[[dict[str, object], int], None] | None = None,
    ) -> None:
        """Train until ``seen_patches`` reaches ``target_patches``.

        This is the PaDIS-native budget: the underlying PaDIS repository uses an
        image/patch counter rather than epochs to drive LR ramp-up, EMA, and run
        duration.
        """
        if target_patches <= 0:
            raise ValueError("target_patches must be positive.")
        if validation_interval_patches is not None and validation_interval_patches <= 0:
            raise ValueError("validation_interval_patches must be positive.")
        if validation_max_patches is not None and validation_max_patches <= 0:
            raise ValueError("validation_max_patches must be positive.")
        if checkpoint_interval_patches is not None and checkpoint_interval_patches <= 0:
            raise ValueError("checkpoint_interval_patches must be positive.")
        if log_interval_patches is not None and log_interval_patches <= 0:
            raise ValueError("log_interval_patches must be positive.")
        if max_train_seconds is not None and max_train_seconds <= 0:
            raise ValueError("max_train_seconds must be positive.")
        self.check_training_ready(verbose=False)
        if self.do_load_checkpoint:
            print("Loading checkpoint...")
            self.current_epoch = self.load_checkpoint()
        if self.seen_patches >= target_patches:
            return

        has_validation = self.check_validation_ready(verbose=False) == 0
        if has_validation:
            existing_min_validation_loss = self._existing_min_validation_loss()
            if existing_min_validation_loss is None:
                self.validation_loss = np.zeros(0)
            else:
                self.validation_loss = np.array([existing_min_validation_loss])
                if self.verbose:
                    print(
                        "Loaded existing minimum validation loss: "
                        f"{existing_min_validation_loss}"
                    )
        else:
            self.validation_loss = None
        self.train_loss = []
        self.model.train()

        data_iter = iter(self.train_loader)
        next_validation = (
            self.seen_patches + int(validation_interval_patches)
            if validation_interval_patches is not None
            else None
        )
        next_checkpoint = (
            self.seen_patches + int(checkpoint_interval_patches)
            if checkpoint_interval_patches is not None
            else None
        )
        next_log = (
            self.seen_patches + int(log_interval_patches)
            if log_interval_patches is not None
            else None
        )
        timing_acc = {
            "data_wait_s": 0.0,
            "train_step_s": 0.0,
            "total_step_s": 0.0,
            "patches": 0,
            "steps": 0,
        }
        train_start_wall = time.monotonic()
        checkpoint_index = int(self.current_epoch)
        progress = tqdm(
            total=target_patches, initial=min(self.seen_patches, target_patches)
        )
        try:
            while self.seen_patches < target_patches:
                previous_seen = self.seen_patches
                patch_size = self._choose_patch_size()
                data_start = time.perf_counter()
                target, _, data_iter = self._collect_training_targets(
                    data_iter, patch_size, restart_on_stop=True
                )
                data_wait_s = time.perf_counter() - data_start
                if target is None:
                    raise ValueError("Training dataloader produced no targets.")
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                train_start = time.perf_counter()
                loss_value = self._optimizer_step(target, patch_size)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                train_step_s = time.perf_counter() - train_start
                step_patches = self.seen_patches - previous_seen
                total_step_s = data_wait_s + train_step_s
                timing_acc["data_wait_s"] += data_wait_s
                timing_acc["train_step_s"] += train_step_s
                timing_acc["total_step_s"] += total_step_s
                timing_acc["patches"] += step_patches
                timing_acc["steps"] += 1
                self.train_loss.append(loss_value)
                progress.update(min(self.seen_patches, target_patches) - previous_seen)
                should_log = False
                if log_fn is not None:
                    if next_log is None:
                        should_log = True
                    elif next_log is not None and self.seen_patches >= next_log:
                        should_log = True
                if should_log:
                    total_s = max(float(timing_acc["total_step_s"]), 1e-12)
                    steps = max(int(timing_acc["steps"]), 1)
                    patches = max(int(timing_acc["patches"]), 1)
                    log_fn(
                        {
                            "train/loss": loss_value,
                            "train/patch_size": patch_size,
                            "train/prior_mode": self.solver_params.prior_mode,
                            "train/seen_patches": self.seen_patches,
                            "train/step": len(self.train_loss),
                            "optimizer/lr": self.optimizer.param_groups[0]["lr"],
                            "timing/data_wait_s_per_step": timing_acc["data_wait_s"]
                            / steps,
                            "timing/train_step_s_per_step": timing_acc["train_step_s"]
                            / steps,
                            "timing/total_s_per_step": timing_acc["total_step_s"]
                            / steps,
                            "timing/data_wait_fraction": timing_acc["data_wait_s"]
                            / total_s,
                            "timing/train_step_fraction": timing_acc["train_step_s"]
                            / total_s,
                            "timing/patches_per_second": patches / total_s,
                            "timing/steps_per_second": steps / total_s,
                        },
                        self.seen_patches,
                    )
                    if self.verbose:
                        print(
                            f"Patches {self.seen_patches} - loss {loss_value:.4g} - "
                            f"data wait {timing_acc['data_wait_s'] / steps:.3f}s/step - "
                            f"train {timing_acc['train_step_s'] / steps:.3f}s/step - "
                            f"{patches / total_s:.1f} patches/s"
                        )
                    timing_acc = {
                        "data_wait_s": 0.0,
                        "train_step_s": 0.0,
                        "total_step_s": 0.0,
                        "patches": 0,
                        "steps": 0,
                    }
                    if next_log is not None:
                        next_log += int(log_interval_patches)

                if next_validation is not None and self.seen_patches >= next_validation:
                    validation_loss = (
                        self.validate(max_patches=validation_max_patches)
                        if validation_max_patches is not None
                        else self.validate()
                    )
                    self.validation_loss = np.append(
                        self.validation_loss, validation_loss
                    )
                    if self.verbose:
                        print(
                            f"Patches {self.seen_patches} - Training loss: {loss_value} "
                            f"- Validation loss: {validation_loss}"
                        )
                    if self.validation_fname is not None and validation_loss <= np.min(
                        self.validation_loss
                    ):
                        self.save_validation(len(self.validation_loss) - 1)
                    if log_fn is not None:
                        validation_metrics = {
                            "validation/loss": validation_loss,
                            "validation/index": len(self.validation_loss),
                            "validation/seen_patches": self.seen_patches,
                            "validation/evaluated_patches": self.last_validation_patches,
                        }
                        if validation_max_patches is not None:
                            validation_metrics[
                                "validation/max_patches"
                            ] = validation_max_patches
                        log_fn(validation_metrics, self.seen_patches)
                    next_validation += int(validation_interval_patches)

                if next_checkpoint is not None and self.seen_patches >= next_checkpoint:
                    checkpoint_index += 1
                    if self.checkpoint_save_folder is not None:
                        self.save_checkpoint(checkpoint_index - 1)
                    next_checkpoint += int(checkpoint_interval_patches)

                if (
                    max_train_seconds is not None
                    and time.monotonic() - train_start_wall >= max_train_seconds
                ):
                    if self.verbose:
                        print(
                            "Reached max_train_seconds "
                            f"({max_train_seconds:g}); stopping training cleanly."
                        )
                    break
        finally:
            progress.close()

    def validate(self, max_patches: int | None = None):
        if max_patches is not None and max_patches <= 0:
            raise ValueError("max_patches must be positive.")
        if self.validation_loader is None:
            return 0.0
        was_training = self.model.training
        self.model.eval()
        raw_state = self._apply_ema_weights()
        validation_loss_total = 0.0
        validation_patches = 0
        try:
            with torch.no_grad():
                for _, target in tqdm(self.validation_loader):
                    if max_patches is not None:
                        remaining = int(max_patches) - validation_patches
                        if remaining <= 0:
                            break
                        if target.shape[0] > remaining:
                            target = target[:remaining]
                    target = target.float()
                    self._check_data_range(target)
                    clean_patch, position_patch = self._sample_training_patch(target)
                    non_blocking = self.device.type == "cuda"
                    clean_patch = clean_patch.to(self.device, non_blocking=non_blocking)
                    if position_patch is not None:
                        position_patch = position_patch.to(
                            self.device, non_blocking=non_blocking
                        )
                    loss = self.loss_fn(self.model, clean_patch, position_patch)
                    batch_patches = int(clean_patch.shape[0])
                    validation_loss_total += float(loss.cpu().item()) * batch_patches
                    validation_patches += batch_patches
        finally:
            self.last_validation_patches = validation_patches
            self._restore_raw_weights(raw_state)
            if was_training:
                self.model.train()
        if validation_patches == 0:
            return 0.0
        return validation_loss_total / validation_patches

    def set_checkpoint_retention(self, max_periodic_checkpoints: int | None) -> None:
        if max_periodic_checkpoints is not None and max_periodic_checkpoints <= 0:
            raise ValueError("max_periodic_checkpoints must be positive or None.")
        self.max_periodic_checkpoints = max_periodic_checkpoints

    def _periodic_checkpoint_sidecars(
        self, checkpoint_path: pathlib.Path
    ) -> list[pathlib.Path]:
        return [
            checkpoint_path.with_suffix(".pt"),
            checkpoint_path.with_suffix(".json"),
            checkpoint_path.with_suffix(".ema.pt"),
        ]

    def prune_periodic_checkpoints(
        self, max_periodic_checkpoints: int | None = None
    ) -> None:
        if max_periodic_checkpoints is None:
            return
        if max_periodic_checkpoints <= 0:
            raise ValueError("max_periodic_checkpoints must be positive or None.")
        if self.checkpoint_save_folder is None or self.checkpoint_fname is None:
            return
        checkpoints = sorted(
            path
            for path in self.checkpoint_save_folder.glob(self.checkpoint_fname)
            if not path.name.endswith(".ema.pt")
        )
        stale_checkpoints = checkpoints[:-max_periodic_checkpoints]
        for checkpoint in stale_checkpoints:
            for path in self._periodic_checkpoint_sidecars(checkpoint):
                if path.exists():
                    path.unlink()

    def save_checkpoint(self, epoch):
        super().save_checkpoint(epoch)
        if self.ema_state is not None and self.checkpoint_save_folder is not None:
            ema_fname = pathlib.Path(
                str(self.checkpoint_fname).replace("*", f"{epoch+1:04d}")
            )
            torch.save(
                {"ema_state_dict": self.ema_state, "seen_patches": self.seen_patches},
                self.checkpoint_save_folder.joinpath(ema_fname).with_suffix(".ema.pt"),
            )
        self.prune_periodic_checkpoints(self.max_periodic_checkpoints)

    @staticmethod
    def _full_state_base_path(path: pathlib.Path) -> pathlib.Path:
        return path.with_name(f"{path.stem}_full")

    def _save_full_training_state(
        self,
        path: pathlib.Path,
        *,
        epoch: int | None = None,
        kind: str,
        validation_loss: float | None = None,
    ) -> None:
        if epoch is None:
            epoch = self.current_epoch
        full_base_path = self._full_state_base_path(path)
        self.model.save_checkpoint(
            full_base_path,
            epoch,
            self.train_loss,
            self.optimizer,
            self.metadata,
            dataset=self.dataset_param,
        )
        full_pt_path = full_base_path.with_suffix(".pt")
        data = torch.load(full_pt_path, map_location=self.device, weights_only=False)
        data["full_save_kind"] = kind
        data["seen_patches"] = self.seen_patches
        data["training_steps"] = len(self.train_loss)
        if validation_loss is not None:
            data["validation_loss"] = float(validation_loss)
        if self.ema_state is not None:
            data["ema_state_dict"] = {
                name: tensor.detach().cpu() for name, tensor in self.ema_state.items()
            }
        torch.save(data, full_pt_path)

    def save_validation(self, epoch):
        raw_state = self._apply_ema_weights()
        try:
            super().save_validation(epoch)
        finally:
            self._restore_raw_weights(raw_state)
        self._save_full_training_state(
            self.validation_save_folder.joinpath(self.validation_fname),
            kind="validation",
            validation_loss=float(self.validation_loss[epoch]),
        )

    def save_final_results(self, final_result_fname=None, save_folder=None, epoch=None):
        raw_state = self._apply_ema_weights()
        try:
            super().save_final_results(final_result_fname, save_folder, epoch)
        finally:
            self._restore_raw_weights(raw_state)
        self._save_full_training_state(
            self.save_folder.joinpath(self.final_result_fname),
            epoch=epoch,
            kind="final",
        )

    def _load_ema_sidecar(self) -> None:
        if self.checkpoint_save_folder is None or self.checkpoint_fname is None:
            return
        ema_pattern = self.checkpoint_fname.replace(".pt", ".ema.pt")
        ema_files = sorted(self.checkpoint_save_folder.glob(ema_pattern))
        if ema_files:
            data = torch.load(
                ema_files[-1], map_location=self.device, weights_only=False
            )
            self.ema_state = {
                name: tensor.to(self.device)
                for name, tensor in data["ema_state_dict"].items()
            }
            self.seen_patches = int(data.get("seen_patches", 0))

    def _load_periodic_checkpoint(self, checkpoint_path: pathlib.Path) -> int:
        data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(data["model_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.current_epoch = int(data.get("epoch", 0))
        self.train_loss = data.get("loss", self.train_loss)
        self.model.train()
        if self.verbose:
            print(f"Loaded PaDIS checkpoint from {checkpoint_path}")
        return self.current_epoch

    def _load_full_training_state(self, path: pathlib.Path) -> int | None:
        full_path = self._full_state_base_path(path).with_suffix(".pt")
        if not full_path.is_file():
            return None
        data = torch.load(full_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(data["model_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.current_epoch = int(data.get("epoch", 0))
        self.train_loss = data.get("loss", self.train_loss)
        if "ema_state_dict" in data:
            self.ema_state = {
                name: tensor.to(self.device)
                for name, tensor in data["ema_state_dict"].items()
            }
        self.seen_patches = int(data.get("seen_patches", self.seen_patches))
        self.model.train()
        if self.verbose:
            print(
                f"Loaded PaDIS full {data.get('full_save_kind', 'training')} state "
                f"from {full_path}"
            )
        return self.current_epoch

    def _full_training_state_fallbacks(self) -> list[pathlib.Path]:
        paths = []
        if self.save_folder is not None and self.final_result_fname is not None:
            paths.append(self.save_folder.joinpath(self.final_result_fname))
        validation_save_folder = getattr(self, "validation_save_folder", None)
        if validation_save_folder is not None and self.validation_fname is not None:
            paths.append(validation_save_folder.joinpath(self.validation_fname))
        return paths

    def load_checkpoint(self):
        if self.checkpoint_save_folder is None or self.checkpoint_fname is None:
            return self.current_epoch
        checkpoints = sorted(
            path
            for path in self.checkpoint_save_folder.glob(self.checkpoint_fname)
            if not path.name.endswith(".ema.pt")
        )
        if checkpoints:
            epoch = self._load_periodic_checkpoint(checkpoints[-1])
            self._load_ema_sidecar()
            return epoch

        for path in self._full_training_state_fallbacks():
            epoch = self._load_full_training_state(path)
            if epoch is not None:
                return epoch

        print(
            f"checkpoint {self.checkpoint_save_folder.joinpath(self.checkpoint_fname)} "
            "not found, failed to load."
        )
        epoch = self.current_epoch
        return epoch
