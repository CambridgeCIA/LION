"""LION-native solver for PaDIS patch-prior training."""

from __future__ import annotations

from typing import Callable, Optional
import pathlib

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
    sample_patch_pair,
    sample_patch_size,
    validate_patch_schedule,
    zero_pad_images,
)


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
        validate_patch_schedule(
            self.solver_params.patch_sizes, self.solver_params.patch_probabilities
        )
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
        self.metadata = LIONParameter()
        self.metadata.method = "PaDIS paper patch denoising"
        self.metadata.paper_preset = self.solver_params.paper_preset
        self.metadata.patch_sizes = self.solver_params.patch_sizes
        self.metadata.patch_probabilities = self.solver_params.patch_probabilities
        self.metadata.pad_width = self.solver_params.pad_width
        self.metadata.sigma_min = self.solver_params.sigma_min
        self.metadata.sigma_max = self.solver_params.sigma_max
        self.metadata.ema_half_life_patches = self.solver_params.ema_half_life_patches
        self.metadata.ema_rampup_ratio = self.solver_params.ema_rampup_ratio
        self.metadata.lr_rampup_kimg = self.solver_params.lr_rampup_kimg
        self.metadata.lidc_image_scaling = getattr(self.geometry, "image_scaling", None)
        self.metadata.lidc_resize_policy = (
            "CPU resize and patch extraction before final device transfer"
        )

    @staticmethod
    def default_parameters(mode: str = "padis-paper-ct-256") -> SolverParams:
        params = SolverParams()
        params.paper_preset = mode
        params.sigma_min = 0.002
        params.sigma_max = 40.0
        params.use_position_channels = True
        params.use_ema = True
        params.ema_half_life_patches = 500_000
        params.ema_rampup_ratio = 0.05
        params.lr_rampup_kimg = 10_000
        params.enforce_data_range = True
        params.input_mode = "image"
        if mode == "padis-paper-ct-256":
            params.patch_sizes = [16, 32, 56]
            params.patch_probabilities = [0.2, 0.3, 0.5]
            params.patch_batch_multipliers = {16: 4, 32: 2, 56: 1}
            params.pad_width = 24
            params.largest_patch_size = 56
        elif mode == "padis-paper-ct-512":
            params.patch_sizes = [16, 32, 64]
            params.patch_probabilities = [0.2, 0.3, 0.5]
            params.patch_batch_multipliers = {16: 4, 32: 2, 64: 1}
            params.pad_width = 64
            params.largest_patch_size = 64
        else:
            raise ValueError(f"Mode {mode} not recognized.")
        return params

    def _sample_training_patch(
        self, images: torch.Tensor, patch_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        padded = zero_pad_images(images, int(self.solver_params.pad_width))
        positions = None
        if self.solver_params.use_position_channels:
            positions = build_position_grid(
                padded.shape[0],
                padded.shape[2],
                padded.shape[3],
                device=padded.device,
                dtype=padded.dtype,
            )
        if patch_size is None:
            patch_size = sample_patch_size(
                self.solver_params.patch_sizes,
                self.solver_params.patch_probabilities,
                device=padded.device,
            )
        if positions is None:
            _, _, height, width = padded.shape
            top = torch.randint(
                0, height - patch_size + 1, (1,), device=padded.device
            ).item()
            left = torch.randint(
                0, width - patch_size + 1, (1,), device=padded.device
            ).item()
            return padded[:, :, top : top + patch_size, left : left + patch_size], None
        return sample_patch_pair(padded, positions, patch_size)

    def mini_batch_step(
        self, sino_batch, target_batch, patch_size: int | None = None
    ) -> torch.Tensor:
        del sino_batch
        clean_images = target_batch.float()
        self._check_data_range(clean_images)
        clean_patch, position_patch = self._sample_training_patch(
            clean_images, patch_size
        )
        clean_patch = clean_patch.to(self.device)
        if position_patch is not None:
            position_patch = position_patch.to(self.device)
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

    def _optimizer_step(self, target: torch.Tensor, patch_size: int) -> float:
        self.optimizer.zero_grad()
        batch_loss = self.mini_batch_step(None, target, patch_size=patch_size)
        batch_loss.backward()
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
        return float(batch_loss.item())

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

    def train_for_patches(
        self,
        target_patches: int,
        *,
        validation_interval_patches: int | None = None,
        checkpoint_interval_patches: int | None = None,
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
        if checkpoint_interval_patches is not None and checkpoint_interval_patches <= 0:
            raise ValueError("checkpoint_interval_patches must be positive.")
        self.check_training_ready(verbose=False)
        if self.do_load_checkpoint:
            print("Loading checkpoint...")
            self.current_epoch = self.load_checkpoint()
        if self.seen_patches >= target_patches:
            return

        has_validation = self.check_validation_ready(verbose=False) == 0
        if has_validation:
            self.validation_loss = np.zeros(0)
        else:
            self.validation_loss = None
        self.train_loss = np.zeros(0)
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
        checkpoint_index = int(self.current_epoch)
        progress = tqdm(
            total=target_patches, initial=min(self.seen_patches, target_patches)
        )
        try:
            while self.seen_patches < target_patches:
                previous_seen = self.seen_patches
                patch_size = self._choose_patch_size()
                target, _, data_iter = self._collect_training_targets(
                    data_iter, patch_size, restart_on_stop=True
                )
                if target is None:
                    raise ValueError("Training dataloader produced no targets.")
                loss_value = self._optimizer_step(target, patch_size)
                self.train_loss = np.append(self.train_loss, loss_value)
                progress.update(min(self.seen_patches, target_patches) - previous_seen)

                if next_validation is not None and self.seen_patches >= next_validation:
                    validation_loss = self.validate()
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
                    next_validation += int(validation_interval_patches)

                if next_checkpoint is not None and self.seen_patches >= next_checkpoint:
                    checkpoint_index += 1
                    if self.checkpoint_save_folder is not None:
                        self.save_checkpoint(checkpoint_index - 1)
                    next_checkpoint += int(checkpoint_interval_patches)
        finally:
            progress.close()

    def validate(self):
        if self.validation_loader is None:
            return 0.0
        was_training = self.model.training
        self.model.eval()
        raw_state = self._apply_ema_weights()
        validation_loss = np.array([])
        try:
            with torch.no_grad():
                for _, target in tqdm(self.validation_loader):
                    target = target.float()
                    self._check_data_range(target)
                    clean_patch, position_patch = self._sample_training_patch(target)
                    clean_patch = clean_patch.to(self.device)
                    if position_patch is not None:
                        position_patch = position_patch.to(self.device)
                    loss = self.loss_fn(self.model, clean_patch, position_patch)
                    validation_loss = np.append(validation_loss, loss.cpu().item())
        finally:
            self._restore_raw_weights(raw_state)
            if was_training:
                self.model.train()
        return float(np.mean(validation_loss))

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

    def save_validation(self, epoch):
        raw_state = self._apply_ema_weights()
        try:
            super().save_validation(epoch)
        finally:
            self._restore_raw_weights(raw_state)

    def save_final_results(self, final_result_fname=None, save_folder=None, epoch=None):
        raw_state = self._apply_ema_weights()
        try:
            super().save_final_results(final_result_fname, save_folder, epoch)
        finally:
            self._restore_raw_weights(raw_state)

    def load_checkpoint(self):
        epoch = super().load_checkpoint()
        if self.checkpoint_save_folder is None or self.checkpoint_fname is None:
            return epoch
        ema_pattern = self.checkpoint_fname.replace(".pt", ".ema.pt")
        ema_files = sorted(self.checkpoint_save_folder.glob(ema_pattern))
        if ema_files:
            data = torch.load(ema_files[-1], map_location=self.device)
            self.ema_state = {
                name: tensor.to(self.device)
                for name, tensor in data["ema_state_dict"].items()
            }
            self.seen_patches = int(data.get("seen_patches", 0))
        return epoch
