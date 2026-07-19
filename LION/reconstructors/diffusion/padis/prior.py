"""Patch layout and prior-assembly internals for the PaDIS reconstructor."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class PatchLayout:
    """Describe a set of image patches and its unpadded image extent."""

    indices: list[tuple[int, int, int, int]]
    image_height: int
    image_width: int


class PaDISPrior:
    """Assemble position-aware patch denoising into a whole-image prior."""

    def patch_layout(
        self,
        image_shape: tuple[int, int],
        params,
        device: torch.device,
        generator: torch.Generator | None = None,
        *,
        fixed_offset: bool = False,
    ) -> PatchLayout:
        """Build the randomly offset PaDIS patch partition for one image."""
        height, width = image_shape
        patch_size = int(params.patch_size)
        pad = int(params.pad_width)
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if pad < 0:
            raise ValueError("pad_width must be non-negative.")
        n_rows = height // patch_size + 1
        n_cols = width // patch_size + 1
        offset_rng = getattr(params, "patch_offset_rng", "torch")
        row_offset = self._random_offset(
            pad, device, generator, fixed_offset, rng_source=offset_rng
        )
        col_offset = self._random_offset(
            pad, device, generator, fixed_offset, rng_source=offset_rng
        )
        row_spaced = torch.arange(n_rows, device=device, dtype=torch.int64) * patch_size
        col_spaced = torch.arange(n_cols, device=device, dtype=torch.int64) * patch_size
        indices = []
        padded_height = height + 2 * pad
        padded_width = width + 2 * pad
        for row_start in row_spaced.tolist():
            for col_start in col_spaced.tolist():
                top = int(row_start) + row_offset
                left = int(col_start) + col_offset
                bottom = top + patch_size
                right = left + patch_size
                if bottom <= padded_height and right <= padded_width:
                    indices.append((top, bottom, left, right))
        if not indices:
            raise ValueError("Patch layout produced no valid patches.")
        return PatchLayout(indices, height, width)

    @staticmethod
    def _random_offset(
        pad: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        fixed_offset: bool = False,
        rng_source: str = "torch",
    ) -> int:
        if pad <= 0 or fixed_offset:
            return 0
        if rng_source == "python":
            return random.randint(0, pad - 1)
        return int(
            torch.randint(0, pad, (1,), device=device, generator=generator).item()
        )

    def _position_grid(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        y = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
        x_coord = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(y, x_coord, indexing="ij")
        grid = torch.stack((xx, yy), dim=0).unsqueeze(0)
        return grid.expand(batch_size, -1, -1, -1)

    def denoise_patches(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        layout: PatchLayout,
        params,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Denoise and reassemble one randomly offset PaDIS patch layout."""
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError(
                "PaDIS patch denoising expects a single padded image batch."
            )

        positions = (
            self._position_grid(x)
            if int(getattr(self.model.model_parameters, "input_position_channels", 2))
            > 0
            else None
        )
        patch_batch_size = getattr(params, "patch_batch_size", None)
        if patch_batch_size is None:
            patch_batch_size = len(layout.indices)
        patch_batch_size = int(patch_batch_size)
        if patch_batch_size <= 0:
            raise ValueError("patch_batch_size must be positive or None.")
        use_checkpoint = self._use_patch_checkpoint_denoiser(params)

        output = torch.zeros_like(x)
        for chunk_start in range(0, len(layout.indices), patch_batch_size):
            chunk_indices = layout.indices[chunk_start : chunk_start + patch_batch_size]
            image_batch = torch.cat(
                [
                    x[:, :, top:bottom, left:right]
                    for top, bottom, left, right in chunk_indices
                ],
                dim=0,
            )
            position_batch = None
            if positions is not None:
                position_batch = torch.cat(
                    [
                        positions[:, :, top:bottom, left:right]
                        for top, bottom, left, right in chunk_indices
                    ],
                    dim=0,
                )
            denoised_batch = self.edm_denoise_batch(
                image_batch,
                position_batch,
                sigma,
                params,
                use_checkpoint=use_checkpoint,
            )
            for offset, (top, bottom, left, right) in enumerate(chunk_indices):
                output[:, :, top:bottom, left:right] += denoised_batch[
                    offset : offset + 1
                ]
                output[:, :, top:bottom, left:right] -= x[:, :, top:bottom, left:right]
        denoised = x + output
        if bool(getattr(params, "consume_denoise_output_noise", False)):
            _ = self._sample_noise(denoised, generator)
        pad = int(params.pad_width)
        if pad == 0:
            return denoised
        zero_border = torch.zeros_like(denoised)
        zero_border[
            :, :, pad : pad + layout.image_height, pad : pad + layout.image_width
        ] = denoised[
            :, :, pad : pad + layout.image_height, pad : pad + layout.image_width
        ]
        return zero_border

    @staticmethod
    def _fixed_patch_starts(
        padded_length: int,
        *,
        pad: int,
        patch_size: int,
        overlap: int,
        layout: str = "lion_clipped",
    ) -> list[int]:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if overlap < 0 or overlap >= patch_size:
            raise ValueError("patch_overlap must satisfy 0 <= overlap < patch_size.")
        if patch_size > padded_length:
            raise ValueError("patch_size cannot exceed padded image dimensions.")
        stride = patch_size - overlap
        last_valid_start = padded_length - patch_size
        if layout == "lion_clipped":
            starts = [pad]
            while starts[-1] < last_valid_start:
                starts.append(starts[-1] + stride)
            starts[-1] = min(starts[-1], last_valid_start)
        elif layout in ("public_overlap", "public_tile"):
            start = pad if layout == "public_overlap" else 4
            if start < 0 or start + patch_size > padded_length:
                raise ValueError(
                    f"fixed_overlap_layout={layout!r} cannot place its first "
                    "patch inside the padded image."
                )
            starts = [start]
            public_stop = padded_length - pad - patch_size
            while starts[-1] < public_stop:
                next_start = starts[-1] + stride
                if next_start + patch_size > padded_length:
                    next_start = last_valid_start
                if next_start <= starts[-1]:
                    break
                starts.append(next_start)
        else:
            raise ValueError(
                "fixed_overlap_layout must be 'lion_clipped', "
                "'public_overlap', or 'public_tile'."
            )
        return sorted(set(int(start) for start in starts))

    def fixed_overlap_patch_layout(
        self,
        image_shape: tuple[int, int],
        params,
    ) -> PatchLayout:
        """Build the deterministic fixed-overlap comparison layout."""
        height, width = image_shape
        pad = int(params.pad_width)
        patch_size = int(params.patch_size)
        overlap = int(getattr(params, "patch_overlap", 8))
        fixed_overlap_layout = getattr(params, "fixed_overlap_layout", "lion_clipped")
        padded_height = height + 2 * pad
        padded_width = width + 2 * pad
        row_starts = self._fixed_patch_starts(
            padded_height,
            pad=pad,
            patch_size=patch_size,
            overlap=overlap,
            layout=fixed_overlap_layout,
        )
        col_starts = self._fixed_patch_starts(
            padded_width,
            pad=pad,
            patch_size=patch_size,
            overlap=overlap,
            layout=fixed_overlap_layout,
        )
        indices = [
            (top, top + patch_size, left, left + patch_size)
            for top in row_starts
            for left in col_starts
        ]
        if not indices:
            raise ValueError("Fixed-overlap patch layout produced no valid patches.")
        return PatchLayout(indices, height, width)

    @staticmethod
    def _use_patch_checkpoint_denoiser(params) -> bool:
        return bool(
            getattr(params, "patch_checkpoint_denoiser", False)
            or getattr(params, "fixed_overlap_checkpoint_denoiser", False)
        )

    def denoise_fixed_overlap_patches(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        layout: PatchLayout,
        params,
        *,
        assembly: Literal["fixed_average", "fixed_stitch"],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Denoise patches and combine them by averaging or stitching."""
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError(
                "Fixed-overlap patch denoising expects a single padded image batch."
            )

        positions = (
            self._position_grid(x)
            if int(getattr(self.model.model_parameters, "input_position_channels", 2))
            > 0
            else None
        )
        patch_batch_size = getattr(params, "patch_batch_size", None)
        if patch_batch_size is None:
            patch_batch_size = len(layout.indices)
        patch_batch_size = int(patch_batch_size)
        if patch_batch_size <= 0:
            raise ValueError("patch_batch_size must be positive or None.")
        use_checkpoint = self._use_patch_checkpoint_denoiser(params)

        output = torch.zeros_like(x)
        if assembly == "fixed_average":
            counts = torch.zeros_like(x)
        else:
            counts = None
        for chunk_start in range(0, len(layout.indices), patch_batch_size):
            chunk_indices = layout.indices[chunk_start : chunk_start + patch_batch_size]
            image_batch = torch.cat(
                [
                    x[:, :, top:bottom, left:right]
                    for top, bottom, left, right in chunk_indices
                ],
                dim=0,
            )
            position_batch = None
            if positions is not None:
                position_batch = torch.cat(
                    [
                        positions[:, :, top:bottom, left:right]
                        for top, bottom, left, right in chunk_indices
                    ],
                    dim=0,
                )
            denoised_batch = self.edm_denoise_batch(
                image_batch,
                position_batch,
                sigma,
                params,
                use_checkpoint=use_checkpoint,
            )

            for offset, (top, bottom, left, right) in enumerate(chunk_indices):
                patch = denoised_batch[offset : offset + 1]
                if assembly == "fixed_average":
                    output[:, :, top:bottom, left:right] += patch
                    counts[:, :, top:bottom, left:right] += 1
                elif assembly == "fixed_stitch":
                    output[:, :, top:bottom, left:right] = patch
                else:
                    raise ValueError(
                        "patch_assembly must be 'padis', 'fixed_average', "
                        "or 'fixed_stitch'."
                    )

        if assembly == "fixed_average":
            output = torch.where(counts > 0, output / counts.clamp_min(1), x)

        if bool(getattr(params, "consume_denoise_output_noise", False)):
            _ = self._sample_noise(output, generator)
        pad = int(params.pad_width)
        if pad == 0:
            return output
        zero_border = torch.zeros_like(output)
        zero_border[
            :, :, pad : pad + layout.image_height, pad : pad + layout.image_width
        ] = output[
            :, :, pad : pad + layout.image_height, pad : pad + layout.image_width
        ]
        return zero_border

    def denoise_whole_image(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        params,
    ) -> torch.Tensor:
        """Denoise a complete image with a whole-image prior."""
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError("Whole-image denoising expects a single image batch.")
        if int(params.pad_width) != 0:
            raise ValueError(
                "Whole-image diffusion reconstruction expects pad_width=0."
            )
        positions = (
            self._position_grid(x)
            if int(getattr(self.model.model_parameters, "input_position_channels", 2))
            > 0
            else None
        )
        return self.edm_denoise_batch(x, positions, sigma, params)

    def _validate_prior_configuration(
        self, params, image_shape: tuple[int, int]
    ) -> None:
        height, width = image_shape
        prior_mode = getattr(params, "prior_mode", "patch")
        patch_size = int(params.patch_size)
        pad_width = int(params.pad_width)
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if pad_width < 0:
            raise ValueError("pad_width must be non-negative.")
        if prior_mode == "whole_image":
            if pad_width != 0:
                raise ValueError(
                    "Whole-image diffusion reconstruction expects pad_width=0."
                )
            if patch_size != height or patch_size != width:
                raise ValueError(
                    "Whole-image diffusion reconstruction expects patch_size to "
                    "match the image height and width."
                )
            model_prior_mode = getattr(
                getattr(self.model, "model_parameters", None),
                "prior_mode",
                "patch",
            )
            if model_prior_mode != "whole_image":
                raise ValueError(
                    "Whole-image reconstruction requires a whole-image model preset."
                )
        elif prior_mode == "patch":
            if (
                patch_size > height + 2 * pad_width
                or patch_size > width + 2 * pad_width
            ):
                raise ValueError("patch_size cannot exceed padded image dimensions.")
            patch_assembly = getattr(params, "patch_assembly", "padis")
            if patch_assembly not in ("padis", "fixed_average", "fixed_stitch"):
                raise ValueError(
                    "patch_assembly must be 'padis', 'fixed_average', or 'fixed_stitch'."
                )
            if patch_assembly in ("fixed_average", "fixed_stitch"):
                patch_overlap = int(getattr(params, "patch_overlap", 8))
                if patch_overlap < 0 or patch_overlap >= patch_size:
                    raise ValueError(
                        "patch_overlap must satisfy 0 <= patch_overlap < patch_size."
                    )
                fixed_overlap_layout = getattr(
                    params, "fixed_overlap_layout", "lion_clipped"
                )
                if fixed_overlap_layout not in (
                    "lion_clipped",
                    "public_overlap",
                    "public_tile",
                ):
                    raise ValueError(
                        "fixed_overlap_layout must be 'lion_clipped', "
                        "'public_overlap', or 'public_tile'."
                    )
        else:
            raise ValueError("prior_mode must be 'patch' or 'whole_image'.")

    def denoise_prior(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        params,
        image_shape: tuple[int, int],
        generator: torch.Generator | None = None,
        *,
        layout_override: PatchLayout | None = None,
    ) -> torch.Tensor:
        """Dispatch denoising to the configured patch or whole-image prior."""
        self._validate_prior_configuration(params, image_shape)
        prior_mode = getattr(params, "prior_mode", "patch")
        if prior_mode == "whole_image":
            if layout_override is not None:
                raise ValueError("layout_override is only valid for patch priors.")
            return self.denoise_whole_image(x, sigma, params)
        if prior_mode != "patch":
            raise ValueError("prior_mode must be 'patch' or 'whole_image'.")
        patch_assembly = getattr(params, "patch_assembly", "padis")
        if patch_assembly in ("fixed_average", "fixed_stitch"):
            if layout_override is not None:
                raise ValueError(
                    "layout_override is only valid for padis patch assembly."
                )
            layout = self.fixed_overlap_patch_layout(image_shape, params)
            return self.denoise_fixed_overlap_patches(
                x,
                sigma,
                layout,
                params,
                assembly=patch_assembly,
                generator=generator,
            )
        layout = (
            layout_override
            if layout_override is not None
            else self.patch_layout(image_shape, params, x.device, generator)
        )
        return self.denoise_patches(x, sigma, layout, params, generator)
