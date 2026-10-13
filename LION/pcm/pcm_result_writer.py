from __future__ import annotations

import numpy as np
import torch

from LION.config.LogConfig import (
    ResultOutputConfig,
    SAPMRenderConfig,
    ScalarImageRenderSpec,
    ScalarMapRenderConfig,
)
from LION.pcm.pcm_visualize import absolute_error
from LION.train.training_components import ModelInput
from LION.utils.artifact_contract import ResultPaths, write_result_manifest
from LION.visualize.plotting_functions import (
    make_colormap,
    save_colorbar,
    save_image_formats,
)
from LION.visualize.scalar_map_rendering import log10_positive, save_histogram


class PCMResultRenderer:
    """Render 2-D PCM reconstruction, model-input, and parameter-map results."""

    def __init__(self, config: ResultOutputConfig) -> None:
        self.config = config

    @staticmethod
    def _to_2d(tensor: torch.Tensor, *, name: str) -> np.ndarray:
        tensor = tensor.detach().cpu().squeeze()
        if tensor.ndim != 2:
            raise ValueError(
                f"Expected {name} to reduce to a 2-D image, got {tuple(tensor.shape)}."
            )
        return np.asarray(tensor, dtype=np.float64)

    def render_scalar_image(
        self,
        *,
        values: torch.Tensor | np.ndarray,
        out_dir,
        filename_stem: str,
        label: str,
        spec: ScalarImageRenderSpec,
        default_vmin: float | None = None,
        default_vmax: float | None = None,
        render_histogram: bool = False,
    ) -> None:
        """Render one scalar image using the shared image/colorbar configuration."""
        image = (
            self._to_2d(values, name=label)
            if isinstance(values, torch.Tensor)
            else np.asarray(values, dtype=np.float64)
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        vmin = (
            (float(np.nanmin(image)) if default_vmin is None else float(default_vmin))
            if spec.vmin is None
            else float(spec.vmin)
        )
        vmax = (
            (float(np.nanmax(image)) if default_vmax is None else float(default_vmax))
            if spec.vmax is None
            else float(spec.vmax)
        )
        output_path = out_dir / f"{filename_stem}.png"

        save_image_formats(
            output_path,
            image,
            self.config.image_formats,
            cmap=make_colormap(spec.cmap, spec.cmap_max),
            vmin=vmin,
            vmax=vmax,
        )

        if self.config.colorbar is not None:
            save_colorbar(self.config, spec, path=output_path, values=image)

        if render_histogram:
            save_histogram(
                image,
                output_path,
                self.config.render_histograms,
                self.config.image_formats,
                label,
            )

    def render_reconstruction(
        self, *, recon: torch.Tensor, target: torch.Tensor | None, out_dir
    ) -> None:
        """Render reconstruction, target, and absolute error."""
        assert self.config.reconstruction is not None
        recon_np = self._to_2d(recon, name="reconstruction")
        target_np = self._to_2d(target, name="target") if target is not None else None

        default_vmin = (
            float(np.nanmin(target_np))
            if target_np is not None
            else float(np.nanmin(recon_np))
        )
        default_vmax = (
            float(np.nanmax(target_np))
            if target_np is not None
            else float(np.nanmax(recon_np))
        )

        self.render_scalar_image(
            values=recon_np,
            out_dir=out_dir,
            filename_stem="reconstruction",
            label="reconstruction",
            spec=self.config.reconstruction,
            default_vmin=default_vmin,
            default_vmax=default_vmax,
        )

        if target_np is None:
            return

        self.render_scalar_image(
            values=target_np,
            out_dir=out_dir,
            filename_stem="ground_truth",
            label="ground truth",
            spec=self.config.reconstruction,
            default_vmin=default_vmin,
            default_vmax=default_vmax,
        )

        assert self.config.absolute_error is not None
        abs_error = self._to_2d(absolute_error(recon, target), name="absolute error")
        self.render_scalar_image(
            values=abs_error,
            out_dir=out_dir,
            filename_stem="absolute_error",
            label="absolute error",
            spec=self.config.absolute_error,
            default_vmin=0.0,
            default_vmax=float(np.nanmax(abs_error)),
        )

    def render_model_input(self, *, model_input: ModelInput, out_dir) -> None:
        """Render the image supplied to the UNet and the initial PDHG iterate."""
        if self.config.input_image is not None:
            self.render_scalar_image(
                values=model_input.input_image,
                out_dir=out_dir,
                filename_stem="unet_input",
                label="UNet input",
                spec=self.config.input_image,
            )

        if (
            self.config.initial_image is not None
            and model_input.initial_image is not None
        ):
            self.render_scalar_image(
                values=model_input.initial_image,
                out_dir=out_dir,
                filename_stem="pdhg_initial",
                label="PDHG initial image",
                spec=self.config.initial_image,
            )

    def render_scalar_map(
        self,
        *,
        values: torch.Tensor,
        out_dir,
        filename_stem: str,
        label: str,
        render_config: ScalarMapRenderConfig | None,
    ) -> None:
        """Render one 2-D lambda map in linear/log10 form."""
        values_np = self._to_2d(values, name=label)

        if render_config is None:
            render_config = ScalarMapRenderConfig(
                linear=ScalarImageRenderSpec(cmap="magma")
            )

        if render_config.linear is not None:
            assert isinstance(
                render_config.linear, ScalarImageRenderSpec
            ), f"Got {type(render_config.linear)}"
            self.render_scalar_image(
                values=values_np,
                out_dir=out_dir,
                filename_stem=filename_stem,
                label=label,
                spec=render_config.linear,
                render_histogram=True,
            )

        if render_config.log10 is not None:
            assert isinstance(
                render_config.log10, ScalarImageRenderSpec
            ), f"Got {type(render_config.log10)}"
            log_values = log10_positive(values_np)
            if np.isfinite(log_values).any():
                self.render_scalar_image(
                    values=log_values,
                    out_dir=out_dir,
                    filename_stem=f"{filename_stem}_log10",
                    label=f"log10({label})",
                    spec=render_config.log10,
                    render_histogram=True,
                )


class PCMResultWriter:
    """Save and render PCM reconstruction, model-input, and TV/TGV lambda-map artifacts."""

    def __init__(self, config: ResultOutputConfig) -> None:
        self.config = config
        self.renderer = PCMResultRenderer(config)

    @staticmethod
    def split_lambda_maps(
        lambdas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(lambda_1_v, lambda_0_w)`` for TV/TGV PCM output."""
        lambdas = lambdas.detach().cpu()

        if lambdas.ndim == 4:
            if lambdas.shape[0] != 1:
                raise ValueError(
                    f"Expected a single lambda-map batch, got {tuple(lambdas.shape)}."
                )
            lambdas = lambdas.squeeze(0)

        if lambdas.ndim != 3:
            raise ValueError(
                "Expected PCM lambda maps with shape [channels, H, W] "
                f"or [1, channels, H, W], got {tuple(lambdas.shape)}."
            )

        if lambdas.shape[0] == 1:
            return lambdas[0], None
        if lambdas.shape[0] == 2:
            return lambdas[0], lambdas[1]

        raise ValueError(
            f"Expected 1 TV map or 2 TGV maps, got {tuple(lambdas.shape)}."
        )

    def write_reconstruction(
        self,
        paths: ResultPaths,
        *,
        reconstruction: torch.Tensor,
        target: torch.Tensor | None,
        save_tensor: bool,
    ) -> None:
        paths.mkdir()

        if save_tensor:
            torch.save(
                reconstruction.detach().cpu().squeeze(), paths.reconstruction_tensor
            )

        if self.config.render_reconstruction:
            self.renderer.render_reconstruction(
                recon=reconstruction,
                target=target,
                out_dir=paths.images_dir,
            )

    def write_model_input(self, paths: ResultPaths, *, model_input: ModelInput) -> None:
        paths.mkdir()
        self.renderer.render_model_input(
            model_input=model_input, out_dir=paths.images_dir
        )

    def write_lambda_maps(self, paths: ResultPaths, *, lambdas: torch.Tensor) -> None:
        paths.mkdir()
        lambda_1_v, lambda_0_w = self.split_lambda_maps(lambdas)

        if self.config.save_lambda_maps_tensor:
            torch.save(lambda_1_v, paths.lambda_1_v_tensor)
            if lambda_0_w is not None:
                torch.save(lambda_0_w, paths.lambda_0_w_tensor)

        if self.config.sapm is None:
            return

        map_tensors = [lambda_1_v]
        map_names = ["lambda_1_v"]
        if lambda_0_w is not None:
            map_tensors.append(lambda_0_w)
            map_names.append("lambda_0_w")

        sapm: SAPMRenderConfig = self.config.sapm
        for map_idx, (values, default_name) in enumerate(
            zip(map_tensors, map_names, strict=True)
        ):
            map_name = (
                sapm.map_names[map_idx]
                if map_idx < len(sapm.map_names)
                else default_name
            )
            render_config = (
                sapm.lambda_map_render_configs[map_idx]
                if map_idx < len(sapm.lambda_map_render_configs)
                else None
            )
            self.renderer.render_scalar_map(
                values=values,
                out_dir=paths.lambda_maps_images_dir,
                filename_stem=map_name,
                label=map_name,
                render_config=render_config,
            )

    def write(
        self,
        paths: ResultPaths,
        *,
        recon: torch.Tensor | None,
        target: torch.Tensor | None,
        lambdas: torch.Tensor | None,
        model_input: ModelInput | None = None,
        save_reconstruction_tensor: bool,
    ) -> None:
        """Write any requested combination of PCM result artifacts."""
        if recon is not None:
            self.write_reconstruction(
                paths,
                reconstruction=recon,
                target=target,
                save_tensor=save_reconstruction_tensor,
            )

        if model_input is not None:
            self.write_model_input(paths, model_input=model_input)

        if lambdas is not None and (
            self.config.sapm is not None or self.config.save_lambda_maps_tensor
        ):
            self.write_lambda_maps(paths, lambdas=lambdas)

        write_result_manifest(
            paths,
            reconstruction_tensor_written=paths.reconstruction_tensor.exists(),
            reconstruction_images_written=paths.images_dir.exists(),
            lambda_1_v_tensor_written=paths.lambda_1_v_tensor.exists(),
            lambda_0_w_tensor_written=paths.lambda_0_w_tensor.exists(),
            lambda_images_written=(
                paths.lambda_maps_images_dir.exists()
                and any(paths.lambda_maps_images_dir.iterdir())
            ),
        )
