from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from LION.config.config_utils import make_dataclass

ImageFormat = Literal["png", "pdf"]
ColorbarOrientation = Literal["horizontal", "vertical"]


@dataclass(kw_only=True)
class ScalarImageRenderSpec:
    vmin: float | None = None
    vmax: float | None = None
    cmap: str | None = None
    cmap_max: float = 1.0
    cbar_tick_format: str = ".2f"
    cbar_first_tick_leq: bool = False
    cbar_last_tick_geq: bool = False
    gif_index_annotation_label: str | None = "t"
    gif_fps: int = 8

    @staticmethod
    def from_list(value: list[float | str | bool | None]) -> ScalarImageRenderSpec:
        (
            vmin,
            vmax,
            cmap,
            cbar_tick_format,
            cbar_first_tick_leq,
            cbar_last_tick_geq,
        ) = value[:6]
        gif_index_annotation_label = value[6] if len(value) > 6 else "t"
        gif_fps = value[7] if len(value) > 7 else 8
        cmap_max = value[8] if len(value) > 8 else 1.0
        assert isinstance(vmin, (float, type(None))), f"Got {type(vmin)}"
        assert isinstance(vmax, (float, type(None))), f"Got {type(vmax)}"
        assert isinstance(cmap, (str, type(None))), f"Got {type(cmap)}"
        assert isinstance(cbar_tick_format, str), f"Got {type(cbar_tick_format)}"
        assert isinstance(cbar_first_tick_leq, bool), f"Got {type(cbar_first_tick_leq)}"
        assert isinstance(cbar_last_tick_geq, bool), f"Got {type(cbar_last_tick_geq)}"
        assert isinstance(
            gif_index_annotation_label, (str, type(None))
        ), f"Got {type(gif_index_annotation_label)}"
        assert isinstance(gif_fps, int), f"Got {type(gif_fps)}"
        assert isinstance(cmap_max, float), f"Got {type(cmap_max)}"

        return ScalarImageRenderSpec(
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            cmap_max=cmap_max,
            cbar_tick_format=cbar_tick_format,
            cbar_first_tick_leq=cbar_first_tick_leq,
            cbar_last_tick_geq=cbar_last_tick_geq,
            gif_index_annotation_label=gif_index_annotation_label,
            gif_fps=gif_fps,
        )


@dataclass(kw_only=True)
class ScalarMapRenderConfig:
    """Rendering configuration for one scalar map."""

    linear: ScalarImageRenderSpec | list[float | str | bool | None] | None = None
    log10: ScalarImageRenderSpec | list[float | str | bool | None] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.linear, dict):
            self.linear = ScalarImageRenderSpec(**self.linear)
        if self.linear is not None and isinstance(self.linear, list):
            self.linear = ScalarImageRenderSpec.from_list(self.linear)
        if isinstance(self.log10, dict):
            self.log10 = ScalarImageRenderSpec(**self.log10)
        if self.log10 is not None and isinstance(self.log10, list):
            self.log10 = ScalarImageRenderSpec.from_list(self.log10)


@dataclass(kw_only=True)
class CommonColorbarConfig:
    orientation: ColorbarOrientation = "horizontal"
    figsize: list[float] | None = None
    axes_rect: tuple[float, float, float, float] | list[float] | None = None
    tick_labelsize: float | None = None
    label_fontsize: float | None = None
    fontfamily: str | None = "serif"
    default_tick_format: str = ".2f"
    dpi: int = 200

    def __post_init__(self) -> None:
        if self.figsize is None:
            self.figsize = (
                [4.0, 0.7] if self.orientation == "horizontal" else [0.9, 4.0]
            )
        if self.axes_rect is None:
            self.axes_rect = (
                [0.08, 0.42, 0.84, 0.35]
                if self.orientation == "horizontal"
                else [0.30, 0.06, 0.30, 0.88]
            )
        left, bottom, width, height = self.axes_rect
        self.axes_rect = (left, bottom, width, height)


@dataclass(kw_only=True)
class MRIRenderConfig:
    factor: float = 6e-3
    frame_idx: int = 0
    row_idx: int = 103
    col_idx: int = 103
    top_left: tuple[int, int] = (20, 20)
    bottom_right: tuple[int, int] = (180, 40)


def empty_list():
    return field(default_factory=list)


@dataclass(kw_only=True)
class SAPMRenderConfig:
    lambda_map_render_configs: list[ScalarMapRenderConfig | None] = empty_list()
    lambda_ratio_render_configs: list[ScalarMapRenderConfig | None] = empty_list()

    lambda_0_w_over_lambda_1_v_render_config: ScalarMapRenderConfig | None = None
    lambda_1_v_over_lambda_0_w_render_config: ScalarMapRenderConfig | None = None

    map_indices: list[int] = empty_list()
    map_names: list[str] = empty_list()

    def __post_init__(self) -> None:
        self.lambda_map_render_configs = [
            self._coerce_scalar_map_render_config(value)
            for value in self.lambda_map_render_configs
        ]
        self.lambda_ratio_render_configs = [
            self._coerce_scalar_map_render_config(value)
            for value in self.lambda_ratio_render_configs
        ]
        self.lambda_0_w_over_lambda_1_v_render_config = (
            self._coerce_scalar_map_render_config(
                self.lambda_0_w_over_lambda_1_v_render_config
            )
        )
        self.lambda_1_v_over_lambda_0_w_render_config = (
            self._coerce_scalar_map_render_config(
                self.lambda_1_v_over_lambda_0_w_render_config
            )
        )

    @staticmethod
    def _coerce_scalar_map_render_config(
        value: dict | ScalarMapRenderConfig | None,
    ) -> ScalarMapRenderConfig | None:
        if value is not None and isinstance(value, dict):
            return ScalarMapRenderConfig(**value)
        return value


@dataclass(kw_only=True)
class ResultOutputConfig:
    """Result rendering/saving configuration.

    This configuration is used by both self-supervised training finalization
    and testing. The caller decides which result is being written and where;
    this class only controls which artifacts are produced.

    Attributes:
        visualize_ids: Sample indices to visualize. If empty, no images will be visualized.
        results_dir_suffix: Suffix to append to the results directory name.
    """

    render_reconstruction: bool = False
    render_csm: bool = False
    save_lambda_maps_tensor: bool = False
    display_interactively: bool = False
    render_histograms: bool = False
    # image_formats: list[ImageFormat] = field(default_factory=lambda: ['png'])
    image_formats: list[ImageFormat] = field(default_factory=lambda: ["pdf"])

    visualize_ids: set[int] = field(default_factory=set)
    results_dir_suffix: str = ""

    colorbar: CommonColorbarConfig | None = None
    reconstruction: ScalarImageRenderSpec | list[
        float | str | bool | None
    ] | None = field(default_factory=lambda: ScalarImageRenderSpec(cmap="gray"))
    absolute_error: ScalarImageRenderSpec | list[
        float | str | bool | None
    ] | None = None
    input_image: ScalarImageRenderSpec | list[float | str | bool | None] | None = field(
        default_factory=lambda: ScalarImageRenderSpec(cmap="gray")
    )
    initial_image: ScalarImageRenderSpec | list[
        float | str | bool | None
    ] | None = field(default_factory=lambda: ScalarImageRenderSpec(cmap="gray"))
    sapm: SAPMRenderConfig | None = None
    mri: MRIRenderConfig | None = None
    csm: ScalarImageRenderSpec | None = field(
        default_factory=lambda: ScalarImageRenderSpec(
            vmin=0.0,
            vmax=0.4,
            cbar_tick_format=".1f",
            cbar_last_tick_geq=True,
            cmap="gray",
            gif_index_annotation_label="coil",
            gif_fps=1,
        )
    )

    cmap_abs_error: str = "viridis"

    def __post_init__(self) -> None:
        self.image_formats = list(dict.fromkeys(self.image_formats))
        self.visualize_ids = set(self.visualize_ids)
        # if 'png' not in self.image_formats:
        #     raise ValueError("image_formats must include 'png'; add 'pdf' to save PDF copies alongside canonical PNGs.")

        if isinstance(self.colorbar, dict):
            self.colorbar = CommonColorbarConfig(**self.colorbar)

        self.reconstruction = self._coerce_scalar_image_render_spec(self.reconstruction)
        self.absolute_error = self._coerce_scalar_image_render_spec(self.absolute_error)
        if self.absolute_error is None:
            self.absolute_error = ScalarImageRenderSpec(
                vmin=0.0, cmap=self.cmap_abs_error
            )
        self.input_image = self._coerce_scalar_image_render_spec(self.input_image)
        self.initial_image = self._coerce_scalar_image_render_spec(self.initial_image)

        if isinstance(self.sapm, dict):
            self.sapm = SAPMRenderConfig(**self.sapm)
        if isinstance(self.mri, dict):
            self.mri = MRIRenderConfig(**self.mri)

    @staticmethod
    def _coerce_scalar_image_render_spec(
        value: ScalarImageRenderSpec | list[float | str | bool | None] | dict | None,
    ) -> ScalarImageRenderSpec | None:
        if isinstance(value, dict):
            return ScalarImageRenderSpec(**value)
        if isinstance(value, list):
            return ScalarImageRenderSpec.from_list(value)
        return value

    def rendering_lambda_maps(self) -> bool:
        return self.sapm is not None

    def rendering_outputs(self) -> bool:
        return (
            self.render_reconstruction
            or self.input_image is not None
            or self.initial_image is not None
            or self.rendering_lambda_maps()
            or self.save_lambda_maps_tensor
        )


@dataclass(kw_only=True)
class CheckpointConfig:
    # Assumes 1-indexing
    every_n_epochs: int | None  # Set <= 0 to disable checkpointing
    start_epoch: int | None  # If None or <= 0, log untrained model as epoch 0
    end_epoch: int | None  # Inclusive. If None, log until the end of training
    epochs: list[int] | set[
        int
    ] | None = None  # Log these specific epochs even if disabled by every_n_epochs or out of range

    every_n_steps: int | None = None  # Set <= 0 to disable step checkpointing
    start_step: int | None = None
    end_step: int | None = None
    steps: list[int] | set[int] | None = None

    log_best: bool = False  # Best validation loss checkpointing
    log_cur: bool = False  # Log even if disabled by every_n_epochs or out of range
    log_final: bool = (
        True  # Log final epoch even if disabled by every_n_epochs or out of range
    )

    best_checkpoint_epoch_block_size: int | None = None
    """When a new best is found, save an additional checkpoint labelled by
    the epoch block containing that improvement.

    For example, with a block size of 50, an improvement at epoch 73 is saved
    as ``..._new_best_in_epochs_0051_0100.pt``.

    If no file exists for a later block, no new best occurred in that block.
    For example, there maybe ``..._new_best_in_epochs_0001_0050.pt``
    and ``..._new_best_in_epochs_0101_0150.pt`` but no ``..._new_best_in_epochs_051_0100.pt``
    if the criterion's best was not improved in epochs 51-100.
    This is intentional to reduce storage usage.
    """

    local: bool
    mlflow: bool

    def __post_init__(self) -> None:
        # Make explicit checkpoint indices sets for faster lookup.
        if self.epochs is not None:
            object.__setattr__(self, "epochs", set(self.epochs))
        if self.steps is not None:
            object.__setattr__(self, "steps", set(self.steps))


@dataclass(kw_only=True)
class LogConfig:
    """Configuration for logging, metrics, checkpoints, and output paths."""

    exp_code: str
    results_dir: Path
    run_suffix: str | None = None

    log_mlflow: bool
    log_csv: bool

    log_train_loss_every_epochs: int = 1
    log_train_metrics_every_epochs: int = -1
    log_train_params_every_epochs: int = 1
    log_val_loss_every_epochs: int = 1
    log_val_metrics_every_epochs: int = -1
    log_test_loss_every_epochs: int = -1
    log_test_metrics_every_epochs: int = -1

    log_train_loss_every_n_steps: int = -1
    log_train_metrics_every_n_steps: int = -1
    log_train_params_every_n_steps: int = -1
    log_val_loss_every_n_steps: int = -1
    log_val_metrics_every_n_steps: int = -1
    log_test_loss_every_n_steps: int = -1
    log_test_metrics_every_n_steps: int = -1

    val_before_start: bool = True

    ckpt: CheckpointConfig

    saving_initial: bool = False

    saving_initial_images: bool = False
    saving_train_recon_images: bool = False
    saving_val_recon_images: bool = False
    saving_test_recon_images: bool = False

    # Shared final artifact configuration. ``saving_test_recon_images`` is
    # retained for backward compatibility and is mapped to
    # ``final_result.render_reconstruction`` by ``make_log_config``.
    final_result: ResultOutputConfig = field(default_factory=ResultOutputConfig)

    save_images_every_n_epochs: int = -1
    save_images_start_epoch: int | None = None
    save_images_stop_epoch: int | None = None

    saving_image_locally: bool = False
    saving_image_mlflow: bool = False

    final_metrics_path: Path | None = None

    def __post_init__(self) -> None:
        """Normalise path-like fields."""
        object.__setattr__(self, "results_dir", Path(self.results_dir))

        if self.final_metrics_path is not None:
            object.__setattr__(
                self, "final_metrics_path", Path(self.final_metrics_path)
            )


def make_log_config(data: dict[str, Any] | None) -> LogConfig:
    """Build ``LogConfig`` including nested checkpoint/result configs.

    Older YAML files using ``saving_test_recon_images: true`` remain supported;
    when no explicit ``final_result.render_reconstruction`` is enabled, that
    legacy flag turns it on.
    """
    log_data = dict(data or {})
    ckpt_data = log_data.pop("ckpt", {})
    final_result_data = log_data.pop("final_result", {})

    final_result = make_dataclass(ResultOutputConfig, final_result_data)
    assert final_result is not None

    if log_data.get("saving_test_recon_images", False):
        final_result.render_reconstruction = True

    return LogConfig(
        **log_data,
        ckpt=make_dataclass(CheckpointConfig, ckpt_data),
        final_result=final_result,
    )
