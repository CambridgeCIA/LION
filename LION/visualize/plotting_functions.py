from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap, Normalize

from LION.config.LogConfig import ImageFormat, ResultOutputConfig, ScalarImageRenderSpec


def make_colormap(cmap_name: str | None, cmap_max: float = 1.0) -> ListedColormap:
    """Return the first ``cmap_max`` fraction of a Matplotlib colormap."""
    return ListedColormap(plt.get_cmap(cmap_name)(np.linspace(0.0, cmap_max, 256)))


def save_image_formats(
    path: Path | str, image: np.ndarray, image_formats: Sequence[str], **imsave_kwargs
) -> list[Path]:
    """Save one scalar image in each requested static format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output_paths = [
        path.with_suffix(f".{image_format}") for image_format in image_formats
    ]
    for output_path in output_paths:
        plt.imsave(output_path, image, **imsave_kwargs)
    return output_paths


def save_figure_formats(
    fig, path: Path | str, image_formats: Sequence[ImageFormat], **savefig_kwargs
) -> list[Path]:
    """Save one Matplotlib figure in each requested static format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output_paths = [
        path.with_suffix(f".{image_format}") for image_format in image_formats
    ]
    for output_path in output_paths:
        fig.savefig(output_path, **savefig_kwargs)
    return output_paths


def save_histogram_formats(
    path: Path | str,
    values: np.ndarray,
    image_formats: Sequence[ImageFormat],
    *,
    name: str,
    bins: int = 100,
    value_range: tuple[float, float] | None = None,
) -> list[Path]:
    """Save a finite-value histogram beside a scalar image."""
    values = np.asarray(values)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return []

    path = Path(path)
    histogram_path = path.with_name(f"{path.stem}_histogram.png")
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.hist(finite_values.ravel(), bins=bins, range=value_range)
    if value_range is not None:
        axis.set_xlim(*value_range)
    axis.set_xlabel(name)
    axis.set_ylabel("Count")
    fig.tight_layout()
    output_paths = save_figure_formats(
        fig, histogram_path, image_formats=image_formats, dpi=200, bbox_inches="tight"
    )
    plt.close(fig)
    return output_paths


def save_colorbar(
    config: ResultOutputConfig,
    render_spec: ScalarImageRenderSpec | None,
    path: Path,
    values: np.ndarray | None = None,
) -> Path:
    """Save a standalone colorbar for an image.

    ``path`` is the path of the image. The colorbar is saved beside it as ``<stem>_colorbar.<ext>``.
    """
    output_path = path.with_name(f"{path.stem}_colorbar.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vmin, vmax, cmap = (
        (render_spec.vmin, render_spec.vmax, render_spec.cmap)
        if render_spec
        else (None, None, None)
    )
    if vmin is None or vmax is None:
        assert values is not None
        vmin = float(np.nanmin(values)) if vmin is None else float(vmin)
        vmax = float(np.nanmax(values)) if vmax is None else float(vmax)
    cmap = make_colormap(cmap, render_spec.cmap_max if render_spec is not None else 1.0)

    cbar_conf = config.colorbar
    assert cbar_conf is not None

    fig = plt.figure(figsize=cbar_conf.figsize, dpi=cbar_conf.dpi)
    assert isinstance(cbar_conf.axes_rect, tuple), f"Got {type(cbar_conf.axes_rect)}"
    cbar_axis = fig.add_axes(cbar_conf.axes_rect)

    cbar = ColorbarBase(
        cbar_axis,
        cmap=cmap,
        norm=Normalize(vmin, vmax),
        orientation=cbar_conf.orientation,
    )
    tick_values = np.linspace(vmin, vmax, 3).tolist()
    cbar.set_ticks(tick_values)
    tick_format = render_spec.cbar_tick_format if render_spec else None
    tick_format = tick_format if tick_format else cbar_conf.default_tick_format
    tick_labels = [format(tick, tick_format) for tick in tick_values]
    if (
        render_spec and render_spec.cbar_first_tick_leq
    ):  # Add less-than-or-equal sign to first tick label
        tick_labels[0] = f"≤ {tick_labels[0]}"
    if (
        render_spec and render_spec.cbar_last_tick_geq
    ):  # Add greater-than-or-equal sign to last tick label
        tick_labels[-1] = f"≥ {tick_labels[-1]}"
    cbar.set_ticklabels(tick_labels)
    if cbar_conf.tick_labelsize is not None:
        cbar.ax.tick_params(labelsize=cbar_conf.tick_labelsize)
    tick_labels = (
        cbar.ax.get_xticklabels()
        if cbar_conf.orientation == "horizontal"
        else cbar.ax.get_yticklabels()
    )

    if cbar_conf.orientation == "horizontal":
        tick_labels[0].set_horizontalalignment("left")
        tick_labels[-1].set_horizontalalignment("right")
    else:
        tick_labels[0].set_verticalalignment("bottom")
        tick_labels[-1].set_verticalalignment("top")

    if cbar_conf.fontfamily is not None:
        for tick_label in tick_labels:
            tick_label.set_fontfamily(cbar_conf.fontfamily)
    output_paths = save_figure_formats(
        fig,
        output_path,
        config.image_formats,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    return output_paths[0]


def tensor_evolution_gif(
    tensor,
    dim=0,
    filename: Path | str | None = None,
    output_config: ResultOutputConfig | None = None,
    fps=8,
    cmap="magma",
    vmin=None,
    vmax=None,
    normalize=True,
    annotate_index=True,
    add_colorbar=False,
    saving_colorbar=False,
    colorbar_label=None,
    dpi=100,
) -> list[np.ndarray]:
    """Create a GIF showing the evolution of a tensor/ndarray along a chosen dimension.
    Works with NumPy arrays or PyTorch tensors (auto-converted to NumPy).

    - 2D frames -> heatmap with colormap
    - (H,W,3/4) -> treated as RGB/RGBA
    - Higher dims (excluding a trailing 3/4) are averaged away per frame
    """
    try:
        import torch

        is_torch = isinstance(tensor, torch.Tensor)
    except Exception:
        is_torch = False

    # Convert to NumPy
    if is_torch:
        tensor = tensor.detach().cpu().numpy()
    else:
        tensor = np.asarray(tensor)

    # Move chosen dim to front (time axis)
    dim = dim % tensor.ndim
    arr = np.moveaxis(tensor, dim, 0)  # shape: (T, ...)

    # Global vmin/vmax for scalar data if not provided
    if vmin is None or vmax is None:
        finite_vals = np.asarray(arr[np.isfinite(arr)])
        if finite_vals.size > 0:
            gmin, gmax = finite_vals.min(), finite_vals.max()
            if vmin is None:
                vmin = gmin
            if vmax is None:
                vmax = gmax
        else:
            if vmin is None:
                vmin, vmax = 0.0, 1.0
            if vmax is None:
                vmax = 1.0

    frames: list[np.ndarray] = []
    T = arr.shape[0]

    def _to_rgb_uint8(frame2d):
        # Normalize and map via colormap
        if normalize and (vmax > vmin):
            scaled = (frame2d - vmin) / (vmax - vmin)
        else:
            scaled = np.zeros_like(frame2d, dtype=float)
        scaled = np.clip(scaled, 0.0, 1.0)
        m = plt.get_cmap(cmap)
        rgba = m(scaled)
        return (rgba[..., :3] * 255.0).astype(np.uint8)

    def _grab_canvas_rgb(fig):
        # Robustly capture rendered RGB pixels from the figure
        fig.canvas.draw()
        try:
            buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        except Exception:
            # Agg fallback
            w, h = fig.canvas.get_width_height()
            raw, _ = fig.canvas.print_to_buffer()
            buf = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        return buf[:, :, :3].copy()

    for t in range(T):
        slice_t = arr[t]

        # Reduce dims except keep trailing RGB/A if present
        if slice_t.ndim > 2:
            if slice_t.shape[-1] in (3, 4):
                if slice_t.ndim > 3:
                    kept = slice_t
                    for ax_idx in reversed(range(slice_t.ndim - 1)):
                        kept = kept.mean(axis=ax_idx, keepdims=True)
                    slice_t = np.squeeze(kept, axis=tuple(range(slice_t.ndim - 1)))
            else:
                slice_t = slice_t.mean(axis=tuple(range(slice_t.ndim - 2)))

        # Prepare frame (RGB uint8)
        if slice_t.ndim == 2:
            frame = _to_rgb_uint8(np.nan_to_num(slice_t, copy=False))
        elif slice_t.ndim == 3 and slice_t.shape[-1] in (3, 4):
            img = np.nan_to_num(slice_t, copy=False)
            if img.dtype != np.uint8:
                mn, mx = np.nanmin(img), np.nanmax(img)
                if mx > mn:
                    img = (img - mn) / (mx - mn)
                else:
                    img = np.zeros_like(img)
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            frame = img[..., :3]
        else:
            frame = _to_rgb_uint8(np.nan_to_num(slice_t.mean(axis=-1), copy=False))

        if annotate_index or add_colorbar or slice_t.ndim == 2:
            # Render via Matplotlib to allow text and/or colorbar and consistent sizing
            fig = plt.figure(
                figsize=(frame.shape[1] / dpi, frame.shape[0] / dpi), dpi=dpi
            )
            ax = plt.axes((0, 0, 1, 1))
            ax.axis("off")

            if slice_t.ndim == 2:
                im = ax.imshow(slice_t, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
                if annotate_index:
                    ax.text(
                        0.01,
                        0.99,
                        f"t={t}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=10,
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.4),
                    )
                if add_colorbar:
                    cax = fig.add_axes((0.88, 0.1, 0.03, 0.8))
                    fig.colorbar(im, cax=cax)
            else:
                ax.imshow(frame, aspect="equal")
                if annotate_index:
                    ax.text(
                        0.01,
                        0.99,
                        f"t={t}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=10,
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.4),
                    )

            frames.append(_grab_canvas_rgb(fig))
            plt.close(fig)
        else:
            frames.append(frame)

    if filename is not None:
        assert output_config is not None
        # imageio.mimsave(filename, frames, duration=1.0 / max(fps, 1), loop=0)
        imageio.mimsave(
            uri=filename, ims=frames, format="GIF", duration=1.0 / max(fps, 1), loop=0
        )
        is_rgb = arr.ndim >= 4 and arr.shape[-1] in (3, 4)
        if saving_colorbar and not is_rgb:
            cbar_render_spec = ScalarImageRenderSpec(vmin=vmin, vmax=vmax, cmap=cmap)
            save_colorbar(
                config=output_config, render_spec=cbar_render_spec, path=Path(filename)
            )
    return frames


# ---------------------------
# Quick smoke test examples:
if __name__ == "__main__":
    # Example 1: scalar heatmap animation
    x = np.random.randn(20, 64, 64)
    tensor_evolution_gif(
        x, dim=0, filename="scalar.gif", fps=10, cmap="viridis", add_colorbar=True
    )

    # Example 2: RGB animation
    rgb = np.random.rand(30, 96, 96, 3)
    tensor_evolution_gif(rgb, dim=0, filename="rgb.gif", fps=12, annotate_index=False)
