"""Visual outputs for PaDIS LIDC reconstruction runs."""

from __future__ import annotations

import pathlib

import torch


def save_preview(
    path: pathlib.Path,
    sinogram,
    fdk_recon,
    recon,
    target,
    *,
    reference=None,
    body_mask,
    error_vmax: float,
    recon_label: str,
) -> None:
    """Save preview."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = 5 if reference is not None else 4
    fig, axes = plt.subplots(2, columns, figsize=(3.25 * columns, 6.2))
    image_kwargs = {"cmap": "gray", "vmin": 0, "vmax": 1}
    axes[0, 0].imshow(sinogram.detach().cpu().squeeze().T, cmap="gray")
    axes[0, 0].set_title("Sinogram")
    axes[0, 1].imshow(fdk_recon.detach().cpu().squeeze(), **image_kwargs)
    axes[0, 1].set_title("FDK")
    axes[0, 2].imshow(recon.detach().cpu().squeeze(), **image_kwargs)
    axes[0, 2].set_title(recon_label)
    axes[0, 3].imshow(target.detach().cpu().squeeze(), **image_kwargs)
    axes[0, 3].set_title("Target")
    if reference is not None:
        axes[0, 4].imshow(reference.detach().cpu().squeeze(), **image_kwargs)
        axes[0, 4].set_title("Public ref")
    axes[1, 0].imshow(body_mask.detach().cpu().squeeze(), cmap="gray")
    axes[1, 0].set_title("Body ROI")
    axes[1, 1].imshow(
        torch.abs(fdk_recon - target).detach().cpu().squeeze(),
        cmap="magma",
        vmin=0,
        vmax=error_vmax,
    )
    axes[1, 1].set_title("|FDK error|")
    axes[1, 2].imshow(
        torch.abs(recon - target).detach().cpu().squeeze(),
        cmap="magma",
        vmin=0,
        vmax=error_vmax,
    )
    axes[1, 2].set_title(f"|{recon_label} error|")
    axes[1, 3].imshow(
        (recon - fdk_recon).detach().cpu().squeeze(),
        cmap="coolwarm",
        vmin=-error_vmax,
        vmax=error_vmax,
    )
    axes[1, 3].set_title(f"{recon_label} - FDK")
    if reference is not None:
        axes[1, 4].imshow(
            torch.abs(recon - reference).detach().cpu().squeeze(),
            cmap="magma",
            vmin=0,
            vmax=error_vmax,
        )
        axes[1, 4].set_title(f"|{recon_label} - ref|")
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_visual_comparison(
    path: pathlib.Path,
    fdk_recon,
    recon,
    target,
    *,
    reference=None,
    error_vmax: float,
    image_vmax: float,
    recon_label: str,
) -> None:
    """Save visual comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    image_kwargs = {"cmap": "gray", "vmin": 0.0, "vmax": image_vmax}
    error_kwargs = {"cmap": "magma", "vmin": 0.0, "vmax": error_vmax}
    compare_image = reference if reference is not None else target
    compare_label = "Public ref" if reference is not None else "Target"

    panels = [
        ("Target", target.detach().cpu().squeeze(), image_kwargs),
        (compare_label, compare_image.detach().cpu().squeeze(), image_kwargs),
        (recon_label, recon.detach().cpu().squeeze(), image_kwargs),
        ("FDK", fdk_recon.detach().cpu().squeeze(), image_kwargs),
        (
            f"|{recon_label} - Target|",
            torch.abs(recon - target).detach().cpu().squeeze(),
            error_kwargs,
        ),
        (
            f"|{recon_label} - {compare_label}|",
            torch.abs(recon - compare_image).detach().cpu().squeeze(),
            error_kwargs,
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax, (title, image, kwargs) in zip(axes.ravel(), panels):
        im = ax.imshow(image, **kwargs)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=160)
    plt.close(fig)

    individual_folder = path.with_suffix("")
    individual_folder.mkdir(parents=True, exist_ok=True)
    individual_images = {
        "target": (target.detach().cpu().squeeze(), image_kwargs),
        compare_label.lower().replace(" ", "_"): (
            compare_image.detach().cpu().squeeze(),
            image_kwargs,
        ),
        "recon": (recon.detach().cpu().squeeze(), image_kwargs),
        "fdk": (fdk_recon.detach().cpu().squeeze(), image_kwargs),
        "abs_recon_target": (
            torch.abs(recon - target).detach().cpu().squeeze(),
            error_kwargs,
        ),
        "abs_recon_reference": (
            torch.abs(recon - compare_image).detach().cpu().squeeze(),
            error_kwargs,
        ),
    }
    for stem, (image, kwargs) in individual_images.items():
        plt.imsave(individual_folder / f"{stem}.png", image.numpy(), **kwargs)


def save_tensor_image(
    path: pathlib.Path,
    tensor: torch.Tensor,
    *,
    transpose: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Save tensor image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = tensor.detach().cpu().squeeze()
    while image.ndim > 2:
        image = image[0]
    if transpose and image.ndim == 2:
        image = image.T
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, image.numpy(), cmap="gray", vmin=vmin, vmax=vmax)


def save_trace_images(
    output_folder: pathlib.Path,
    sample_index: int,
    snapshots: list[dict],
) -> dict | None:
    """Save trace images."""
    if not snapshots:
        return None

    trace_root = output_folder / "trace_images"
    sample_folder = trace_root / f"sample_{sample_index:04d}"
    sample_folder.mkdir(parents=True, exist_ok=True)
    tensor_path = trace_root / f"sample_{sample_index:04d}.pt"
    torch.save({"index": int(sample_index), "snapshots": snapshots}, tensor_path)

    image_records = []
    for snapshot in snapshots:
        stem = (
            f"step_{int(snapshot['step']):04d}_"
            f"inner_{int(snapshot['inner']):02d}_"
            f"{snapshot['algorithm']}"
        )
        current_path = sample_folder / f"{stem}_current.png"
        denoised_path = sample_folder / f"{stem}_denoised.png"
        projected_path = sample_folder / f"{stem}_projected.png"
        x_next_path = sample_folder / f"{stem}_x_next.png"
        forward_path = sample_folder / f"{stem}_forward_projected.png"
        save_tensor_image(
            current_path,
            snapshot["x"],
            vmin=0.0,
            vmax=1.0,
        )
        save_tensor_image(
            denoised_path,
            snapshot["denoised"],
            vmin=0.0,
            vmax=1.0,
        )
        save_tensor_image(
            projected_path,
            snapshot["projected"],
            vmin=0.0,
            vmax=1.0,
        )
        save_tensor_image(
            x_next_path,
            snapshot["x_next"],
            vmin=0.0,
            vmax=1.0,
        )
        save_tensor_image(
            forward_path,
            snapshot["forward_projected"],
            transpose=True,
        )
        image_records.append(
            {
                "step": int(snapshot["step"]),
                "inner": int(snapshot["inner"]),
                "algorithm": snapshot["algorithm"],
                "sigma": float(snapshot["sigma"]),
                "current_png": str(current_path),
                "denoised_png": str(denoised_path),
                "projected_png": str(projected_path),
                "x_next_png": str(x_next_path),
                "forward_projected_png": str(forward_path),
            }
        )

    return {
        "index": int(sample_index),
        "tensor_path": str(tensor_path),
        "folder": str(sample_folder),
        "images": image_records,
    }
