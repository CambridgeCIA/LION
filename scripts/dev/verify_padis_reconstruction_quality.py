"""Verify PaDIS reconstruction tensors against target and public references."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def as_numpy_image(tensor_or_array) -> np.ndarray:
    if isinstance(tensor_or_array, torch.Tensor):
        array = tensor_or_array.detach().cpu().numpy()
    else:
        array = np.asarray(tensor_or_array)
    array = np.squeeze(array).astype(np.float64)
    return np.clip(array, 0.0, 1.0)


def psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def ssim_value(image: np.ndarray, reference: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(reference, image, data_range=1.0))


def edge_ssim_value(image: np.ndarray, reference: np.ndarray) -> float:
    from scipy.ndimage import sobel
    from skimage.metrics import structural_similarity

    image_edges = sobel(image)
    reference_edges = sobel(reference)
    data_range = float(reference_edges.max() - reference_edges.min())
    if data_range == 0:
        return 1.0 if float(np.max(np.abs(image_edges - reference_edges))) == 0 else 0.0
    return float(
        structural_similarity(reference_edges, image_edges, data_range=data_range)
    )


def image_metrics(image: np.ndarray, reference: np.ndarray) -> dict:
    error = np.abs(image - reference)
    mse = float(np.mean((image - reference) ** 2))
    return {
        "mse": mse,
        "psnr": psnr_from_mse(mse),
        "ssim": ssim_value(image, reference),
        "edge_ssim": edge_ssim_value(image, reference),
        "mae": float(error.mean()),
        "abs_error_p90": float(np.quantile(error, 0.90)),
        "abs_error_p95": float(np.quantile(error, 0.95)),
        "abs_error_p99": float(np.quantile(error, 0.99)),
        "abs_error_max": float(error.max()),
        "mean_delta": float(image.mean() - reference.mean()),
    }


def load_public_reference(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if "recon" not in data.files:
            raise ValueError(f"{path} does not contain a 'recon' array.")
        return np.asarray(data["recon"])
    if path.suffix in (".pt", ".pth"):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            for key in ("reconstructions", "recon", "images"):
                if key in data:
                    return data[key].detach().cpu().numpy()
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
    raise ValueError(f"Unsupported public reference format: {path}")


def save_visual_outputs(
    output_dir: Path,
    *,
    sample_index: int,
    recon: np.ndarray,
    target: np.ndarray,
    reference: np.ndarray | None,
    fdk: np.ndarray | None,
    image_vmax: float,
    error_vmax: float,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_dir = output_dir / f"sample_{sample_index:04d}_visual_compare"
    sample_dir.mkdir(parents=True, exist_ok=True)

    images: dict[str, tuple[np.ndarray, str, float, float]] = {
        "target": (target, "gray", 0.0, image_vmax),
        "recon": (recon, "gray", 0.0, image_vmax),
        "abs_recon_target": (np.abs(recon - target), "magma", 0.0, error_vmax),
    }
    if reference is not None:
        images["public_ref"] = (reference, "gray", 0.0, image_vmax)
        images["abs_recon_reference"] = (
            np.abs(recon - reference),
            "magma",
            0.0,
            error_vmax,
        )
    if fdk is not None:
        images["fdk"] = (fdk, "gray", 0.0, image_vmax)

    paths = {}
    for stem, (image, cmap, vmin, vmax) in images.items():
        path = sample_dir / f"{stem}.png"
        plt.imsave(path, image, cmap=cmap, vmin=vmin, vmax=vmax)
        paths[stem] = str(path)

    panels = [
        ("Target", target, "gray", 0.0, image_vmax),
        (
            "Public ref" if reference is not None else "Target",
            reference if reference is not None else target,
            "gray",
            0.0,
            image_vmax,
        ),
        ("PaDIS", recon, "gray", 0.0, image_vmax),
        (
            "FDK" if fdk is not None else "|PaDIS - Target|",
            fdk if fdk is not None else np.abs(recon - target),
            "gray" if fdk is not None else "magma",
            0.0,
            image_vmax if fdk is not None else error_vmax,
        ),
        ("|PaDIS - Target|", np.abs(recon - target), "magma", 0.0, error_vmax),
        (
            "|PaDIS - Public ref|" if reference is not None else "|PaDIS - Target|",
            np.abs(recon - reference)
            if reference is not None
            else np.abs(recon - target),
            "magma",
            0.0,
            error_vmax,
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax, (title, image, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    montage_path = output_dir / f"sample_{sample_index:04d}_visual_compare.png"
    fig.savefig(montage_path, dpi=160)
    plt.close(fig)
    paths["montage"] = str(montage_path)
    return paths


def fail_if_needed(args, summary: dict) -> None:
    failures = []
    checks = [
        ("mean_target_psnr", "min_mean_target_psnr", ">="),
        ("min_target_ssim", "min_sample_target_ssim", ">="),
        ("mean_target_mae", "max_mean_target_mae", "<="),
        ("max_target_abs_error_p95", "max_sample_target_abs_error_p95", "<="),
        ("mean_public_ssim", "min_mean_public_ssim", ">="),
        ("mean_public_mae", "max_mean_public_mae", "<="),
        ("max_public_abs_error_p95", "max_sample_public_abs_error_p95", "<="),
    ]
    for metric_key, arg_key, op in checks:
        threshold = getattr(args, arg_key)
        value = summary.get(metric_key)
        if threshold is None or value is None:
            continue
        failed = value < threshold if op == ">=" else value > threshold
        if failed:
            failures.append(
                f"{metric_key}={value:.6g} does not satisfy {op} {threshold:.6g}"
            )
    if failures:
        raise RuntimeError("Quality verification failed:\n  " + "\n  ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstructions", type=Path, required=True)
    parser.add_argument("--public-reference", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-vmax", type=float, default=0.75)
    parser.add_argument("--error-vmax", type=float, default=0.10)
    parser.add_argument("--min-mean-target-psnr", type=float, default=None)
    parser.add_argument("--min-sample-target-ssim", type=float, default=None)
    parser.add_argument("--max-mean-target-mae", type=float, default=None)
    parser.add_argument("--max-sample-target-abs-error-p95", type=float, default=None)
    parser.add_argument("--min-mean-public-ssim", type=float, default=None)
    parser.add_argument("--max-mean-public-mae", type=float, default=None)
    parser.add_argument("--max-sample-public-abs-error-p95", type=float, default=None)
    args = parser.parse_args()

    data = torch.load(args.reconstructions, map_location="cpu", weights_only=False)
    reconstructions = data["reconstructions"]
    targets = data["targets"]
    fdk_reconstructions = data.get("fdk_reconstructions")
    public_reference = (
        load_public_reference(args.public_reference)
        if args.public_reference is not None
        else None
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(len(reconstructions)):
        recon = as_numpy_image(reconstructions[index])
        target = as_numpy_image(targets[index])
        fdk = (
            as_numpy_image(fdk_reconstructions[index])
            if fdk_reconstructions is not None
            else None
        )
        reference = None
        if public_reference is not None:
            ref_index = 0 if len(public_reference) == 1 else index
            reference = as_numpy_image(public_reference[ref_index])
            if reference.shape != recon.shape:
                raise ValueError(
                    f"Public reference shape {reference.shape} does not match "
                    f"reconstruction shape {recon.shape} for sample {index}."
                )

        record = {
            "index": int(index),
            "target": image_metrics(recon, target),
            "visual_outputs": save_visual_outputs(
                args.output_dir,
                sample_index=index,
                recon=recon,
                target=target,
                reference=reference,
                fdk=fdk,
                image_vmax=float(args.image_vmax),
                error_vmax=float(args.error_vmax),
            ),
        }
        if reference is not None:
            record["public_reference"] = image_metrics(recon, reference)
        records.append(record)

    summary = {
        "num_samples": len(records),
        "mean_target_psnr": float(np.mean([r["target"]["psnr"] for r in records])),
        "min_target_psnr": float(np.min([r["target"]["psnr"] for r in records])),
        "mean_target_ssim": float(np.mean([r["target"]["ssim"] for r in records])),
        "min_target_ssim": float(np.min([r["target"]["ssim"] for r in records])),
        "mean_target_mae": float(np.mean([r["target"]["mae"] for r in records])),
        "max_target_abs_error_p95": float(
            np.max([r["target"]["abs_error_p95"] for r in records])
        ),
    }
    if public_reference is not None:
        summary.update(
            {
                "mean_public_ssim": float(
                    np.mean([r["public_reference"]["ssim"] for r in records])
                ),
                "min_public_ssim": float(
                    np.min([r["public_reference"]["ssim"] for r in records])
                ),
                "mean_public_mae": float(
                    np.mean([r["public_reference"]["mae"] for r in records])
                ),
                "max_public_abs_error_p95": float(
                    np.max([r["public_reference"]["abs_error_p95"] for r in records])
                ),
            }
        )

    payload = {
        "reconstructions": str(args.reconstructions),
        "public_reference": str(args.public_reference)
        if args.public_reference is not None
        else None,
        "summary": summary,
        "records": records,
    }
    output_path = args.output_dir / "quality_verification.json"
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved quality verification to {output_path}")
    fail_if_needed(args, summary)


if __name__ == "__main__":
    main()
