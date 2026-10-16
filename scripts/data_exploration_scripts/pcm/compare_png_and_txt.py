"""Compare PNG current maps and TXT measurements from the input images (real data).

The PNG and TXT files should pair with each other, but the PNG images are normalized to [0, 1].

This script compares the files.

I found that the PNG images are not normalized directly from the TXT data, but from a clipped version.

Some I'm not even sure how.

I'm not sure what to use.

Pretrained denoisers are usually trained on normalized images.

The synthetic data is only PNGs in [0, 1].

But I also need to know the range of actual current map values for the physics.

This script tries to find the mapping based on the real input images.
"""

# %% Imports
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

print(f"Current working directory: {os.getcwd()}")

# %%
pcm_input_data_dir = Path(
    # "../../../data/PCM Synthetic Data Set/Synthetic Data/Input Images"
    "/store/LION/ttv22/pcm/data/PCM Synthetic Data Set/Synthetic Data/Input Images"
)
assert pcm_input_data_dir.exists(), f"Directory {pcm_input_data_dir} does not exist."
raw_txt_dir = pcm_input_data_dir / "txt"
normalised_png_dir = pcm_input_data_dir / "png"


# %%
def get_sample_paths(*, raw_txt_dir: Path, normalised_png_dir: Path):
    txt_paths = list(raw_txt_dir.glob("*.txt"))
    txt_paths.sort()
    png_paths = list(normalised_png_dir.glob("*.png"))
    png_paths.sort()
    txt_stems = [path.stem for path in txt_paths]
    png_stems = [path.stem for path in png_paths]
    assert txt_stems == png_stems, "Mismatch between txt and png files."
    print(f"Found {len(txt_stems)} samples")
    print(f"Sample names: {txt_stems}")

    return {
        sample_name: {"txt": txt_path, "png": png_path}
        for sample_name, txt_path, png_path in zip(txt_stems, txt_paths, png_paths)
    }


sample_paths = get_sample_paths(
    raw_txt_dir=raw_txt_dir, normalised_png_dir=normalised_png_dir
)


# %%
def plot(suptitle: str, *, png_image_np: np.ndarray, txt_image_np: np.ndarray):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(png_image_np, cmap="gray")
    plt.colorbar(label="Normalized Pixel Value")
    plt.title("PNG Image")

    plt.subplot(2, 2, 2)
    plt.imshow(txt_image_np, cmap="gray")
    plt.colorbar(label="Photocurrent Value")
    plt.title("TXT Image")

    plt.subplot(2, 2, 3)
    plt.hist(png_image_np.ravel(), bins=256, color="blue", alpha=0.7)
    plt.xlabel("Normalized Pixel Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of PNG Image")

    plt.subplot(2, 2, 4)
    plt.hist(txt_image_np.ravel(), bins=256, color="red", alpha=0.7)
    plt.xlabel("Photocurrent Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of TXT Image")

    plt.suptitle(f"{suptitle}")
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(txt_image_np.ravel(), png_image_np.ravel(), s=2, alpha=0.2)
    plt.xlabel("TXT value")
    plt.ylabel("PNG value")
    plt.title("PNG value vs TXT value")
    plt.grid(True)
    plt.show()


def plot_png_and_txt(
    suptitle: str,
    *,
    png_image_np: np.ndarray,
    txt_image_np: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Plot PNG and TXT images side by side with colorbars."""
    plt.figure(figsize=(9, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(png_image_np, cmap="gray")
    plt.colorbar(label="Normalized Pixel Value")
    plt.title("PNG Image")

    plt.subplot(1, 2, 2)
    plt.imshow(txt_image_np, cmap="gray")
    plt.colorbar(label="Photocurrent Value")
    plt.title("TXT Image")

    plt.suptitle(f"{suptitle}")

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    plt.show()


def plot_subplot(sample_name: str, *, total_samples: int, sample_index: int) -> None:
    png_image_np_uint, txt_image_np, sign_flipped = load_np(sample_name)
    png_image_np = png_image_np_uint.astype(np.float32) / 255.0  # Normalize to [0, 1]

    plt.subplot(total_samples, 2, 2 * sample_index + 1)
    plt.imshow(png_image_np, cmap="gray")
    plt.colorbar(label="Normalized Pixel Value")
    plt.title(f"{sample_name} PNG")

    try:
        plt.subplot(total_samples, 2, 2 * sample_index + 2)
        # plt.imshow(txt_image_np.clip(0), cmap="gray")
        plt.imshow(txt_image_np, cmap="gray")
        plt.colorbar(label="Photocurrent Value")
        plt.title(f"{sample_name} TXT")
    except ValueError as e:
        print(f"Error plotting TXT image for sample {sample_name}: {e}")
        plt.subplot(total_samples, 2, 2 * sample_index + 2)
        plt.text(
            0.5,
            0.5,
            f"Error plotting TXT image:\n{e}",
            ha="center",
            va="center",
            fontsize=10,
        )
        plt.axis("off")


# %%
def load_np(sample_name: str):
    txt_path = sample_paths[sample_name]["txt"]
    png_path = sample_paths[sample_name]["png"]

    png_image_pil = Image.open(png_path).convert("L")
    png_image_np_uint = np.array(png_image_pil)

    txt_image_np = np.loadtxt(txt_path)

    # print(f"TXT: [{txt_image_np.min():.3e}, {txt_image_np.max():.3e}]")

    # Shape it like the PNG image
    try:
        txt_image_np_reshaped = txt_image_np.reshape(png_image_np_uint.shape)
        txt_image_np = txt_image_np_reshaped

        # Check TXT polarity against PNG target.
        png_for_corr = png_image_np_uint.astype(np.float64)
        txt_for_corr = txt_image_np.astype(np.float64)

        txt_for_corr_norm = (txt_for_corr - txt_for_corr.min()) / (
            txt_for_corr.max() - txt_for_corr.min()
        )

        correlation = np.corrcoef(txt_for_corr_norm.ravel(), png_for_corr.ravel())[0, 1]
        # print(f"Correlation between TXT and PNG: {correlation:.4f}")

        sign_flip = correlation < 0
        if sign_flip:
            txt_image_np = -txt_image_np

    except ValueError as e:
        print(f"Error reshaping TXT image: {e}")
        print(f"TXT shape: {txt_image_np.shape}, PNG shape: {png_image_np_uint.shape}")
        sign_flip = False  # Default to no sign flip if reshaping fails

    return png_image_np_uint, txt_image_np, sign_flip


# %%
def make_expected_png(txt: np.ndarray, vmin: float, vmax: float):
    txt_clipped = np.clip(txt, vmin, vmax)
    txt_normalized = (txt_clipped - vmin) / (vmax - vmin)
    # Note: np.rint rounds to the nearest integer
    png = np.rint(txt_normalized * 255).astype(np.uint8)
    return png


def fit_non_saturated(
    *, png_image_np_uint: np.ndarray, txt_image_np: np.ndarray, verbose: bool = False
):
    png = png_image_np_uint.astype(np.float64)
    txt = txt_image_np.astype(np.float64)
    txt_norm = (txt - txt.min()) / (txt.max() - txt.min())
    # Fit only non-saturated PNG pixels.
    # This avoids pixels clipped to 0 or 255.
    mask = (png > 0) & (png < 255)
    if mask.sum() < 2:
        raise ValueError("Not enough non-saturated pixels to fit display map.")

    A = np.column_stack([txt_norm[mask].ravel(), np.ones(mask.sum())])
    a, b = np.linalg.lstsq(A, png[mask].ravel(), rcond=None)[0]

    low_norm = -b / a
    high_norm = (255 - b) / a

    if verbose:
        print("a:", a)
        print("b:", b)
        print("low_norm:", low_norm)
        print("high_norm:", high_norm)

    z = a * txt_norm + b

    target = png_image_np_uint.astype(np.float64)
    z_flat = z.ravel()
    target_flat = target.ravel()

    lower = -np.inf
    upper = np.inf
    eps = 1e-12

    # Pixels that should round to 1, ..., 254.
    mid = (target_flat > 0) & (target_flat < 255)
    if np.any(mid):
        lower = max(lower, np.max(target_flat[mid] - 0.5 - z_flat[mid] + eps))
        upper = min(upper, np.min(target_flat[mid] + 0.5 - z_flat[mid] - eps))
    else:
        raise ValueError(
            "No mid-range pixels found. Cannot adjust bias for exact match."
        )

    # Pixels that should clip/round to 0.
    black = target_flat == 0
    if np.any(black):
        upper = min(upper, np.min(0.5 - z_flat[black] - eps))

    # Pixels that should clip/round to 255.
    white = target_flat == 255
    if np.any(white):
        lower = max(lower, np.max(254.5 - z_flat[white] + eps))

    if lower > upper:
        raise ValueError(
            f"Got lower={lower} > upper={upper}. No exact bias correction found. Need to refit both a and b."
        )

    delta_b = 0.5 * (lower + upper)
    b_exact = b + delta_b

    if verbose:
        print("delta_b for exact match:", delta_b)

    # png_pred = np.rint(a * txt_norm + b_exact)
    # png_pred = np.clip(png_pred, 0, 255).astype(np.uint8)

    low_norm = -b_exact / a
    high_norm = (255 - b_exact) / a

    vmin = txt.min() + low_norm * (txt.max() - txt.min())
    vmax = txt.min() + high_norm * (txt.max() - txt.min())
    return vmin, vmax


# %%
def process_sample(sample_name: str, verbose: bool = False):
    png_image_np_uint, txt_image_np, sign_flipped = load_np(sample_name)
    png_image_np = png_image_np_uint.astype(np.float32) / 255.0  # Normalize to [0, 1]

    if verbose:
        # print(f"Image shape: {png_image_np.shape}")
        # print(f"TXT: [{txt_image_np.min():.3e}, {txt_image_np.max():.3e}]")
        # if sign_flipped:
        #     print("TXT was sign-flipped to match PNG.")
        # else:
        #     print("TXT was kept as-is (no sign flipping).")
        # plot(
        #     f"Sample {sample_name} (before clipping TXT)",
        #     png_image_np=png_image_np,
        #     # txt_image_np=txt_image_np.clip(0),
        #     txt_image_np=txt_image_np,
        # )
        output_dir = Path("png_vs_txt")
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_png_and_txt(
            f"Sample {sample_name} (before clipping TXT)",
            png_image_np=png_image_np,
            txt_image_np=txt_image_np.clip(0),
            # txt_image_np=txt_image_np,
            output_path=output_dir / f"sample_{sample_name}.png",
        )

    # vmin, vmax = fit_non_saturated(
    #     png_image_np_uint=png_image_np_uint, txt_image_np=txt_image_np, verbose=verbose
    # )
    # png_generated_uint = make_expected_png(txt_image_np, vmin, vmax)

    # if verbose:
    #     print(f"vmin: {vmin:.3e}, vmax: {vmax:.3e}")
    #     diff = png_image_np_uint.astype(np.int16) - png_generated_uint.astype(np.int16)
    #     print("max abs diff:", np.abs(diff).max())
    #     print("mean abs diff:", np.abs(diff).mean())
    #     print("number different:", np.count_nonzero(diff))
    #     print("unique diffs:", np.unique(diff))
    # assert np.array_equal(png_image_np_uint, png_generated_uint)
    # print("Exact match after bias correction.")

    # if verbose:
    #     txt_image_np_converted = txt_image_np.clip(vmin, vmax)
    #     plot(
    #         f"Sample {sample_name} (after clipping TXT)",
    #         png_image_np=png_image_np,
    #         txt_image_np=txt_image_np_converted,
    #     )


# Found 23 samples
# Sample names: [
#   '100_CIGS1',
#   '100_P1', '100_P2', '100_P3', '100_P4', '100_P5',
#   '256_P1',
#   '512_CIGS1', '512_CIGS2', '512_CIGS3', '512_CIGS4', '512_CIGS5', '512_CIGS6',
#   '512_OPV1', '512_OPV2', '512_OPV3', '512_OPV4',
#   '512_R1', '512_R2', '512_R3',
#   '512_S1', '512_S2', '512_S3'
# ]

# # %%
# process_sample("100_CIGS1", verbose=True)  # OK, vmin: 3.532e-07, vmax: 4.632e-07

# # %%
# process_sample("100_P1", verbose=True)  # OK, vmin: 4.235e-08, vmax: 2.254e-07

# # # %%
# # process_sample("100_P2", verbose=True)  # Fails

# # # %%
# # process_sample("100_P3", verbose=True)  # Fails

# # %%
# process_sample("100_P4", verbose=True)  # OK, vmin: 1.016e-08, vmax: 1.082e-07

# # %%
# process_sample("100_P5", verbose=True)  # OK, vmin: 2.061e-07, vmax: 2.741e-07

# # # %%
# # process_sample("256_P1", verbose=True)  # Fails

# # # %%
# # process_sample("512_CIGS1", verbose=True)  # Fails

# # # %%
# # process_sample("512_CIGS2", verbose=True)  # Fails


# The failed cases are due to the fact that the PNG images and the TXT images are not aligned.
# They may have been incorrectly paired. Just ignore them.
# I only need a few examples to get an idea of the scaling.


# %%
def process_all_samples(all_samples: list[str] | None = None, verbose: bool = False):
    ok_samples = []
    failed_samples = []
    if all_samples is None:
        all_samples = list(sample_paths.keys())
    # for sample_name in tqdm(all_samples, desc="Processing samples", unit="sample"):
    plt.figure(figsize=(10, 5 * len(all_samples)))
    for sample_name in all_samples:
        # print(f"Processing sample: {sample_name}")
        # Make the {sample_name} a fixed-width string with trailing whitespaces to fill missing characters
        # print(f"{sample_name:<12}   ", end="")
        # try:
        # process_sample(sample_name, verbose=verbose)
        plot_subplot(
            sample_name,
            total_samples=len(all_samples),
            sample_index=all_samples.index(sample_name),
        )
        ok_samples.append(sample_name)
    # except Exception as e:
    #     print(f"Error processing sample {sample_name}: {e}")
    #     failed_samples.append(sample_name)
    plt.title("PNG vs TXT Comparison")
    plt.savefig(f"png_vs_txt/{len(all_samples)}_samples.png", bbox_inches="tight")
    # plt.show()
    print(f"OK samples: {ok_samples}")
    print(f"Failed samples: {failed_samples}")


# %%
process_all_samples(
    [
        "100_CIGS1",  # CIGS
        "512_CIGS1",  # CIGS
        "100_P1",  # Perovskite
        "100_P2",  # Perovskite
        "256_P1",  # Perovskite
        "512_OPV1",  # Organic
        "512_R1",  # Reference
        "512_S1",  # Silicone
    ],
    verbose=True,
)


process_all_samples(verbose=True)
# OK samples: ['100_CIGS1', '100_P1', '100_P4', '100_P5']
# Failed samples: ['100_P2', '100_P3', '256_P1', '512_CIGS1', '512_CIGS2', '512_CIGS3', '512_CIGS4', '512_CIGS5', '512_CIGS6', '512_OPV1', '512_OPV2', '512_OPV3', '512_OPV4', '512_R1', '512_R2', '512_R3', '512_S1', '512_S2', '512_S3']

# 100_CIGS1      TXT: [-4.780e-07, -3.440e-07]

# 100_P1         TXT: [4.100e-08, 2.380e-07]
# 100_P2         TXT: [1.800e-08, 1.210e-07]
# 100_P3         TXT: [2.100e-08, 1.290e-07]
# 100_P4         TXT: [1.000e-08, 1.140e-07]
# 100_P5         TXT: [-2.870e-07, -1.320e-07]

# 256_P1         TXT: [-1.674e-05, -1.466e-05]

# 512_CIGS1      TXT: [-5.077e-04, 6.995e-02]
# 512_CIGS2      TXT: [-1.977e-04, 3.633e-03]
# 512_CIGS3      TXT: [-2.068e-02, 2.839e-03]
# 512_CIGS4      TXT: [-1.778e-02, 2.803e-03]
# 512_CIGS5      TXT: [-3.803e-04, 1.768e-03]
# 512_CIGS6      TXT: [-3.433e-03, 1.795e-03]

# 512_OPV1       TXT: [-5.339e-03, 1.068e-03]
# 512_OPV2       TXT: [-5.962e-03, 1.902e-03]
# 512_OPV3       TXT: [-2.244e-03, 3.235e-03]
# 512_OPV4       TXT: [-1.046e-03, 2.174e-04]

# 512_R1         TXT: [-2.501e-05, 6.429e-06]
# Error processing sample 512_R1: cannot reshape array of size 786432 into shape (512,512)
# 512_R2         TXT: [-1.249e-05, 2.117e-06]
# Error processing sample 512_R2: cannot reshape array of size 786432 into shape (512,512)
# 512_R3         TXT: [-2.534e-05, 5.145e-06]
# Error processing sample 512_R3: cannot reshape array of size 786432 into shape (512,512)

# 512_S1         TXT: [-1.359e-01, 1.364e-01]
# 512_S2         TXT: [-1.540e-02, 1.294e-01]
# 512_S3         TXT: [-1.018e-01, 9.083e-03]

# %%
process_all_samples(["100_CIGS1", "100_P1", "100_P4", "100_P5"], verbose=True)
