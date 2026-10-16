# %% Imports
from pathlib import Path

import numpy as np
from PIL import Image

# %%
pcm_input_data_dir = Path(
    "../../data/PCM Synthetic Data Set/Synthetic Data/Input Images"
)
assert pcm_input_data_dir.exists(), f"Directory {pcm_input_data_dir} does not exist."
raw_txt_dir = pcm_input_data_dir / "txt"
normalised_png_dir = pcm_input_data_dir / "png"

# %%
txt_paths = list(raw_txt_dir.glob("*.txt"))
txt_paths.sort()
png_paths = list(normalised_png_dir.glob("*.png"))
png_paths.sort()
txt_stems = [path.stem for path in txt_paths]
png_stems = [path.stem for path in png_paths]
assert txt_stems == png_stems, "Mismatch between txt and png files."
print(f"Found {len(txt_stems)} samples")

sample_paths = {
    sample_name: {"txt": txt_path, "png": png_path}
    for sample_name, txt_path, png_path in zip(txt_stems, txt_paths, png_paths)
}


# %%
def find_vmin_vmax(sample_name: str):
    txt_path = sample_paths[sample_name]["txt"]
    png_path = sample_paths[sample_name]["png"]

    png_image_pil = Image.open(png_path)
    png_image_np_uint = np.array(png_image_pil)
    png_image_np = png_image_np_uint.astype(np.float32) / 255.0  # Normalize to [0, 1]

    txt_image_np = np.loadtxt(txt_path)
    # Shape it like the PNG image
    txt_image_np = txt_image_np.reshape(png_image_np.shape)

    txt_image_np_normalized = (txt_image_np - txt_image_np.min()) / (
        txt_image_np.max() - txt_image_np.min()
    )
    txt_image_np_uint_round = np.rint(txt_image_np_normalized * 255).astype(np.uint8)
    diff = png_image_np_uint.astype(np.int16) - txt_image_np_uint_round.astype(np.int16)

    png = png_image_np_uint.astype(np.float64)
    txt = txt_image_np.astype(np.float64)

    txt_norm = (txt - txt.min()) / (txt.max() - txt.min())

    # Fit: png ≈ a * txt_norm + b
    A = np.column_stack([txt_norm.ravel(), np.ones(txt_norm.size)])
    a, b = np.linalg.lstsq(A, png.ravel(), rcond=None)[0]

    png_pred = np.rint(a * txt_norm + b)
    png_pred = np.clip(png_pred, 0, 255).astype(np.uint8)

    diff = png_image_np_uint.astype(np.int16) - png_pred.astype(np.int16)

    png = png_image_np_uint.astype(np.float64)
    txt = txt_image_np.astype(np.float64)

    txt_min = txt.min()
    txt_max = txt.max()
    txt_range = txt_max - txt_min

    txt_norm = (txt - txt_min) / txt_range

    # Fit only non-saturated PNG pixels.
    # This avoids pixels clipped to 0 or 255.
    mask = (png > 0) & (png < 255)

    A = np.column_stack([txt_norm[mask].ravel(), np.ones(mask.sum())])
    a, b = np.linalg.lstsq(A, png[mask].ravel(), rcond=None)[0]

    low_norm = -b / a
    high_norm = (255 - b) / a

    vmin_est = txt_min + low_norm * txt_range
    vmax_est = txt_min + high_norm * txt_range

    png_pred = np.rint(a * txt_norm + b)
    png_pred = np.clip(png_pred, 0, 255).astype(np.uint8)

    diff = png_image_np_uint.astype(np.int16) - png_pred.astype(np.int16)

    assert np.abs(diff).max() == 0

    return vmin_est, vmax_est


# %%
for sample_name in sample_paths:
    vmin_est, vmax_est = find_vmin_vmax(sample_name)
    print(f"Sample: {sample_name}, vmin_est: {vmin_est:.3e}, vmax_est: {vmax_est:.3e}")

# %%
