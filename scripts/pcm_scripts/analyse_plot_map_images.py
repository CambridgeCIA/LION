from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_and_plot_map_image(
    example_pcm_data_dir: Path, map_image_file_name: str
) -> None:
    name = map_image_file_name.split(".")[0]
    map_image_file = example_pcm_data_dir / map_image_file_name
    print(f"Loading map image from {map_image_file} ...")

    map_image_np = np.loadtxt(map_image_file)

    # Strip the last row until the last row is not all zeros
    while map_image_np.shape[0] > 0 and np.all(map_image_np[-1, :] == 0):
        map_image_np = map_image_np[:-1, :]
    # Strip the last column until the last column is not all zeros
    while map_image_np.shape[1] > 0 and np.all(map_image_np[:, -1] == 0):
        map_image_np = map_image_np[:, :-1]
    # Strip the first row until the first row is not all zeros
    while map_image_np.shape[0] > 0 and np.all(map_image_np[0, :] == 0):
        map_image_np = map_image_np[1:, :]
    # Strip the first column until the first column is not all zeros
    while map_image_np.shape[1] > 0 and np.all(map_image_np[:, 0] == 0):
        map_image_np = map_image_np[:, 1:]

    print(f"{name} shape: {map_image_np.shape}")
    print(f"{name} data type: {map_image_np.dtype}")
    print(f"{name} min: {map_image_np.min()}")
    print(f"{name} max: {map_image_np.max()}")
    np.save(example_pcm_data_dir / f"{name}.npy", map_image_np)

    plt.imshow(map_image_np, cmap="gray")
    plt.colorbar()
    plt.title(
        f"Loaded map image from {map_image_file_name}\n"
        f"shape: {map_image_np.shape}\n"
        f"min: {map_image_np.min():.6g}, max: {map_image_np.max():.6g}"
    )
    plt.tight_layout()
    plt.savefig(example_pcm_data_dir / f"{name}.png", dpi=300)
    plt.close()

    clipped_map_image_np = map_image_np.clip(max=0)
    plt.imshow(clipped_map_image_np, cmap="gray")
    plt.colorbar()
    plt.title(
        f"Loaded map image from {map_image_file_name} and clipped\n"
        f"shape: {clipped_map_image_np.shape}\n"
        f"min: {clipped_map_image_np.min():.6g}, max: {clipped_map_image_np.max():.6g}"
    )
    plt.tight_layout()
    plt.savefig(example_pcm_data_dir / f"{name}_clipped.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    data_dir = Path("data")
    measurement_file_name = (
        "GaAs ref cell closer 20ms 16repeat 100mAbiasLED 800mV bias "
        "256x256 +390+330.txt"
    )
    output_measurement_file_name_only = "GaAs"

    example_pcm_data_dir = data_dir / "Example PCM data"
    load_and_plot_map_image(example_pcm_data_dir, "output.txt")

    photocurrent_data_dir = data_dir / "photocurrent_data"
    load_and_plot_map_image(photocurrent_data_dir, "Si_256.TXT.map")
    load_and_plot_map_image(photocurrent_data_dir, "Si_2_256.TXT.map")
