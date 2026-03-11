from __future__ import annotations

from pathlib import Path
from turtle import back
import matplotlib.pyplot as plt
import numpy as np
import torch

from spyrit.core.torch import fwht, ifwht

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp


if __name__ == "__main__":
    data_dir = Path("data")

    # measurement_file_dir = data_dir / "Example PCM data"
    # measurement_file_name = (
    #     "GaAs ref cell closer 20ms 16repeat 100mAbiasLED 800mV bias "
    #     "256x256 +390+330.txt"
    # )
    # measurement_file_name = "TEST 512x512.txt"
    # J = 9  # 2^J x 2^J = 512 x 512 map

    measurement_file_dir = data_dir / "photocurrent_data"
    # measurement_file_name = "Si_256.TXT"
    measurement_file_name = "Si_2_256.TXT"
    J = 8  # 2^J x 2^J = 256 x 256 map

    with (measurement_file_dir / measurement_file_name).open(
        "r", encoding="utf-8", errors="ignore"
    ) as f:
        lines = f.readlines()

        # The content of the measurement file looks like this:
        # (1)    Output from Current mapping V2
        # (2)    (blank line)
        # (3)    Pattern Index	Measured current (A)
        # (4)    0.00000000E+0	2.13886190E-5
        # (5)    0.00000000E+0	1.30551160E-7
        # (6)    1.00000000E+0	8.66430485E-6
        # (7)    -1.00000000E+0	1.35798561E-5
        # (8)    2.00000000E+0	1.02335228E-5
        # (9)    -2.00000000E+0	1.20306721E-5

        # So we skip the first 3 header lines and parse from line 4 onwards
        lines = lines[3:]  # skip first 3 header lines
        print(f"Loaded {len(lines)} measurement lines from '{measurement_file_name}'")
        # Each line in the block contains 2 numbers: pattern index and measured current
        # Convert them as a float np.array with shape (num_lines, 2).
        out = np.fromstring("\n".join(lines), sep=" ").reshape((-1, 2))

        name = measurement_file_name.split(".")[0]
        np.save(measurement_file_dir / f"{name}_measurement_data.npy", out)

        # Make a 1D array with num_lines//2 elements,
        # where each element is the sum of the measured current multiplied by the pattern index sign.
        num_measurements = out.shape[0]
        assert num_measurements % 2 == 0, "Number of measurements should be even."

        sign_vector = np.round(np.sign(out[:, 0]))
        sign_vector[:2] = [1.0, -1.0]  # Ensure the first two patterns are +0 and -0

        hadamard_measurement_vector = (
            (out[:, 1] * sign_vector).reshape(-1, 2).sum(axis=1)
        )

        # background_current = out[1, 1]  # Measured current for pattern index -0
        # hadamard_measurement_vector -= background_current

        print(f"hadamard_measurement_vector shape: {hadamard_measurement_vector.shape}")
        print(f"hadamard_measurement_vector dtype: {hadamard_measurement_vector.dtype}")
        np.save(
            measurement_file_dir / f"{name}_hadamard_measurement_vector.npy",
            hadamard_measurement_vector,
        )

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    pcm_op = PhotocurrentMapOp(J=J, device=device)  # 2^Jx2^J map
    hadamard_measurement_vector_torch = torch.tensor(
        hadamard_measurement_vector, dtype=torch.float32, device=device
    )
    reconstructed_image_torch = pcm_op.pseudo_inv(hadamard_measurement_vector_torch)
    reconstructed_image_np = (
        reconstructed_image_torch.cpu().numpy().reshape((2**J, 2**J))
    )
    print(f"reconstructed_image_np shape: {reconstructed_image_np.shape}")
    print(f"reconstructed_image_np dtype: {reconstructed_image_np.dtype}")
    np.save(
        measurement_file_dir / f"{name}_reconstructed_image.npy", reconstructed_image_np
    )
    plt.imshow(reconstructed_image_np, cmap="gray")
    plt.colorbar()
    plt.title(
        f"Reconstructed image from {measurement_file_name}\n"
        f"shape: {reconstructed_image_np.shape}\n"
        f"min: {reconstructed_image_np.min():.6g}, max: {reconstructed_image_np.max():.6g}"
    )
    plt.tight_layout()
    plt.savefig(measurement_file_dir / f"{name}_reconstructed_image.png", dpi=300)
    plt.close()

    spyrit_reconstructed_image_torch = ifwht(
        hadamard_measurement_vector_torch, order=True
    )
    spyrit_reconstructed_image_np = (
        spyrit_reconstructed_image_torch.cpu().numpy().reshape((2**J, 2**J))
    )
    print(f"spyrit_reconstructed_image_np shape: {spyrit_reconstructed_image_np.shape}")
    print(f"spyrit_reconstructed_image_np dtype: {spyrit_reconstructed_image_np.dtype}")
    np.save(
        measurement_file_dir / f"{name}_spyrit_reconstructed_image.npy",
        spyrit_reconstructed_image_np,
    )
    plt.imshow(spyrit_reconstructed_image_np, cmap="gray")
    plt.colorbar()
    plt.title(
        f"Spyrit Reconstructed image from {measurement_file_name}\n"
        f"shape: {spyrit_reconstructed_image_np.shape}\n"
        f"min: {spyrit_reconstructed_image_np.min():.6g}, max: {spyrit_reconstructed_image_np.max():.6g}"
    )
    plt.tight_layout()
    plt.savefig(
        measurement_file_dir / f"{name}_spyrit_reconstructed_image.png", dpi=300
    )
    plt.close()
