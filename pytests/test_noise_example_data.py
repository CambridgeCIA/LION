from pathlib import Path

import numpy as np
import torch
from LION.data_loaders.pcm.data_utils import (
    read_pcm_current_map_from_txt,
    read_pcm_measurement_data_from_txt,
)
from LION.experiments.pcm_experiments import PCMNoiseSimulation
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm_tools.build_measurement import build_real_measurement_vector
from matplotlib import pyplot as plt


def test_compare_noise_txt_and_map_files() -> None:
    """Test that the example data is consistent with the noise simulation."""
    example_dir = Path("example_data/E27/pos1")
    noise_simulation_params = PCMNoiseSimulation.npl_default(
        current_to_model_scale=1.0, j_order=10
    )
    # device = "cuda:0"
    device = "cpu"

    # Read ground truth and noisy measurement
    measurements_df = read_pcm_measurement_data_from_txt(
        noise_simulation_params.noisy_measurement_filepath(example_dir)
    )
    measurements_np = measurements_df.to_numpy(dtype=np.float64)

    y_from_file_np = build_real_measurement_vector(measurements_np)
    y_from_file = torch.as_tensor(y_from_file_np, dtype=torch.float64, device=device)

    x_from_file = read_pcm_current_map_from_txt(
        # noise_simulation_params.ground_truth_filepath(example_dir),
        example_dir / "E27-full-map-1-default-settings.txt.map",
        device=device,
        dtype=torch.float64,
    )

    x_from_file = x_from_file / 64.0

    pcm_op = PhotocurrentMapOp(j_order=10, device=device, normalized=False)

    x_computed = pcm_op.pseudo_inv(y_from_file)

    assert tuple(y_from_file.shape) == (1048576,)
    assert tuple(x_computed.shape) == (1024, 1024)
    assert tuple(x_from_file.shape) == (768, 1024)

    x_computed_cropped = x_computed[:768, :1024]

    min_abs_x_from_file = x_from_file.abs().min().item()
    max_abs_x_from_file = x_from_file.abs().max().item()
    min_abs_x_computed = x_computed_cropped.abs().min().item()
    max_abs_x_computed = x_computed_cropped.abs().max().item()

    print(
        f"min_abs_x_from_file: {min_abs_x_from_file}, max_abs_x_from_file: {max_abs_x_from_file}"
    )
    print(
        f"min_abs_x_computed: {min_abs_x_computed}, max_abs_x_computed: {max_abs_x_computed}"
    )

    diff = torch.abs(x_computed_cropped - x_from_file)

    print("max absolute difference:", diff.max().item())
    print("mean absolute difference:", diff.mean().item())

    idx = torch.argmax(diff)
    print("worst computed:", x_computed_cropped.flatten()[idx].item())
    print("worst from file:", x_from_file.flatten()[idx].item())

    factor = x_computed_cropped / x_from_file
    print("min factor:", factor.min().item())
    print("max factor:", factor.max().item())

    factor_int = np.round(factor.cpu().numpy()).astype(int)
    unique_factors, counts = np.unique(factor_int, return_counts=True)
    counts_dict = dict(zip(unique_factors, counts))
    print("Unique factors and their counts:")
    for uf, count in zip(unique_factors, counts):
        print(f"Factor: {uf}, Count: {count}")
    assert counts_dict[1] == 786418  # Most pixels are equal

    # plt.figure(figsize=(12, 6))
    # plt.hist(factor.flatten().cpu().numpy(), bins=1000)
    # plt.title("Histogram of factor (computed / from file)")
    # plt.xlabel("Factor")
    # plt.ylabel("Count")
    # plt.grid()
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(x_computed_cropped.cpu().numpy(), cmap="gray")
    # plt.title("Computed from measurement vector")
    # plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.imshow(x_from_file.cpu().numpy(), cmap="gray")
    # plt.title("Read from ground truth file")
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()

    # assert torch.allclose(x_computed_cropped, x_from_file / 64.0, rtol=1e-8, atol=1e-8)
    # assert torch.allclose(x_computed_cropped, x_from_file, rtol=1e-18, atol=1e-18)


if __name__ == "__main__":
    test_compare_noise_txt_and_map_files()
