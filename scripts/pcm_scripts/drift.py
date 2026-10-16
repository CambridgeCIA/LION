from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from LION.pcm_tools.noise_utils import read_pcm_measurement_data_from_txt


def plot_summed_measurements(
    inverted_summed_data: np.ndarray, data_name: str, output_path: Path
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(inverted_summed_data)), inverted_summed_data)
    plt.xlabel("Measurement index (sum of positive and negative patterns)")
    plt.ylabel("Current (A)")
    plt.title(f"Drift of measured current for {data_name} over time")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(output_path)
    print(f"Saved drift plot to '{output_path}'")


def plot_drift():
    data_dir = Path("/store/LION/ttv22/pcm/data/photocurrent_data")
    data_name = "Si_2_256"
    # data_name = "Si_256"
    raw_data_path = data_dir / f"{data_name}_measurement_data.npy"

    if raw_data_path.suffix == ".npy":
        measurements = np.load(raw_data_path)
    elif raw_data_path.suffix == ".txt":
        # shape (N, 2): first column is index of Hadamard pattern, second column is measured current in Amperes
        measurements_df = read_pcm_measurement_data_from_txt(raw_data_path)
        measurements = measurements_df.to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Unsupported data file format: {raw_data_path.suffix}")

    # Add every two consecutive measurements
    summed_data = measurements[::2, 1] + measurements[1::2, 1]  # shape (N/2,)
    num_summed_measurements = summed_data.shape[0]
    print(
        f"Loaded {num_summed_measurements} summed measurements from '{raw_data_path}'"
    )

    inverted_summed_data = -summed_data

    output_path = data_dir / f"{data_name}_drift_plot_4.png"

    plot_summed_measurements(inverted_summed_data, data_name, output_path)

    first_measurement = measurements[0, 1]
    diff_from_first = first_measurement - measurements[2::2, 1]

    plot_summed_measurements(
        diff_from_first,
        f"{data_name} (diff from first measurement)",
        data_dir / f"{data_name}_diff_from_first_measurement_plot_1.png",
    )


if __name__ == "__main__":
    plot_drift()
