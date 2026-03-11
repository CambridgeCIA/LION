from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_drift():
    data_dir = Path("data/photocurrent_data")
    # data_name = "Si_2_256"
    data_name = "Si_256"
    file_path = data_dir / f"{data_name}_measurement_data.npy"
    data = np.load(
        file_path
    )  # shape (N, 2): first column is index of Hadamard pattern, second column is measured current in Amperes
    # Add every two consecutive measurements
    summed_data = data[::2, 1] + data[1::2, 1]  # shape (N/2,)
    num_measurements = summed_data.shape[0]
    print(f"Loaded {num_measurements} summed measurements from '{file_path}'")

    inverted_summed_data = -summed_data
    plt.plot(np.arange(num_measurements), inverted_summed_data)
    plt.xlabel("Measurement index (sum of positive and negative patterns)")
    plt.ylabel("Current (A)")
    plt.title(f"Drift of measured current for {data_name} over time")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(data_dir / f"{data_name}_drift_plot.png")


if __name__ == "__main__":
    plot_drift()
