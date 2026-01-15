from __future__ import annotations

from pathlib import Path

import numpy as np


if __name__ == "__main__":
    data_dir = Path("data")
    measurement_file_dir = data_dir / "Example PCM data"
    measurement_file_name = (
        "GaAs ref cell closer 20ms 16repeat 100mAbiasLED 800mV bias "
        "256x256 +390+330.txt"
    )
    # measurement_file_name = "TEST 512x512.txt"
    with (measurement_file_dir / measurement_file_name).open("r", encoding="utf-8", errors="ignore") as f:

        measurements: list[tuple[int, float]] = []
        indices: list[int] = []
        measured_currents_ampere: list[float] = []

        line_index = 0
        for line in f:
            line_index += 1  # 1-based line index
            if line_index < 4:
                continue
            pattern_index_str, measured_current_ampere_str = line.split()
            pattern_index: int = round(float(pattern_index_str))
            measured_current_ampere: float = float(measured_current_ampere_str)
            measurements.append((pattern_index, measured_current_ampere))
            indices.append(pattern_index)
            measured_currents_ampere.append(measured_current_ampere)

    print(f"Loaded {len(measurements)} measurements from '{measurement_file_name}'")

    indices_np = np.array(indices, dtype=np.int64)

    print(f"indices_np shape: {indices_np.shape}")
    print(f"indices_np dtype: {indices_np.dtype}")
    print(f"indices_np min: {indices_np.min()}")
    print(f"indices_np max: {indices_np.max()}")
    print(f"indices_np unique count: {np.unique(indices_np).size}")

    counts: dict[int, int] = {}
    for index in indices:
        counts[index] = counts.get(index, 0) + 1
    reversed_counts: dict[int, list[int]] = {}
    for index, count in counts.items():
        if count not in reversed_counts:
            reversed_counts[count] = []
        reversed_counts[count].append(index)
    print(f"counts: {reversed_counts.keys()}")
    for count in sorted(reversed_counts.keys()):
        print(f"indices with count {count}: ", end="")
        if len(reversed_counts[count]) <= 5:
            print(reversed_counts[count])
        else:
            print(f"{len(reversed_counts[count])} indices")

    measured_currents_ampere_np = np.array(measured_currents_ampere, dtype=np.float64)
