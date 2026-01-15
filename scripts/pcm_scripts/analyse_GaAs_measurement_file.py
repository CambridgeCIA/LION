from __future__ import annotations

from pathlib import Path

import numpy as np


if __name__ == "__main__":
    data_dir = Path("data")
    measurement_file_name = (
        "GaAs ref cell closer 20ms 16repeat 100mAbiasLED 800mV bias "
        "256x256 +390+330.txt"
    )
    output_measurement_file_name_only = "GaAs"

    example_pcm_data_dir = data_dir / "Example PCM data"

    # Map image size is 2^order_row x 2^order_col.
    # We have 2^order_row * 2^order_col * 2 measurements (positive and negative patterns).
    order_row = 8
    order_col = 8
    num_measurements_expected = (2 ** order_row) * (2 ** order_col) * 2
    print(f"Expecting {num_measurements_expected} measurements for {2**order_row}x{2**order_col} map.")
    with (example_pcm_data_dir / measurement_file_name).open("r", encoding="utf-8", errors="ignore") as f:
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
        num_lines = len(lines)
        # The file should contain a multiple of num_measurements_expected lines
        assert num_lines % num_measurements_expected == 0, (
            "Unexpected number of lines. "
            f"Should be a multiple of {num_measurements_expected}, but got {num_lines}"
        )
        num_blocks = num_lines // num_measurements_expected
        print(f"File contains {num_lines} lines, which is {num_blocks} blocks of {num_measurements_expected} measurements each.")
        for block_index in range(num_blocks):
            line_block_start = block_index * num_measurements_expected
            print(f"Processing block {block_index} with lines {line_block_start} to {line_block_start + num_measurements_expected - 1}...")
            block = lines[line_block_start:line_block_start + num_measurements_expected]

            # Each line in the block contains 2 numbers: pattern index and measured current
            # Let's save them as a float np.array with shape (num_measurements_expected, 2)
            out = np.fromstring("\n".join(block), sep=" ").reshape((-1, 2))
            assert out.shape == (num_measurements_expected, 2), (
                f"Unexpected shape of parsed data: {out.shape}, "
                f"expected ({num_measurements_expected}, 2)"
            )
            # The first row should contain index 0
            assert out[0, 0] == 0.0, f"First pattern index in block is not 0, got {out[0, 0]}"
            # Every consecutive pair of rows should have opposite indices
            # and the same absolute value, with the pair's absolute index increasing by 1 each time.
            # The order of the two rows in the pair may vary.
            for i in range(num_measurements_expected // 2):
                index_1 = out[2 * i, 0]
                index_2 = out[2 * i + 1, 0]
                print(f"Pair {i}: indices {index_1}, {index_2}")
                pos_index = max(index_1, index_2)
                neg_index = min(index_1, index_2)
                expected_abs_index = float(i)
                assert abs(pos_index) == abs(neg_index) == expected_abs_index, (
                    f"Pattern index pair at rows {2*i} and {2*i+1} do not match expected absolute index {expected_abs_index}, "
                    f"got {index_1} at row {2*i} and {index_2} at row {2*i+1}"
                )
