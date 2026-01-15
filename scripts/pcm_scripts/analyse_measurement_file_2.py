from __future__ import annotations

from pathlib import Path

import numpy as np


if __name__ == "__main__":
    data_dir = Path("data")
    measurement_file_dir = data_dir / "Example PCM data"
    # measurement_file_name = (
    #     "GaAs ref cell closer 20ms 16repeat 100mAbiasLED 800mV bias "
    #     "256x256 +390+330.txt"
    # )
    measurement_file_name = "TEST 512x512.txt"
    with (measurement_file_dir / measurement_file_name).open("r", encoding="utf-8", errors="ignore") as f:
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

        # Every consecutive pair of rows should have opposite indices and the same absolute value.
        # The order of the two rows in the pair may vary.
        # Let's divide into blocks of pairs where the next pair's absolute index increases by 1 each time.
        blocks: list[tuple[int, int, int]] = []
        cur_start = 0
        cur_start_absolute_index = abs(int(out[0, 0]))
        prev_absolute_index = -1

        block_lengths: dict[int, int] = {}

        for i in range(len(out) // 2):
            index_1 = round(out[2 * i, 0])
            index_2 = round(out[2 * i + 1, 0])
            assert index_1 == -index_2, (
                f"Pattern indices at rows {2*i} and {2*i+1} are not opposites: "
                f"{index_1} vs {index_2}"
            )
            absolute_index = abs(index_1)
            if i != 0 and absolute_index != prev_absolute_index + 1:
                block_length = i - cur_start
                blocks.append((cur_start, cur_start_absolute_index, block_length))
                cur_start = i
                cur_start_absolute_index = absolute_index
                block_lengths[block_length] = block_lengths.get(block_length, 0) + 1
                # break
            prev_absolute_index = absolute_index
        block_length = len(out) // 2 - cur_start
        blocks.append((cur_start, cur_start_absolute_index, block_length))
        block_lengths[block_length] = block_lengths.get(block_length, 0) + 1
        print(f"Found {len(blocks)} blocks of pattern index pairs with increasing absolute indices.")
        print("Block lengths and their counts:")
        for length, count in sorted(block_lengths.items()):
            print(f"  Length {length}: {count} blocks")
        # for block_index, (start_pair_index, start_absolute_index, block_length) in enumerate(blocks):
        #     print(
        #         f"Block {block_index}: "
        #         f"start from index {start_pair_index} "
        #         f"with absolute value {start_absolute_index}, "
        #         f"length {block_length} pairs."
        #     )