"""Make paired box-plot CSVs by adding scalar TV from a separate dataset run."""

from __future__ import annotations

import argparse
from pathlib import Path

from LION.pcm.box_result_merge import merge_scalar_tv_box_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--scalar-tv-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = merge_scalar_tv_box_results(
        args.reference_run_dir, args.scalar_tv_run_dir, args.output_dir
    )
    print(f"Merged paired CSVs written to: {result}")


if __name__ == "__main__":
    main()
