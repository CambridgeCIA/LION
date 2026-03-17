# %% [markdown]
# # Image reconstruction for Photocurrent Mapping
#
# This script is the script-first entry point for the PCM experiments.
# It can still be executed cell-by-cell in editors that understand ``# %%``.

# %%
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LION.pcm.config import PRESETS, get_preset_config
from LION.pcm.experiment import override_device, run_experiment


# %%
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Photocurrent Mapping reconstruction experiments.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="si_2_256_512x512_image",
        help="Name of the experiment preset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device override, for example 'cpu', 'cuda', 'cuda:0', or 'mps'.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List the available preset names and exit.",
    )
    return parser.parse_args()


# %%
def main() -> None:
    """Run the selected PCM experiment preset."""
    args = parse_args()

    if args.list_presets:
        for preset_name in sorted(PRESETS):
            print(preset_name)
        return

    config = get_preset_config(args.preset)
    config = override_device(config, args.device)
    output_dir = run_experiment(config)
    print(f"Results written to: {output_dir}")


# %%
if __name__ == "__main__":
    main()
