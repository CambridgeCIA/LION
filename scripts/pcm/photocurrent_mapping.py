from __future__ import annotations

from LION.pcm.config import parse_experiment_config
from LION.pcm.run_experiment import run_experiment


def main() -> None:
    """Run the selected PCM experiment preset.

    Example usage:
    - To see all options:
        ```
        python scripts/pcm/photocurrent_mapping.py --help
        ```
    - Example of running a preset experiment:
        ```
        python scripts/pcm/photocurrent_mapping.py --preset cigs_example_256x256
        ```
    """
    config = parse_experiment_config()
    output_dir = run_experiment(config)
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
