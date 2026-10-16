"""Script to run a PCM dataset experiment."""

from __future__ import annotations

from LION.pcm.config_cli import parse_pcm_experiment_config_cli
from LION.pcm.run_dataset_experiment import run_dataset_experiment


def main() -> None:
    """Run the selected PCM experiment preset.

    Example usage:
    - To see all options:
        ```
        python scripts/pcm/<this_script_name>.py --help
        ```
    - Example of running a preset experiment:
        ```
        python scripts/pcm/<this_script_name>.py --preset synthetic_png_dataset
        ```
    """
    experiment_config = parse_pcm_experiment_config_cli()
    experiment_log_dir = run_dataset_experiment(experiment_config)
    print(f"Results written to: {experiment_log_dir}")


if __name__ == "__main__":
    main()
