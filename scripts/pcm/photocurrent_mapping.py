from __future__ import annotations

from LION.pcm.config import parse_experiment_config
from LION.pcm.experiment import run_experiment


def main() -> None:
    """Run the selected PCM experiment preset."""
    config = parse_experiment_config()
    output_dir = run_experiment(config)
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
