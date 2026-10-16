"""Compare PnP-ADMM iteration metrics with realistic PCM noise and no noise."""

from LION.pcm.pnp_iteration_metrics import (
    parse_pnp_iteration_metrics_cli,
    run_pnp_iteration_metrics_analysis,
)


def main() -> None:
    """Run the PnP-ADMM iteration-metric comparison."""
    comparison_config, experiment_config = parse_pnp_iteration_metrics_cli()
    run_pnp_iteration_metrics_analysis(comparison_config, experiment_config)


if __name__ == "__main__":
    main()
