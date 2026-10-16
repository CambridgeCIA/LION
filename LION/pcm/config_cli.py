"""CLI configuration parsing for PCM experiment."""

from __future__ import annotations

import sys

import tyro
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from LION.experiments.pcm_experiments import PCMExperimentParameter
from LION.experiments.pcm_presets.presets import PRESETS


def show_preset_help() -> None:
    """Show help message about using presets and exit."""
    preset_names = "\n".join(f"  - {name}" for name in sorted(PRESETS))

    body = Text()
    body.append("--preset [PRESET_NAME]\n", style="bold cyan")
    body.append(
        (
            "      Use a saved ExperimentConfig as the base configuration.\n"
            "      Without --preset, every field marked '(required)' below must be passed.\n"
            "      Any explicitly passed flag overrides the preset value.\n\n"
        ),
        style="cyan",
    )
    body.append("Available preset names:\n", style="bold")
    body.append(preset_names)

    Console().print(
        Panel(
            body, title="[bold red]NOTE: Preset support[/bold red]", border_style="red"
        )
    )


def pop_preset_arg(argv: list[str]) -> tuple[str | None, list[str]]:
    """Pop the --preset argument from argv, if present, and return the preset name and remaining args."""
    preset_name: str | None = None
    remaining: list[str] = []

    i = 0
    while i < len(argv):
        arg_name = argv[i]

        if arg_name == "--preset":
            if i + 1 >= len(argv):
                raise SystemExit("Error: --preset requires a value.")
            if preset_name is not None:
                raise SystemExit("Error: --preset was passed more than once.")
            preset_name = argv[i + 1]
            i += 2
            continue

        if arg_name.startswith("--preset="):
            if preset_name is not None:
                raise SystemExit("Error: --preset was passed more than once.")
            preset_name = arg_name.split("=", 1)[1]
            i += 1
            continue

        remaining.append(arg_name)
        i += 1

    if preset_name is not None and preset_name not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise SystemExit(
            f"Error: invalid preset {preset_name!r}. Valid presets are: {valid}."
        )

    return preset_name, remaining


def parse_pcm_experiment_config_cli(
    argv: list[str] | None = None,
) -> PCMExperimentParameter:
    """Parse command-line arguments into a custom dataclass instance."""
    if argv is None:
        argv = sys.argv[1:]

    if "-h" in argv or "--help" in argv:
        show_preset_help()

    preset_name, remaining = pop_preset_arg(argv)

    preset_help_reminder = (
        "    NOTE: Preset support is available through `--preset`! See details at the top.\n"
        "    If you want to use a preset as the base configuration, pass `--preset PRESET_NAME`.\n"
        "    Without --preset, every field marked '(required)' below must be passed.\n"
    )

    base_description = (PCMExperimentParameter.__doc__ or "").strip()
    description = f"{base_description}\n\n{preset_help_reminder}"

    if preset_name is None:
        return tyro.cli(
            PCMExperimentParameter,
            args=remaining,
            description=description,
            config=(tyro.conf.CascadeSubcommandArgs,),
        )

    return tyro.cli(
        PCMExperimentParameter,
        args=remaining,
        default=PRESETS[preset_name],
        description=description,
        config=(tyro.conf.CascadeSubcommandArgs,),
    )
