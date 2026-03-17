from __future__ import annotations

import sys

import tyro

from LION.pcm.config import PRESETS, ExperimentConfig

PRESET_HELP = (
    "Preset support\n"
    "--preset {" + ", ".join(PRESETS) + "}\n"
    "    Use a saved ExperimentConfig as the base configuration.\n"
    "    Without --preset, every field marked '(required)' below must be passed.\n"
    "    Any explicitly passed flag overrides the preset value."
)


def pop_preset_arg(argv: list[str]) -> tuple[str | None, list[str]]:
    """
    Remove ``--preset NAME`` or ``--preset=NAME`` from ``argv``.
    """
    preset_name: str | None = None
    remaining: list[str] = []

    i = 0
    while i < len(argv):
        token = argv[i]

        if token == "--preset":
            if i + 1 >= len(argv):
                raise SystemExit("Error: --preset requires a value.")
            if preset_name is not None:
                raise SystemExit("Error: --preset was passed more than once.")
            preset_name = argv[i + 1]
            i += 2
            continue

        if token.startswith("--preset="):
            if preset_name is not None:
                raise SystemExit("Error: --preset was passed more than once.")
            preset_name = token.split("=", 1)[1]
            i += 1
            continue

        remaining.append(token)
        i += 1

    if preset_name is not None and preset_name not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise SystemExit(
            f"Error: invalid preset {preset_name!r}. " f"Valid presets are: {valid}."
        )

    return preset_name, remaining


def parse_experiment_config(argv: list[str] | None = None) -> ExperimentConfig:
    """
    Parse CLI arguments into an ``ExperimentConfig``.
    """
    if argv is None:
        argv = sys.argv[1:]

    preset_name, remaining = pop_preset_arg(argv)

    base_description = (ExperimentConfig.__doc__ or "").strip()
    description = f"{base_description}\n\n{PRESET_HELP}"

    if preset_name is None:
        return tyro.cli(
            ExperimentConfig,
            args=remaining,
            description=description,
        )

    return tyro.cli(
        ExperimentConfig,
        args=remaining,
        default=PRESETS[preset_name],
        description=description,
    )


if __name__ == "__main__":
    cfg = parse_experiment_config()
    print(cfg)
