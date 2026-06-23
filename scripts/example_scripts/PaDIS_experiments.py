"""Run named PaDIS reconstruction presets with separated result namespaces."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import pathlib
import subprocess
import sys

from LION.utils.paths import LION_EXPERIMENTS_PATH


@dataclass(frozen=True)
class ReconstructionPreset:
    implementation: str
    experiment: str
    description: str
    arguments: tuple[str, ...]


PRESETS = {
    "paper-fan-180": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeamCTRecon",
        description="Noise-free PaDIS fan beam with LION's LIDC geometry and 180 views.",
        arguments=(
            "--num-steps",
            "30",
            "--inner-steps",
            "6",
            "--sigma-min",
            "0.003",
            "--sigma-max",
            "0.05",
            "--zeta",
            "0.3",
            "--langevin-noise-scale",
            "0.25",
            "--data-consistency-scale",
            "5.0",
            "--initial-reconstruction",
            "fdk",
            "--clip-denoised",
        ),
    ),
    "whole-paper-fan-180": ReconstructionPreset(
        implementation="lion-whole-image",
        experiment="PaDISFanBeamCTRecon",
        description="Noise-free whole-image diffusion fan beam with LION's LIDC geometry and 180 views.",
        arguments=(
            "--prior-mode",
            "whole-image",
            "--num-steps",
            "30",
            "--inner-steps",
            "6",
            "--sigma-min",
            "0.003",
            "--sigma-max",
            "0.05",
            "--zeta",
            "0.3",
            "--langevin-noise-scale",
            "0.25",
            "--data-consistency-scale",
            "5.0",
            "--initial-reconstruction",
            "fdk",
            "--clip-denoised",
        ),
    ),
    "lion-clinical": ReconstructionPreset(
        implementation="lion-native",
        experiment="clinicalCTRecon",
        description="LION clinical-dose 360-view fan-beam experiment.",
        arguments=(
            "--num-steps",
            "30",
            "--inner-steps",
            "6",
            "--sigma-min",
            "0.003",
            "--sigma-max",
            "0.05",
            "--langevin-noise-scale",
            "0.25",
            "--data-consistency-scale",
            "5.0",
            "--clip-denoised",
        ),
    ),
    "lion-low-dose": ReconstructionPreset(
        implementation="lion-native",
        experiment="LowDoseCTRecon",
        description="LION low-dose 360-view fan-beam experiment.",
        arguments=(
            "--num-steps",
            "30",
            "--inner-steps",
            "6",
            "--sigma-min",
            "0.003",
            "--sigma-max",
            "0.05",
            "--langevin-noise-scale",
            "0.25",
            "--data-consistency-scale",
            "5.0",
            "--clip-denoised",
        ),
    ),
    "lion-sparse-50": ReconstructionPreset(
        implementation="lion-native",
        experiment="SparseAngleCTRecon",
        description="LION clinical-dose 50-view fan-beam experiment.",
        arguments=(
            "--num-steps",
            "100",
            "--inner-steps",
            "2",
            "--sigma-min",
            "0.003",
            "--sigma-max",
            "0.05",
            "--langevin-noise-scale",
            "0.15",
            "--data-consistency-scale",
            "3.0",
            "--clip-denoised",
        ),
    ),
    "lion-sparse-low-dose-50": ReconstructionPreset(
        implementation="lion-native",
        experiment="SparseAngleLowDoseCTRecon",
        description="LION low-dose 50-view fan-beam experiment.",
        arguments=(
            "--num-steps",
            "50",
            "--inner-steps",
            "4",
            "--sigma-min",
            "0.003",
            "--sigma-max",
            "0.2",
            "--langevin-noise-scale",
            "0.1",
            "--data-consistency-scale",
            "5.0",
            "--clip-denoised",
        ),
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List available reconstruction presets.")

    run_parser = subparsers.add_parser("run", help="Run one named preset.")
    run_parser.add_argument("preset", choices=tuple(PRESETS))
    run_parser.add_argument("--checkpoint", type=pathlib.Path, default=None)
    run_parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=LION_EXPERIMENTS_PATH / "PaDIS" / "reconstruction_presets",
    )
    run_parser.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    run_parser.add_argument("--start-index", type=int, default=0)
    run_parser.add_argument("--max-samples", type=int, default=1)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--dry-run", action="store_true")
    return parser


def command_for(args, passthrough: list[str]) -> tuple[list[str], pathlib.Path]:
    preset = PRESETS[args.preset]
    engine = pathlib.Path(__file__).with_name("PaDIS_LIDC_reconstruction.py")
    output_folder = args.output_root / preset.implementation / args.preset
    command = [
        sys.executable,
        "-u",
        str(engine),
        "--experiment",
        preset.experiment,
        "--output-folder",
        str(output_folder),
        "--split",
        args.split,
        "--start-index",
        str(args.start_index),
        "--max-samples",
        str(args.max_samples),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        *preset.arguments,
    ]
    if args.checkpoint is not None:
        command.extend(("--checkpoint", str(args.checkpoint)))
    command.extend(passthrough)
    return command, output_folder


def main() -> None:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]
    if args.command == "list":
        if passthrough:
            parser.error(f"Unexpected arguments: {' '.join(passthrough)}")
        for name, preset in PRESETS.items():
            print(f"{name:28} [{preset.implementation}] {preset.description}")
        return

    command, output_folder = command_for(args, passthrough)
    print("Running:", " ".join(command))
    if args.dry_run:
        return

    output_folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "preset": args.preset,
        "preset_config": asdict(PRESETS[args.preset]),
        "command": command,
        "passthrough_overrides": passthrough,
    }
    with open(output_folder / "launcher_manifest.json", "w") as file:
        json.dump(manifest, file, indent=2)
    with open(output_folder / "run.log", "a", buffering=1) as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


if __name__ == "__main__":
    main()
