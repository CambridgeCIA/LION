#!/usr/bin/env python3
"""Run the compact validation tuning scheme used by PaDIS-Reproduction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import pathlib
import subprocess
import sys


HERE = pathlib.Path(__file__).resolve().parent
TUNER = HERE / "PaDIS_tune_reconstruction_hyperparameters.py"


@dataclass(frozen=True)
class Sweep:
    """Named, reproducible subset of the reconstruction tuning design."""

    name: str
    models: str
    method: str
    implementation: str
    experiments: str
    group: str
    candidates: tuple[str, ...]


def names(prefix: str, values: tuple[float, ...], suffix: str = "") -> tuple[str, ...]:
    def safe(value: float) -> str:
        return f"{value:g}".replace("-", "_").replace(".", "p")

    return tuple(f"{prefix}{safe(value)}{suffix}" for value in values)


SWEEPS = (
    Sweep(
        "cp",
        "patch_lidc_default",
        "cp_tv",
        "lion_physics",
        "ct_20,ct_8",
        "admm_tv__lion_physics__any",
        names("lambda_", (5e-4, 1e-3, 2e-3)),
    ),
    Sweep(
        "pnp",
        "patch_lidc_default",
        "pnp_admm",
        "lion_physics",
        "ct_20,ct_8",
        "pnp_admm__lion_physics__any",
        (
            "eta_5e_06__iters_100",
            "eta_1e_05__iters_40",
            "eta_1e_05__iters_60",
            "eta_1e_05__iters_100",
            "eta_2e_05__iters_40",
            "eta_2e_05__iters_60",
            "eta_2e_05__iters_100",
            "eta_3e_05__iters_40",
            "eta_3e_05__iters_60",
            "eta_3e_05__iters_100",
            "eta_5e_05__iters_60",
            "eta_5e_05__iters_100",
        ),
    ),
    Sweep(
        "pnp_noise",
        "patch_lidc_default",
        "pnp_admm",
        "lion_physics",
        "ct_20,ct_8",
        "pnp_admm__lion_physics__any",
        names("noise_", (0.01, 0.03, 0.05)),
    ),
    Sweep(
        "dps_lion",
        "patch_lidc_default",
        "padis_dps",
        "lion_physics",
        "ct_20,ct_8",
        "padis_dps__lion_physics__any",
        tuple(
            f"zeta_{z}__eps_{e}"
            for z in ("3p5", "3p75", "4", "4p25")
            for e in ("0p3", "0p5")
        ),
    ),
    Sweep(
        "dps_public",
        "patch_lidc_default",
        "padis_dps",
        "public_repo",
        "ct_20,ct_8",
        "padis_dps__public_repo__any",
        tuple(f"zeta_{z}__eps_{e}" for z in ("0p15", "0p2") for e in ("0p5", "0p75")),
    ),
    Sweep(
        "dps_paper",
        "patch_lidc_default",
        "padis_dps",
        "paper",
        "ct_20,ct_8",
        "padis_dps__paper__any",
        tuple(
            f"zeta_{z}__eps_{e}"
            for z in ("0p0075", "0p01", "0p015")
            for e in ("0p5", "1")
        ),
    ),
    Sweep(
        "langevin_lion_patch",
        "patch_lidc_default",
        "langevin",
        "lion_physics",
        "ct_20,ct_8",
        "langevin__lion_physics__patch",
        tuple(
            f"zeta_{z}__eps_{e}" for z in ("3p5", "4", "4p5") for e in ("0p5", "0p75")
        ),
    ),
    Sweep(
        "langevin_public",
        "patch_lidc_default",
        "langevin",
        "public_repo",
        "ct_20,ct_8",
        "langevin__public_repo__patch",
        tuple(f"zeta_{z}__eps_{e}" for z in ("0p2", "0p3") for e in ("0p5", "0p75")),
    ),
    Sweep(
        "langevin_paper",
        "patch_lidc_default",
        "langevin",
        "paper",
        "ct_20,ct_8",
        "langevin__paper__patch",
        tuple(f"zeta_{z}__eps_{e}" for z in ("0p01", "0p03") for e in ("0p5", "0p75")),
    ),
    Sweep(
        "pc_lion_patch",
        "patch_lidc_default",
        "predictor_corrector",
        "lion_physics",
        "ct_20,ct_8",
        "predictor_corrector__lion_physics__patch",
        tuple(
            f"zeta_{z}__r_{r}"
            for z in ("3p75", "4", "4p25", "4p5", "4p75")
            for r in ("0p01", "0p015")
        ),
    ),
    Sweep(
        "pc_public",
        "patch_lidc_default",
        "predictor_corrector",
        "public_repo",
        "ct_20,ct_8",
        "predictor_corrector__public_repo__patch",
        ("zeta_0p5__r_0p08", "zeta_0p5__r_0p16"),
    ),
    Sweep(
        "pc_paper",
        "patch_lidc_default",
        "predictor_corrector",
        "paper",
        "ct_20,ct_8",
        "predictor_corrector__paper__patch",
        tuple(
            f"zeta_{z}__r_{r}"
            for z in ("0p01", "0p02", "0p03")
            for r in ("0p04", "0p08", "0p16")
        ),
    ),
    *(
        Sweep(
            f"veddnm_{impl}",
            "patch_lidc_default",
            "ve_ddnm",
            impl,
            "ct_20,ct_8",
            f"ve_ddnm__{impl}__patch",
            names("eps_", (0.05, 0.1, 0.2)),
        )
        for impl in ("lion_physics", "public_repo", "paper")
    ),
    Sweep(
        "veddnm_lion_noise",
        "patch_lidc_default",
        "ve_ddnm",
        "lion_physics",
        "ct_20,ct_8",
        "ve_ddnm__lion_physics__patch",
        names("noise_scale_", (0.0, 0.5, 1.5)),
    ),
    Sweep(
        "patch_average",
        "patch_lidc_default",
        "patch_average",
        "lion_physics",
        "ct_20,ct_8",
        "patch_average__lion_physics__any",
        names("zeta_", (3.5, 4.0)),
    ),
    Sweep(
        "whole_dps_paper",
        "whole_lidc_default",
        "whole_image_diffusion",
        "paper",
        "ct_20,ct_8",
        "whole_image_diffusion__paper__any",
        names("zeta_", (0.0075, 0.01, 0.015)),
    ),
    Sweep(
        "whole_langevin_lion",
        "whole_lidc_default",
        "langevin",
        "lion_physics",
        "ct_20",
        "langevin__lion_physics__whole_image",
        tuple(
            f"zeta_{z}__eps_{e}" for z in ("3p5", "4", "4p5") for e in ("0p5", "0p75")
        ),
    ),
    Sweep(
        "whole_pc_lion",
        "whole_lidc_default",
        "predictor_corrector",
        "lion_physics",
        "ct_20",
        "predictor_corrector__lion_physics__whole_image",
        tuple(
            f"zeta_{z}__r_{r}"
            for z in ("3p75", "4", "4p25", "4p5", "4p75")
            for r in ("0p01", "0p015")
        ),
    ),
    Sweep(
        "whole_veddnm_lion",
        "whole_lidc_default",
        "ve_ddnm",
        "lion_physics",
        "ct_20",
        "ve_ddnm__lion_physics__whole_image",
        names("eps_", (0.05, 0.1, 0.2)),
    ),
    Sweep(
        "native512_public",
        "patch_lidc_512",
        "padis_dps",
        "public_repo",
        "ct_512_60",
        "padis_dps__public_repo__any",
        names("native512_zeta_", (0.4, 0.5, 0.8, 1.2, 1.6))
        + ("native512_zeta_1p6__eps_0p75",),
    ),
    Sweep(
        "native512_lion",
        "patch_lidc_512",
        "padis_dps",
        "lion_physics",
        "ct_512_60",
        "padis_dps__lion_physics__any",
        names("native512_zeta_", (0.8, 1.2, 1.6, 2.0)),
    ),
    Sweep(
        "native512_cp",
        "patch_lidc_512",
        "cp_tv",
        "lion_physics",
        "ct_512_60",
        "admm_tv__lion_physics__any",
        names("lambda_", (1e-3, 2e-3)),
    ),
    Sweep(
        "full_patch",
        "patch_lidc_full",
        "padis_dps",
        "lion_physics",
        "ct_20",
        "padis_dps__lion_physics__any",
        names("full_patch_zeta_", (2.0, 4.5, 4.75, 5.0)),
    ),
    Sweep(
        "full_whole",
        "whole_lidc_full",
        "whole_image_diffusion",
        "lion_physics",
        "ct_20",
        "whole_image_diffusion__lion_physics__any",
        names("full_whole_zeta_", (4.0, 5.0)),
    ),
)

FAST_SMOKE_CANDIDATES = {
    "cp": ("lambda_0p001",),
    "pnp": ("eta_3e_05__iters_60",),
    "dps_lion": ("zeta_4p25__eps_0p5",),
    "dps_public": ("zeta_0p2__eps_0p5",),
    "dps_paper": ("zeta_0p0075__eps_0p5",),
    "langevin_lion_patch": ("zeta_4__eps_0p5",),
    "pc_lion_patch": ("zeta_4p25__r_0p01",),
    "veddnm_lion_physics": ("eps_0p1",),
    "whole_dps_paper": ("zeta_0p01",),
    "native512_lion": ("native512_zeta_2",),
    "full_patch": ("full_patch_zeta_4p5",),
    "full_whole": ("full_whole_zeta_4",),
}


def main() -> None:
    """Run the documented tuning sweep collection sequentially."""
    default_output = pathlib.Path(
        os.environ.get(
            "PADIS_TUNING_ROOT", pathlib.Path.cwd() / "padis_tuning_reproduction"
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, default=default_output)
    parser.add_argument("--only", default="all", help="Comma-separated sweep names.")
    parser.add_argument("--data-folder", type=pathlib.Path)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=4)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    smoke_group = parser.add_mutually_exclusive_group()
    smoke_group.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Execute every tuning candidate with one sample, one diffusion outer "
            "step, one inner denoiser evaluation, and one TV/PnP iteration on "
            "the first configured experiment for each model; outputs remain "
            "isolated under --output-root."
        ),
    )
    smoke_group.add_argument(
        "--fast-smoke",
        action="store_true",
        help=(
            "Exercise a representative candidate from each major tuning family "
            "with the same one-sample/one-experiment/one-NFE execution limits."
        ),
    )
    args = parser.parse_args()
    selected = None if args.only == "all" else set(args.only.split(","))
    unknown = set() if selected is None else selected - {s.name for s in SWEEPS}
    if unknown:
        parser.error(f"unknown sweep(s): {', '.join(sorted(unknown))}")

    for sweep in SWEEPS:
        if selected is not None and sweep.name not in selected:
            continue
        if args.fast_smoke and sweep.name not in FAST_SMOKE_CANDIDATES:
            continue
        smoke = args.smoke or args.fast_smoke
        experiments = sweep.experiments.split(",", 1)[0] if smoke else sweep.experiments
        candidates = (
            FAST_SMOKE_CANDIDATES[sweep.name] if args.fast_smoke else sweep.candidates
        )
        command = [
            sys.executable,
            "-u",
            str(TUNER),
            "--use-existing-training-root",
            "--checkpoint-policy",
            "min_intense_val",
            "--training-root",
            str(args.training_root),
            "--output-root",
            str(args.output_root),
            "--run-name",
            sweep.name,
            "--candidate-set",
            "reproduction",
            "--models",
            sweep.models,
            "--methods",
            sweep.method,
            "--implementations",
            sweep.implementation,
            "--experiments",
            experiments,
            "--only-groups",
            sweep.group,
            "--only-candidates",
            ",".join(candidates),
            "--max-samples",
            str(args.max_samples),
            "--start-index",
            str(args.start_index),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        if args.data_folder:
            command += ["--data-folder", str(args.data_folder)]
        for flag in ("dry_run", "rerun_existing", "stop_on_failure"):
            if getattr(args, flag):
                command.append("--" + flag.replace("_", "-"))
        if smoke:
            command += [
                "--stop-after-outer-steps",
                "1",
                "--reconstruction-arg=--inner-steps",
                "--reconstruction-arg=1",
                "--reconstruction-arg=--tv-iterations",
                "--reconstruction-arg=1",
                "--reconstruction-arg=--pnp-iterations",
                "--reconstruction-arg=1",
                "--reconstruction-arg=--pnp-cg-iterations",
                "--reconstruction-arg=1",
                "--reconstruction-arg=--patch-batch-size",
                "--reconstruction-arg=1",
            ]
        print(f"\n=== {sweep.name} ===", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
