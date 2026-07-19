"""Core candidate types and builders for PaDIS hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Iterable

import PaDIS_run_reconstruction_matrix as matrix


@dataclass(frozen=True)
class Candidate:
    """One named set of sampler arguments in a tuning sweep."""

    name: str
    method: str
    implementation: str
    prior: str | None
    args: tuple[str, ...]
    notes: str = ""

    @property
    def group(self) -> str:
        """Handle group for the PaDIS workflow."""
        prior = self.prior if self.prior is not None else "any"
        return f"{self.method}__{self.implementation}__{prior}"


@dataclass(frozen=True)
class RunRecord:
    """Completed or failed tuning run and its recorded provenance."""

    candidate: Candidate
    job: matrix.ReconstructionJob
    command: tuple[str, ...]
    metrics_path: pathlib.Path
    log_path: pathlib.Path
    status: str
    elapsed_seconds: float | None
    summary: dict
    error: str | None = None


def safe_name(value: str) -> str:
    """Return a filesystem-safe name."""
    return (
        value.replace("-", "_")
        .replace(".", "p")
        .replace("/", "_")
        .replace("=", "_")
        .replace("+", "plus")
    )


def flag_value_args(flag: str, values: Iterable[object]) -> tuple[tuple[str, ...], ...]:
    """Handle flag value args for the PaDIS workflow."""
    return tuple((flag, str(value)) for value in values)


def zeta_candidates(
    *,
    method: str,
    implementation: str,
    prior: str | None,
    values: Iterable[float],
    prefix: str = "zeta",
) -> list[Candidate]:
    """Handle zeta candidates for the PaDIS workflow."""
    return [
        Candidate(
            name=f"{prefix}_{safe_name(f'{value:g}')}",
            method=method,
            implementation=implementation,
            prior=prior,
            args=("--zeta", f"{value:g}"),
        )
        for value in values
    ]


def current_default_candidates() -> list[Candidate]:
    """Handle current default candidates for the PaDIS workflow."""
    candidates: list[Candidate] = []
    for method, implementations in matrix.CORE_IMPLEMENTATIONS_BY_METHOD.items():
        for implementation in implementations:
            priors: tuple[str | None, ...] = (None,)
            if method in {"langevin", "predictor_corrector", "ve_ddnm"}:
                priors = ("patch", "whole_image")
            for prior in priors:
                candidates.append(
                    Candidate(
                        name="current_defaults",
                        method=method,
                        implementation=implementation,
                        prior=prior,
                        args=(),
                        notes="Matrix/reconstruction defaults with no tuner override.",
                    )
                )
    return candidates


def pilot_candidates() -> list[Candidate]:
    """Handle pilot candidates for the PaDIS workflow."""
    candidates = current_default_candidates()

    candidates.extend(
        Candidate(
            name=f"tv_lam_{safe_name(f'{lam:g}')}",
            method="cp_tv",
            implementation="lion_physics",
            prior=None,
            args=("--tv-lambda", f"{lam:g}", "--tv-iterations", "500"),
        )
        for lam in (3e-4, 1e-3, 3e-3, 1e-2)
    )

    candidates.extend(
        Candidate(
            name=(f"eta_{safe_name(f'{eta:g}')}__iters_{iterations}"),
            method="pnp_admm",
            implementation="lion_physics",
            prior=None,
            args=(
                "--pnp-eta",
                f"{eta:g}",
                "--pnp-iterations",
                str(iterations),
            ),
        )
        for eta in (3e-6, 1e-5, 3e-5)
        for iterations in (10, 20, 40)
    )

    candidates.extend(
        zeta_candidates(
            method="padis_dps",
            implementation="lion_physics",
            prior=None,
            values=(2.0, 3.0, 4.0, 5.0, 6.0),
        )
    )
    candidates.extend(
        Candidate(
            name=f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
            method="padis_dps",
            implementation="lion_physics",
            prior=None,
            args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
        )
        for zeta in (3.0, 4.0, 5.0)
        for eps in (0.5, 0.75, 1.0)
    )
    candidates.extend(
        zeta_candidates(
            method="padis_dps",
            implementation="public_repo",
            prior=None,
            values=(0.1, 0.2, 0.3, 0.4),
        )
    )
    candidates.extend(
        zeta_candidates(
            method="padis_dps",
            implementation="paper",
            prior=None,
            values=(0.005, 0.01, 0.03, 0.1, 0.3),
        )
    )

    for prior in ("patch", "whole_image"):
        candidates.extend(
            Candidate(
                name=f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
                method="langevin",
                implementation="lion_physics",
                prior=prior,
                args=(
                    "--zeta",
                    f"{zeta:g}",
                    "--sampling-epsilon",
                    f"{eps:g}",
                ),
            )
            for zeta in (3.0, 4.0, 5.0)
            for eps in (0.25, 0.5, 0.75)
        )
        candidates.extend(
            Candidate(
                name=f"zeta_{safe_name(f'{zeta:g}')}__snr_{safe_name(f'{snr:g}')}",
                method="predictor_corrector",
                implementation="lion_physics",
                prior=prior,
                args=("--zeta", f"{zeta:g}", "--pc-snr", f"{snr:g}"),
            )
            for zeta in (3.5, 4.25, 5.0)
            for snr in (0.04, 0.08, 0.12)
        )
        candidates.extend(
            Candidate(
                name=f"eps_{safe_name(f'{eps:g}')}",
                method="ve_ddnm",
                implementation="lion_physics",
                prior=prior,
                args=("--sampling-epsilon", f"{eps:g}"),
            )
            for eps in (0.05, 0.1, 0.2)
        )

    for method in ("langevin", "predictor_corrector", "ve_ddnm"):
        candidates.extend(
            zeta_candidates(
                method=method,
                implementation="public_repo",
                prior="patch",
                values=(0.1, 0.2, 0.3, 0.5),
            )
        )
        candidates.extend(
            zeta_candidates(
                method=method,
                implementation="paper",
                prior="patch",
                values=(0.005, 0.01, 0.03, 0.1, 0.3),
            )
        )

    for method in ("patch_average", "patch_stitch"):
        candidates.extend(
            Candidate(
                name=f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
                method=method,
                implementation="lion_physics",
                prior=None,
                args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
            )
            for zeta in (3.0, 4.0, 5.0)
            for eps in (0.3, 0.5, 0.75)
        )
        candidates.extend(
            zeta_candidates(
                method=method,
                implementation="public_repo",
                prior=None,
                values=(0.1, 0.2, 0.3, 0.5),
            )
        )

    candidates.extend(
        Candidate(
            name=f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
            method="whole_image_diffusion",
            implementation="lion_physics",
            prior=None,
            args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
        )
        for zeta in (3.0, 4.0, 5.0)
        for eps in (0.5, 0.75, 1.0)
    )
    candidates.extend(
        zeta_candidates(
            method="whole_image_diffusion",
            implementation="paper",
            prior=None,
            values=(0.005, 0.01, 0.03, 0.1, 0.3),
        )
    )

    return unique_candidates(candidates)


def broad_candidates() -> list[Candidate]:
    """Handle broad candidates for the PaDIS workflow."""
    candidates = pilot_candidates()
    candidates.extend(
        zeta_candidates(
            method="padis_dps",
            implementation="lion_physics",
            prior=None,
            values=(1.0, 1.5, 2.5, 3.5, 4.5, 5.5, 7.0),
        )
    )
    candidates.extend(
        Candidate(
            name=(
                f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}"
                f"__dc_{schedule}"
            ),
            method="padis_dps",
            implementation="lion_physics",
            prior=None,
            args=(
                "--zeta",
                f"{zeta:g}",
                "--dps-epsilon",
                f"{eps:g}",
                "--data-consistency-scale-schedule",
                schedule,
                "--data-consistency-scale-floor",
                "0.05",
            ),
            notes="Check whether sigma-dependent LION data weighting improves cross-experiment robustness.",
        )
        for zeta in (4.0, 5.0)
        for eps in (0.5, 1.0)
        for schedule in ("edm", "inverse_sigma")
    )
    candidates.extend(
        Candidate(
            name=f"tv_lam_{safe_name(f'{lam:g}')}__iters_{iterations}",
            method="cp_tv",
            implementation="lion_physics",
            prior=None,
            args=(
                "--tv-lambda",
                f"{lam:g}",
                "--tv-iterations",
                str(iterations),
            ),
        )
        for lam in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
        for iterations in (250, 500, 1000)
    )
    return unique_candidates(candidates)


def focused_candidates() -> list[Candidate]:
    """Handle focused candidates for the PaDIS workflow."""
    candidates = current_default_candidates()
    candidates.extend(
        Candidate(
            name=f"tv_lam_{safe_name(f'{lam:g}')}",
            method="cp_tv",
            implementation="lion_physics",
            prior=None,
            args=("--tv-lambda", f"{lam:g}", "--tv-iterations", "500"),
        )
        for lam in (0.002, 0.003, 0.005, 0.0075, 0.01, 0.015)
    )
    candidates.extend(
        Candidate(
            name=f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
            method="padis_dps",
            implementation="lion_physics",
            prior=None,
            args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
        )
        for zeta in (3.5, 4.0, 4.5, 4.55, 4.6, 4.7, 4.8, 4.9, 5.0)
        for eps in (0.5, 0.75, 1.0)
    )
    candidates.extend(
        zeta_candidates(
            method="padis_dps",
            implementation="public_repo",
            prior=None,
            values=(0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3),
        )
    )
    candidates.extend(
        zeta_candidates(
            method="padis_dps",
            implementation="paper",
            prior=None,
            values=(0.005, 0.0075, 0.01, 0.015, 0.02),
        )
    )
    candidates.extend(
        Candidate(
            name=f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
            method="whole_image_diffusion",
            implementation="lion_physics",
            prior=None,
            args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
        )
        for zeta in (3.5, 4.0, 4.5)
        for eps in (0.5, 0.75, 1.0)
    )
    candidates.extend(
        Candidate(
            name=f"eta_{safe_name(f'{eta:g}')}__iters_{iterations}",
            method="pnp_admm",
            implementation="lion_physics",
            prior=None,
            args=(
                "--pnp-eta",
                f"{eta:g}",
                "--pnp-iterations",
                str(iterations),
            ),
        )
        for eta in (5e-6, 1e-5, 2e-5)
        for iterations in (20, 40, 60)
    )
    return unique_candidates(candidates)


def lion_physics_candidate(
    *,
    method: str,
    name: str,
    args: tuple[str, ...],
    prior: str | None = None,
    notes: str = "",
) -> Candidate:
    """Handle lion physics candidate for the PaDIS workflow."""
    return Candidate(
        name=name,
        method=method,
        implementation="lion_physics",
        prior=prior,
        args=args,
        notes=notes,
    )


def sampler_candidate(
    *,
    method: str,
    implementation: str,
    name: str,
    args: tuple[str, ...],
    prior: str | None = None,
    notes: str = "",
) -> Candidate:
    """Handle sampler candidate for the PaDIS workflow."""
    return Candidate(
        name=name,
        method=method,
        implementation=implementation,
        prior=prior,
        args=args,
        notes=notes,
    )


def unique_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Return unique candidates."""
    seen: set[tuple] = set()
    unique: list[Candidate] = []
    for candidate in candidates:
        key = (
            candidate.method,
            candidate.implementation,
            candidate.prior,
            candidate.name,
            candidate.args,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique
