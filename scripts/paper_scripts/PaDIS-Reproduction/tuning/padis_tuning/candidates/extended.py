"""Extended LION and reproduction candidate sweeps for PaDIS tuning."""

from __future__ import annotations

from padis_tuning.candidates.core import (
    Candidate,
    broad_candidates,
    current_default_candidates,
    focused_candidates,
    lion_physics_candidate,
    pilot_candidates,
    safe_name,
    sampler_candidate,
    unique_candidates,
)
from padis_tuning.candidates.paper import (
    padis_dps_lion_full_candidates,
    paper_full_candidates,
    public_paper_sampler_candidates,
    public_repo_full_candidates,
)


def consensus_24h_candidates() -> list[Candidate]:
    """Compact validation grid for broad method coverage within a day locally.

    This set is intended for staged sweeps with --only-methods and
    --only-implementations. It keeps the sigma endpoints of Hu et al., LION-physics
    data objective/scaling, FDK settings, and patch geometry fixed; only
    method-level solver and sampler strengths are varied.
    """
    candidates = current_default_candidates()

    candidates.extend(
        lion_physics_candidate(
            method="cp_tv",
            name=f"tv_lam_{safe_name(f'{lam:g}')}__iters_{iterations}",
            args=("--tv-lambda", f"{lam:g}", "--tv-iterations", str(iterations)),
        )
        for lam in (0.003, 0.005, 0.0075, 0.01)
        for iterations in (500, 1000)
    )

    candidates.extend(
        lion_physics_candidate(
            method="pnp_admm",
            name=f"eta_{safe_name(f'{eta:g}')}__iters_{iterations}",
            args=("--pnp-eta", f"{eta:g}", "--pnp-iterations", str(iterations)),
        )
        for eta in (1e-5, 2e-5, 3e-5)
        for iterations in (40, 60)
    )
    for noise_level in (0.01, 0.03):
        candidates.append(
            lion_physics_candidate(
                method="pnp_admm",
                name=f"pnp_noise_level_{safe_name(f'{noise_level:g}')}",
                args=("--pnp-noise-level", f"{noise_level:g}"),
            )
        )

    for zeta in (4.25, 4.5, 4.75, 5.0):
        for eps in (0.5, 0.65, 0.75):
            candidates.append(
                lion_physics_candidate(
                    method="padis_dps",
                    name=(
                        f"zeta_{safe_name(f'{zeta:g}')}"
                        f"__eps_{safe_name(f'{eps:g}')}"
                    ),
                    args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
                )
            )

    for zeta in (0.15, 0.2):
        for eps in (0.5, 0.75):
            candidates.append(
                sampler_candidate(
                    method="padis_dps",
                    implementation="public_repo",
                    name=(
                        f"zeta_{safe_name(f'{zeta:g}')}"
                        f"__eps_{safe_name(f'{eps:g}')}"
                    ),
                    args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
                )
            )
    for zeta in (0.0075, 0.01, 0.015):
        for eps in (0.5, 1.0):
            candidates.append(
                sampler_candidate(
                    method="padis_dps",
                    implementation="paper",
                    name=(
                        f"zeta_{safe_name(f'{zeta:g}')}"
                        f"__eps_{safe_name(f'{eps:g}')}"
                    ),
                    args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
                )
            )

    for method in ("whole_image_diffusion", "patch_average", "patch_stitch"):
        for zeta in (4.0, 4.5, 5.0):
            for eps in (0.5, 0.75):
                candidates.append(
                    lion_physics_candidate(
                        method=method,
                        name=(
                            f"zeta_{safe_name(f'{zeta:g}')}"
                            f"__eps_{safe_name(f'{eps:g}')}"
                        ),
                        args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
                    )
                )

    for method in ("patch_average", "patch_stitch"):
        for zeta in (0.15, 0.2):
            for eps in (0.5, 0.75):
                candidates.append(
                    sampler_candidate(
                        method=method,
                        implementation="public_repo",
                        name=(
                            f"zeta_{safe_name(f'{zeta:g}')}"
                            f"__eps_{safe_name(f'{eps:g}')}"
                        ),
                        args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
                    )
                )

    for zeta in (0.0075, 0.01, 0.015):
        for eps in (0.5, 1.0):
            candidates.append(
                sampler_candidate(
                    method="whole_image_diffusion",
                    implementation="paper",
                    name=(
                        f"zeta_{safe_name(f'{zeta:g}')}"
                        f"__eps_{safe_name(f'{eps:g}')}"
                    ),
                    args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
                )
            )

    for prior in ("patch", "whole_image"):
        for zeta in (3.5, 4.0, 4.5):
            for eps in (0.5, 0.75):
                candidates.append(
                    lion_physics_candidate(
                        method="langevin",
                        prior=prior,
                        name=(
                            f"zeta_{safe_name(f'{zeta:g}')}"
                            f"__eps_{safe_name(f'{eps:g}')}"
                        ),
                        args=("--zeta", f"{zeta:g}", "--sampling-epsilon", f"{eps:g}"),
                    )
                )
        for zeta in (4.0, 4.25, 4.5):
            for snr in (0.02, 0.04, 0.08):
                candidates.append(
                    lion_physics_candidate(
                        method="predictor_corrector",
                        prior=prior,
                        name=(
                            f"zeta_{safe_name(f'{zeta:g}')}"
                            f"__snr_{safe_name(f'{snr:g}')}"
                        ),
                        args=("--zeta", f"{zeta:g}", "--pc-snr", f"{snr:g}"),
                    )
                )
        for eps in (0.05, 0.1, 0.2):
            candidates.append(
                lion_physics_candidate(
                    method="ve_ddnm",
                    prior=prior,
                    name=f"sampling_eps_{safe_name(f'{eps:g}')}",
                    args=("--sampling-epsilon", f"{eps:g}"),
                )
            )

    for implementation, zetas in (
        ("public_repo", (0.2, 0.3, 0.4)),
        ("paper", (0.01, 0.03, 0.1)),
    ):
        for zeta in zetas:
            for eps in (0.5, 0.75):
                candidates.append(
                    sampler_candidate(
                        method="langevin",
                        implementation=implementation,
                        prior="patch",
                        name=(
                            f"zeta_{safe_name(f'{zeta:g}')}"
                            f"__eps_{safe_name(f'{eps:g}')}"
                        ),
                        args=("--zeta", f"{zeta:g}", "--sampling-epsilon", f"{eps:g}"),
                    )
                )
        for zeta in zetas:
            for snr in (0.08, 0.16):
                candidates.append(
                    sampler_candidate(
                        method="predictor_corrector",
                        implementation=implementation,
                        prior="patch",
                        name=(
                            f"zeta_{safe_name(f'{zeta:g}')}"
                            f"__snr_{safe_name(f'{snr:g}')}"
                        ),
                        args=("--zeta", f"{zeta:g}", "--pc-snr", f"{snr:g}"),
                    )
                )
        for eps in (0.05, 0.1, 0.2):
            candidates.append(
                sampler_candidate(
                    method="ve_ddnm",
                    implementation=implementation,
                    prior="patch",
                    name=f"sampling_eps_{safe_name(f'{eps:g}')}",
                    args=("--sampling-epsilon", f"{eps:g}"),
                )
            )

    return unique_candidates(candidates)


def consensus_24h_no_defaults_candidates() -> list[Candidate]:
    """Consensus candidates excluding current defaults already run by smoke."""
    return [
        candidate
        for candidate in consensus_24h_candidates()
        if candidate.name != "current_defaults"
    ]


def lion_physics_pc_public_gap_candidates() -> list[Candidate]:
    """Focused PC candidates for closing the public-compatible validation gap."""
    candidates: list[Candidate] = []
    for zeta in (3.75, 4.0, 4.25, 4.5):
        for snr in (0.01, 0.015, 0.02, 0.03, 0.04, 0.06):
            candidates.append(
                lion_physics_candidate(
                    method="predictor_corrector",
                    prior="patch",
                    name=(
                        f"zeta_{safe_name(f'{zeta:g}')}"
                        f"__snr_{safe_name(f'{snr:g}')}"
                    ),
                    args=("--zeta", f"{zeta:g}", "--pc-snr", f"{snr:g}"),
                    notes=(
                        "Patch-prior PC refinement around the current "
                        "LION-physics best point; keeps CT physics settings fixed."
                    ),
                )
            )
    for snr in (0.01, 0.015, 0.02, 0.03):
        candidates.append(
            lion_physics_candidate(
                method="predictor_corrector",
                prior="patch",
                name=f"zeta_4p75__snr_{safe_name(f'{snr:g}')}",
                args=("--zeta", "4.75", "--pc-snr", f"{snr:g}"),
                notes=(
                    "High-zeta, low-SNR PC refinement; zeta=5 collapsed in "
                    "earlier validation."
                ),
            )
        )
    return unique_candidates(candidates)


def sampler_full_candidates() -> list[Candidate]:
    """Handle sampler full candidates for the PaDIS workflow."""
    candidates = lion_physics_full_candidates()
    candidates.extend(public_paper_sampler_candidates())
    candidates.extend(public_repo_full_candidates())
    candidates.extend(paper_full_candidates())
    return unique_candidates(candidates)


def lion_physics_full_candidates() -> list[Candidate]:
    """Validation candidates for all implemented LION-physics reconstruction rows."""
    candidates = [
        candidate
        for candidate in current_default_candidates()
        if candidate.implementation == "lion_physics"
    ]
    candidates.extend(padis_dps_lion_full_candidates())
    candidates.extend(lion_physics_pc_public_gap_candidates())

    candidates.extend(
        lion_physics_candidate(
            method="cp_tv",
            name=f"tv_lam_{safe_name(f'{lam:g}')}__iters_{iterations}",
            args=("--tv-lambda", f"{lam:g}", "--tv-iterations", str(iterations)),
        )
        for lam in (5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7.5e-3, 1e-2, 1.5e-2)
        for iterations in (250, 500, 1000)
    )
    candidates.append(
        lion_physics_candidate(
            method="cp_tv",
            name="tv_non_negative",
            args=("--tv-non-negativity",),
        )
    )

    candidates.extend(
        lion_physics_candidate(
            method="pnp_admm",
            name=f"eta_{safe_name(f'{eta:g}')}__iters_{iterations}",
            args=("--pnp-eta", f"{eta:g}", "--pnp-iterations", str(iterations)),
        )
        for eta in (2e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5)
        for iterations in (20, 40, 60, 100)
    )
    for cg_iterations in (50, 100, 200):
        candidates.append(
            lion_physics_candidate(
                method="pnp_admm",
                name=f"pnp_cg_iters_{cg_iterations}",
                args=("--pnp-cg-iterations", str(cg_iterations)),
            )
        )
    for tolerance in (1e-6, 1e-7, 1e-8):
        candidates.append(
            lion_physics_candidate(
                method="pnp_admm",
                name=f"pnp_cg_tol_{safe_name(f'{tolerance:g}')}",
                args=("--pnp-cg-tolerance", f"{tolerance:g}"),
            )
        )
    for name, args in (
        ("pnp_no_clip", ("--no-pnp-clip",)),
        ("pnp_noise_level_0p01", ("--pnp-noise-level", "0.01")),
        ("pnp_noise_level_0p03", ("--pnp-noise-level", "0.03")),
        ("pnp_noise_level_0p05", ("--pnp-noise-level", "0.05")),
    ):
        candidates.append(
            lion_physics_candidate(method="pnp_admm", name=name, args=args)
        )

    def add_dps_like(
        method: str,
        name: str,
        *args: str,
        prior: str | None = None,
        notes: str = "",
    ) -> None:
        """Add dps like."""
        candidates.append(
            lion_physics_candidate(
                method=method,
                name=name,
                prior=prior,
                args=tuple(args),
                notes=notes,
            )
        )

    for method in ("whole_image_diffusion", "patch_average", "patch_stitch"):
        for zeta in (3.5, 4.0, 4.5, 5.0):
            for eps in (0.3, 0.5, 0.75, 1.0):
                add_dps_like(
                    method,
                    f"core_zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
                    "--zeta",
                    f"{zeta:g}",
                    "--dps-epsilon",
                    f"{eps:g}",
                )

    for prior in ("patch", "whole_image"):
        for zeta in (3.0, 3.5, 4.0, 4.5, 5.0):
            for eps in (0.25, 0.5, 0.75, 1.0):
                add_dps_like(
                    "langevin",
                    f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
                    "--zeta",
                    f"{zeta:g}",
                    "--sampling-epsilon",
                    f"{eps:g}",
                    prior=prior,
                )
        for scale in (0.0, 0.5, 1.0, 1.5):
            add_dps_like(
                "langevin",
                f"noise_scale_{safe_name(f'{scale:g}')}",
                "--langevin-noise-scale",
                f"{scale:g}",
                prior=prior,
            )
        for zeta in (3.5, 4.0, 4.25, 4.5, 5.0):
            for snr in (0.02, 0.04, 0.08, 0.12, 0.16):
                add_dps_like(
                    "predictor_corrector",
                    f"zeta_{safe_name(f'{zeta:g}')}__snr_{safe_name(f'{snr:g}')}",
                    "--zeta",
                    f"{zeta:g}",
                    "--pc-snr",
                    f"{snr:g}",
                    prior=prior,
                )
        for rule in ("paper_linear", "score_sde_squared"):
            add_dps_like(
                "predictor_corrector",
                f"pc_step_{safe_name(rule)}",
                "--pc-corrector-step-rule",
                rule,
                prior=prior,
            )
        for sigma_choice in ("next", "current"):
            add_dps_like(
                "predictor_corrector",
                f"pc_denoise_sigma_{sigma_choice}",
                "--pc-corrector-denoise-sigma",
                sigma_choice,
                prior=prior,
            )
        for eps in (0.05, 0.075, 0.1, 0.125, 0.2, 0.3):
            add_dps_like(
                "ve_ddnm",
                f"sampling_eps_{safe_name(f'{eps:g}')}",
                "--sampling-epsilon",
                f"{eps:g}",
                prior=prior,
            )
        for scale in (0.0, 0.5, 1.5):
            add_dps_like(
                "ve_ddnm",
                f"noise_scale_{safe_name(f'{scale:g}')}",
                "--langevin-noise-scale",
                f"{scale:g}",
                prior=prior,
            )
        for name, args in (
            ("clip_denoised", ("--clip-denoised",)),
            ("clip_state", ("--clip-state",)),
            ("clip_denoised_and_state", ("--clip-denoised", "--clip-state")),
        ):
            add_dps_like("ve_ddnm", name, *args, prior=prior)
        for name, args in (
            ("ddnm_no_corrected_clip", ("--no-ddnm-corrected-clip",)),
            ("ddnm_no_pinv_clip", ("--no-ddnm-pseudoinverse-clip",)),
            (
                "ddnm_no_projected_pinv_clip",
                ("--no-ddnm-projected-pseudoinverse-clip",),
            ),
            ("ddnm_public_inner", ("--ve-ddnm-nfe-layout", "public_inner")),
            ("ddnm_init_fdk", ("--initial-reconstruction", "fdk")),
        ):
            add_dps_like("ve_ddnm", name, *args, prior=prior)

    return unique_candidates(candidates)


def reproduction_candidates() -> list[Candidate]:
    """Compact union of the candidates used by the published tuning scheme."""
    candidates: list[Candidate] = []

    def add(
        method: str, implementation: str, prior: str | None, name: str, *args: str
    ) -> None:
        """Provide the add callback used by the enclosing operation."""
        candidates.append(Candidate(name, method, implementation, prior, tuple(args)))

    for lam in (5e-4, 1e-3, 2e-3):
        add(
            "cp_tv",
            "lion_physics",
            None,
            f"lambda_{safe_name(f'{lam:g}')}",
            "--tv-lambda",
            f"{lam:g}",
            "--tv-iterations",
            "1000",
        )
    for eta, iterations in (
        (5e-6, 100),
        (1e-5, 40),
        (1e-5, 60),
        (1e-5, 100),
        (2e-5, 40),
        (2e-5, 60),
        (2e-5, 100),
        (3e-5, 40),
        (3e-5, 60),
        (3e-5, 100),
        (5e-5, 60),
        (5e-5, 100),
    ):
        add(
            "pnp_admm",
            "lion_physics",
            None,
            f"eta_{safe_name(f'{eta:g}')}__iters_{iterations}",
            "--pnp-eta",
            f"{eta:g}",
            "--pnp-iterations",
            str(iterations),
        )
    for noise in (0.01, 0.03, 0.05):
        add(
            "pnp_admm",
            "lion_physics",
            None,
            f"noise_{safe_name(f'{noise:g}')}",
            "--pnp-eta",
            "3e-5",
            "--pnp-iterations",
            "60",
            "--pnp-noise-level",
            f"{noise:g}",
        )

    for implementation, zetas, epsilons in (
        ("lion_physics", (3.5, 3.75, 4.0, 4.25), (0.3, 0.5)),
        ("public_repo", (0.15, 0.2), (0.5, 0.75)),
        ("paper", (0.0075, 0.01, 0.015), (0.5, 1.0)),
    ):
        for zeta in zetas:
            for eps in epsilons:
                add(
                    "padis_dps",
                    implementation,
                    None,
                    f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
                    "--zeta",
                    f"{zeta:g}",
                    "--dps-epsilon",
                    f"{eps:g}",
                    *(
                        (
                            "--initial-reconstruction",
                            "noise",
                            "--no-clip-initial",
                            "--no-clip-output",
                        )
                        if implementation == "lion_physics"
                        else ()
                    ),
                )

    for implementation, zetas, epsilons in (
        ("lion_physics", (3.5, 4.0, 4.5), (0.5, 0.75)),
        ("public_repo", (0.2, 0.3), (0.5, 0.75)),
        ("paper", (0.01, 0.03), (0.5, 0.75)),
    ):
        for prior in (
            ("patch", "whole_image") if implementation == "lion_physics" else ("patch",)
        ):
            for zeta in zetas:
                for eps in epsilons:
                    add(
                        "langevin",
                        implementation,
                        prior,
                        f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
                        "--zeta",
                        f"{zeta:g}",
                        "--sampling-epsilon",
                        f"{eps:g}",
                    )

    for implementation, zetas, snrs in (
        ("lion_physics", (3.75, 4.0, 4.25, 4.5, 4.75), (0.01, 0.015)),
        ("public_repo", (0.5,), (0.08, 0.16)),
        ("paper", (0.01, 0.02, 0.03), (0.04, 0.08, 0.16)),
    ):
        for prior in (
            ("patch", "whole_image") if implementation == "lion_physics" else ("patch",)
        ):
            for zeta in zetas:
                for snr in snrs:
                    add(
                        "predictor_corrector",
                        implementation,
                        prior,
                        f"zeta_{safe_name(f'{zeta:g}')}__r_{safe_name(f'{snr:g}')}",
                        "--zeta",
                        f"{zeta:g}",
                        "--pc-snr",
                        f"{snr:g}",
                    )

    for implementation in ("lion_physics", "public_repo", "paper"):
        for prior in (
            ("patch", "whole_image") if implementation == "lion_physics" else ("patch",)
        ):
            for eps in (0.05, 0.1, 0.2):
                add(
                    "ve_ddnm",
                    implementation,
                    prior,
                    f"eps_{safe_name(f'{eps:g}')}",
                    "--sampling-epsilon",
                    f"{eps:g}",
                )
    for scale in (0.0, 0.5, 1.5):
        add(
            "ve_ddnm",
            "lion_physics",
            "patch",
            f"noise_scale_{safe_name(f'{scale:g}')}",
            "--sampling-epsilon",
            "0.1",
            "--langevin-noise-scale",
            f"{scale:g}",
        )

    for zeta in (3.5, 4.0):
        add(
            "patch_average",
            "lion_physics",
            None,
            f"zeta_{safe_name(f'{zeta:g}')}",
            "--zeta",
            f"{zeta:g}",
            "--dps-epsilon",
            "0.5",
        )
    for zeta in (0.0075, 0.01, 0.015):
        add(
            "whole_image_diffusion",
            "paper",
            None,
            f"zeta_{safe_name(f'{zeta:g}')}",
            "--zeta",
            f"{zeta:g}",
            "--dps-epsilon",
            "0.5",
        )

    # Native-512 and full-data brackets use the same final checkpoint policy as inference.
    for implementation, zetas in (
        ("public_repo", (0.4, 0.5, 0.8, 1.2, 1.6)),
        ("lion_physics", (0.8, 1.2, 1.6, 2.0)),
    ):
        for zeta in zetas:
            add(
                "padis_dps",
                implementation,
                None,
                f"native512_zeta_{safe_name(f'{zeta:g}')}",
                "--zeta",
                f"{zeta:g}",
                "--dps-epsilon",
                "0.5",
                *(
                    (
                        "--initial-reconstruction",
                        "noise",
                        "--no-clip-initial",
                        "--no-clip-output",
                    )
                    if implementation == "lion_physics"
                    else (
                        "--initial-reconstruction",
                        "fdk",
                        "--clip-initial",
                        "--clip-output",
                    )
                ),
            )
    add(
        "padis_dps",
        "public_repo",
        None,
        "native512_zeta_1p6__eps_0p75",
        "--zeta",
        "1.6",
        "--dps-epsilon",
        "0.75",
    )
    for zeta in (2.0, 4.5, 4.75, 5.0):
        add(
            "padis_dps",
            "lion_physics",
            None,
            f"full_patch_zeta_{safe_name(f'{zeta:g}')}",
            "--zeta",
            f"{zeta:g}",
            "--dps-epsilon",
            "0.5",
            "--initial-reconstruction",
            "noise",
            "--no-clip-initial",
            "--no-clip-output",
        )
    for zeta in (4.0, 5.0):
        add(
            "whole_image_diffusion",
            "lion_physics",
            None,
            f"full_whole_zeta_{safe_name(f'{zeta:g}')}",
            "--zeta",
            f"{zeta:g}",
            "--dps-epsilon",
            "0.5",
        )
    return unique_candidates(candidates)


def candidate_set(name: str) -> list[Candidate]:
    """Return the named reproducible candidate collection."""
    if name == "reproduction":
        return reproduction_candidates()
    if name == "smoke":
        return current_default_candidates()
    if name == "pilot":
        return pilot_candidates()
    if name == "broad":
        return broad_candidates()
    if name == "focused":
        return focused_candidates()
    if name == "padis_dps_lion_full":
        return padis_dps_lion_full_candidates()
    if name == "lion_physics_full":
        return lion_physics_full_candidates()
    if name == "public_paper_sampler":
        return public_paper_sampler_candidates()
    if name == "public_repo_full":
        return public_repo_full_candidates()
    if name == "paper_full":
        return paper_full_candidates()
    if name == "consensus_24h":
        return consensus_24h_candidates()
    if name == "consensus_24h_no_defaults":
        return consensus_24h_no_defaults_candidates()
    if name == "sampler_full":
        return sampler_full_candidates()
    raise ValueError(f"Unknown candidate set {name!r}.")
