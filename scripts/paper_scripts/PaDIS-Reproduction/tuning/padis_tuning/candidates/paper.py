"""Paper and public-repository candidate sweeps for PaDIS tuning."""

from __future__ import annotations

from padis_tuning.candidates.core import (
    Candidate,
    current_default_candidates,
    lion_physics_candidate,
    safe_name,
    sampler_candidate,
    unique_candidates,
)


def padis_dps_lion_full_candidates() -> list[Candidate]:
    """One-factor-plus-core-grid sweep for PaDIS DPS LION-physics settings.

    This intentionally excludes sigma_min and sigma_max so the CT noise range
    remains the range used by Hu et al. for each experiment. It also keeps the LION-physics
    data objective, CT scaling, FDK filter, and patch geometry fixed, so the
    sweep focuses on sampler strength, schedule shape, NFE allocation,
    initialization, clipping, and Langevin noise scale.
    """
    candidates: list[Candidate] = [
        lion_physics_candidate(
            method="padis_dps",
            name="current_defaults",
            args=(),
            notes="Matrix/reconstruction defaults with no tuner override.",
        )
    ]

    def add(name: str, *args: str, notes: str = "") -> None:
        """Provide the add callback used by the enclosing operation."""
        candidates.append(
            lion_physics_candidate(
                method="padis_dps",
                name=name,
                args=tuple(args),
                notes=notes,
            )
        )

    for zeta in (
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.25,
        3.5,
        3.75,
        4.0,
        4.25,
        4.5,
        4.55,
        4.6,
        4.7,
        4.8,
        5.0,
    ):
        for eps in (0.3, 0.5, 0.65, 0.75, 1.0):
            add(
                f"core_zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
                "--zeta",
                f"{zeta:g}",
                "--dps-epsilon",
                f"{eps:g}",
            )

    for num_steps, inner_steps in (
        (50, 20),
        (100, 5),
        (100, 20),
        (200, 5),
        (1000, 1),
    ):
        add(
            f"nfe_{num_steps}x{inner_steps}",
            "--num-steps",
            str(num_steps),
            "--inner-steps",
            str(inner_steps),
            notes="Change NFE allocation while keeping the sigma range of Hu et al.",
        )

    for rho in (3, 5, 7, 10):
        add(
            f"schedule_edm_rho_{safe_name(f'{rho:g}')}",
            "--noise-schedule",
            "edm",
            "--rho",
            f"{rho:g}",
            notes="EDM schedule-shape diagnostic; sigma_min/sigma_max are unchanged.",
        )

    for name, args in (
        (
            "init_noise_unclipped",
            (
                "--initial-reconstruction",
                "noise",
                "--no-clip-initial",
                "--no-clip-output",
            ),
        ),
        (
            "init_noise_clipped_output",
            (
                "--initial-reconstruction",
                "noise",
            ),
        ),
        (
            "init_fdk_clipped",
            (
                "--initial-reconstruction",
                "fdk",
                "--clip-initial",
                "--clip-output",
            ),
        ),
        (
            "init_inverse_clipped",
            (
                "--initial-reconstruction",
                "inverse",
                "--clip-initial",
                "--clip-output",
            ),
        ),
        (
            "init_fdk_clipped_no_langevin_noise",
            (
                "--initial-reconstruction",
                "fdk",
                "--clip-initial",
                "--clip-output",
                "--disable-langevin-noise",
            ),
        ),
        (
            "init_fdk_clipped_no_prior",
            (
                "--initial-reconstruction",
                "fdk",
                "--clip-initial",
                "--clip-output",
                "--disable-prior-score",
            ),
        ),
        ("init_inverse", ("--initial-reconstruction", "inverse")),
        ("no_clip_initial", ("--no-clip-initial",)),
        ("no_clip_output", ("--no-clip-output",)),
        ("clip_denoised", ("--clip-denoised",)),
        ("clip_state", ("--clip-state",)),
        ("clip_denoised_and_state", ("--clip-denoised", "--clip-state")),
    ):
        add(name, *args)

    for scale in (0.0, 0.5, 1.5):
        add(
            f"langevin_noise_scale_{safe_name(f'{scale:g}')}",
            "--langevin-noise-scale",
            f"{scale:g}",
        )

    # 512-specific stability diagnostics: keep the Hu et al./LION sigma schedule
    # and default NFE allocation, but reduce stochasticity without disabling it.
    for zeta, eps, noise_scale in (
        (4.25, 0.5, 0.1),
        (4.25, 0.5, 0.25),
        (4.25, 0.3, 0.1),
        (3.0, 0.5, 0.1),
        (2.0, 0.5, 0.1),
        (2.0, 0.3, 0.1),
        (3.0, 0.2, 0.05),
        (3.0, 0.1, 0.05),
        (2.0, 0.2, 0.05),
        (2.0, 0.1, 0.05),
        (1.0, 0.2, 0.05),
        (1.0, 0.1, 0.05),
        (2.0, 0.05, 0.05),
        (1.0, 0.05, 0.05),
        (3.0, 0.2, 0.02),
        (2.0, 0.2, 0.02),
        (1.0, 0.2, 0.02),
        (3.0, 0.1, 0.02),
        (2.0, 0.1, 0.02),
        (1.0, 0.1, 0.02),
        (2.0, 0.2, 0.01),
        (1.0, 0.2, 0.01),
        (2.0, 0.1, 0.01),
        (1.0, 0.1, 0.01),
        (2.0, 0.3, 0.01),
        (1.0, 0.3, 0.01),
        (2.0, 0.2, 0.005),
        (1.0, 0.2, 0.005),
        (2.0, 0.1, 0.005),
        (1.0, 0.1, 0.005),
        (1.0, 0.3, 0.005),
        (1.0, 0.2, 0.002),
    ):
        add(
            (
                f"init_fdk_clipped_zeta_{safe_name(f'{zeta:g}')}"
                f"__eps_{safe_name(f'{eps:g}')}"
                f"__noise_{safe_name(f'{noise_scale:g}')}"
            ),
            "--initial-reconstruction",
            "fdk",
            "--clip-initial",
            "--clip-output",
            "--zeta",
            f"{zeta:g}",
            "--dps-epsilon",
            f"{eps:g}",
            "--langevin-noise-scale",
            f"{noise_scale:g}",
            notes=(
                "512 PaDIS-DPS stability candidate: unchanged sigma schedule "
                "and NFE, nonzero Langevin noise."
            ),
        )

    return unique_candidates(candidates)


def public_paper_sampler_candidates() -> list[Candidate]:
    """Sampler-only candidates for public-compatible and paper implementations."""
    candidates = [
        candidate
        for candidate in current_default_candidates()
        if candidate.implementation in {"public_repo", "paper"}
    ]

    def add(
        *,
        method: str,
        implementation: str,
        name: str,
        args: tuple[str, ...],
        prior: str | None = None,
        notes: str = "",
    ) -> None:
        """Provide the add callback used by the enclosing operation."""
        candidates.append(
            sampler_candidate(
                method=method,
                implementation=implementation,
                name=name,
                args=args,
                prior=prior,
                notes=notes,
            )
        )

    dps_like_methods = ("padis_dps", "patch_average", "patch_stitch")
    for method in dps_like_methods:
        for zeta in (0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3):
            for eps in (0.3, 0.5, 0.75):
                add(
                    method=method,
                    implementation="public_repo",
                    name=(
                        f"zeta_{safe_name(f'{zeta:g}')}"
                        f"__eps_{safe_name(f'{eps:g}')}"
                    ),
                    args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
                )
    for method in ("padis_dps", "whole_image_diffusion"):
        for zeta in (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.1, 0.3):
            for eps in (0.5, 1.0):
                add(
                    method=method,
                    implementation="paper",
                    name=(
                        f"zeta_{safe_name(f'{zeta:g}')}"
                        f"__eps_{safe_name(f'{eps:g}')}"
                    ),
                    args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
                )

    for implementation, zetas in (
        ("public_repo", (0.1, 0.2, 0.3, 0.4, 0.5)),
        ("paper", (0.005, 0.01, 0.03, 0.1, 0.3)),
    ):
        for prior in ("patch", "whole_image"):
            for zeta in zetas:
                for eps in (0.25, 0.5, 0.75, 1.0):
                    add(
                        method="langevin",
                        implementation=implementation,
                        prior=prior,
                        name=(
                            f"zeta_{safe_name(f'{zeta:g}')}"
                            f"__eps_{safe_name(f'{eps:g}')}"
                        ),
                        args=(
                            "--zeta",
                            f"{zeta:g}",
                            "--sampling-epsilon",
                            f"{eps:g}",
                        ),
                    )
            for scale in (0.0, 0.5, 1.0, 1.5):
                add(
                    method="langevin",
                    implementation=implementation,
                    prior=prior,
                    name=f"noise_scale_{safe_name(f'{scale:g}')}",
                    args=("--langevin-noise-scale", f"{scale:g}"),
                )
            for zeta in zetas:
                for snr in (0.04, 0.08, 0.12, 0.16, 0.24):
                    add(
                        method="predictor_corrector",
                        implementation=implementation,
                        prior=prior,
                        name=(
                            f"zeta_{safe_name(f'{zeta:g}')}"
                            f"__snr_{safe_name(f'{snr:g}')}"
                        ),
                        args=("--zeta", f"{zeta:g}", "--pc-snr", f"{snr:g}"),
                    )
            for eps in (0.05, 0.1, 0.2, 0.5, 1.0):
                add(
                    method="ve_ddnm",
                    implementation=implementation,
                    prior=prior,
                    name=f"sampling_eps_{safe_name(f'{eps:g}')}",
                    args=("--sampling-epsilon", f"{eps:g}"),
                )

    for schedule in ("geometric", "edm"):
        add(
            method="padis_dps",
            implementation="public_repo",
            name=f"schedule_{schedule}",
            args=("--noise-schedule", schedule),
        )
        add(
            method="padis_dps",
            implementation="paper",
            name=f"schedule_{schedule}",
            args=("--noise-schedule", schedule),
        )

    return unique_candidates(candidates)


def paper_full_candidates() -> list[Candidate]:
    """Full sampler sweep for paper-mode reconstruction implementations.

    This mirrors the breadth of the LION-physics sampler sweep while keeping
    paper-mode CT semantics fixed: paper data objective, paper data-step
    schedule, paper sigma endpoints, and the default LION experiment geometry.
    The sweep intentionally excludes sigma_min/sigma_max, CT data objective,
    physical scaling, FDK filter, and patch-geometry changes.
    """
    candidates = [
        candidate
        for candidate in current_default_candidates()
        if candidate.implementation == "paper"
    ]

    def add(
        *,
        method: str,
        name: str,
        args: tuple[str, ...],
        prior: str | None = None,
        notes: str = "",
    ) -> None:
        """Provide the add callback used by the enclosing operation."""
        candidates.append(
            sampler_candidate(
                method=method,
                implementation="paper",
                name=name,
                args=args,
                prior=prior,
                notes=notes,
            )
        )

    paper_zeta_values = (
        0.001,
        0.002,
        0.003,
        0.005,
        0.0075,
        0.01,
        0.015,
        0.02,
        0.03,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.3,
        0.5,
        0.75,
        1.0,
    )

    def add_common_sampler_controls(method: str, *, prior: str | None = None) -> None:
        """Add common sampler controls."""
        for num_steps, inner_steps in (
            (50, 20),
            (100, 5),
            (100, 20),
            (200, 5),
            (1000, 1),
        ):
            add(
                method=method,
                prior=prior,
                name=f"nfe_{num_steps}x{inner_steps}",
                args=("--num-steps", str(num_steps), "--inner-steps", str(inner_steps)),
                notes="Change NFE allocation while keeping paper sigma endpoints.",
            )
        for rho in (3, 5, 7, 10):
            add(
                method=method,
                prior=prior,
                name=f"schedule_edm_rho_{safe_name(f'{rho:g}')}",
                args=("--noise-schedule", "edm", "--rho", f"{rho:g}"),
                notes="EDM schedule-shape diagnostic; sigma_min/sigma_max are unchanged.",
            )
        for name, args in (
            (
                "init_fdk_clipped",
                ("--initial-reconstruction", "fdk", "--clip-initial", "--clip-output"),
            ),
            ("init_inverse", ("--initial-reconstruction", "inverse")),
            ("clip_initial", ("--clip-initial",)),
            ("clip_output", ("--clip-output",)),
            ("clip_initial_output", ("--clip-initial", "--clip-output")),
            ("clip_denoised", ("--clip-denoised",)),
            ("clip_state", ("--clip-state",)),
            ("clip_denoised_and_state", ("--clip-denoised", "--clip-state")),
        ):
            add(method=method, prior=prior, name=name, args=args)
        for scale in (0.0, 0.5, 1.5):
            add(
                method=method,
                prior=prior,
                name=f"langevin_noise_scale_{safe_name(f'{scale:g}')}",
                args=("--langevin-noise-scale", f"{scale:g}"),
            )

    for method in ("padis_dps", "whole_image_diffusion"):
        for zeta in paper_zeta_values:
            for eps in (0.25, 0.5, 0.75, 1.0):
                add(
                    method=method,
                    name=(
                        f"core_zeta_{safe_name(f'{zeta:g}')}"
                        f"__eps_{safe_name(f'{eps:g}')}"
                    ),
                    args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
                )
        add_common_sampler_controls(method)

    for prior in ("patch", "whole_image"):
        for zeta in paper_zeta_values:
            for eps in (0.1, 0.25, 0.5, 0.75, 1.0, 1.25):
                add(
                    method="langevin",
                    prior=prior,
                    name=(
                        f"zeta_{safe_name(f'{zeta:g}')}"
                        f"__eps_{safe_name(f'{eps:g}')}"
                    ),
                    args=("--zeta", f"{zeta:g}", "--sampling-epsilon", f"{eps:g}"),
                )
        add_common_sampler_controls("langevin", prior=prior)

        for zeta in paper_zeta_values:
            for snr in (0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.24):
                add(
                    method="predictor_corrector",
                    prior=prior,
                    name=(
                        f"zeta_{safe_name(f'{zeta:g}')}"
                        f"__snr_{safe_name(f'{snr:g}')}"
                    ),
                    args=("--zeta", f"{zeta:g}", "--pc-snr", f"{snr:g}"),
                )
        for rule in ("paper_linear", "score_sde_squared"):
            add(
                method="predictor_corrector",
                prior=prior,
                name=f"pc_step_{safe_name(rule)}",
                args=("--pc-corrector-step-rule", rule),
            )
        for sigma_choice in ("next", "current"):
            add(
                method="predictor_corrector",
                prior=prior,
                name=f"pc_denoise_sigma_{sigma_choice}",
                args=("--pc-corrector-denoise-sigma", sigma_choice),
            )
        add(
            method="predictor_corrector",
            prior=prior,
            name="pc_reuse_predictor_layout",
            args=("--pc-reuse-predictor-layout",),
        )
        add_common_sampler_controls("predictor_corrector", prior=prior)

        for eps in (0.025, 0.05, 0.1, 0.2, 0.5, 1.0):
            add(
                method="ve_ddnm",
                prior=prior,
                name=f"sampling_eps_{safe_name(f'{eps:g}')}",
                args=("--sampling-epsilon", f"{eps:g}"),
            )
        for name, args in (
            ("ddnm_no_pinv_clip", ("--no-ddnm-pseudoinverse-clip",)),
            (
                "ddnm_no_projected_pinv_clip",
                ("--no-ddnm-projected-pseudoinverse-clip",),
            ),
            ("ddnm_corrected_clip", ("--ddnm-corrected-clip",)),
            ("ddnm_public_inner", ("--ve-ddnm-nfe-layout", "public_inner")),
            (
                "ddnm_init_fdk_clipped",
                ("--initial-reconstruction", "fdk", "--clip-initial", "--clip-output"),
            ),
        ):
            add(method="ve_ddnm", prior=prior, name=name, args=args)
        add_common_sampler_controls("ve_ddnm", prior=prior)

    return unique_candidates(candidates)


def public_repo_full_candidates() -> list[Candidate]:
    """Full sampler sweep for public-compatible reconstruction implementations.

    This keeps public-compatible CT semantics fixed, including the calibrated
    public LION-geometry data-consistency scales and public FDK/filter setup.
    The sweep excludes sigma_min/sigma_max, CT data objective, CT scaling, FDK
    filter, and patch-geometry changes.
    """
    candidates = [
        candidate
        for candidate in current_default_candidates()
        if candidate.implementation == "public_repo"
    ]

    def add(
        *,
        method: str,
        name: str,
        args: tuple[str, ...],
        prior: str | None = None,
        notes: str = "",
    ) -> None:
        """Provide the add callback used by the enclosing operation."""
        candidates.append(
            sampler_candidate(
                method=method,
                implementation="public_repo",
                name=name,
                args=args,
                prior=prior,
                notes=notes,
            )
        )

    public_zeta_values = (
        0.05,
        0.075,
        0.1,
        0.125,
        0.15,
        0.175,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.75,
    )

    def add_common_sampler_controls(method: str, *, prior: str | None = None) -> None:
        """Add common sampler controls."""
        for num_steps, inner_steps in (
            (50, 20),
            (100, 5),
            (100, 20),
            (200, 5),
            (1000, 1),
        ):
            add(
                method=method,
                prior=prior,
                name=f"nfe_{num_steps}x{inner_steps}",
                args=("--num-steps", str(num_steps), "--inner-steps", str(inner_steps)),
                notes="Change NFE allocation while keeping sigma endpoints fixed.",
            )
        for rho in (3, 5, 7, 10):
            add(
                method=method,
                prior=prior,
                name=f"schedule_edm_rho_{safe_name(f'{rho:g}')}",
                args=("--noise-schedule", "edm", "--rho", f"{rho:g}"),
                notes="EDM schedule-shape diagnostic; sigma_min/sigma_max are unchanged.",
            )
        for name, args in (
            ("init_noise", ("--initial-reconstruction", "noise")),
            ("init_inverse", ("--initial-reconstruction", "inverse")),
            ("no_clip_initial", ("--no-clip-initial",)),
            ("no_clip_output", ("--no-clip-output",)),
            ("clip_denoised", ("--clip-denoised",)),
            ("clip_state", ("--clip-state",)),
            ("clip_denoised_and_state", ("--clip-denoised", "--clip-state")),
        ):
            add(method=method, prior=prior, name=name, args=args)
        for scale in (0.0, 0.5, 1.5):
            add(
                method=method,
                prior=prior,
                name=f"langevin_noise_scale_{safe_name(f'{scale:g}')}",
                args=("--langevin-noise-scale", f"{scale:g}"),
            )

    for method in ("padis_dps", "patch_average", "patch_stitch"):
        for zeta in public_zeta_values:
            for eps in (0.25, 0.5, 0.75, 1.0):
                add(
                    method=method,
                    name=(
                        f"core_zeta_{safe_name(f'{zeta:g}')}"
                        f"__eps_{safe_name(f'{eps:g}')}"
                    ),
                    args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
                )
        add_common_sampler_controls(method)

    for prior in ("patch", "whole_image"):
        for zeta in public_zeta_values:
            for eps in (0.1, 0.25, 0.5, 0.75, 1.0, 1.25):
                add(
                    method="langevin",
                    prior=prior,
                    name=(
                        f"zeta_{safe_name(f'{zeta:g}')}"
                        f"__eps_{safe_name(f'{eps:g}')}"
                    ),
                    args=("--zeta", f"{zeta:g}", "--sampling-epsilon", f"{eps:g}"),
                )
        add_common_sampler_controls("langevin", prior=prior)

        for zeta in public_zeta_values:
            for snr in (0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.24):
                add(
                    method="predictor_corrector",
                    prior=prior,
                    name=(
                        f"zeta_{safe_name(f'{zeta:g}')}"
                        f"__snr_{safe_name(f'{snr:g}')}"
                    ),
                    args=("--zeta", f"{zeta:g}", "--pc-snr", f"{snr:g}"),
                )
        for rule in ("paper_linear", "score_sde_squared"):
            add(
                method="predictor_corrector",
                prior=prior,
                name=f"pc_step_{safe_name(rule)}",
                args=("--pc-corrector-step-rule", rule),
            )
        for sigma_choice in ("next", "current"):
            add(
                method="predictor_corrector",
                prior=prior,
                name=f"pc_denoise_sigma_{sigma_choice}",
                args=("--pc-corrector-denoise-sigma", sigma_choice),
            )
        for layout_flag, name in (
            ("--pc-reuse-predictor-layout", "pc_reuse_predictor_layout"),
            ("--no-pc-reuse-predictor-layout", "pc_no_reuse_predictor_layout"),
        ):
            add(
                method="predictor_corrector",
                prior=prior,
                name=name,
                args=(layout_flag,),
            )
        add_common_sampler_controls("predictor_corrector", prior=prior)

        for eps in (0.025, 0.05, 0.1, 0.2, 0.5, 1.0):
            add(
                method="ve_ddnm",
                prior=prior,
                name=f"sampling_eps_{safe_name(f'{eps:g}')}",
                args=("--sampling-epsilon", f"{eps:g}"),
            )
        for name, args in (
            ("ddnm_no_pinv_clip", ("--no-ddnm-pseudoinverse-clip",)),
            (
                "ddnm_no_projected_pinv_clip",
                ("--no-ddnm-projected-pseudoinverse-clip",),
            ),
            ("ddnm_corrected_clip", ("--ddnm-corrected-clip",)),
            ("ddnm_paper_layout", ("--ve-ddnm-nfe-layout", "paper_1000x1")),
            ("ddnm_init_noise", ("--initial-reconstruction", "noise")),
        ):
            add(method="ve_ddnm", prior=prior, name=name, args=args)
        add_common_sampler_controls("ve_ddnm", prior=prior)

    return unique_candidates(candidates)
