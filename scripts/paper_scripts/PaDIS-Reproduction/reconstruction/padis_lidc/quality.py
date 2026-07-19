"""Quality gates for PaDIS LIDC reconstruction runs."""

from __future__ import annotations


def enforce_quality_gates(args, summaries: list[dict]) -> None:
    """Enforce quality gates."""
    failures = []
    for summary in summaries:
        name = summary["name"]
        mean_psnr = summary["mean_psnr"]
        if args.min_mean_psnr is not None and mean_psnr < args.min_mean_psnr:
            failures.append(
                f"{name}: mean PSNR {mean_psnr:.4g} dB < {args.min_mean_psnr:.4g} dB"
            )

        mean_mae = summary["mean_mae"]
        if args.max_mean_mae is not None and mean_mae > args.max_mean_mae:
            failures.append(
                f"{name}: mean MAE {mean_mae:.4g} > {args.max_mean_mae:.4g}"
            )

        max_mae = summary["max_mae"]
        if args.max_sample_mae is not None and max_mae > args.max_sample_mae:
            failures.append(
                f"{name}: maximum sample MAE {max_mae:.4g} > "
                f"{args.max_sample_mae:.4g}"
            )

        mean_abs_error_p95 = summary["mean_abs_error_p95"]
        if (
            args.max_mean_abs_error_p95 is not None
            and mean_abs_error_p95 > args.max_mean_abs_error_p95
        ):
            failures.append(
                f"{name}: mean p95 abs error {mean_abs_error_p95:.4g} > "
                f"{args.max_mean_abs_error_p95:.4g}"
            )

        max_abs_error_p95 = summary["max_abs_error_p95"]
        if (
            args.max_sample_abs_error_p95 is not None
            and max_abs_error_p95 > args.max_sample_abs_error_p95
        ):
            failures.append(
                f"{name}: maximum sample p95 abs error "
                f"{max_abs_error_p95:.4g} > "
                f"{args.max_sample_abs_error_p95:.4g}"
            )

        mean_ssim = summary.get("mean_ssim")
        if args.min_mean_ssim is not None:
            if mean_ssim is None:
                failures.append(f"{name}: SSIM was not computed")
            elif mean_ssim < args.min_mean_ssim:
                failures.append(
                    f"{name}: mean SSIM {mean_ssim:.4g} < {args.min_mean_ssim:.4g}"
                )

        min_ssim = summary.get("min_ssim")
        if args.min_sample_ssim is not None:
            if min_ssim is None:
                failures.append(f"{name}: sample SSIM was not computed")
            elif min_ssim < args.min_sample_ssim:
                failures.append(
                    f"{name}: minimum sample SSIM {min_ssim:.4g} < "
                    f"{args.min_sample_ssim:.4g}"
                )

        mean_edge_ssim = summary.get("mean_edge_ssim")
        if args.min_mean_edge_ssim is not None:
            if mean_edge_ssim is None:
                failures.append(f"{name}: edge SSIM was not computed")
            elif mean_edge_ssim < args.min_mean_edge_ssim:
                failures.append(
                    f"{name}: mean edge SSIM {mean_edge_ssim:.4g} < "
                    f"{args.min_mean_edge_ssim:.4g}"
                )

        min_edge_ssim = summary.get("min_edge_ssim")
        if args.min_sample_edge_ssim is not None:
            if min_edge_ssim is None:
                failures.append(f"{name}: sample edge SSIM was not computed")
            elif min_edge_ssim < args.min_sample_edge_ssim:
                failures.append(
                    f"{name}: minimum sample edge SSIM {min_edge_ssim:.4g} < "
                    f"{args.min_sample_edge_ssim:.4g}"
                )

        mean_reference_ssim = summary.get("mean_reference_ssim")
        if args.min_mean_reference_ssim is not None:
            if mean_reference_ssim is None:
                failures.append(f"{name}: public-reference SSIM was not computed")
            elif mean_reference_ssim < args.min_mean_reference_ssim:
                failures.append(
                    f"{name}: public-reference mean SSIM "
                    f"{mean_reference_ssim:.4g} < "
                    f"{args.min_mean_reference_ssim:.4g}"
                )

        min_reference_ssim = summary.get("min_reference_ssim")
        if args.min_sample_reference_ssim is not None:
            if min_reference_ssim is None:
                failures.append(
                    f"{name}: public-reference sample SSIM was not computed"
                )
            elif min_reference_ssim < args.min_sample_reference_ssim:
                failures.append(
                    f"{name}: public-reference minimum sample SSIM "
                    f"{min_reference_ssim:.4g} < "
                    f"{args.min_sample_reference_ssim:.4g}"
                )

        mean_reference_edge_ssim = summary.get("mean_reference_edge_ssim")
        if args.min_mean_reference_edge_ssim is not None:
            if mean_reference_edge_ssim is None:
                failures.append(f"{name}: public-reference edge SSIM was not computed")
            elif mean_reference_edge_ssim < args.min_mean_reference_edge_ssim:
                failures.append(
                    f"{name}: public-reference mean edge SSIM "
                    f"{mean_reference_edge_ssim:.4g} < "
                    f"{args.min_mean_reference_edge_ssim:.4g}"
                )

        mean_reference_mae = summary.get("mean_reference_mae")
        if args.max_mean_reference_mae is not None:
            if mean_reference_mae is None:
                failures.append(f"{name}: public-reference MAE was not computed")
            elif mean_reference_mae > args.max_mean_reference_mae:
                failures.append(
                    f"{name}: public-reference mean MAE "
                    f"{mean_reference_mae:.4g} > "
                    f"{args.max_mean_reference_mae:.4g}"
                )

        mean_reference_abs_error_p95 = summary.get("mean_reference_abs_error_p95")
        if args.max_mean_reference_abs_error_p95 is not None:
            if mean_reference_abs_error_p95 is None:
                failures.append(
                    f"{name}: public-reference p95 abs error was not computed"
                )
            elif mean_reference_abs_error_p95 > args.max_mean_reference_abs_error_p95:
                failures.append(
                    f"{name}: public-reference mean p95 abs error "
                    f"{mean_reference_abs_error_p95:.4g} > "
                    f"{args.max_mean_reference_abs_error_p95:.4g}"
                )

        if args.require_better_than_fdk:
            mean_fdk_psnr = summary["mean_fdk_psnr"]
            if mean_psnr <= mean_fdk_psnr:
                failures.append(
                    f"{name}: PaDIS mean PSNR {mean_psnr:.4g} dB <= "
                    f"FDK mean PSNR {mean_fdk_psnr:.4g} dB"
                )
        if args.min_sample_psnr is not None:
            min_psnr = summary["min_psnr"]
            if min_psnr < args.min_sample_psnr:
                failures.append(
                    f"{name}: minimum sample PSNR {min_psnr:.4g} dB < "
                    f"{args.min_sample_psnr:.4g} dB"
                )
        if args.require_each_better_than_fdk and summary["min_fdk_margin"] <= 0:
            failures.append(
                f"{name}: at least one PaDIS sample did not improve over FDK "
                f"(minimum margin {summary['min_fdk_margin']:.4g} dB)"
            )

    if failures:
        message = "Quality gate failed:\n  " + "\n  ".join(failures)
        raise RuntimeError(message)
