"""Run PaDIS DPS or Langevin CT reconstruction on LIDC-IDRI splits."""

from __future__ import annotations

import argparse
import pathlib

from padis_lidc.experiments import (
    DEFAULT_CHECKPOINT,
    EXPERIMENT_ALIASES,
    GEOMETRY_CHOICES,
    IMPLEMENTATION_CHOICES,
    LIDC_EXPERIMENTS,
    LION_EXPERIMENTS_PATH,
    PAPER_CT_EXPERIMENTS,
    RECONSTRUCTION_METHOD_CHOICES,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the PaDIS/LION reconstruction command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    experiment_choices = sorted(
        {
            "none",
            *LIDC_EXPERIMENTS.keys(),
            *PAPER_CT_EXPERIMENTS.keys(),
            *EXPERIMENT_ALIASES.keys(),
        }
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-folder", type=pathlib.Path, default=None)
    parser.add_argument(
        "--image-scaling",
        type=float,
        default=0.5,
        help=(
            "Fallback LIDC image scaling when a method does not need a PaDIS "
            "checkpoint and no checkpoint sidecar JSON is available."
        ),
    )
    parser.add_argument(
        "--output-folder",
        type=pathlib.Path,
        default=LION_EXPERIMENTS_PATH / "PaDIS" / "LIDC_reconstruction",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--experiment",
        choices=experiment_choices,
        default="none",
        help=(
            "Run a LION LIDC CT experiment. Paper aliases include ct_8, "
            "ct_20, ct_60, ct_20_limited_angle_120, and ct_512_60. "
            "image_scaling is read from the PaDIS checkpoint geometry."
        ),
    )
    parser.add_argument(
        "--implementation",
        choices=IMPLEMENTATION_CHOICES,
        default="custom",
        help=(
            "Sampler preset. 'paper' uses the CT schedule described by Hu et al. and "
            "data step; 'public_repo' keeps the PaDIS README reconstruction "
            "mechanics but uses the CT sigma schedule of Hu et al.; "
            "'lion_physics' uses LION CT operators with operator-normalized "
            "least-squares data updates and no public matching constants; "
            "'lion_quality' is the LION-native quality preset; 'custom' uses "
            "the explicit sampler flags."
        ),
    )
    parser.add_argument(
        "--geometry",
        choices=GEOMETRY_CHOICES,
        default="lion",
        help=(
            "CT geometry family for paper CT experiment aliases. Only 'lion' "
            "is currently executable for LIDC-IDRI. PaDIS geometry tags are "
            "accepted only to fail with a physical-correctness explanation."
        ),
    )
    parser.add_argument(
        "--matrix-group",
        default="main",
        help=(
            "Matrix row/group label propagated from "
            "PaDIS_run_reconstruction_matrix.py for verification."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--algorithm",
        choices=("dps_langevin", "langevin", "pc"),
        default="dps_langevin",
    )
    parser.add_argument(
        "--method",
        choices=RECONSTRUCTION_METHOD_CHOICES,
        default="padis_dps",
        help=(
            "Paper-comparison reconstruction method. Diffusion methods reuse "
            "the PaDIS sampler with method-specific prior/algorithm settings; "
            "baseline, CP-TV, and PnP-ADMM use LION-native reconstruction paths."
        ),
    )
    parser.add_argument(
        "--prior-mode",
        choices=("auto", "patch", "whole-image"),
        default="auto",
        help="Use checkpoint prior mode, or override with patch PaDIS / whole-image diffusion.",
    )
    parser.add_argument(
        "--measurement-source",
        choices=("normal", "reconstruction"),
        default="normal",
        help="Manual dataset mode only. Ignored when --experiment is set.",
    )
    parser.add_argument(
        "--public-padis-image-dir",
        type=pathlib.Path,
        default=None,
        help="Use PaDIS-style PNG slices as the image-prior dataset.",
    )
    parser.add_argument(
        "--public-reference-reconstructions",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional public PaDIS reference reconstructions as .npz, .pt, or "
            "a single PNG. When supplied, public-reference similarity metrics "
            "and gates can be used."
        ),
    )
    parser.add_argument("--noise", choices=("none", "low-dose"), default="none")
    parser.add_argument("--noise-i0", type=float, default=3500)
    parser.add_argument("--noise-sigma", type=float, default=5)
    parser.add_argument("--noise-cross-talk", type=float, default=0.05)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=25,
        help="Number of test/validation slices to reconstruct. Default 25 matches the CT evaluation budget of Hu et al.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument(
        "--paper-ct-sampling",
        action="store_true",
        help=(
            "Deprecated alias for --implementation paper. Use the strict "
            "PaDIS paper CT sampler: noise init, 100 outer steps, 10 inner "
            "steps, sigma_max=10."
        ),
    )
    parser.add_argument(
        "--public-padis-ct-sampling",
        action="store_true",
        help=(
            "Deprecated alias for --implementation public_repo. Use the "
            "public PaDIS CT-script compatibility sampler mechanics with the "
            "paper CT sigma schedule."
        ),
    )
    parser.add_argument(
        "--public-repo-sigma-schedule",
        choices=("paper", "readme"),
        default="paper",
        help=(
            "Sigma schedule for --implementation public_repo. Default 'paper' "
            "uses the geometric CT schedule of Hu et al.; 'readme' uses the literal "
            "public README/default EDM schedule for legacy comparisons."
        ),
    )
    parser.add_argument(
        "--public-repo-helper-initialization",
        action="store_true",
        help=(
            "For --implementation public_repo and the public helper methods "
            "predictor_corrector/langevin/ve_ddnm, use the helper functions' "
            "Gaussian initial state instead of the README DPS FDK initial "
            "state. This is for output-level comparisons against "
            "PaDIS_lion_recon --sampler pc|langevin|ddnm."
        ),
    )
    parser.add_argument(
        "--lion-quality-ct-sampling",
        action="store_true",
        help=(
            "Deprecated alias for --implementation lion_quality. "
            "Use the preferred LION-native CT sampler: paper CT schedule and "
            "squared-residual objective with FDK init, Hann 0.9 filtering, "
            "and operator-norm data-consistency scaling."
        ),
    )
    parser.add_argument(
        "--paper-ct-views",
        type=int,
        choices=(8, 20),
        default=20,
        help="Select the PaDIS paper CT sigma_min for --paper-ct-sampling.",
    )
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--inner-steps", type=int, default=None)
    parser.add_argument("--sigma-min", type=float, default=None)
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument(
        "--noise-schedule",
        choices=("edm", "geometric"),
        default=None,
        help="Sigma schedule. Hu et al. use geometric; the public PaDIS script uses edm/rho.",
    )
    parser.add_argument("--zeta", type=float, default=None)
    parser.add_argument(
        "--initial-reconstruction",
        choices=("noise", "fdk", "inverse"),
        default=None,
        help=(
            "Override sampler initialization. Paper preset defaults to noise; "
            "public/default presets default to FDK."
        ),
    )
    parser.add_argument(
        "--initial-fdk-filter-type",
        choices=("none", "ram-lak", "hann", "hamming", "cosine", "shepp-logan"),
        default=None,
        help="Optional FDK ramp-window filter for FDK initialization.",
    )
    parser.add_argument(
        "--initial-fdk-frequency-scaling",
        type=float,
        default=None,
        help="Cutoff frequency fraction for the initial FDK filter window.",
    )
    parser.set_defaults(initial_fdk_padded=None)
    parser.add_argument(
        "--initial-fdk-padded",
        dest="initial_fdk_padded",
        action="store_true",
        help="Pad projections during initial FDK filtering.",
    )
    parser.add_argument(
        "--no-initial-fdk-padded",
        dest="initial_fdk_padded",
        action="store_false",
        help="Do not pad projections during initial FDK filtering.",
    )
    parser.add_argument("--initial-fdk-batch-size", type=int, default=None)
    parser.add_argument("--dps-epsilon", type=float, default=None)
    parser.add_argument("--sampling-epsilon", type=float, default=None)
    parser.add_argument(
        "--data-consistency-gradient",
        choices=("norm", "least_squares", "paper_squared_residual"),
        default=None,
        help=(
            "DPS measurement gradient. Hu et al. use paper_squared_residual; "
            "LION-physics uses least_squares; the public repo uses norm."
        ),
    )
    parser.add_argument(
        "--adjoint-data-step-schedule",
        choices=("paper", "public_repo"),
        default=None,
        help="Adjoint data-step schedule for Langevin/PC samplers.",
    )
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--pad-width", type=int, default=None)
    parser.add_argument("--patch-batch-size", type=int, default=None)
    parser.set_defaults(patch_checkpoint_denoiser=None)
    parser.add_argument(
        "--patch-checkpoint-denoiser",
        dest="patch_checkpoint_denoiser",
        action="store_true",
        help=(
            "Use activation checkpointing for ordinary PaDIS patch denoiser "
            "batches. This reduces peak memory during DPS data-gradient steps "
            "without changing the CT objective or sigma schedule."
        ),
    )
    parser.add_argument(
        "--no-patch-checkpoint-denoiser",
        dest="patch_checkpoint_denoiser",
        action="store_false",
        help="Disable activation checkpointing for ordinary PaDIS patch denoiser batches.",
    )
    parser.add_argument(
        "--patch-assembly",
        choices=("padis", "fixed_average", "fixed_stitch"),
        default=None,
        help=(
            "Patch score assembly mode. The default PaDIS mode uses shifted "
            "non-overlapping layouts; fixed_average/fixed_stitch implement the "
            "paper comparison patch averaging/stitching forms."
        ),
    )
    parser.add_argument(
        "--patch-overlap",
        type=int,
        default=None,
        help="Overlap in pixels for fixed patch averaging/stitching. Paper default is 8.",
    )
    parser.add_argument(
        "--fixed-overlap-layout",
        choices=("lion_clipped", "public_overlap", "public_tile"),
        default=None,
        help=(
            "Patch start rule for fixed patch averaging/stitching. "
            "public_overlap/public_tile mirror the public PaDIS helper "
            "functions; lion_clipped is the original LION-safe default."
        ),
    )
    parser.set_defaults(fixed_overlap_checkpoint_denoiser=None)
    parser.add_argument(
        "--fixed-overlap-checkpoint-denoiser",
        dest="fixed_overlap_checkpoint_denoiser",
        action="store_true",
        help=(
            "Use activation checkpointing for fixed-overlap patch "
            "averaging/stitching denoiser batches. This preserves gradients "
            "but recomputes model batches during the DPS data-gradient step to "
            "reduce peak memory."
        ),
    )
    parser.add_argument(
        "--no-fixed-overlap-checkpoint-denoiser",
        dest="fixed_overlap_checkpoint_denoiser",
        action="store_false",
        help=(
            "Disable activation checkpointing for fixed-overlap patch "
            "averaging/stitching denoiser batches."
        ),
    )
    parser.add_argument(
        "--langevin-ddnm",
        action="store_true",
        help="Use VE-DDNM correction inside the Langevin sampler.",
    )
    parser.set_defaults(
        ddnm_pseudoinverse_clip=None,
        ddnm_projected_pseudoinverse_clip=None,
        ddnm_corrected_clip=None,
    )
    parser.add_argument(
        "--ddnm-pseudoinverse-clip",
        dest="ddnm_pseudoinverse_clip",
        action="store_true",
        help=(
            "Clip the measured pseudoinverse A^dagger y used by VE-DDNM. "
            "Enabled by default for --method ve_ddnm under LION fan-beam geometry."
        ),
    )
    parser.add_argument(
        "--no-ddnm-pseudoinverse-clip",
        dest="ddnm_pseudoinverse_clip",
        action="store_false",
        help="Disable clipping of the measured VE-DDNM pseudoinverse.",
    )
    parser.add_argument(
        "--ddnm-projected-pseudoinverse-clip",
        dest="ddnm_projected_pseudoinverse_clip",
        action="store_true",
        help=(
            "Clip the A^dagger A(D) pseudoinverse term used by VE-DDNM. "
            "Enabled by default for --method ve_ddnm to keep LION fan-beam "
            "runs finite."
        ),
    )
    parser.add_argument(
        "--no-ddnm-projected-pseudoinverse-clip",
        dest="ddnm_projected_pseudoinverse_clip",
        action="store_false",
        help=(
            "Disable clipping of the A^dagger A(D) VE-DDNM term. This is "
            "closer to the formula of Hu et al. and the public implementation but can be unstable with "
            "LION fan-beam FDK."
        ),
    )
    parser.add_argument(
        "--ddnm-corrected-clip",
        dest="ddnm_corrected_clip",
        action="store_true",
        help=(
            "Clip the corrected VE-DDNM clean estimate "
            "A^dagger y + D - A^dagger A(D) before forming the score. "
            "This is a LION-stability diagnostic, not the formula of Hu et al."
        ),
    )
    parser.add_argument(
        "--no-ddnm-corrected-clip",
        dest="ddnm_corrected_clip",
        action="store_false",
        help="Disable clipping of the corrected VE-DDNM clean estimate.",
    )
    parser.add_argument(
        "--diagnose-ddnm-pseudoinverse",
        action="store_true",
        help=(
            "Record DDNM pseudoinverse diagnostics in metrics.json by applying "
            "A^dagger y + x - A^dagger A(x) to the target image. This is a "
            "debugging aid for checking whether the LION pseudoinverse is "
            "accurate enough for VE-DDNM."
        ),
    )
    parser.add_argument(
        "--ve-ddnm-nfe-layout",
        choices=("paper_1000x1", "public_inner"),
        default=None,
        help=(
            "How VE-DDNM spends its 1000 neural function evaluations. "
            "paper_1000x1 uses 1000 descending sigma levels with one denoise "
            "per level, matching Algorithm A.3 literally. public_inner uses "
            "100 outer sigma levels and 10 inner denoising updates per level, "
            "matching the public helper implementation."
        ),
    )
    parser.add_argument("--langevin-noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--pc-corrector-step-rule",
        choices=("paper_linear", "score_sde_squared"),
        default="paper_linear",
        help=(
            "Predictor-corrector corrector step-size rule. Hu et al. and "
            "public PaDIS code use paper_linear; score_sde_squared is retained "
            "only as a diagnostic score-SDE variant."
        ),
    )
    parser.add_argument(
        "--pc-snr",
        type=float,
        default=None,
        help=(
            "Signal-to-noise ratio used by the predictor-corrector "
            "corrector step. Defaults to the sampler preset value."
        ),
    )
    parser.add_argument(
        "--pc-corrector-denoise-sigma",
        choices=("next", "current"),
        default=None,
        help=(
            "Sigma used for the PC corrector denoising call. Paper mode uses "
            "next; public-repo compatibility uses current to mirror the "
            "published code."
        ),
    )
    parser.set_defaults(pc_reuse_predictor_layout=None)
    parser.add_argument(
        "--pc-reuse-predictor-layout",
        dest="pc_reuse_predictor_layout",
        action="store_true",
        help=(
            "Reuse the predictor patch layout for the PC corrector denoising "
            "call. This mirrors the public PaDIS helper; paper mode leaves it "
            "disabled unless this flag is passed."
        ),
    )
    parser.add_argument(
        "--no-pc-reuse-predictor-layout",
        dest="pc_reuse_predictor_layout",
        action="store_false",
        help="Disable PC predictor-layout reuse for the corrector denoising call.",
    )
    parser.add_argument("--data-range", type=float, default=1.0)
    parser.add_argument(
        "--body-threshold",
        type=float,
        default=0.02,
        help="Target-domain threshold for body/anatomy ROI metrics.",
    )
    parser.add_argument(
        "--nonair-threshold",
        type=float,
        default=1e-4,
        help="Target-domain threshold for non-air ROI metrics.",
    )
    parser.add_argument("--body-bbox-padding", type=int, default=8)
    parser.add_argument("--error-vmax", type=float, default=0.10)
    parser.add_argument(
        "--preview-vmax",
        type=float,
        default=0.75,
        help="Upper display window for fixed-window preview comparison images.",
    )
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument(
        "--no-position-channels",
        action="store_true",
        help="Construct the PaDIS prior without x/y position inputs. The checkpoint must use the same architecture.",
    )
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--prog-bar", action="store_true")
    parser.set_defaults(clip_initial=None, clip_output=None)
    parser.add_argument("--clip-initial", dest="clip_initial", action="store_true")
    parser.add_argument("--no-clip-initial", dest="clip_initial", action="store_false")
    parser.add_argument("--clip-output", dest="clip_output", action="store_true")
    parser.add_argument("--no-clip-output", dest="clip_output", action="store_false")
    parser.add_argument(
        "--clip-denoised",
        action="store_true",
        help="Clamp each clean denoised patch-assembled estimate to the model data range before score/data updates.",
    )
    parser.add_argument(
        "--clip-state",
        action="store_true",
        help="Clamp the noisy sampler state to the model data range after each update. Intended as an ablation.",
    )
    parser.add_argument("--disable-data-consistency", action="store_true")
    parser.add_argument("--disable-langevin-noise", action="store_true")
    parser.add_argument("--disable-prior-score", action="store_true")
    parser.add_argument(
        "--tv-lambda",
        type=float,
        default=0.001,
        help="TV regularisation weight. Hu et al. use 0.001 for CT; fixed-validation also selected 0.001 for the LION TV substitute.",
    )
    parser.add_argument("--tv-iterations", type=int, default=1000)
    parser.add_argument("--tv-lipschitz", type=float, default=None)
    parser.add_argument("--tv-non-negativity", action="store_true")
    parser.add_argument(
        "--pnp-checkpoint",
        type=pathlib.Path,
        default=None,
        help="DRUNet denoiser checkpoint for --method pnp_admm.",
    )
    parser.add_argument("--pnp-iterations", type=int, default=60)
    parser.add_argument("--pnp-eta", type=float, default=3e-5)
    parser.add_argument("--pnp-cg-iterations", type=int, default=50)
    parser.add_argument("--pnp-cg-tolerance", type=float, default=1e-7)
    parser.set_defaults(pnp_clip=True)
    parser.add_argument(
        "--pnp-clip",
        dest="pnp_clip",
        action="store_true",
        help="Clip PnP-ADMM iterates and denoiser outputs to the normalized image support [0, 1].",
    )
    parser.add_argument(
        "--no-pnp-clip",
        dest="pnp_clip",
        action="store_false",
        help="Disable PnP-ADMM iterate clipping.",
    )
    parser.add_argument(
        "--pnp-noise-level",
        type=float,
        default=None,
        help="Optional denoiser noise-level input for DRUNet checkpoints trained with noise channels.",
    )
    parser.add_argument(
        "--data-consistency-normalization",
        choices=("operator_norm", "operator_lipschitz", "none"),
        default=None,
        help=(
            "Optionally scale data-consistency updates by the composed "
            "measurement operator norm or least-squares Lipschitz constant."
        ),
    )
    parser.add_argument(
        "--data-consistency-scale",
        type=float,
        default=None,
        help="Extra multiplier after data-consistency normalisation.",
    )
    parser.add_argument(
        "--adjoint-data-consistency-scale",
        type=float,
        default=None,
        help=(
            "Optional separate multiplier for direct adjoint residual updates "
            "used by Langevin and predictor-corrector. Defaults to "
            "--data-consistency-scale when unset."
        ),
    )
    parser.set_defaults(consume_discarded_measurement_noise=None)
    parser.add_argument(
        "--consume-discarded-measurement-noise",
        dest="consume_discarded_measurement_noise",
        action="store_true",
        help=(
            "Burn the public PaDIS script's zero-noise measurement RNG draw. "
            "This preserves exact public RNG alignment."
        ),
    )
    parser.add_argument(
        "--no-consume-discarded-measurement-noise",
        dest="consume_discarded_measurement_noise",
        action="store_false",
        help=(
            "Skip the public PaDIS script's zero-noise measurement RNG draw. "
            "This keeps the public sampler form but can improve reconstruction quality."
        ),
    )
    parser.add_argument(
        "--data-consistency-scale-schedule",
        choices=("constant", "edm", "inverse_sigma"),
        default="constant",
        help=(
            "Sigma-dependent multiplier for data consistency. 'edm' uses "
            "sigma_data^2/(sigma^2+sigma_data^2); 'inverse_sigma' uses "
            "sigma_min/sigma."
        ),
    )
    parser.add_argument(
        "--data-consistency-scale-power",
        type=float,
        default=1.0,
        help="Power applied to the selected sigma-dependent data-consistency factor.",
    )
    parser.add_argument(
        "--data-consistency-scale-floor",
        type=float,
        default=0.0,
        help="Minimum schedule factor before multiplying by --data-consistency-scale.",
    )
    parser.add_argument(
        "--operator-norm",
        type=float,
        default=None,
        help="Optional precomputed norm of the CT operator.",
    )
    parser.add_argument("--operator-norm-iterations", type=int, default=20)
    parser.add_argument("--operator-norm-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--run-ablations",
        action="store_true",
        help=(
            "Run baseline plus no_data_consistency, no_langevin_noise, "
            "and no_prior_score variants into separate folders."
        ),
    )
    parser.add_argument(
        "--min-mean-psnr",
        type=float,
        default=None,
        help="Fail the run if the mean reconstruction PSNR is below this value.",
    )
    parser.add_argument(
        "--min-mean-ssim",
        type=float,
        default=None,
        help="Fail the run if the mean reconstruction SSIM is below this value.",
    )
    parser.add_argument(
        "--max-mean-mae",
        type=float,
        default=None,
        help="Fail the run if mean normalized MAE to the target is above this value.",
    )
    parser.add_argument(
        "--max-sample-mae",
        type=float,
        default=None,
        help="Fail the run if any normalized MAE to the target is above this value.",
    )
    parser.add_argument(
        "--max-mean-abs-error-p95",
        type=float,
        default=None,
        help="Fail the run if mean target p95 absolute error is above this value.",
    )
    parser.add_argument(
        "--max-sample-abs-error-p95",
        type=float,
        default=None,
        help="Fail the run if any target p95 absolute error is above this value.",
    )
    parser.add_argument(
        "--min-sample-ssim",
        type=float,
        default=None,
        help="Fail the run if any individual reconstruction SSIM is below this value.",
    )
    parser.add_argument(
        "--min-mean-edge-ssim",
        type=float,
        default=None,
        help="Fail the run if the mean Sobel-edge SSIM to the target is below this value.",
    )
    parser.add_argument(
        "--min-sample-edge-ssim",
        type=float,
        default=None,
        help="Fail the run if any Sobel-edge SSIM to the target is below this value.",
    )
    parser.add_argument(
        "--min-mean-reference-ssim",
        type=float,
        default=None,
        help="Fail the run if mean SSIM to --public-reference-reconstructions is below this value.",
    )
    parser.add_argument(
        "--min-sample-reference-ssim",
        type=float,
        default=None,
        help="Fail the run if any SSIM to --public-reference-reconstructions is below this value.",
    )
    parser.add_argument(
        "--min-mean-reference-edge-ssim",
        type=float,
        default=None,
        help="Fail the run if mean Sobel-edge SSIM to the public reference is below this value.",
    )
    parser.add_argument(
        "--max-mean-reference-mae",
        type=float,
        default=None,
        help="Fail the run if mean MAE to --public-reference-reconstructions is above this value.",
    )
    parser.add_argument(
        "--max-mean-reference-abs-error-p95",
        type=float,
        default=None,
        help="Fail the run if mean public-reference p95 absolute error is above this value.",
    )
    parser.add_argument(
        "--min-sample-psnr",
        type=float,
        default=None,
        help="Fail the run if any individual reconstruction PSNR is below this value.",
    )
    parser.add_argument(
        "--require-better-than-fdk",
        action="store_true",
        help="Fail the run if PaDIS does not improve mean PSNR over FDK.",
    )
    parser.add_argument(
        "--require-each-better-than-fdk",
        action="store_true",
        help="Fail the run if any individual PaDIS reconstruction does not improve PSNR over FDK.",
    )
    parser.add_argument(
        "--trace-interval",
        type=int,
        default=0,
        help="Save sampler diagnostics every N outer steps. Set 1 for every step.",
    )
    parser.add_argument(
        "--trace-images",
        action="store_true",
        help=(
            "Save denoised, projected, and forward-projected trace snapshots. "
            "Uses --trace-interval; defaults to every 5 outer steps when no "
            "trace interval is set."
        ),
    )
    parser.add_argument(
        "--stop-after-outer-steps",
        type=int,
        default=None,
        help=(
            "Debugging aid: stop after this many outer sampler steps while "
            "preserving the full configured sigma schedule."
        ),
    )
    return parser
