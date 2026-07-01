# PaDIS LIDC Reconstruction Setup

This note documents the PaDIS reconstruction setup used for paper-aligned
experiments and public-repository compatibility checks while still using
LION-native CT operations for the final LION reconstruction path. The
reconstruction script supports both paper-described and public-repository
behavior through an explicit `--implementation` switch.

The verified script is:

```bash
scripts/paper_scripts/PaDIS/PaDIS_LIDC_reconstruction.py
```

The matched public-compatible reference was generated with the companion repo:

```bash
/home/thomas/DiS/Project/PaDIS_lion_recon/inverse_nodist.py
```

## Verified Result

The latest 3-sample verification used the checkpoint:

```bash
/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/patch_lidc_default_10h_local/padis_lidc_256.pt
```

The standalone verifier reported:

| Metric | Value |
|---|---:|
| Mean target PSNR | 33.1606 dB |
| Minimum target PSNR | 32.7360 dB |
| Mean target SSIM | 0.8335 |
| Minimum target SSIM | 0.8075 |
| Mean target MAE | 0.01498 |
| Maximum target p95 absolute error | 0.04587 |
| Mean public-reference SSIM | 0.9960 |
| Minimum public-reference SSIM | 0.9944 |
| Mean public-reference MAE | 0.00160 |
| Maximum public-reference p95 absolute error | 0.00628 |

Images from that run are written under:

```bash
/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/reconstruction_lion_public_preset_strict_visual_gates_3sample/PaDISFanBeam20CTRecon/test/dps_langevin
```

New runs that use `--experiment ct_20 --method padis_dps` write to:

```bash
/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/reconstruction_lion_public_preset_strict_visual_gates_3sample/ct_20/test/padis_dps/dps_langevin
```

Useful files in that directory include:

```bash
sample_0000_visual_compare.png
sample_0001_visual_compare.png
sample_0002_visual_compare.png
quality_verification_public_lion_fanbeam/quality_verification.json
quality_verification_public_lion_fanbeam/sample_0000_visual_compare.png
quality_verification_public_lion_fanbeam/sample_0001_visual_compare.png
quality_verification_public_lion_fanbeam/sample_0002_visual_compare.png
```

## Differences From The Paper

The paper is the primary source of truth for what PaDIS was intended to do.
However, the public repository and README command execute several details
differently. The working LION setup intentionally follows the public repository
where that is needed to reproduce the observed reconstruction behavior.

| Item | Paper | PaDIS README / public repo execution | Working LION setup | Match status |
|---|---|---|---|---|
| Dataset and image scale | CT images are scaled to the range `[0, 1]`. Main CT experiments use AAPM data. | README reconstruction consumes PNG test images scaled to `[0, 1]`. | Uses PaDIS-style LIDC PNG slices scaled to `[0, 1]`. | Agrees on scaling; dataset differs from the paper main CT experiments. |
| Reconstruction geometry | Main CT experiments use parallel beam, 20 or 8 views, detector size 512. Extra experiments include 60-view parallel beam and 180-view fan beam. | README command uses `ct_parbeam`, 20 views, detector size 512. | `--geometry lion` keeps LION fan beam with detector 900. PaDIS geometry tags are accepted only to fail with a physical-correctness warning. | PaDIS geometry is not implemented for LIDC-IDRI because the processed slices do not contain enough physical metadata to convert them correctly. |
| CT operator implementation | Paper says CT projectors are provided by an external implementation. | Public repo uses ODL/Astra operators. | Final reconstruction uses LION-native CT forward projection and FDK. | Intentional LION-native divergence. |
| Noise schedule type | Geometrically spaced descending noise levels. | README command executes an EDM/Karras-style power schedule with `rho=7`; it is not geometric in code unless changed by command-line arguments. | `--implementation paper`, `--implementation public_repo`, and `--implementation lion_physics` use the paper geometric schedule in the reconstruction array. | Matches the paper schedule for array runs; diverges from literal README defaults. |
| CT sigma range | For 20-view CT: `sigma_max=10`, `sigma_min=0.002`. For 8-view CT: `sigma_min=0.003`. 60-view and fan-beam CT use the main 20-view CT schedule. | README command shows `sigma_min=0.003` for 20 views, but this is command-line configurable. | `--implementation paper`, `--implementation public_repo`, and `--implementation lion_physics` use the paper sigma range per experiment. | Matches the paper schedule for array runs; diverges from literal README defaults. |
| Sampler length | 100 outer steps and 10 inner steps, about 1000 neural function evaluations. | README command uses `steps=100`; public DPS performs 10 inner denoising steps per outer step. | Uses 100 outer steps and 10 inner steps. | Matches. |
| Initial sampler state | Paper Langevin pseudocode initializes from Gaussian noise. FBP is discussed as a baseline, not as the sampler initial state. | Public CT DPS computes a filtered backprojection-style reconstruction, clips it, pads it, and starts from that state. | Uses LION-native FDK initialization, clipped to `[0, 1]`, then padded. | Matches public behavior conceptually; differs from the paper. |
| FBP / FDK filter | Not applicable to paper sampler initialization. | Public parallel-beam path uses Hann FBP frequency scaling `0.9`; public fan-beam path uses `0.3`. | `--implementation public_repo --geometry lion` uses Hann FDK `0.3`. | Public-compatible for the validated LION geometry path; differs from the paper. |
| Reconstruction method | PaDIS with DPS or Langevin-style data consistency. | Public README reconstruction uses the `dps()` path in `inverse_nodist.py`. | Uses LION `dps_langevin`. | Matches the public DPS/Langevin method family. |
| Langevin / DPS epsilon | Paper states `epsilon=1` for Langevin and DDNM. | Public DPS code uses `alpha = 0.5 * sigma^2`. | `--implementation paper` uses `dps_epsilon=1`. `--implementation public_repo` uses `dps_epsilon=0.5`. `--implementation lion_physics` keeps `dps_epsilon=1` except for validated stability defaults: fixed-overlap patch rows and the whole-image `ct_fanbeam_180` row use `0.5`. | Toggle implemented; the LION-physics exceptions are documented paper divergences. |
| Data consistency objective | Paper pseudocode applies an adjoint residual step. A strict paper-style LION preset corresponds to a squared-residual objective with residual-normalized step size. | Public DPS uses the gradient of the L2 norm of the residual, computed from `y - A(x0hat)`. | `--implementation paper` uses the squared-residual objective. `--implementation public_repo` uses the norm-gradient DPS objective. `--implementation lion_physics` uses the least-squares objective `0.5 * ||y - A(x0hat)||^2`. | Paper/public toggles implemented; LION-physics intentionally uses the standard least-squares CT data term. |
| Data step size | Paper describes `zeta_i = 0.3 / L2Norm(y - A(x))` for Langevin and PC-style data steps. | Public DPS applies `x = x - zeta * grad(L2Norm(y - A(x0hat)))` with `zeta=0.3`; the norm-gradient already normalizes the gradient direction. Public PC/Langevin helpers use explicit adjoint residual steps. | `--implementation paper` uses residual-normalized paper stepping. `--implementation public_repo --geometry lion` uses calibrated CT norm-gradient scale `0.0405` for DPS and calibrated direct-adjoint scale `0.1022` for PC/Langevin. `--implementation lion_physics` uses `zeta / L` with `L = ||F||^2 = (abs(measurement_scale) * ||A||)^2` for the composed LION measurement map `F(x)=A(measurement_scale*x + measurement_offset)`. The affine offset is not part of the Lipschitz scale. Current method defaults are DPS `zeta=3.0`, PC `zeta=4.25`, and Langevin `zeta=4.0`. | Paper/public toggles implemented. LION-physics avoids public-repo calibration constants but diverges from the paper's `0.3` coefficient. |
| Public-compatible LION-geometry reference | Not in paper. | Not in the original README path. | The companion `PaDIS_lion_recon` repo adds `ct_lion_fanbeam` / `ct_lion_parbeam`; these use `data_gradient_scale=0.09` to normalize the ODL adjoint scale for LION geometry comparisons. | Compatibility shim only; not paper. |
| Patch offsets and random draws | Paper does not specify exact RNG consumption. | Public code uses Python-style patch offset behavior and consumes several otherwise-unused random draws. | LION public preset mirrors those offset and RNG-consumption semantics. | Matches public repo; not paper-specified. |
| Output clipping | Not a central paper reconstruction detail. | Public repo clamps reconstructions to `[0, 1]`. | LION clips initial and final reconstructions to `[0, 1]`. | Matches public behavior. |
| Verified behavior | Paper reports aggregate CT reconstruction quality. | Matched public-compatible LION-geometry 3-sample reference has mean target PSNR 33.12 dB and mean target SSIM 0.836. | LION 3-sample run has mean target PSNR 33.16 dB, mean target SSIM 0.834, and mean public-reference SSIM 0.996. | Working setup is validated against both target and public-compatible reference. |

## Implementation And Geometry Switches

Use the reconstruction script from the LION root:

```bash
scripts/paper_scripts/PaDIS/PaDIS_LIDC_reconstruction.py
```

The important new switches are:

| Switch | Values | Meaning |
|---|---|---|
| `--implementation` | `paper` | Paper-described reconstruction: geometric CT noise schedule, paper CT sigma values, Gaussian initialization, squared-residual data objective, `epsilon=1`. |
| `--implementation` | `public_repo` | Public README/code reconstruction mechanics with paper CT sigma scheduling by default: FDK/FBP initialization, norm-gradient DPS objective, public RNG/patch-offset behavior, geometric schedule, and paper CT sigma range. |
| `--implementation` | `lion_physics` | LION-native physical CT preset: paper geometric CT sigma schedule, LION fan-beam forward/adjoint/FDK, least-squares data objective, and data steps normalized by the composed LION measurement Lipschitz constant. Uses tuned method-specific sampler settings and no public-repo matching constants. |
| `--public-repo-sigma-schedule` | `paper` / `readme` | Select the sigma schedule for `--implementation public_repo`. The array default is `paper`; `readme` is only for reproducing literal public README/default runs. |
| `--public-repo-helper-initialization` | flag | Diagnostic output-comparison mode for `--implementation public_repo` with `predictor_corrector`, `langevin`, or `ve_ddnm`. It uses the public helper functions' Gaussian initial-state convention instead of the README DPS FDK initial-state convention: PC samples central-image noise and then pads it; Langevin/DDNM sample the already padded state. |
| `--implementation` | `lion_quality` | Legacy LION-native quality preset retained for diagnostics. The method-default matrix now uses `lion_physics` for VE-DDNM; `lion_quality` remains selectable to reproduce the earlier stabilized fallback. |
| `--implementation` | `custom` | Use explicit low-level sampler flags. |
| `--geometry` | `lion` | LION fan-beam CT geometry: detector 900, full 360 degree angular range, LION-native FDK. This is the only executable LIDC-IDRI geometry. |
| `--geometry` | `padis` | Accepted only to raise an explicit error. The LIDC-IDRI tensors used here cannot be converted into the PaDIS public-repo geometry in a physically correct way. |
| `--geometry` | `padis_parallel` / `padis_fanbeam` | Also accepted only to raise the same explicit physical-correctness error. |
| `--experiment` | `ct_8`, `ct_20`, `ct_60`, `ct_fanbeam_180`, `ct_512_60` | Paper-facing CT experiment aliases. Old class names such as `PaDISFanBeam20CTRecon` are still accepted as aliases. |
| `--ddnm-corrected-clip` | flag | Diagnostic VE-DDNM stabilization that clips `A^dagger y + D - A^dagger A(D)` before forming the score. This is not part of Algorithm A.3 and is only for LION fan-beam troubleshooting. |
| `--ve-ddnm-nfe-layout` | `paper_1000x1` / `public_inner` | Controls how VE-DDNM spends 1000 NFEs. `paper_1000x1` uses one denoise per sigma level and is the paper-mode default; `public_inner` uses the public helper's 100 outer levels with 10 inner updates. |
| `--diagnose-ddnm-pseudoinverse` | flag | Adds metrics for the perfect-denoiser DDNM correction `A^dagger y + x - A^dagger A(x)` evaluated at the target image. This is a diagnostic for LION pseudoinverse quality, not a reconstruction method. |
| `--fixed-overlap-checkpoint-denoiser` | flag | Recompute fixed-overlap patch denoiser microbatches during the DPS backward pass to reduce memory. This is enabled by default for `patch_average` and `patch_stitch`; use `--no-fixed-overlap-checkpoint-denoiser` only for diagnostics. |

The older flags `--paper-ct-sampling`, `--public-padis-ct-sampling`, and
`--lion-quality-ct-sampling` remain available as deprecated aliases.

## Current LION-Physics Performance Snapshot

These are local validation metrics from the recorded debug runs, not the final
25-sample A100 paper matrix. They are included to show that the LION-native
least-squares/operator-Lipschitz data updates are in the same quality band as
the public-compatible rows without using the public CT matching constants.

| Row | Experiment | Samples | Comparator PSNR / SSIM | LION-physics PSNR / SSIM | Status |
|---|---|---:|---:|---:|---|
| PaDIS DPS vs public-compatible DPS | `ct_20` | 3 | 33.16 / 0.834 | 32.33 / 0.829 | Close; small PSNR drop, similar SSIM |
| Langevin vs public-compatible Langevin | `ct_20` | 3 | 32.66 / 0.819 | 32.92 / 0.837 | Slightly better |
| Predictor-corrector vs public-compatible PC | `ct_20` | 3 | 29.94 / 0.729 | 29.79 / 0.747 | Same quality band; better SSIM |
| VE-DDNM vs legacy `lion_quality` | `ct_20` | 1 | 33.09 / 0.772 | 33.06 / 0.771 | Essentially identical |
| Whole-image diffusion, strict paper vs LION-physics | `ct_20` | 1 vs 3 | 9.67 / -0.002 | 31.84 / 0.812 | LION-physics much better |
| Patch average vs public-helper row | `ct_20` | 1 | 33.30 / 0.801 | 33.77 / 0.813 | Better |
| Patch stitch vs public-helper row | `ct_20` | 1 | 32.89 / 0.787 | 32.33 / 0.789 | Slight PSNR drop, same SSIM band |
| PnP-ADMM old custom run vs clipped matrix row | `ct_20` | 1 | 28.90 / 0.669 | 29.65 / 0.694 | Better |

Cross-experiment LION-physics checks against the LION FDK baseline:

| Method | Experiment | Samples | FDK PSNR / SSIM | LION-physics PSNR / SSIM |
|---|---|---:|---:|---:|
| PaDIS DPS | `ct_8` | 3 | 16.71 / 0.208 | 26.23 / 0.696 |
| PaDIS DPS | `ct_60` | 3 | 27.37 / 0.581 | 35.40 / 0.893 |
| PaDIS DPS | `ct_fanbeam_180` | 3 | 34.68 / 0.872 | 36.52 / 0.907 |
| Whole-image diffusion | `ct_8` | 3 | 16.71 / 0.208 | 26.17 / 0.671 |
| Whole-image diffusion | `ct_60` | 3 | 27.37 / 0.581 | 34.84 / 0.872 |
| Whole-image diffusion | `ct_fanbeam_180` | 3 | 34.68 / 0.872 | 36.21 / 0.892 |

The remaining validation gap is not the physical CT scaling path itself; it is
coverage. The full 25-sample A100 reconstruction matrix still needs to run, and
the `ct_512_60` diffusion row still has only memory/dispatch evidence because
the linked 512 checkpoint is a smoke checkpoint rather than a trained 512 prior.

## Paper Method Rows

`PaDIS_LIDC_reconstruction.py` exposes the paper comparison rows through
`--method`:

| Method | Training dependency | LION implementation | Paper agreement |
|---|---|---|---|
| `baseline` | None | LION FDK/FBP-style analytic reconstruction from the CT operator | Paper says CT baseline is FBP. In LION fan-beam geometry this is implemented with FDK, the corresponding fan-beam analytic baseline. |
| `admm_tv` | None | LION `tv_min` total-variation reconstruction | Similar method only: current LION TV uses Chambolle-Pock, not the paper's ADMM-TV solver. Uses paper CT `lambda=0.001` by default. |
| `pnp_admm` | DRUNet denoiser | LION `PnP(..., algorithm="ADMM")` with a DRUNet denoiser wrapper | Requires a trained denoising CNN. Agreement depends on the denoiser training run; this is not the PaDIS diffusion checkpoint. The PaDIS driver clips PnP-ADMM iterates and denoiser outputs to `[0, 1]`, the normalized LIDC image support, to keep sparse-view ADMM iterates inside the denoiser training domain. |
| `whole_image_diffusion` | Whole-image NCSN++ min-validation checkpoint | LION PaDIS sampler with `prior_mode=whole_image` | Method default now uses `--implementation lion_physics` with the paper geometric CT sigma schedule, LION fan-beam operator/FDK, least-squares data objective, and operator-Lipschitz-normalized data steps. The reconstruction matrix expects `whole_image_lidc_256_min_val.pt`, because the min-validation checkpoint is the validated whole-image reconstruction checkpoint. The `ct_fanbeam_180` whole-image row uses `dps_epsilon=0.5` after a one-slice fanbeam diagnostic improved PSNR while preserving the physical data objective. |
| `langevin` | Patch PaDIS checkpoint | LION PaDIS Langevin sampler | Method default now uses `--implementation lion_physics`: paper geometric CT sigma schedule, LION fan-beam operator/FDK, least-squares data objective, and operator-Lipschitz-normalized direct-adjoint data steps. It uses tuned `zeta=4.0` and `sampling_epsilon=0.5`, which are paper divergences validated against the public-compatible Langevin row. |
| `predictor_corrector` | Patch PaDIS checkpoint | LION PaDIS predictor-corrector sampler | Method default now uses `--implementation lion_physics`: paper geometric CT sigma schedule, LION fan-beam operator/FDK, least-squares data objective, and operator-Lipschitz-normalized direct-adjoint data steps. It uses tuned `zeta=4.25` and `pc_snr=0.08`, denoises the corrector at the next/lower sigma, and does not reuse the public helper patch layout. |
| `ve_ddnm` | Patch PaDIS checkpoint | LION PaDIS Langevin sampler with VE-DDNM correction | Method default now uses `--implementation lion_physics`. It keeps the paper `paper_1000x1` NFE layout but uses LION fan-beam stabilization: noise initialization, clipped LION FDK pseudoinverse terms, clipped corrected DDNM estimate, and `sampling_epsilon=0.1`. This remains a documented paper divergence caused by the LION fan-beam pseudoinverse. |
| `patch_average` | Patch PaDIS checkpoint | Fixed-overlap patch denoising with averaged overlap pixels | Method default now uses `--implementation lion_physics` with the validated public-overlap denoiser layout, checkpointed fixed-overlap denoising, LION fan-beam least-squares data consistency, and `dps_epsilon=0.5`. The driver defaults to `patch_batch_size=1` for this fixed-overlap row unless overridden; fixed-overlap denoising streams patches in chunks of this size for memory, which is not a CT scaling change. The overlap layout mirrors the public PaDIS `denoisedOverlap(...)` helper but the CT update is LION-native and operator-normalized. This is not a faithful implementation of the original conditional patch-DDPM paper cited as `[23]`. |
| `patch_stitch` | Patch PaDIS checkpoint | Fixed-overlap patch denoising with overwrite/stitching semantics | Method default now uses `--implementation lion_physics` with the validated public-tile denoiser layout, checkpointed fixed-overlap denoising, LION fan-beam least-squares data consistency, and `dps_epsilon=0.5`. The driver defaults to `patch_batch_size=1` for this fixed-overlap row unless overridden; fixed-overlap denoising streams patches in chunks of this size for memory, which is not a CT scaling change. The tile layout mirrors the public PaDIS `denoisedTile(...)` helper but the CT update is LION-native and operator-normalized. This is not a faithful implementation of the original tile-and-stitch paper cited as `[66]`. |
| `padis_dps` | Patch PaDIS checkpoint | Main PaDIS patch sampler with DPS/Langevin-style data consistency | Method default now uses `--implementation lion_physics` with the paper geometric CT sigma schedule, LION fan-beam operator/FDK, least-squares data objective, and data steps normalized by the composed LION measurement Lipschitz constant. The `ct_512_60` row defaults to `patch_batch_size=1` plus `patch_checkpoint_denoiser=True` as a memory-only control for ordinary PaDIS patch denoising; this does not change the CT objective or sigma schedule. |

The Slurm reconstruction matrix defaults to `PADIS_RECON_METHODS=all`,
`PADIS_RECON_MODELS=method_default`, and
`PADIS_RECON_IMPLEMENTATIONS=lion_physics`. This means each method selects the
checkpoint family it needs while the reconstruction layer records and uses the
LION-native physical CT preset wherever a sampler implementation is relevant:
patch PaDIS for patch methods, whole-image PaDIS for `whole_image_diffusion`, a
DRUNet denoiser for `pnp_admm`, and LION-physics CT settings for `padis_dps`,
`langevin`, `predictor_corrector`, `ve_ddnm`, `patch_average`, and
`patch_stitch`. To force strict paper mechanics for a diagnostic run, set
`PADIS_RECON_IMPLEMENTATIONS=paper` or pass `--implementations paper` to
`PaDIS_run_reconstruction_matrix.py`. The Python matrix CLI still accepts
`--implementations method_default` for reproducing method-specific diagnostic
defaults.
With `PADIS_RECON_EXPERIMENTS=paper_matrix`, the default array has 26 jobs:
main-table methods run on the main CT experiments, extra-table methods run on
the extra CT experiments they are reported for, and the 512 CT jobs use
`patch_lidc_512`.

## Train The Required Models

To train every checkpoint family required by the reconstruction matrix without
launching reconstruction, use the all-training submitter:

```bash
scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_all_training.sh
```

This submits the PaDIS diffusion-prior training array and the DRUNet denoiser
job used by the `pnp_admm` row into the same training root. Set
`PADIS_SUBMIT_PNP_TRAINING=0` only if you already have a suitable PnP denoiser
checkpoint or have excluded `pnp_admm` from reconstruction.

The full pipeline submitter launches checks, cache preparation, pilot training,
real PaDIS training, and the PnP denoiser job by default:

```bash
scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_pipeline.sh
```

To build the reusable single-file tensor caches and `.pt.zst` archives on a
machine without Slurm, run:

```bash
scripts/paper_scripts/PaDIS/run_prepare_lidc_cache.sh
```

By default this builds `256-default`, `256-full`, and `512-default` under
`$LION_DATA_PATH/processed/LIDC-IDRI-cache`, matching the Slurm cache-prep job.
Set `PADIS_CACHE_PREP_VARIANTS`, `PADIS_CACHE_ROOT`, `PADIS_DATA_FOLDER`, or
`PYTHON` to override the local defaults.

Set `PADIS_SUBMIT_PNP_TRAINING=0` to disable the PnP denoiser job in either
submitter. To submit the reconstruction matrix automatically after real
training and PnP training finish, enable the opt-in reconstruction chain:

```bash
PADIS_SUBMIT_RECONSTRUCTION=1 \
PADIS_RECON_METHODS=all \
PADIS_RECON_MODELS=method_default \
PADIS_RECON_IMPLEMENTATIONS=lion_physics \
PADIS_RECON_GEOMETRIES=lion \
PADIS_RECON_MAX_SAMPLES=25 \
scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_pipeline.sh
```

With `PADIS_SUBMIT_RECONSTRUCTION=1`, the pipeline sets
`PADIS_RECON_VERIFY=1` by default and submits the reconstruction verifier after
the reconstruction array. If the selected reconstruction matrix contains
`pnp_admm`, the pipeline also requires either `PADIS_SUBMIT_PNP_TRAINING=1` or
an existing `PADIS_PNP_CHECKPOINT`; otherwise it fails at submission time rather
than launching an array that cannot complete.

For targeted reruns, the PaDIS patch and whole-image diffusion checkpoints are
trained by the diffusion-only array:

```bash
scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_training_only.sh
```

The PnP-ADMM row additionally needs a DRUNet denoiser checkpoint. Train only
that denoiser into the same training root with:

```bash
PADIS_TRAIN_ROOT=/path/to/a100_training_<stamp> \
scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_pnp_training.sh
```

The reconstruction matrix expects the denoiser at:

```bash
/path/to/a100_training_<stamp>/pnp_lidc_drunet/pnp_lidc_drunet.pt
```

### GCP Spot Training

For a retained `/mnt/data` GCP spot instance, run the local spot orchestrator
from the LION root:

```bash
scripts/paper_scripts/PaDIS/gcp/run_PaDIS_GCP_spot_training.sh
```

On the retained GCP image, the runner auto-activates the `lion` Conda
environment from `/mnt/data/conda` if no Conda environment is already active.
You can still activate it manually first if preferred.

Defaults match the retained instance layout:

```text
LION_DATA_PATH=/mnt/data/Datasets
PADIS_TRAIN_ROOT=/mnt/data/Datasets/experiments/PaDIS/final_real_runs/gcp_spot_training
PADIS_RAM_DISK=/mnt/ram-disk
```

Mount the RAM cache directory before running:

```bash
sudo mkdir -p /mnt/ram-disk
sudo mount -t tmpfs -o size=300g tmpfs /mnt/ram-disk
```

The runner uses up to four visible GPUs and assigns one model per GPU. It trains
patch-based PaDIS models for 6 hours each, whole-image PaDIS models for 18
hours each, and trains the PnP DRUNet to its epoch target. Rerunning the same
command resumes from retained checkpoints and state under `PADIS_TRAIN_ROOT`.

Durable training state stays under `/mnt/data`; only staged LIDC tensor caches
live under `/mnt/ram-disk`. If prepared cache archives are absent, the runner
can build the RAM caches from `/mnt/data/Datasets/processed/LIDC-IDRI`.

Checkpoint behavior:

```text
Periodic resume checkpoint: every 10 minutes
Periodic checkpoints kept during training: 2
Periodic checkpoints kept after completion: 1
Final lightweight checkpoint: kept
Final full training-state checkpoint: kept
Best-validation checkpoint: kept
```

Useful overrides:

```bash
PADIS_GCP_GPU_IDS=0,1,2,3 \
PADIS_GCP_RUN_NAME=gcp_spot_training_20260701 \
PADIS_WANDB_MODE=online \
scripts/paper_scripts/PaDIS/gcp/run_PaDIS_GCP_spot_training.sh
```

### PnP DRUNet Training Defaults

The PnP denoiser script is:

```bash
scripts/paper_scripts/PaDIS/PaDIS_LIDC_PnP_denoiser.py
```

Current defaults are:

| Setting | Default | Notes |
|---|---:|---|
| Validation frequency | 1 epoch | Keeps the best-validation model responsive after the validation-loss fix. |
| Periodic checkpoint frequency | 10 epochs | Periodic resume checkpoints are intentionally sparse. |
| Retained periodic checkpoints | 5 | Use `--max-periodic-checkpoints -1` to keep all periodic checkpoints. |
| Best-validation checkpoint | `pnp_lidc_drunet_min_val.pt` | Saved independently of periodic checkpoint retention. |
| Full final checkpoint | `pnp_lidc_drunet_full.pt` | Contains optimizer state for resuming from the final trained model. |
| W&B resume metadata | `wandb_run.json` in the run folder | Existing run IDs are reused automatically unless `--wandb-id` is given. Metrics use W&B's resumed step counter with `epoch` as the chart axis. |

The local corrected-validation DRUNet run was stopped after epoch 50 completed
and resumed to 100 total epochs with a 7-hour wall-clock budget:

```bash
conda run --no-capture-output -n lion-dev env \
  PYTHONPATH=/home/thomas/DiS/Project/LION \
  MPLCONFIGDIR=/tmp/padis-mpl \
  XDG_CACHE_HOME=/tmp/padis-xdg \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/paper_scripts/PaDIS/PaDIS_LIDC_PnP_denoiser.py \
  --output-root /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_pnp_fixed_validation_20260629 \
  --run-name pnp_drunet_fixed_val_6h_wandb_local \
  --epochs 100 \
  --max-train-seconds 25200 \
  --batch-size 8 \
  --max-slices-per-patient 4 \
  --validation-every 1 \
  --checkpoint-every 10 \
  --max-periodic-checkpoints 5 \
  --device cuda \
  --num-workers 4 \
  --final-name pnp_lidc_drunet.pt \
  --validation-name pnp_lidc_drunet_min_val.pt \
  --wandb-project PaDIS-LIDC \
  --wandb-name pnp_drunet_fixed_val_6h_wandb_local \
  --wandb-id yzjn69ku
```

The resumed local run logs to:

```text
https://wandb.ai/tjh200-university-of-cambridge/PaDIS-LIDC/runs/yzjn69ku
```

After checking the interrupted resume, the remote run was explicitly verified
through W&B's API to contain epoch 50 under the same run id before restarting
from the epoch-50 checkpoint.

The resume issue was traced to manual W&B step logging. The script now logs
metrics without passing a manual `step` and defines `epoch` as the W&B metric
axis. After the repair, W&B API checks verified the same run id at epochs 80,
90, and 100. The completed run has `lastHistoryStep=100`, 100 history rows,
summary `epoch=100`, final `train_loss=8.738872675601616e-05`, final
`validation_loss=0.0001103682602217482`, and
`min_validation_loss=7.537099122046352e-05`.

At resume, the script loaded:

```bash
/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_pnp_fixed_validation_20260629/pnp_drunet_fixed_val_6h_wandb_local/pnp_lidc_drunet_check_0050.pt
```

and restored the previous best validation loss from
`pnp_lidc_drunet_min_val.pt`, so the first resumed validation epoch does not
overwrite the best model unless it is actually better.

The final model and retained resume checkpoints are:

```bash
/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_pnp_fixed_validation_20260629/pnp_drunet_fixed_val_6h_wandb_local/pnp_lidc_drunet.pt
/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_pnp_fixed_validation_20260629/pnp_drunet_fixed_val_6h_wandb_local/pnp_lidc_drunet_min_val.pt
/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_pnp_fixed_validation_20260629/pnp_drunet_fixed_val_6h_wandb_local/pnp_lidc_drunet_check_0060.pt
/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_pnp_fixed_validation_20260629/pnp_drunet_fixed_val_6h_wandb_local/pnp_lidc_drunet_check_0070.pt
/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_pnp_fixed_validation_20260629/pnp_drunet_fixed_val_6h_wandb_local/pnp_lidc_drunet_check_0080.pt
/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_pnp_fixed_validation_20260629/pnp_drunet_fixed_val_6h_wandb_local/pnp_lidc_drunet_check_0090.pt
/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_pnp_fixed_validation_20260629/pnp_drunet_fixed_val_6h_wandb_local/pnp_lidc_drunet_check_0100.pt
```

The `Expected 'geometry' parameter!` message during save is expected for these
denoiser-only checkpoints and did not prevent the final, best-validation, or
periodic checkpoints from being written. The best-validation checkpoint still
uses the legacy scalar-loss checkpoint schema, so its `epoch` field should be
treated as checkpoint metadata rather than a one-indexed W&B epoch label.

The A100 PnP Slurm job now validates every epoch by default, saves periodic
checkpoints every 10 epochs, retains 5 periodic checkpoints, and logs to W&B
unless `PADIS_NO_WANDB=1` is set. The optional wall-clock stop can be supplied
with:

```bash
PADIS_PNP_MAX_TRAIN_SECONDS=25200 \
scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_pnp_training.sh
```

For the matrixed PaDIS diffusion-prior Slurm jobs, W&B model artifacts are now
enabled by default. Set `PADIS_NO_WANDB_ARTIFACT=1` only for runs where artifact
upload should be disabled.

If you customise the PnP denoiser output path, use the same variables for
training and reconstruction:

```bash
PADIS_PNP_OUTPUT_ROOT=/path/to/pnp_outputs \
PADIS_PNP_RUN_NAME=denoiser_run \
PADIS_PNP_FINAL_NAME=final_denoiser.pt
```

The pipeline and standalone reconstruction submitters derive
`PADIS_PNP_CHECKPOINT` as:

```bash
$PADIS_PNP_OUTPUT_ROOT/$PADIS_PNP_RUN_NAME/$PADIS_PNP_FINAL_NAME
```

when the selected matrix contains `pnp_admm` and `PADIS_PNP_CHECKPOINT` is not
set explicitly. This keeps a submitted PnP training job and the later
reconstruction manifest pointed at the same checkpoint.

The PnP Slurm job exposes the training CLI through `PADIS_PNP_*` environment
variables, including `PADIS_PNP_BATCH_SIZE`, `PADIS_PNP_EPOCHS`,
`PADIS_PNP_LR`, `PADIS_PNP_BETA1`, `PADIS_PNP_BETA2`,
`PADIS_PNP_IMAGE_SCALING`, `PADIS_PNP_INT_CHANNELS`,
`PADIS_PNP_N_BLOCKS`, `PADIS_PNP_VALIDATION_EVERY`,
`PADIS_PNP_CHECKPOINT_EVERY`, `PADIS_PNP_FINAL_NAME`,
`PADIS_PNP_CHECKPOINT_PATTERN`, and `PADIS_PNP_VALIDATION_NAME`.

Both training entry points can log to Weights & Biases. For local runs, pass
`--wandb-project PaDIS-LIDC` and optionally `--wandb-name <run_name>`. The PnP
denoiser script also supports `--wandb-entity`, `--wandb-id`, `--wandb-mode`,
`--no-wandb`, and `--max-train-seconds` for clean wall-clock-limited training.
The whole-image diffusion training script supports the same project/name style
and can skip large checkpoint uploads with `--no-wandb-artifact`.

The local six-hour W&B DRUNet run used:

```bash
conda run --no-capture-output -n lion-dev env \
  PYTHONPATH=/home/thomas/DiS/Project/LION \
  MPLCONFIGDIR=/tmp/padis-mpl \
  XDG_CACHE_HOME=/tmp/padis-xdg \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/paper_scripts/PaDIS/PaDIS_LIDC_PnP_denoiser.py \
  --output-root /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_long_training_20260628 \
  --run-name pnp_drunet_6h_wandb_local \
  --epochs 1000 \
  --max-train-seconds 21600 \
  --batch-size 8 \
  --max-slices-per-patient 4 \
  --validation-every 1 \
  --checkpoint-every 1 \
  --device cuda \
  --num-workers 4 \
  --final-name pnp_lidc_drunet.pt \
  --validation-name pnp_lidc_drunet_min_val.pt \
  --wandb-project PaDIS-LIDC \
  --wandb-name pnp_drunet_6h_wandb_local
```

The matching local six-hour W&B whole-image run used:

```bash
conda run --no-capture-output -n lion-dev env \
  PYTHONPATH=/home/thomas/DiS/Project/LION \
  MPLCONFIGDIR=/tmp/padis-mpl \
  XDG_CACHE_HOME=/tmp/padis-xdg \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/paper_scripts/PaDIS/PaDIS_LIDC_256.py \
  --prior-mode whole-image \
  --save-folder /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_long_training_20260628 \
  --run-name whole_image_6h_wandb_local \
  --max-slices-per-patient 4 \
  --max-train-seconds 21600 \
  --batch-size 1 \
  --microbatch-size 1 \
  --validation-interval-patches 64 \
  --validation-max-patches 64 \
  --checkpoint-interval-patches 2048 \
  --log-interval-patches 64 \
  --max-periodic-checkpoints 3 \
  --device cuda \
  --num-workers 4 \
  --wandb-project PaDIS-LIDC \
  --wandb-name whole_image_6h_wandb_local \
  --no-wandb-artifact
```

For a pilot run:

```bash
conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS/PaDIS_LIDC_PnP_denoiser.py \
  --output-root /tmp/padis_pnp_training \
  --run-name pnp_lidc_drunet \
  --epochs 1 \
  --batch-size 2 \
  --max-slices-per-patient 1 \
  --max-train-samples 2 \
  --max-validation-samples 2 \
  --device cuda
```

A tiny CPU smoke test of the same training entry point is:

```bash
python scripts/paper_scripts/PaDIS/PaDIS_LIDC_PnP_denoiser.py \
  --output-root /tmp/padis_pnp_smoke \
  --run-name smoke \
  --epochs 1 \
  --batch-size 1 \
  --max-slices-per-patient 1 \
  --max-train-samples 1 \
  --max-validation-samples 1 \
  --validation-every 999 \
  --checkpoint-every 999 \
  --device cpu \
  --num-workers 0 \
  --int-channels 8 \
  --n-blocks 1
```

The paper-matrix runner expands the shared PaDIS diffusion training task list
used by `submit_PaDIS_A100_all_training.sh` and
`submit_PaDIS_A100_training_only.sh` into reconstruction jobs:

| Training task | Default reconstruction experiments |
|---|---|
| `patch_lidc_default` | `ct_20`, `ct_8`, `ct_60`, `ct_fanbeam_180` |
| `patch_lidc_full` | `ct_20` |
| `patch_lidc_p8_default` | `ct_20` |
| `patch_lidc_p16_default` | `ct_20` |
| `patch_lidc_p32_default` | `ct_20` |
| `patch_lidc_p96_default` | `ct_20` |
| `patch_lidc_no_pos_default` | `ct_20` |
| `whole_lidc_default` | `ct_20`, `ct_8`, `ct_60`, `ct_fanbeam_180` |
| `whole_lidc_full` | `ct_20` |
| `patch_lidc_512` | `ct_512_60` |

When `PADIS_RECON_MODELS=method_default`, the matrix uses the method table
above instead of expanding all trained-model ablations. Set an explicit model
list, for example `PADIS_RECON_MODELS=all`, only when running checkpoint
ablations deliberately.

Explicit `PADIS_RECON_EXPERIMENTS` selections are still checked against the
selected method/model's paper experiment set. For example, `langevin` is a
`ct_20` comparison row, so `PADIS_RECON_METHODS=langevin` with
`PADIS_RECON_EXPERIMENTS=ct_8` fails by default. Set
`PADIS_RECON_ALLOW_OFF_PAPER_EXPERIMENTS=1` only for deliberate diagnostic or
ablation runs outside the paper protocol.

## Why PaDIS Geometry Is Not Implemented

The PaDIS public repo CT geometry is not just a different number of detector
bins. It uses a different physical coordinate system: the public CT operators
place the image on a 40-unit support with an 80-unit detector span, while the
LION LIDC reconstruction geometry uses a 300 mm field of view, detector size
900, DSO 575 mm, and DSD 1050 mm.

The processed LIDC-IDRI slices used by these scripts are saved as `slice_*.npy`
HU arrays. They do not retain enough per-scan physical metadata, such as
pixel spacing and orientation, to derive a physically correct detector/object
transformation into the PaDIS public-repo coordinate system. Rescaling those
arrays into the PaDIS geometry would be a numerical coordinate change, not a
valid CT geometry conversion. For that reason `--geometry padis`,
`--geometry padis_parallel`, and `--geometry padis_fanbeam` fail explicitly.

## Noise Schedule Clarification

The PaDIS paper says inverse-problem sampling uses a geometrically spaced
descending noise level. The PaDIS README command does not execute that schedule.

The README command is:

```bash
python3 inverse_nodist.py \
  --network=training-runs/67-ctaxial/network-snapshot-000800.pkl \
  --outdir=results \
  --image_dir=image_dir \
  --image_size=256 \
  --views=20 \
  --name=ct_parbeam \
  --steps=100 \
  --sigma_min=0.003 \
  --sigma_max=10 \
  --zeta=0.3 \
  --pad=24 \
  --psize=56
```

Since `rho` is not passed, the public script uses its default `rho=7`. The
executed schedule is:

```python
t_steps = (
    sigma_max ** (1 / rho)
    + step_indices / (num_steps - 1)
    * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
) ** rho
```

That is the EDM/Karras-style power schedule. For the reconstruction array,
LION intentionally keeps the public repo reconstruction mechanics but changes
the schedule to the paper setting, because the public script exposes
`--sigma_min`, `--sigma_max`, and `--rho`/schedule-related parameters at the
command line and the paper is the source of truth for the experiment protocol.

## Run The Matched Public-Compatible Reference

Run this from:

```bash
/home/thomas/DiS/Project/PaDIS_lion_recon
```

Command:

```bash
conda run --no-capture-output -n padis-dev env PYTHONPATH=/home/thomas/DiS/Project/LION \
  MPLCONFIGDIR=/tmp/padis-mpl XDG_CACHE_HOME=/tmp/padis-xdg \
  python inverse_nodist.py \
  --network /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/patch_lidc_default_10h_local/padis_lidc_256.pt \
  --lion_repo /home/thomas/DiS/Project/LION \
  --device cuda \
  --ct_impl astra_cuda \
  --image_dir /home/thomas/DiS/Project/Data/processed/LIDC-IDRI-padis-png-256 \
  --outdir /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/reconstruction_public_lion_fanbeam_scaled_default_3sample \
  --name ct_lion_fanbeam \
  --views 20 \
  --steps 100 \
  --sigma_min 0.003 \
  --sigma_max 10 \
  --rho 7 \
  --zeta 0.3 \
  --sigma 0 \
  --intermediate_interval 20 \
  --trace_interval 0 \
  --max_images 3 \
  --seed 2
```

This writes:

```bash
/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/reconstruction_public_lion_fanbeam_scaled_default_3sample/reconstructions.npz
```

That `.npz` file is the public-compatible LION-geometry reference used by the
LION run below.

## Run The Matching LION Reconstruction

Run this from:

```bash
/home/thomas/DiS/Project/LION
```

Command:

```bash
conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS/PaDIS_LIDC_reconstruction.py \
  --checkpoint /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/patch_lidc_default_10h_local/padis_lidc_256.pt \
  --output-folder /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/reconstruction_lion_public_preset_strict_visual_gates_3sample \
  --experiment ct_20 \
  --implementation public_repo \
  --public-repo-sigma-schedule readme \
  --geometry lion \
  --public-padis-image-dir /home/thomas/DiS/Project/Data/processed/LIDC-IDRI-padis-png-256 \
  --public-reference-reconstructions /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/reconstruction_public_lion_fanbeam_scaled_default_3sample/reconstructions.npz \
  --split test \
  --algorithm dps_langevin \
  --max-samples 3 \
  --device cuda \
  --seed 2 \
  --save-previews \
  --prog-bar \
  --trace-interval 20 \
  --trace-images \
  --min-mean-psnr 32.8 \
  --min-sample-psnr 32.5 \
  --min-mean-ssim 0.80 \
  --min-sample-ssim 0.80 \
  --min-mean-edge-ssim 0.45 \
  --min-sample-edge-ssim 0.40 \
  --max-mean-mae 0.016 \
  --max-sample-mae 0.0165 \
  --max-mean-abs-error-p95 0.045 \
  --max-sample-abs-error-p95 0.047 \
  --min-mean-reference-ssim 0.985 \
  --min-sample-reference-ssim 0.98 \
  --min-mean-reference-edge-ssim 0.97 \
  --max-mean-reference-mae 0.003 \
  --max-mean-reference-abs-error-p95 0.009 \
  --require-better-than-fdk \
  --require-each-better-than-fdk
```

This writes:

```bash
/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/reconstruction_lion_public_preset_strict_visual_gates_3sample/ct_20/test/padis_dps/dps_langevin
```

## Run The LION-Physics Reconstruction

This runs the LION-native physical CT preset: LION fan-beam geometry and FDK,
paper geometric CT sigma schedule, least-squares data consistency, and the
measurement Lipschitz normalizer `L = ||F||^2` for the composed measurement map
`F(x)=A(measurement_scale*x + measurement_offset)`. Since the affine offset does
not change Lipschitz constants, the implementation uses
`L = (abs(measurement_scale) * ||A||)^2`, with `||A||` estimated by LION's
generic `power_method` on the actual LION CT operator. The least-squares term is
the summed objective `0.5 * ||F(x) - y||^2`, so this uses `||F||^2` directly
rather than the mean-squared-error scaling `||F||^2 / y.numel()`. It does not
use the public-repo compatibility scale constants.

```bash
conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS/PaDIS_LIDC_reconstruction.py \
  --checkpoint /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/patch_lidc_default_10h_local/padis_lidc_256.pt \
  --output-folder /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/lion_physics_paper_schedule_zeta3 \
  --experiment ct_20 \
  --implementation lion_physics \
  --geometry lion \
  --method padis_dps \
  --algorithm dps_langevin \
  --max-samples 1 \
  --seed 2 \
  --device cuda \
  --prog-bar \
  --save-previews
```

The method-specific physical defaults are DPS `zeta=3.0`, predictor-corrector
`zeta=4.25, pc_snr=0.08`, Langevin `zeta=4.0, sampling_epsilon=0.5`,
VE-DDNM `sampling_epsilon=0.1` with corrected clipping, the whole-image
`ct_fanbeam_180` row with `dps_epsilon=0.5`, and fixed-overlap patch rows with
public overlap/tile denoiser layouts, checkpointed denoising, `patch_batch_size=1`,
and `dps_epsilon=0.5`. Use `--zeta`, `--pc-snr`,
`--sampling-epsilon`, `--dps-epsilon`, `--patch-batch-size`, or
`--fixed-overlap-layout` only for controlled relaxation/layout sweeps.

## Run Matching Public Helper Patch Rows

Patch averaging and patch stitching are exposed through implementation-specific
fixed-overlap layouts. For a quick patch-averaging smoke run:

```bash
conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS/PaDIS_LIDC_reconstruction.py \
  --checkpoint /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/patch_lidc_default_10h_local/padis_lidc_256.pt \
  --output-folder /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/lion_public_helper_patch_average_smoke \
  --experiment ct_20 \
  --implementation public_repo \
  --method patch_average \
  --geometry lion \
  --split test \
  --algorithm dps_langevin \
  --max-samples 1 \
  --device cuda \
  --seed 2 \
  --patch-batch-size 1 \
  --save-previews \
  --prog-bar \
  --trace-interval 1 \
  --trace-images \
  --stop-after-outer-steps 1
```

Use `--method patch_stitch` and a different `--output-folder` for the matching
stitching helper. Under `--implementation public_repo`, LION uses
`fixed_overlap_layout=public_overlap` for `patch_average` and
`fixed_overlap_layout=public_tile` for `patch_stitch`. Under
`--implementation paper`, both rows use the earlier LION `lion_clipped` layout.

## Run The Paper Preset

This runs the paper-described 20-view CT sampler while keeping the LION geometry:

```bash
conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS/PaDIS_LIDC_reconstruction.py \
  --checkpoint /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/patch_lidc_default_10h_local/padis_lidc_256.pt \
  --output-folder /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/reconstruction_lion_paper_preset_example \
  --experiment ct_20 \
  --implementation paper \
  --geometry lion \
  --public-padis-image-dir /home/thomas/DiS/Project/Data/processed/LIDC-IDRI-padis-png-256 \
  --split test \
  --algorithm dps_langevin \
  --max-samples 25 \
  --device cuda \
  --seed 2 \
  --save-previews \
  --prog-bar
```

PaDIS geometry is intentionally not available for this LIDC-IDRI setup. Use
`--geometry lion`.

## Run The Slurm Reconstruction Matrix

After the jobs from
`scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_all_training.sh` have
finished, submit the matching reconstruction array from the LION root. If you
used the diffusion-only submitter instead, make sure the separate PnP denoiser
checkpoint is also available before including `pnp_admm`:

```bash
PADIS_TRAIN_ROOT=/path/to/a100_training_<stamp> \
PADIS_RECON_METHODS=all \
PADIS_RECON_MODELS=method_default \
PADIS_RECON_IMPLEMENTATIONS=lion_physics \
PADIS_RECON_GEOMETRIES=lion \
PADIS_RECON_MAX_SAMPLES=25 \
scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_reconstruction.sh
```

The reconstruction scripts default to `--split test --max-samples 25`, matching
the paper's CT reconstruction evaluation budget. Lower
`PADIS_RECON_MAX_SAMPLES` only for pilot/debug runs.
The Slurm reconstruction array sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default to reduce CUDA
allocator fragmentation in memory-heavy rows such as `ct_512_60`; override the
environment variable before submission if a cluster image requires a different
allocator setting.

For a cheap A100 smoke over methods that do not require a newly trained PnP
denoiser or whole-image prior:

```bash
PADIS_TRAIN_ROOT=/path/to/a100_training_<stamp> \
scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_reconstruction_smoke.sh
```

The smoke defaults to one `ct_20` sample for `baseline`, `admm_tv`,
`padis_dps`, `langevin`, `predictor_corrector`, and `ve_ddnm`, with previews
and trace images enabled. It also submits the post-array verifier by default.
The submitter writes
`reconstruction_matrix_jobs.json`, and the verifier uses that manifest together
with the expected Slurm matrix size and per-job sample count. Missing
`metrics.json` files, wrong job identities, incorrect sampler settings,
incorrect TV/PnP method settings, or incomplete sample outputs fail the
verification job. It writes:

```bash
$PADIS_RECON_ROOT/reconstruction_matrix_verification.json
```

The default smoke also applies method-specific quality gates to the non-baseline
rows that have been locally validated: `admm_tv`, `padis_dps`, `langevin`,
`predictor_corrector`, and `ve_ddnm` must beat FDK in mean PSNR and on the
sample, with conservative method-specific mean PSNR floors:

```bash
PADIS_RECON_VERIFY_REQUIRE_METHOD_MEAN_BETTER_THAN_FDK=admm_tv,padis_dps,langevin,predictor_corrector,ve_ddnm
PADIS_RECON_VERIFY_REQUIRE_METHOD_EACH_BETTER_THAN_FDK=admm_tv,padis_dps,langevin,predictor_corrector,ve_ddnm
PADIS_RECON_VERIFY_MIN_METHOD_MEAN_PSNR="admm_tv=28 padis_dps=33 langevin=32 predictor_corrector=29 ve_ddnm=32"
```

The global switches
`PADIS_RECON_VERIFY_REQUIRE_MEAN_BETTER_THAN_FDK=1` and
`PADIS_RECON_VERIFY_REQUIRE_EACH_BETTER_THAN_FDK=1` apply to every selected
method, including `baseline`, so they are intended for filtered method lists
rather than the default smoke.

For custom smoke subsets, override `PADIS_RECON_METHODS` and either narrow
`PADIS_RECON_SMOKE_QUALITY_METHODS` or set the verifier variables above
explicitly. The verifier also accepts the legacy aliases
`PADIS_RECON_VERIFY_MIN_METHOD_PSNR`, `PADIS_RECON_VERIFY_MIN_METHOD_SSIM`, and
`PADIS_RECON_VERIFY_MAX_METHOD_MAE`, but the explicit `*_MEAN_*` names are
preferred.

After the whole-image or PnP checkpoints exist, override `PADIS_RECON_METHODS`
to include those rows.

Useful selectors:

```bash
PADIS_RECON_METHODS=baseline,admm_tv,padis_dps,whole_image_diffusion
PADIS_RECON_MODELS=method_default
PADIS_RECON_EXPERIMENTS=ct_20,ct_8
PADIS_RECON_IMPLEMENTATIONS=lion_physics
PADIS_RECON_GEOMETRIES=lion
```

Raw reconstruction flags can be passed through the Slurm array with
`PADIS_RECON_EXTRA_ARGS`. Prefer method-specific defaults where possible:
`PADIS_RECON_METHODS=ve_ddnm PADIS_RECON_IMPLEMENTATIONS=lion_physics` runs the
LION-stabilized VE-DDNM row, while
`PADIS_RECON_METHODS=ve_ddnm PADIS_RECON_IMPLEMENTATIONS=paper` runs the
strict paper diagnostic row.

The matrix can also be inspected locally without submitting Slurm:

```bash
python scripts/paper_scripts/PaDIS/PaDIS_run_reconstruction_matrix.py \
  --training-root /path/to/a100_training_<stamp> \
  --output-root /tmp/padis_recon_matrix_preview \
  --models method_default \
  --methods all \
  --implementations lion_physics \
  --geometries lion \
  --count
```

## Run A Local CUDA Matrix Without Slurm

Slurm is not required for local validation. On this machine CUDA is visible
only outside Codex's sandbox, so Codex-run commands need sandbox escalation; in
an ordinary shell the same `conda run` commands below are sufficient.

The existing 10-hour local patch checkpoint can be staged into the matrix
layout with:

```bash
TRAIN_ROOT=/tmp/padis_lion_local_training
mkdir -p "$TRAIN_ROOT/patch_lidc_default"
ln -sf /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/patch_lidc_default_10h_local/padis_lidc_256.pt \
  "$TRAIN_ROOT/patch_lidc_default/padis_lidc_256.pt"
```

That staged root is enough for `baseline`, `admm_tv`, `padis_dps`,
`langevin`, `predictor_corrector`, and the LION-stabilized `ve_ddnm` row. It is
not enough for the full 26-job matrix. A full default matrix additionally
requires:

```bash
$TRAIN_ROOT/pnp_lidc_drunet/pnp_lidc_drunet.pt
$TRAIN_ROOT/whole_lidc_default/whole_image_lidc_256_min_val.pt
$TRAIN_ROOT/patch_lidc_512/padis_lidc_512.pt
```

A one-sample local CUDA smoke over the non-training-dependent rows is:

```bash
OUT_ROOT=/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/local_cuda_smoke
conda run --no-capture-output -n lion-dev env \
  PYTHONPATH=/home/thomas/DiS/Project/LION \
  MPLCONFIGDIR=/tmp/padis-mpl \
  XDG_CACHE_HOME=/tmp/padis-xdg \
  python scripts/paper_scripts/PaDIS/PaDIS_run_reconstruction_matrix.py \
  --training-root "$TRAIN_ROOT" \
  --output-root "$OUT_ROOT" \
  --methods baseline,admm_tv,padis_dps,langevin,predictor_corrector,ve_ddnm \
  --models method_default \
  --experiments ct_20 \
  --implementations lion_physics \
  --geometries lion \
  --max-samples 1 \
  --save-previews \
  --prog-bar \
  --trace-interval 20 \
  --trace-images
```

Save the matching manifest for strict verification:

```bash
conda run --no-capture-output -n lion-dev env \
  PYTHONPATH=/home/thomas/DiS/Project/LION \
  python scripts/paper_scripts/PaDIS/PaDIS_run_reconstruction_matrix.py \
  --training-root "$TRAIN_ROOT" \
  --output-root "$OUT_ROOT" \
  --methods baseline,admm_tv,padis_dps,langevin,predictor_corrector,ve_ddnm \
  --models method_default \
  --experiments ct_20 \
  --implementations lion_physics \
  --geometries lion \
  --max-samples 1 \
  --save-previews \
  --prog-bar \
  --trace-interval 20 \
  --trace-images \
  --list > "$OUT_ROOT/reconstruction_matrix_jobs.json"
```

Then run the verifier:

```bash
conda run --no-capture-output -n lion-dev env \
  PYTHONPATH=/home/thomas/DiS/Project/LION \
  python scripts/paper_scripts/PaDIS/PaDIS_verify_reconstruction_matrix.py \
  --root "$OUT_ROOT" \
  --expected-jobs-json "$OUT_ROOT/reconstruction_matrix_jobs.json" \
  --expected-records 6 \
  --expected-samples 1 \
  --require-methods baseline,admm_tv,padis_dps,langevin,predictor_corrector,ve_ddnm \
  --require-experiments ct_20 \
  --min-method-mean-psnr admm_tv=28 \
  --min-method-mean-psnr padis_dps=33 \
  --min-method-mean-psnr langevin=32 \
  --min-method-mean-psnr predictor_corrector=29 \
  --min-method-mean-psnr ve_ddnm=32 \
  --require-method-mean-better-than-fdk admm_tv,padis_dps,langevin,predictor_corrector,ve_ddnm \
  --require-method-each-better-than-fdk admm_tv,padis_dps,langevin,predictor_corrector,ve_ddnm \
  --output-json "$OUT_ROOT/reconstruction_matrix_verification.json"
```

For the current 25-slice no-prior validation, change only the method list and
sample count:

```bash
OUT_ROOT=/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/local_cuda_baseline_tv_25
conda run --no-capture-output -n lion-dev env \
  PYTHONPATH=/home/thomas/DiS/Project/LION \
  MPLCONFIGDIR=/tmp/padis-mpl \
  XDG_CACHE_HOME=/tmp/padis-xdg \
  python scripts/paper_scripts/PaDIS/PaDIS_run_reconstruction_matrix.py \
  --training-root "$TRAIN_ROOT" \
  --output-root "$OUT_ROOT" \
  --methods baseline,admm_tv \
  --models method_default \
  --experiments ct_20 \
  --implementations lion_physics \
  --geometries lion \
  --max-samples 25 \
  --save-previews \
  --prog-bar \
  --trace-interval 0
```

The PaDIS Slurm scripts default to `LION_MAMBA_ENV=lion-dev` and
`LION_MAMBA_ENV_FALLBACKS=padis-dev`. Override those environment variables if a
different cluster environment should be used.

## Verify A Saved LION Run

Run this from:

```bash
/home/thomas/DiS/Project/LION
```

Command:

```bash
conda run --no-capture-output -n lion-dev python scripts/dev/verify_padis_reconstruction_quality.py \
  --reconstructions /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/reconstruction_lion_public_preset_strict_visual_gates_3sample/PaDISFanBeam20CTRecon/test/dps_langevin/reconstructions.pt \
  --public-reference /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/reconstruction_public_lion_fanbeam_scaled_default_3sample/reconstructions.npz \
  --output-dir /home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/reconstruction_lion_public_preset_strict_visual_gates_3sample/PaDISFanBeam20CTRecon/test/dps_langevin/quality_verification_public_lion_fanbeam \
  --min-mean-target-psnr 33.0 \
  --min-sample-target-ssim 0.80 \
  --max-mean-target-mae 0.016 \
  --max-sample-target-abs-error-p95 0.047 \
  --min-mean-public-ssim 0.995 \
  --max-mean-public-mae 0.002 \
  --max-sample-public-abs-error-p95 0.0065
```

## Inspect Outputs

The most useful visual files are:

```bash
sample_0000_visual_compare.png
sample_0001_visual_compare.png
sample_0002_visual_compare.png
quality_verification_public_lion_fanbeam/sample_0000_visual_compare.png
quality_verification_public_lion_fanbeam/sample_0001_visual_compare.png
quality_verification_public_lion_fanbeam/sample_0002_visual_compare.png
```

Trace images are written under:

```bash
trace_images/
```

Each trace snapshot includes current state, denoised state, data-consistency
projected state, next state, and forward-projected sinogram image.

## Local Validation Evidence

The following checks were run on the local GTX 1070 with CUDA outside the
sandbox. LION-native checks used `lion-dev`; public-fork checks used
`padis-dev` because the fork depends on ODL. They are smoke/quality checks, not
a replacement for the A100 paper matrix.

| Output root | Methods | Samples | Result |
|---|---|---|---|
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_local_cuda_smoke_post_slurm_note_20260628` | `baseline`, `admm_tv`, `padis_dps`, `langevin`, `predictor_corrector`, `ve_ddnm` | 1 `ct_20` test slice per method | No-Slurm local CUDA matrix smoke using `lion-dev` with sandbox escalation and the existing patch PaDIS checkpoint staged under `/tmp/padis_lion_local_training`. The verifier passed 6 records, 1 sample each, exact manifest identities, expected sampler/method settings, method-specific PSNR floors, and better-than-FDK gates for all non-baseline rows. Mean PSNRs: baseline/FDK 22.15 dB, ADMM-TV 29.49 dB, PaDIS DPS 34.10 dB, Langevin 33.57 dB, predictor-corrector 30.26 dB, and LION-stabilized VE-DDNM 33.10 dB. Trace images were written for the diffusion rows. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_512_training_smoke_20260628/train_root/patch_lidc_512` | `patch_lidc_512` training | 1 target patch, 1 LIDC slice per patient | No-Slurm local CUDA smoke of the 512 training entrypoint produced `padis_lidc_512.pt` and `padis_lidc_512_full.pt`. Together with the existing local patch, whole-image, and PnP smoke checkpoints, a temporary staged root at `/tmp/padis_lion_full_matrix_smoke_root` passed `PaDIS_run_reconstruction_matrix.py --check-inputs` for all 26 method-default reconstruction jobs. This validates checkpoint layout and 512 training dispatch only; it is not a paper-quality 512 prior. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_current_smoke_noiseinit_rerun_20260628` | `baseline`, `admm_tv`, `padis_dps`, `langevin`, `predictor_corrector`, `ve_ddnm` | 1 `ct_20` test slice per method | Current-code escaped-CUDA matrix smoke using the existing patch PaDIS checkpoint through a temporary matrix-compatible symlink root. The manifest verifier passed 6 records, 1 sample each, expected sampler/method settings including `noise_initialization`, and method-specific quality gates. Mean PSNRs: baseline/FDK 22.15 dB, ADMM-TV 29.49 dB, PaDIS DPS 34.10 dB, Langevin 33.57 dB, predictor-corrector 30.26 dB, and LION-stabilized VE-DDNM 33.10 dB. All non-baseline gated rows beat FDK on the sample and wrote metrics, tensors, traces, and trace images. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_current_nontraining_smoke_20260628` | `baseline`, `admm_tv`, `padis_dps`, `langevin`, `predictor_corrector`, `ve_ddnm` | 1 `ct_20` test slice per method | Current-code escaped-CUDA matrix smoke using the existing patch PaDIS checkpoint. The manifest verifier passed 6 records, 1 sample each, and method-specific quality gates. Mean PSNRs: baseline/FDK 22.15 dB, ADMM-TV 29.49 dB, PaDIS DPS 34.10 dB, Langevin 33.57 dB, predictor-corrector 30.26 dB, and LION-stabilized VE-DDNM 33.10 dB. All non-baseline gated rows beat FDK on the sample and wrote metrics, tensors, traces, and trace images. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_current_fixed_overlap_probe_20260628` | `patch_average`, `patch_stitch` | 1 `ct_20` test slice per method, stopped after 5 outer steps | Current-code escaped-CUDA fixed-overlap probe using `--patch-batch-size 1`, checkpointed denoising, trace JSON, trace images, previews, and tensors. The manifest verifier passed structurally for both records. Quality is intentionally meaningless for this truncated noise-initialized run; both rows remained far below FDK. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_cuda_training_dependent_rerun_20260628` | `whole_image_diffusion` training, `pnp_admm` denoiser training, `whole_image_diffusion`, `pnp_admm` | 1 tiny training smoke each; 1 `ct_20` reconstruction slice each | Current-code escaped-CUDA smoke for the two training-dependent rows. Whole-image diffusion trained one patch budget unit and produced final and min-validation whole-image checkpoints; PnP trained a deliberately tiny DRUNet and produced `pnp_lidc_drunet/pnp_lidc_drunet.pt`. The verifier passed structurally for both rows and applied a quality gate to PnP-ADMM. PnP-ADMM reached 28.03 dB versus FDK 22.15 dB in this tiny smoke; whole-image diffusion was intentionally not quality-meaningful because the checkpoint was one-step trained and the reconstruction was stopped after 2 outer steps. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_training_watch_20260628/pnp_drunet_watch_local` | DRUNet denoiser training for `pnp_admm` | 64 training samples, 16 validation samples, local subset | Escaped-CUDA monitored training run using the LION-native DRUNet PnP denoiser script. Validation was checked every epoch. The run was manually interrupted after epoch 12 once the validation loss had worsened after the best epoch: validation loss improved from 0.00342 at epoch 1 to a best observed 0.000665 at epoch 9, then rose to 0.000722, 0.000804, and 0.000701 at epochs 10-12 while training loss kept dropping. Saved artifacts include `pnp_lidc_drunet_min_val.pt` plus periodic checkpoints at epochs 5 and 10. No final `pnp_lidc_drunet.pt` was written because the run was intentionally interrupted after the overfitting signal. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_training_watch_20260628/whole_image_watch_local_v2` | Whole-image diffusion training | 224 whole-image samples before 240 s cap | Escaped-CUDA monitored whole-image NCSN++ training run using `--prior-mode whole-image`, batch size 1, validation every 64 samples, and metrics JSONL at `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_training_watch_20260628/whole_image_watch_local_v2_metrics.jsonl`. The run stopped cleanly at `--max-train-seconds 240`. Validation worsened from 44064.8 at 64 samples to 51765.1 at 128 and 52530.0 at 192, so this checkpoint is not quality-useful. It did write final and min-validation checkpoints plus `loss.png` and `validation_loss.png`. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_pnp_watch_recon_20260628` | `pnp_admm` | 1 `ct_20` test slice | Escaped-CUDA reconstruction smoke using the watched DRUNet best-validation checkpoint `pnp_lidc_drunet_min_val.pt`. The PnP-ADMM pipeline completed 10 ADMM iterations and beat FDK on the sample: PSNR 28.90 dB and SSIM 0.669 versus FDK PSNR 22.15 dB and SSIM 0.328. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_whole_image_watch_recon_20260628` | `whole_image_diffusion` | 1 `ct_20` test slice, stopped after 1 outer step | Escaped-CUDA reconstruction smoke using the watched whole-image checkpoint `whole_image_lidc_256_min_val.pt`. The pipeline completed and wrote metrics, tensors, trace JSON, trace images, and previews, but quality was poor: PSNR -21.23 dB versus FDK 22.15 dB. This verifies dispatch/output plumbing only; the short whole-image training watch did not produce a useful prior. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_long_training_20260628/pnp_drunet_6h_wandb_local` | DRUNet denoiser training for `pnp_admm` | 6-hour local CUDA run, 53 epochs | W&B run `wh0t93qz` (`https://wandb.ai/tjh200-university-of-cambridge/PaDIS-LIDC/runs/wh0t93qz`). The run stopped cleanly at `--max-train-seconds 21600`, saved `pnp_lidc_drunet.pt` and `pnp_lidc_drunet_min_val.pt`, and reached min validation loss `0.0004345`; final epoch train loss was `8.48e-05` and validation loss was `0.000757`. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_pnp_fixed_validation_20260629/pnp_drunet_fixed_val_6h_wandb_local` | DRUNet denoiser training for `pnp_admm` | Resumed local CUDA run from epoch 50 to epoch 100 | W&B run `yzjn69ku` (`https://wandb.ai/tjh200-university-of-cambridge/PaDIS-LIDC/runs/yzjn69ku`). The initial resume concern was traced to manual W&B step logging; after removing the manual step and using `epoch` as the metric axis, W&B API verification reported `lastHistoryStep=100` with 100 history rows. Final train loss was `8.7389e-05`, final validation loss was `0.00011037`, and min validation loss was `7.5371e-05`. Periodic checkpoint retention kept exactly epochs 60, 70, 80, 90, and 100, with final and best-validation checkpoints also present. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_long_training_20260628/whole_image_6h_wandb_local` | Whole-image diffusion training | 6-hour local CUDA run, 17,664 seen patches | W&B run `pk2frvqf` (`https://wandb.ai/tjh200-university-of-cambridge/PaDIS-LIDC/runs/pk2frvqf`). The run stopped cleanly at `--max-train-seconds 21600`, saved final and min-validation checkpoints including `whole_image_lidc_256_min_val.pt`, and reached min validation loss `12261.434`. Validation was still high and only trending downward, so this is not enough evidence that the whole-image prior is trained to paper quality. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_long_training_recon_20260629/pnp_admm_6h_minval` | `pnp_admm` | 1 `ct_20` test slice | Escaped-CUDA reconstruction using the six-hour DRUNet min-validation checkpoint. The run completed 10 ADMM iterations and beat FDK clearly: PSNR 28.90 dB, SSIM 0.669, MAE 0.0242, and p95 absolute error 0.0711 versus FDK PSNR 22.15 dB and SSIM 0.328. This gives useful confidence that the LION PnP-ADMM wiring and denoiser training path are functioning, though it is still a LION-native DRUNet surrogate rather than an exact paper denoiser. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_long_training_recon_20260629/whole_image_6h_minval_paper` | `whole_image_diffusion` | 1 `ct_20` test slice, full 100 outer / 10 inner paper preset | Escaped-CUDA reconstruction using the six-hour whole-image min-validation checkpoint with `--implementation paper`, geometric schedule, `sigma_max=10`, `sigma_min=0.002`, Gaussian initialization, and `prior_mode=whole_image`. The full sampler completed and wrote trace images, but quality was poor: PSNR 9.67 dB, SSIM -0.0017, MAE 0.307, and relative sinogram residual 1.19 versus FDK PSNR 22.15 dB and SSIM 0.328. This verifies loader/sampler/output plumbing only; the six-hour local whole-image checkpoint is not a reliable reconstruction prior. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/training_dependent_lion_physics_1sample` | `pnp_admm`, `whole_image_diffusion` | 1 `ct_20` test slice each, Slurm-default forced `lion_physics` matrix path | Escaped-CUDA run of the training-dependent rows using `--implementations lion_physics`. The staged matrix root linked `whole_lidc_default/whole_image_lidc_256.pt` to the six-hour whole-image final checkpoint, and `pnp_admm` used the fixed-validation DRUNet min-validation checkpoint. The verifier passed 2 records with expected sampler/method settings. PnP-ADMM passed a quality gate with PSNR 28.90 dB, SSIM 0.669, MAE 0.0242, and relative sinogram residual 0.00816 versus FDK PSNR 23.17 dB. Whole-image diffusion was intentionally stopped after one outer step and remained a dispatch/checkpoint-loading smoke only: PSNR 4.78 dB versus FDK 23.17 dB. The matrix now expects the whole-image min-validation checkpoint instead; see the full LION-physics whole-image row below. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_whole_image_full_20260630/whole_image_6h_minval_lion_physics_full` | `whole_image_diffusion` | 1 `ct_20` test slice, full 100 outer / 10 inner LION-physics sampler | Escaped-CUDA full reconstruction using the six-hour whole-image min-validation checkpoint. The sampler used `--implementation lion_physics`, paper geometric schedule with `sigma_min=0.002` and `sigma_max=10`, LION FDK initialization, least-squares data consistency, and `operator_lipschitz` normalization. It completed in 9:41 on the local GTX 1070 and reached PSNR 32.58 dB, SSIM 0.789, edge SSIM 0.555, MAE 0.01599, p95 error 0.0465, and relative sinogram residual 0.00233 versus FDK PSNR 23.17 dB and SSIM 0.373. This promotes the whole-image row from plumbing-only to one-slice quality evidence under the physical preset, but a 25-sample A100 matrix is still required for final paper-array claims. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/whole_image_full_minval_default_path` | `whole_image_diffusion` | 1 `ct_20` test slice, full matrix-wrapper `method_default` path | Escaped-CUDA rerun through `PaDIS_run_reconstruction_matrix.py --models method_default --methods whole_image_diffusion --experiments ct_20 --implementations method_default`. The matrix selected `implementation=lion_physics`, `whole_lidc_default/whole_image_lidc_256_min_val.pt`, and `prior_mode=whole-image` without manual reconstruction overrides. It completed in 9:41, reached the same PSNR 32.58 dB versus FDK 23.17 dB, and `PaDIS_verify_reconstruction_matrix.py` passed 1 expected record with 1 sample and the mean-better-than-FDK gate. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/whole_image_remaining_minval_default_path` | `whole_image_diffusion` | 1 test slice each for `ct_8`, `ct_60`, and `ct_fanbeam_180`, full matrix-wrapper path | Escaped-CUDA matrix-wrapper run with the whole-image min-validation checkpoint and the pre-promotion `dps_epsilon=1` LION-physics default. The verifier passed 3 structural records. `ct_8` reached PSNR 27.65 dB versus FDK 18.94 dB, SSIM 0.681 versus 0.226, and relative sinogram residual 0.00215. `ct_60` reached PSNR 34.58 dB versus FDK 29.09 dB, SSIM 0.834 versus 0.629, and relative residual 0.00322. The fanbeam row was finite but slightly below FDK on PSNR and SSIM: 35.33 dB versus 35.45 dB and SSIM 0.851 versus 0.883, motivating the targeted fanbeam relaxation check below. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/whole_image_fanbeam_eps0p5` | `whole_image_diffusion` | 1 `ct_fanbeam_180` test slice, full LION-physics sampler with `dps_epsilon=0.5` | Escaped-CUDA targeted fanbeam diagnostic using the same whole-image min-validation checkpoint, paper geometric sigma schedule, LION fan-beam FDK initialization, least-squares data consistency, and `operator_lipschitz` normalization. Only the DPS/Langevin relaxation changed from `dps_epsilon=1` to `0.5`. The verifier passed 1 record with the mean-better-than-FDK gate: PSNR 35.78 dB versus FDK 35.45 dB, MAE 0.0122, p95 error 0.0327, and relative sinogram residual 0.00311. SSIM remains below FDK, 0.859 versus 0.883. The `whole_image_diffusion` plus `ct_fanbeam_180` LION-physics matrix row now defaults to this `dps_epsilon=0.5` relaxation; this is a documented paper divergence but not a CT scaling constant. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/whole_image_fanbeam_default_matrix_path` | `whole_image_diffusion` | 1 `ct_fanbeam_180` test slice, promoted full matrix-wrapper `method_default` path | Escaped-CUDA rerun after promoting the fanbeam whole-image relaxation into the driver and matrix manifest. The command used `PaDIS_run_reconstruction_matrix.py --models method_default --methods whole_image_diffusion --experiments ct_fanbeam_180 --implementations method_default`, so this validates the production default path rather than a manual reconstruction override. The verifier passed 1 expected-job record with the mean-better-than-FDK gate and confirmed the sampler payload: geometric schedule, `sigma_min=0.002`, `sigma_max=10`, `dps_epsilon=0.5`, least-squares data consistency, and `operator_lipschitz` normalization. Metrics matched the targeted diagnostic: PSNR 35.78 dB versus FDK 35.45 dB, SSIM 0.859 versus FDK 0.883, MAE 0.0122, p95 error 0.0327, and relative sinogram residual 0.00311. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/whole_image_ct20_default_3sample` | `whole_image_diffusion` | 3 `ct_20` test slices, promoted full matrix-wrapper path | Escaped-CUDA 3-sample validation of the production whole-image LION-physics row using the six-hour min-validation checkpoint. The command used `PaDIS_run_reconstruction_matrix.py --models method_default --methods whole_image_diffusion --experiments ct_20 --implementations lion_physics`, so this validates the matrix default rather than a manual reconstruction override. The verifier passed 1 expected-job record with 3 samples, required every sample to beat FDK, and confirmed `prior_mode=whole_image`, paper geometric sigma schedule, least-squares data consistency, and `operator_lipschitz` normalization. Mean PSNR was 31.84 dB, minimum PSNR 30.85 dB, mean SSIM 0.812, minimum SSIM 0.789, MAE 0.0163, and relative sinogram residual 0.00285 versus FDK mean PSNR 21.00 dB. This strengthens whole-image confidence from one-slice evidence to a small multi-sample check, but it is still not a substitute for the full 25-sample A100 matrix. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/whole_image_cross_experiment_3sample` | `whole_image_diffusion` | 3 test slices each for `ct_8`, `ct_60`, and `ct_fanbeam_180` | Escaped-CUDA production matrix validation of the whole-image LION-physics rows outside `ct_20` using the six-hour min-validation checkpoint. The command used `PaDIS_run_reconstruction_matrix.py --models method_default --methods whole_image_diffusion --experiments ct_8,ct_60,ct_fanbeam_180 --implementations lion_physics --max-samples 3`, so this validates matrix dispatch rather than a manual reconstruction override. The verifier passed 3 expected records with 3 samples each, required every sample to beat FDK by PSNR, and confirmed paper geometric sigma schedules, whole-image prior mode, least-squares data consistency, `operator_lipschitz` normalization, `data_consistency_scale=1.0`, and no public-repo adjoint scale. `ct_8` reached mean PSNR 26.17 dB, SSIM 0.671, MAE 0.0278, residual 0.00290, and minimum FDK margin 8.71 dB versus FDK mean PSNR 16.71 dB. `ct_60` reached mean PSNR 34.84 dB, SSIM 0.871, MAE 0.0127, residual 0.00397, and minimum FDK margin 5.49 dB versus FDK mean PSNR 27.37 dB. `ct_fanbeam_180` reached mean PSNR 36.21 dB, SSIM 0.892, MAE 0.0114, residual 0.00379, and minimum FDK margin 0.326 dB versus FDK mean PSNR 34.68 dB. The fan-beam first sample still has lower SSIM than FDK despite higher PSNR, so treat this as a strong PSNR/data-consistency pass rather than a uniform win on every image metric. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/pnp_admm_cross_experiment_clipped_1sample` | `pnp_admm` | 1 `ct_20` and 1 `ct_8` test slice, forced `lion_physics` matrix path | Escaped-CUDA validation after adding default PnP iterate clipping to `[0, 1]`. Before this change, `ct_8` returned NaNs with `pnp_eta=1e-4`; increasing `eta` alone either remained non-finite or collapsed to poor near-constant output. With clipping and the original matrix penalty, the verifier passed both records and each beat FDK: `ct_20` PSNR 29.65 dB versus FDK 23.17 dB, and `ct_8` PSNR 20.89 dB versus FDK 18.94 dB. This validates the full PnP paper experiment set locally for one slice each. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_sampler_dps_smoke_20260628` | public-fork `dps` helper | 1 LIDC PNG, stopped after 1 outer step | Escaped-CUDA smoke of the LION-compatible public fork's default DPS sampler after adding public helper sampler selection and patch-denoiser microbatch plumbing. The run completed with the default gradient-tracking DPS path and wrote `reconstructions.npz` plus `trace.json`. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_sampler_pc_trace_20260628` | public-fork `pc_sampling` helper | 1 LIDC PNG, stopped after 1 outer step | Escaped-CUDA smoke of the LION-compatible public fork's predictor-corrector helper via `--sampler pc --patch_batch_size 1 --trace_interval 1`. The helper path completed on the local 8GB GPU after disabling denoiser-gradient storage for non-DPS public helpers and wrote `reconstructions.npz` plus PC predictor/corrector trace statistics. This is a reference-generation smoke, not a quality run. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_sampler_langevin_smoke_20260628` | public-fork `langevin` helper | 1 LIDC PNG, stopped after 1 outer step | Escaped-CUDA smoke of the LION-compatible public fork's Langevin helper via `--sampler langevin --patch_batch_size 1`. The helper branch completed and wrote `reconstructions.npz`; this is a reference-generation smoke, not a quality run. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_sampler_ddnm_smoke_20260628` | public-fork DDNM helper | 1 LIDC PNG, stopped after 1 outer step | Escaped-CUDA smoke of the LION-compatible public fork's DDNM helper via `--sampler ddnm --patch_batch_size 1`. This executes the public `langevin(..., ddnm=True)` branch and wrote `reconstructions.npz`; it is a reference-generation smoke, not a quality run. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_patch_average_1step_checkpoint_20260628` | public-fork `patch_average` helper | 1 LIDC PNG, stopped after 1 outer step | Escaped-CUDA smoke of the LION-compatible public fork's `denoisedOverlap(...)` helper inside the DPS loop via `--sampler patch_average --patch_batch_size 1 --checkpoint_denoiser`. The upstream helper's default overrun is bounded to the last valid patch in this fork path. The run completed and wrote `reconstructions.npz`, intermediates, and trace statistics. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_patch_stitch_1step_checkpoint_20260628` | public-fork `patch_stitch` helper | 1 LIDC PNG, stopped after 1 outer step | Escaped-CUDA smoke of the LION-compatible public fork's `denoisedTile(...)` helper inside the DPS loop via `--sampler patch_stitch --patch_batch_size 1 --checkpoint_denoiser`. This keeps the public helper's hard-coded start index `4`. The run completed and wrote `reconstructions.npz`, intermediates, and trace statistics. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_pc_1step_central_noise_20260628` | LION `predictor_corrector` versus public-fork `pc_sampling` helper | 1 `ct_20` test slice, stopped after 1 outer step | Escaped-CUDA comparison using `--implementation public_repo --public-repo-sigma-schedule readme --public-repo-helper-initialization` and `--public-reference-reconstructions` from the traced public-fork PC smoke. After matching the helper's central-noise-then-pad initialization, LION reached public-reference SSIM 0.991, edge SSIM 0.980, MAE 0.0120, and p95 absolute error 0.0255. This validates one-step output-level alignment for the public PC helper under LION geometry. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_pc_5step_20260628` | public-fork `pc_sampling` helper | 1 LIDC PNG, stopped after 5 outer steps | Escaped-CUDA reference-generation run for a longer public predictor-corrector helper comparison using the literal README sigma schedule, `ct_lion_fanbeam`, `--patch_batch_size 1`, and seed 2. It wrote `reconstructions.npz` and trace statistics. The public-fork quality is intentionally poor for this truncated diagnostic: PSNR 4.69 dB and SSIM 0.0023. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_pc_5step_reuse_layout_20260628` | LION `predictor_corrector` versus public-fork `pc_sampling` helper | 1 `ct_20` test slice, stopped after 5 outer steps | Escaped-CUDA comparison after adding the public-helper PC behavior that reuses the predictor patch layout for the corrector denoising call. The focused tests passed (`79 passed`), but output-level agreement did not improve versus the previous 5-step run: public-reference SSIM 0.9649, edge SSIM 0.9225, MAE 0.0303, and p95 absolute error 0.258. Trace comparison showed the first denoised prior output matched exactly at step 0, but the CT data update differed immediately; LION's raw adjoint correction norm was about 10x the public fork's, so the remaining PC drift was a data-consistency/operator-scaling issue rather than a patch-denoiser issue. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_pc_5step_split_adjscale_default_20260628` | LION `predictor_corrector` versus public-fork `pc_sampling` helper | 1 `ct_20` test slice, stopped after 5 outer steps | Escaped-CUDA comparison after splitting public-compatible DPS norm-gradient scaling from direct-adjoint PC/Langevin scaling. The public-repo preset now keeps `data_consistency_scale=0.0405` for DPS and uses `adjoint_data_consistency_scale=0.1022` for direct adjoint residual updates. Focused tests passed (`80 passed`). With no manual scale override, LION reached public-reference SSIM 0.9987, edge SSIM 0.9967, MAE 0.00459, and p95 absolute error 0.0133 against the 5-step public PC helper reference. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_pc_100step_20260628` | public-fork `pc_sampling` helper | 1 LIDC PNG, full public helper run | Escaped-CUDA 100-step reference-generation run using the literal README sigma schedule, `ct_lion_fanbeam`, `--patch_batch_size 1`, trace interval 10, and seed 2. The helper completed with finite output: PSNR 30.40 dB and SSIM 0.7130. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_pc_100step_20260628` | LION `predictor_corrector` versus public-fork `pc_sampling` helper | 1 `ct_20` test slice, full public-helper-compatible run | Escaped-CUDA comparison using `--implementation public_repo --public-repo-sigma-schedule readme --public-repo-helper-initialization` and the 100-step public PC reference. LION reached target PSNR 30.17 dB and SSIM 0.7092; against the public reference it reached SSIM 0.9913, edge SSIM 0.9874, MAE 0.00419, and p95 absolute error 0.00771. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_langevin_1step_helper_init_fixed_20260628` | LION `langevin` versus public-fork `langevin` helper | 1 `ct_20` test slice, stopped after 1 outer step | Escaped-CUDA comparison using `--implementation public_repo --public-repo-sigma-schedule readme --public-repo-helper-initialization` and `--public-reference-reconstructions` from the public-fork Langevin smoke. With the public helper's padded-state noise convention, LION reached public-reference SSIM 0.980, edge SSIM 0.954, MAE 0.0196, and p95 absolute error 0.0612. This validates one-step output-level alignment for the public Langevin helper under LION geometry. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_langevin_5step_20260628` | public-fork `langevin` helper | 1 LIDC PNG, stopped after 5 outer steps | Escaped-CUDA reference-generation run for a longer public Langevin helper comparison using the literal README sigma schedule, `ct_lion_fanbeam`, `--patch_batch_size 1`, and seed 2. It wrote `reconstructions.npz` and intermediate PNGs. The public-fork quality is intentionally poor for this truncated diagnostic: PSNR 5.06 dB and SSIM 0.00284. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_langevin_5step_split_adjscale_default_20260628` | LION `langevin` versus public-fork `langevin` helper | 1 `ct_20` test slice, stopped after 5 outer steps | Escaped-CUDA comparison after adding the split direct-adjoint scale parameter but before routing the Langevin hand-written data update through it. The trace still showed `data_consistency_scale=0.0405`, and output-level agreement stayed poor: public-reference SSIM 0.9662, edge SSIM 0.9247, MAE 0.0290, and p95 absolute error 0.258. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_langevin_5step_split_adjscale_applied_20260628` | LION `langevin` versus public-fork `langevin` helper | 1 `ct_20` test slice, stopped after 5 outer steps | Escaped-CUDA comparison after routing Langevin's data step through the shared direct-adjoint correction helper. Focused tests still passed (`80 passed`). LION reached public-reference SSIM 0.9998, edge SSIM 0.9995, MAE 0.00165, and p95 absolute error 0.0 against the 5-step public Langevin helper reference. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_langevin_100step_20260628` | public-fork `langevin` helper | 1 LIDC PNG, full 100 outer / 10 inner sampler | Escaped-CUDA 100-step reference-generation run using the literal README sigma schedule, `ct_lion_fanbeam`, `--patch_batch_size 1`, and seed 2. The helper completed with finite output: PSNR 31.92 dB and SSIM 0.7541. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_langevin_100step_20260628` | LION `langevin` versus public-fork `langevin` helper | 1 `ct_20` test slice, full public-helper-compatible run | Escaped-CUDA comparison using `--implementation public_repo --public-repo-sigma-schedule readme --public-repo-helper-initialization` and the 100-step public Langevin reference. LION reached target PSNR 31.99 dB and SSIM 0.7536; against the public reference it reached SSIM 0.9983, edge SSIM 0.9920, MAE 0.00132, and p95 absolute error 0.00401. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_ddnm_1step_helper_init_20260628` | LION `ve_ddnm` versus public-fork DDNM helper | 1 `ct_20` test slice, stopped after 1 outer step | Escaped-CUDA comparison using `--implementation public_repo --public-repo-sigma-schedule readme --public-repo-helper-initialization --ve-ddnm-nfe-layout public_inner` and `--public-reference-reconstructions` from the public-fork DDNM smoke. With padded-state noise and the public helper's 100x10 loop layout, LION reached public-reference SSIM 1.000, edge SSIM 0.999, MAE 0.00182, and p95 absolute error 0.0. This validates one-step output-level alignment for the public DDNM helper under LION geometry. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_ddnm_5step_20260628` | public-fork DDNM helper | 1 LIDC PNG, stopped after 5 outer steps | Escaped-CUDA reference-generation run for a longer public DDNM helper comparison using the literal README sigma schedule, `ct_lion_fanbeam`, `--patch_batch_size 1`, and seed 2. It wrote `reconstructions.npz` and intermediate PNGs. The public-fork quality is intentionally poor for this truncated diagnostic: PSNR 4.86 dB and SSIM 0.00295. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_ddnm_5step_split_adjscale_default_20260628` | LION `ve_ddnm` versus public-fork DDNM helper | 1 `ct_20` test slice, stopped after 5 outer steps | Escaped-CUDA comparison using `--implementation public_repo --public-repo-sigma-schedule readme --public-repo-helper-initialization --ve-ddnm-nfe-layout public_inner` and the new split-scale defaults. LION reached public-reference SSIM 0.9996, edge SSIM 0.9991, MAE 0.00217, and p95 absolute error 0.0 against the 5-step public DDNM helper reference. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_ddnm_100step_20260628` | public-fork DDNM helper | 1 LIDC PNG, full 100 outer / 10 inner sampler | Escaped-CUDA run of the public DDNM branch without LION's VE-DDNM stabilisation, using the literal README sigma schedule, `ct_lion_fanbeam`, `--patch_batch_size 1`, and seed 2. The run executed to completion but produced an all-NaN reconstruction array: public-repo PSNR and SSIM were both `nan`, and `reconstructions.npz` contained 65,536 NaN pixels out of 65,536. This confirms the un-stabilised public DDNM formula is executable but not numerically usable in this matched CT setup. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_ddnm_parallel_100step_20260630/ct_lion_parbeam` | public-fork DDNM helper | 1 LIDC PNG, full 100 outer / 10 inner sampler | Escaped-CUDA run of the public DDNM branch using the LION-scale parallel-beam shim `ct_lion_parbeam`, literal README sigma schedule, `--patch_batch_size 1`, and seed 2. A pseudoinverse overflow/invalid warning appeared around outer step 72, and the completed `reconstructions.npz` was all NaNs: 65,536 NaN pixels out of 65,536, with `nan` PSNR and SSIM. This shows that simply switching the matched LION comparison from fan-beam to the LION-scale parallel shim does not stabilise the public DDNM formula. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_ddnm_parallel_100step_20260630/ct_parbeam` | public-fork DDNM helper | 1 LIDC PNG, full 100 outer / 10 inner sampler | Escaped-CUDA run of the public DDNM branch using the original public-repo `ct_parbeam` geometry, literal README sigma schedule, `--patch_batch_size 1`, and seed 2. This run completed with finite output: 65,536 finite pixels, no NaNs/Infs, PSNR 4.65 dB, and SSIM 0.00264. Numerically stable here means finite, not useful reconstruction quality. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_patch_average_1step_20260628` | LION `patch_average` versus public-fork `patch_average` helper | 1 `ct_20` test slice, stopped after 1 outer step | Escaped-CUDA comparison using `--implementation public_repo --public-repo-sigma-schedule readme`, `fixed_overlap_layout=public_overlap`, checkpointed denoising, and `--public-reference-reconstructions` from the public-fork patch-average smoke. LION reached public-reference SSIM 1.000, edge SSIM 1.000, MAE 1.35e-5, and p95 absolute error 0.0. This validates one-step output-level alignment for the public overlap-averaging helper under LION geometry. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_patch_stitch_1step_20260628` | LION `patch_stitch` versus public-fork `patch_stitch` helper | 1 `ct_20` test slice, stopped after 1 outer step | Escaped-CUDA comparison using `--implementation public_repo --public-repo-sigma-schedule readme`, `fixed_overlap_layout=public_tile`, checkpointed denoising, and `--public-reference-reconstructions` from the public-fork patch-stitch smoke. LION reached public-reference SSIM 1.000, edge SSIM 1.000, MAE 1.32e-5, and p95 absolute error 0.0. This validates one-step output-level alignment for the public tile/stitch helper under LION geometry. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_patch_average_100step_20260628` | public-fork `patch_average` helper | 1 LIDC PNG, attempted full 100 outer / 10 inner sampler | Earlier escaped-CUDA attempt using the literal README sigma schedule, `ct_lion_fanbeam`, `--patch_batch_size 1`, `--checkpoint_denoiser`, trace interval 10, and seed 2. The helper fit in memory and wrote intermediates through step 40, but local runtime degraded sharply; it was interrupted at 48/100 after 38 minutes before `reconstructions.npz` was written. This was superseded by the detach-fixed public-fork run below. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_patch_average_100step_detach_20260628_retry` | public-fork `patch_average` helper | 1 LIDC PNG, full 100 outer / 10 inner sampler | Escaped-CUDA 100-step reference-generation run after detaching the public fork's fixed-overlap iterate between inner updates to avoid autograd graph retention. It used the literal README sigma schedule, `ct_lion_fanbeam`, `--patch_batch_size 1`, `--checkpoint_denoiser`, no trace/intermediate output, and seed 2. The helper completed in 48:21 with finite output: PSNR 33.36 dB and SSIM 0.7998. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_patch_average_100step_detach_20260628` | LION `patch_average` versus public-fork `patch_average` helper | 1 `ct_20` test slice, full public-helper-compatible run | Escaped-CUDA comparison using `--implementation public_repo --public-repo-sigma-schedule readme`, `fixed_overlap_layout=public_overlap`, checkpointed denoising, and the detach-fixed 100-step public patch-average reference. LION reached target PSNR 33.39 dB and SSIM 0.7990; against the public reference it reached SSIM 0.9973, edge SSIM 0.9880, MAE 0.00120, and p95 absolute error 0.00387. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_patch_stitch_100step_detach_20260628` | public-fork `patch_stitch` helper | 1 LIDC PNG, full 100 outer / 10 inner sampler | Escaped-CUDA 100-step reference-generation run after the same public-fork fixed-overlap detach fix. It used the literal README sigma schedule, `ct_lion_fanbeam`, `--patch_batch_size 1`, `--checkpoint_denoiser`, no trace/intermediate output, and seed 2. The helper completed in 48:16 with finite output: PSNR 32.10 dB and SSIM 0.7799. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_patch_stitch_100step_detach_20260628` | LION `patch_stitch` versus public-fork `patch_stitch` helper | 1 `ct_20` test slice, full public-helper-compatible run | Escaped-CUDA comparison using `--implementation public_repo --public-repo-sigma-schedule readme`, `fixed_overlap_layout=public_tile`, checkpointed denoising, and the detach-fixed 100-step public patch-stitch reference. LION reached target PSNR 32.04 dB and SSIM 0.7780; against the public reference it reached SSIM 0.9966, edge SSIM 0.9845, MAE 0.00157, and p95 absolute error 0.00500. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_baseline_tv_25slice_noiseinit_20260628` | `baseline`, `admm_tv` | 25 `ct_20` test slices | Current-code escaped-CUDA no-prior validation with a saved reconstruction-matrix manifest. The strict verifier passed 2 records, 25 samples each, expected sampler/method settings including `noise_initialization`, and ADMM-TV beating FDK in mean and on every slice. Mean PSNRs: baseline/FDK 20.59 dB and LION TV substitute 27.67 dB. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_local_cuda_validation_20260627` | `langevin` | 1 `ct_20` test slice | Full paper-preset run, mean PSNR 26.08 dB versus FDK 22.15 dB; verifier passed the method-specific better-than-FDK gate. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_local_cuda_validation_20260627` | `predictor_corrector` | 1 `ct_20` test slice | Paper/public linear PC corrector step completed with finite output, but mean PSNR was 11.51 dB versus FDK 22.15 dB. This row needs full-array/A100 validation or tuning before treating quality as paper-like. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_local_cuda_validation_20260627` | `ve_ddnm` | 1 `ct_20` test slice | Corrected VE-DDNM pseudoinverse path completed with finite output, but mean PSNR was -7.35 dB versus FDK 22.15 dB. Output-clipped and filtered-FDK variants were also below FDK. This row is implemented but not quality-matched in LION fan-beam geometry. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_gpu_smoke_20260627` | `padis_dps` | 1 `ct_20` test slice | Public-compatible full run, mean PSNR 34.10 dB versus FDK 25.43 dB. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_pc_ddnm_ablation_20260627/langevin_public_full` | `langevin` | 1 `ct_20` test slice | Public-compatible full run with paper geometric schedule, mean PSNR 33.14 dB and SSIM 0.803 versus FDK 25.43 dB. This is close to the paper's reported 20-view Langevin PSNR. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_pc_ddnm_ablation_20260627/pc_public_full` | `predictor_corrector` | 1 `ct_20` test slice | Public-compatible full run with paper geometric schedule, mean PSNR 30.17 dB and SSIM 0.716 versus FDK 25.43 dB. This is substantially better than strict paper-mode PC locally, but still below the paper's reported aggregate PC PSNR. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_pc_ddnm_ablation_20260627/ve_ddnm_public_full` | `ve_ddnm` | 1 `ct_20` test slice | Earlier public-compatible helper diagnostic with only one inner denoising update remained poor, mean PSNR 12.39 dB and SSIM 0.079 versus FDK 25.43 dB. The explicit `--ve-ddnm-nfe-layout public_inner` mode uses 10 inner updates to match the public helper loop. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_ve_ddnm_inner10_20260627/paper_default` | `ve_ddnm` | 1 `ct_20` test slice | Older paper-mode 10-inner run completed with finite output but diverged badly, mean PSNR -25.42 dB versus FDK 22.15 dB. The strict paper diagnostic now uses `paper_1000x1` instead. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_ve_ddnm_inner10_20260627/public_repo_paper_schedule` | `ve_ddnm` | 1 `ct_20` test slice | Current public-compatible 10-inner run with paper geometric schedule completed with finite output but remained below FDK, mean PSNR 5.68 dB versus FDK 25.43 dB. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_ve_ddnm_split_clip_20260627/public_repo_paper_schedule` | `ve_ddnm` | 1 `ct_20` test slice | Diagnostic run with unclipped `A^dagger A(D)` produced non-finite metrics. The finite matrix default therefore clips both VE-DDNM pseudoinverse terms under LION fan-beam geometry. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_ve_ddnm_no_noise_probe_20260627` | `ve_ddnm` | 1 `ct_20` test slice | Public-compatible 10-inner run with Langevin noise disabled remained poor, mean PSNR 6.00 dB versus FDK 25.42 dB, so stochastic noise is not the main failure mode. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_ve_ddnm_corrected_clip_probe_20260627` | `ve_ddnm` | 1 `ct_20` test slice | Public-compatible 10-inner run with `--ddnm-corrected-clip` improved substantially to mean PSNR 19.97 dB, but still remained below FDK 25.42 dB. This confirms the corrected DDNM estimate leaves the stable CT image domain without clipping. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_ve_ddnm_paper_corrected_clip_probe_20260627` | `ve_ddnm` | 1 `ct_20` test slice | Strict paper-mode run with only `--ddnm-corrected-clip` added reached mean PSNR 15.70 dB versus FDK 22.15 dB. This is still not quality-matched and is a paper divergence. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_ve_ddnm_paper_layout_full_20260628` | `ve_ddnm` | 1 `ct_20` test slice, full `paper_1000x1` layout | Strict paper-layout VE-DDNM completed with trace JSON, trace images, previews, and tensors, but mean PSNR was -23.07 dB versus FDK 22.15 dB. This confirms the strict paper diagnostic is runnable but not quality-matched. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_ve_ddnm_paper_layout_corrected_clip_full_20260628` | `ve_ddnm` | 1 `ct_20` test slice, full `paper_1000x1` layout | Adding the non-paper diagnostic `--ddnm-corrected-clip` improved the current layout to mean PSNR 15.64 dB and SSIM 0.133, but it still remained below FDK 22.15 dB and is not used as the paper default. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_ve_ddnm_paper_layout_eps0p1_corrected_clip_full_20260628` | `ve_ddnm` | 1 `ct_20` test slice, full `paper_1000x1` layout | LION-stabilized VE-DDNM with `sampling_epsilon=0.1` and `--ddnm-corrected-clip` reached mean PSNR 32.94 dB and SSIM 0.766 versus FDK 22.15 dB. This was the predecessor to the current `lion_physics` VE-DDNM default. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_ve_ddnm_lion_quality_full_20260628` | `ve_ddnm` | 1 `ct_20` test slice, full `paper_1000x1` layout | Legacy `--implementation lion_quality` VE-DDNM completed with trace JSON, trace images, previews, and tensors. It reached mean PSNR 33.09 dB and SSIM 0.772 versus FDK 22.15 dB; the newer `lion_physics` VE-DDNM default reproduces this quality while using the physical least-squares/operator-Lipschitz data objective. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_tuning_20260630/padis_dps_lipschitz_eta_zeta0p3` | `padis_dps` | 1 `ct_20` test slice, LION-physics least-squares/Lipschitz run | With the paper `zeta=0.3`, the physically normalized mode was stable but under-relaxed: PSNR 26.12 dB, SSIM 0.620, edge SSIM 0.271, MAE 0.0283, p95 error 0.1019, and relative sinogram residual 0.0162 versus FDK PSNR 23.17 dB. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_tuning_20260630/padis_dps_lipschitz_eta_zeta1p0_fixed` | `padis_dps` | 1 `ct_20` test slice, LION-physics least-squares/Lipschitz run | After fixing CLI override ordering, `--zeta 1.0` reached PSNR 29.39 dB, SSIM 0.726, edge SSIM 0.373, MAE 0.0206, p95 error 0.0632, and relative sinogram residual 0.00582. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_tuning_20260630/padis_dps_lipschitz_eta_zeta3p0` | `padis_dps` | 1 `ct_20` test slice, LION-physics least-squares/Lipschitz run | Current tuned LION-physics preset evidence. `zeta=3.0` reached PSNR 32.83 dB, SSIM 0.800, edge SSIM 0.466, MAE 0.0153, p95 error 0.0451, and relative sinogram residual 0.00228 versus FDK PSNR 23.17 dB. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_tuning_20260630/padis_dps_lipschitz_eta_zeta5p0` | `padis_dps` | 1 `ct_20` test slice, LION-physics least-squares/Lipschitz run | `zeta=5.0` was too aggressive: the run completed but produced NaN reconstruction metrics. This brackets the current default at the smaller finite value `zeta=3.0`. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_validation_20260630/padis_dps_3sample_zeta3` | `padis_dps` | 3 `ct_20` test slices, LION-physics least-squares/Lipschitz run | Broader validation of the tuned DPS preset. With `zeta=3.0`, the run reached mean PSNR 32.33 dB, SSIM 0.829, edge SSIM 0.584, MAE 0.0152, p95 error 0.0487, and relative sinogram residual 0.00277 versus FDK PSNR 21.00 dB. This is close to the public-compatible 3-sample DPS reference without public-repo CT scaling constants. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_validation_20260630/pc_1sample_zeta3p75_lsadj` | `predictor_corrector` | 1 `ct_20` test slice, LION-physics direct-adjoint least-squares/Lipschitz run | Best constant-zeta PC point tested so far. `zeta=3.75` reached PSNR 29.26 dB, SSIM 0.684, edge SSIM 0.460, MAE 0.0221, p95 error 0.0630, and relative sinogram residual 0.00517 versus FDK PSNR 23.17 dB. This remains below the public-compatible PC reference and should not yet drive matrix defaults. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_validation_20260630/pc_1sample_zeta3p75_lsadj_public_pc_layout` | `predictor_corrector` | 1 `ct_20` test slice, LION-physics data step plus public PC denoising-layout diagnostic | Keeping the LION-physics data objective and Lipschitz normalizer but switching to public-helper PC denoising conventions did not close the gap: PSNR 29.00 dB, SSIM 0.690, edge SSIM 0.454, MAE 0.0224, p95 error 0.0641, and relative sinogram residual 0.00564. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_validation_20260630/pc_1sample_zeta4p25_snr0p08` | `predictor_corrector` | 1 `ct_20` test slice, LION-physics direct-adjoint least-squares/Lipschitz run | Current PC physical default evidence. `zeta=4.25 --pc-snr 0.08` reached PSNR 30.99 dB, SSIM 0.738, edge SSIM 0.519, MAE 0.0189, p95 error 0.0535, and relative sinogram residual 0.00558. This exceeds the 1-slice public-compatible PC reference without public CT scale constants. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_validation_20260630/pc_3sample_default` | `predictor_corrector` | 3 `ct_20` test slices, current LION-physics default | No manual sampler flags; this validates the promoted driver default. Mean PSNR 29.79 dB, SSIM 0.747, edge SSIM 0.576, MAE 0.0207, p95 error 0.0660, and relative sinogram residual 0.00758 versus FDK PSNR 21.00 dB. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_validation_20260630/pc_3sample_public_repo_reference` | `predictor_corrector` | 3 `ct_20` test slices, public-compatible reference | Matching reference for the same samples. Mean PSNR 29.94 dB, SSIM 0.729, edge SSIM 0.566, MAE 0.0221, p95 error 0.0639, and relative sinogram residual 0.0466. LION-physics PC is therefore within 0.15 dB PSNR, better on SSIM/edge SSIM/MAE/residual, and no longer depends on the public CT scale constants. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_validation_20260630/langevin_1sample_zeta3p5_lsadj` | `langevin` | 1 `ct_20` test slice, LION-physics direct-adjoint least-squares/Lipschitz run | Best Langevin constant-zeta point tested so far. `zeta=3.5` reached PSNR 31.05 dB, SSIM 0.786, edge SSIM 0.455, MAE 0.0171, p95 error 0.0494, and relative sinogram residual 0.00242 versus FDK PSNR 23.17 dB. This is finite and useful but still below the public-compatible Langevin reference. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_validation_20260630/langevin_1sample_zeta3p75_lsadj` | `langevin` | 1 `ct_20` test slice, LION-physics direct-adjoint least-squares/Lipschitz run | `zeta=3.75` crossed the stability boundary for constant data relaxation: PSNR collapsed to 4.65 dB, SSIM 0.0025, MAE 0.500, p95 error 0.994, and relative sinogram residual 1.07. This brackets the direct-adjoint constant default below 3.75. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_validation_20260630/langevin_1sample_zeta4_edm_scale_lsadj` | `langevin` | 1 `ct_20` test slice, LION-physics direct-adjoint run with EDM data-weight schedule | `zeta=4.0 --data-consistency-scale-schedule edm` stayed finite, so high-noise overcorrection explains part of the collapse. It reached PSNR 30.62 dB, SSIM 0.781, edge SSIM 0.458, MAE 0.0174, p95 error 0.0493, and relative sinogram residual 0.00226, but it did not beat the constant `zeta=3.5` run. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_validation_20260630/langevin_1sample_zeta4_eps0p5` | `langevin` | 1 `ct_20` test slice, LION-physics direct-adjoint least-squares/Lipschitz run | Current Langevin physical default evidence. `zeta=4.0 --sampling-epsilon 0.5` reached PSNR 32.95 dB, SSIM 0.808, edge SSIM 0.512, MAE 0.0151, p95 error 0.0430, and relative sinogram residual in the same range as the best physical direct-adjoint runs. This is close to the 1-slice public-compatible Langevin PSNR and slightly above its SSIM without public CT scale constants. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_validation_20260630/langevin_3sample_default` | `langevin` | 3 `ct_20` test slices, current LION-physics default | No manual sampler flags; this validates the promoted Langevin physical default. The sampler used `zeta=4.0`, `sampling_epsilon=0.5`, a least-squares data objective, and `operator_lipschitz` normalization. Mean PSNR was 32.92 dB, SSIM 0.837, edge SSIM 0.634, MAE 0.0145, p95 error 0.0457, and relative sinogram residual 0.00155 versus FDK PSNR 21.00 dB. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_validation_20260630/langevin_3sample_public_repo_reference` | `langevin` | 3 `ct_20` test slices, public-compatible reference | Matching public-compatible reference for the same samples. The sampler used public-style direct-adjoint scaling (`adjoint_data_consistency_scale=0.1022`), norm-gradient settings, `zeta=0.3`, and the same paper CT sigma range. Mean PSNR was 32.66 dB, SSIM 0.819, edge SSIM 0.613, MAE 0.0160, p95 error 0.0475, and relative sinogram residual 0.0521. LION-physics Langevin is therefore slightly better on these aggregate metrics without public CT scale constants. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_remaining_20260630/ve_ddnm_1sample_default` | `ve_ddnm` | 1 `ct_20` test slice, current LION-physics default | No manual sampler flags; this validates the VE-DDNM physical default against the previous `lion_quality` fallback. The sampler used `paper_1000x1`, `sampling_epsilon=0.1`, noise initialization, `ddnm_corrected_clip=True`, least-squares data objective, and `operator_lipschitz` normalization. It reached PSNR 33.06 dB, SSIM 0.771, edge SSIM 0.586, MAE 0.0164, p95 error 0.0452, and relative sinogram residual 0.00264 versus FDK PSNR 22.15 dB. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/ve_ddnm_ct20_default_3sample` | `ve_ddnm` | 3 `ct_20` test slices, promoted full matrix-wrapper path | Escaped-CUDA 3-sample validation of the production VE-DDNM LION-physics row using the 10-hour patch checkpoint. The command used `PaDIS_run_reconstruction_matrix.py --models method_default --methods ve_ddnm --experiments ct_20 --implementations lion_physics`, so this validates the matrix default rather than a manual reconstruction override. The verifier passed 1 expected-job record with 3 samples, required every sample to beat FDK, and confirmed `paper_1000x1`, paper geometric sigma schedule, `sampling_epsilon=0.1`, noise initialization, `ddnm_corrected_clip=True`, least-squares data consistency, and `operator_lipschitz` normalization. Mean PSNR was 32.66 dB, minimum PSNR 32.10 dB, mean SSIM 0.795, minimum SSIM 0.767, MAE 0.0165, relative sinogram residual 0.00194, and minimum FDK margin 10.80 dB versus FDK mean PSNR 20.29 dB. The run was completed before adding explicit runtime Lipschitz metadata to `metrics.json`; the sampler settings are correct, and future runs now record the estimated `||A||`, `||F||`, and `||F||^2`. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_remaining_20260630/patch_average_1sample_public_overlap` | `patch_average` | 1 `ct_20` test slice, LION-physics public-overlap layout with paper `dps_epsilon=1` | The public-overlap layout fixed the old LION-clipped failure mode, but paper `dps_epsilon=1` underperformed the public-compatible patch-average reference: PSNR 32.23 dB, SSIM 0.791, edge SSIM 0.443, MAE 0.0158, p95 error 0.0457, and relative sinogram residual 0.00231. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_remaining_20260630/patch_average_1sample_public_overlap_eps0p5` | `patch_average` | 1 `ct_20` test slice, current LION-physics default | Validated patch-average physical default. The sampler used public-overlap patch assembly, checkpointed fixed-overlap denoising, `dps_epsilon=0.5`, least-squares data objective, and `operator_lipschitz` normalization. It reached PSNR 33.94 dB, SSIM 0.814, edge SSIM 0.512, MAE 0.0141, p95 error 0.0402, and relative sinogram residual 0.00155 versus FDK PSNR 23.17 dB. This beats the public-compatible patch-average reference on PSNR/SSIM/MAE/p95 without public CT scale constants, though edge SSIM remains slightly lower. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_remaining_20260630/patch_stitch_1sample_public_tile_eps0p5` | `patch_stitch` | 1 `ct_20` test slice, current LION-physics default | Validated patch-stitch physical default. The sampler used public-tile patch assembly, checkpointed fixed-overlap denoising, `dps_epsilon=0.5`, least-squares data objective, and `operator_lipschitz` normalization. It reached PSNR 32.39 dB, SSIM 0.792, edge SSIM 0.505, MAE 0.0158, p95 error 0.0457, and relative sinogram residual 0.00167 versus FDK PSNR 23.17 dB. This is within the public-compatible patch-stitch quality band: PSNR is about 0.5 dB lower, while SSIM and MAE are slightly better. |
| Current worktree Slurm/matrix audit | Full A100 reconstruction matrix defaults | 26 paper-matrix jobs, non-GPU manifest and submitter validation | Slurm reconstruction wrappers now default to `PADIS_RECON_IMPLEMENTATIONS=lion_physics`, while the Python matrix CLI now also maps the whole-image `method_default` row to `lion_physics`. `PaDIS_run_reconstruction_matrix.py --implementations lion_physics --models method_default --methods all --experiments paper_matrix --geometries lion --max-samples 25 --count` returns 26 jobs. The expected sampler records `patch_checkpoint_denoiser=True` for the ordinary `padis_dps ct_512_60` row, while `fixed_overlap_checkpoint_denoiser=True` remains specific to patch averaging/stitching. The four whole-image jobs now use `whole_image_lidc_256_min_val.pt`, and the whole-image `ct_fanbeam_180` expected sampler records `dps_epsilon=0.5`. Broader non-GPU validation passed with `pytest -q tests/experiments/test_padis_*.py tests/models/test_padis_reconstructor.py` (175 tests), shell syntax checks passed for the current Slurm wrappers, `git diff --check` passed, and staged input checking passed for all 26 forced-`lion_physics` jobs. The regression suite now includes a full-matrix guard that all 14 diffusion-sampler rows in a forced `lion_physics` paper matrix use geometric paper sigmas, least-squares data consistency, `operator_lipschitz`, `data_consistency_scale=1.0`, and no public-repo adjoint scale. The A100 training-task tests also verify that `patch_lidc_512` uses the native 512 engine, produces `padis_lidc_512.pt`, and keeps W&B artifact logging enabled by default for the real training array. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/full_26_lion_physics_jobs.json` | Full forced-`lion_physics` paper matrix manifest | 26 jobs, `--max-samples 25` | The staged debug training root now passes `PaDIS_run_reconstruction_matrix.py --check-inputs` for all 26 forced-`lion_physics` jobs when supplied with the fixed-validation DRUNet min-validation checkpoint. The saved manifest was refreshed from the supported `--list` output and validated as 26 jobs with 14 diffusion-sampler rows, all using least-squares `operator_lipschitz` data updates with no public compatibility scale constants. The root links the 10-hour patch checkpoint, the six-hour whole-image checkpoint, and a 512 training-smoke checkpoint. This proves matrix layout and default Slurm input resolution, not 25-sample reconstruction quality; the 512 and whole-image linked checkpoints are not paper-quality evidence. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/cross_experiment_baseline_1sample` | `baseline` | 1 test slice for each paper CT experiment: `ct_20`, `ct_8`, `ct_60`, `ct_fanbeam_180`, `ct_512_60` | Escaped-CUDA cross-experiment smoke of the forced-`lion_physics` matrix baseline row. The verifier passed 5 records with expected identities/settings and no failures, validating the LION FDK/data path across the paper-facing experiment aliases including the 512 row. Baseline/FDK PSNRs were 23.17, 18.94, 29.09, 35.45, and 27.86 dB respectively. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_cross_experiment_20260630/admm_tv_padis_dps_1sample` | `admm_tv`, `padis_dps` | 1 test slice per completed paper CT experiment | Escaped-CUDA cross-experiment validation of the forced `lion_physics` matrix path. The completed 8-record verifier passed with finite metrics. `padis_dps` completed `ct_20`, `ct_8`, and `ct_60`; all three beat FDK with PSNRs 32.29, 26.36, and 34.27 dB versus FDK 23.17, 18.94, and 29.09 dB, and relative sinogram residuals 0.00248, 0.00276, and 0.00321. This is direct evidence that the least-squares `operator_lipschitz` normalization transfers across 8, 20, and 60 views without public-repo matching constants. `admm_tv` completed all five paper CT aliases; it beat FDK on `ct_20`, `ct_8`, `ct_60`, and `ct_512_60` but not on `ct_fanbeam_180`, where FDK was already 35.45 dB and TV reached 31.71 dB. The parent matrix command was intentionally interrupted after starting the next long `padis_dps ct_fanbeam_180` row; no metrics were written for that partial row. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_cross_experiment_20260630/padis_dps_fanbeam_1sample` | `padis_dps` | 1 `ct_fanbeam_180` test slice | Escaped-CUDA full fan-beam validation of the promoted LION-physics DPS default. The verifier passed 1 record with the better-than-FDK PSNR gate. The sampler used the paper geometric schedule, `sigma_min=0.002`, `sigma_max=10`, least-squares data consistency, `operator_lipschitz` normalization, and `zeta=3.0`. It reached PSNR 35.82 dB versus FDK 35.45 dB, MAE 0.0115, p95 error 0.0322, and relative sinogram residual 0.00345. SSIM was slightly below FDK, 0.868 versus 0.883, so treat this as finite and competitive fan-beam evidence rather than a uniform win on every metric. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/padis_dps_cross_experiment_3sample` | `padis_dps` | 3 test slices each for `ct_8`, `ct_60`, and `ct_fanbeam_180` | Escaped-CUDA production matrix validation of the promoted LION-physics DPS default outside the already-covered `ct_20` row. The command used `PaDIS_run_reconstruction_matrix.py --models method_default --methods padis_dps --experiments ct_8,ct_60,ct_fanbeam_180 --implementations lion_physics --max-samples 3`, so this validates matrix dispatch, not a manual reconstruction override. The verifier passed 3 expected records with 3 samples each, required every sample to beat FDK, and confirmed paper geometric sigma schedules, least-squares data consistency, `operator_lipschitz` normalization, `data_consistency_scale=1.0`, and no public-repo adjoint scale. `ct_8` reached mean PSNR 26.23 dB, SSIM 0.696, MAE 0.0260, residual 0.00286, and minimum FDK margin 7.42 dB versus FDK mean PSNR 16.71 dB. `ct_60` reached mean PSNR 35.40 dB, SSIM 0.893, MAE 0.0113, residual 0.00372, and minimum FDK margin 5.18 dB versus FDK mean PSNR 27.37 dB. `ct_fanbeam_180` reached mean PSNR 36.52 dB, SSIM 0.907, MAE 0.0103, residual 0.00412, and minimum FDK margin 0.367 dB versus FDK mean PSNR 34.68 dB. This strengthens the claim that the LION-native Lipschitz normalization transfers across sparse-view, high-view, and fan-beam rows without public compatibility scale constants. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/metadata_payload_check` | `padis_dps` | 1 `ct_20` test slice, stopped after 1 outer step | Escaped-CUDA metadata-only check after fixing runtime Lipschitz reporting for CUDA cache keys. Quality is intentionally meaningless because the sampler stopped after one outer step. The saved `metrics.json` now records the actual cached LION operator norm used for `operator_lipschitz`: `operator_norm_estimate=269.3168`, `measurement_operator_norm=269.3168`, `data_lipschitz=72531.52`, `data_lipschitz_objective=sum_squared_residual`, and `data_lipschitz_offset_included=false`. This verifies the reporting path for the physical scaling; the data-step math was already using the cached normalizer during reconstruction. |
| Current worktree Lipschitz audit | `lion_physics` data-consistency implementation | Code/readme/test inspection, no new reconstruction samples | The physical-mode scaling matches the LION/tomosipo operator-norm convention used elsewhere in LION: `power_method(op)` estimates `||A||`, and least-squares data steps use the composed-map gradient Lipschitz constant `L = ||F||^2 = (abs(measurement_scale) * ||A||)^2` for `F(x)=A(measurement_scale*x + measurement_offset)`. Because the sampler objective is the summed term `0.5 * ||F(x)-y||^2`, not a mean-squared residual, there is no division by `y.numel()`. The affine `measurement_offset` is excluded from `L` because it does not change the linear operator norm. Public-repo constants remain confined to `--implementation public_repo`; forced `lion_physics` matrix rows are guarded by tests to use `operator_lipschitz`, `data_consistency_scale=1.0`, and no public adjoint scale. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_cross_experiment_20260630/padis_dps_512_1step_default_microbatch` | `padis_dps` | 1 `ct_512_60` test slice, stopped after 1 outer step | Escaped-CUDA 512-row memory/dispatch smoke after promoting the memory-safe 512 defaults. The original default local 8GB run OOMed before writing metrics; the driver and matrix manifest now default `padis_dps ct_512_60 --implementation lion_physics` to `patch_batch_size=1` and `patch_checkpoint_denoiser=True`. With only `--stop-after-outer-steps 1` as an extra debug flag, the row completed and the verifier passed 1 record. The sampler used the paper geometric schedule, `sigma_min=0.002`, `sigma_max=10`, least-squares data consistency, and `operator_lipschitz` normalization. The single outer step took about 71 s locally and produced intentionally poor quality, PSNR 4.83 dB versus FDK 27.86 dB. This is dispatch and memory evidence only: the linked 512 checkpoint is a training smoke checkpoint, and full-quality `ct_512_60` diffusion evidence still requires A100 time plus a proper 512 prior. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_cross_experiment_20260630/padis_dps_512_1step_patch_checkpoint_default_v2` | `padis_dps` | 1 `ct_512_60` test slice, stopped after 1 outer step | Escaped-CUDA rerun after splitting the ordinary PaDIS patch checkpoint flag from the fixed-overlap patch-average/stitching flag. No explicit checkpointing sampler flag was passed; the matrix default supplied `patch_batch_size=1`, `patch_checkpoint_denoiser=True`, and `fixed_overlap_checkpoint_denoiser=False`. The row completed in about 70 s on the local GTX 1070, wrote metrics/tensors, and the matrix verifier passed 1 expected record with 1 sample. The sampler payload recorded the paper geometric schedule, `sigma_min=0.002`, `sigma_max=10`, least-squares data consistency, and `operator_lipschitz` normalization. Quality is intentionally poor because this was stopped after one outer step and uses the 512 training-smoke checkpoint: PSNR 4.83 dB versus FDK 27.86 dB. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/nontraining_1sample` | `baseline`, `admm_tv`, `padis_dps`, `langevin`, `predictor_corrector`, `ve_ddnm` | 1 `ct_20` test slice each, Slurm-default LION-physics matrix path | Escaped-CUDA end-to-end matrix run using `--implementations lion_physics`, `--models method_default`, `--max-samples 1`, and the linked 10-hour patch checkpoint. The verifier passed 6 records with expected sampler/method settings and all non-baseline rows beating FDK. Mean PSNRs were baseline/FDK 23.17 dB, ADMM-TV 29.49 dB, PaDIS DPS 32.29 dB, Langevin 34.02 dB, PC 29.51 dB, and VE-DDNM 32.95 dB. This validates the current production-default non-training-dependent reconstruction path, but is still a one-slice check rather than the final 25-sample A100 matrix. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/patch_rows_1step_batchfix` | `patch_average`, `patch_stitch` | 1 `ct_20` test slice each, Slurm-default LION-physics matrix path stopped after 1 outer step | Escaped-CUDA smoke after promoting fixed-overlap patch rows to default `patch_batch_size=1` when no explicit override is supplied. The verifier passed 2 records and confirmed `public_overlap`/`public_tile`, checkpointed fixed-overlap denoising, `dps_epsilon=0.5`, `patch_batch_size=1`, and `operator_lipschitz` normalization. Quality is intentionally meaningless because the run stopped after one outer step; the full one-slice patch rows above remain the quality evidence. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/patch_rows_1step_streaming` | `patch_average`, `patch_stitch` | 1 `ct_20` test slice each, Slurm-default LION-physics matrix path stopped after 1 outer step | Escaped-CUDA smoke after changing fixed-overlap denoising to stream patch chunks through the denoiser instead of materializing one full patch batch first. Both rows completed on the local 8GB GPU, and the verifier passed 2 records with `public_overlap`/`public_tile`, checkpointed fixed-overlap denoising, `dps_epsilon=0.5`, `patch_batch_size=1`, and `operator_lipschitz` normalization. Quality is intentionally meaningless because the run stopped after one outer step. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_physics_matrix_20260630/patch_rows_full_default_1sample` | `patch_average`, `patch_stitch` | 1 `ct_20` test slice each, Slurm-default LION-physics matrix path | Escaped-CUDA full fixed-overlap matrix run with no reconstruction override flags beyond previews/progress. Both rows completed on the local 8GB GPU through the production matrix command and the verifier passed 2 records against the saved manifest, confirming `lion_physics`, paper geometric sigma schedule, public-overlap/public-tile denoiser layouts, checkpointed fixed-overlap denoising, `patch_batch_size=1`, least-squares data consistency, and `operator_lipschitz` normalization. Patch-average reached PSNR 33.77 dB, SSIM 0.813, edge SSIM 0.518, MAE 0.0143, and relative sinogram residual 0.00152 versus FDK PSNR 23.17 dB. Patch-stitch reached PSNR 32.33 dB, SSIM 0.789, edge SSIM 0.505, MAE 0.0161, and relative sinogram residual 0.00169 versus FDK PSNR 23.17 dB. This replaces the earlier one-step matrix smokes as the production-path quality evidence for the fixed-overlap rows, but it is still one-slice evidence rather than the final 25-sample A100 result. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_ddnm_pseudoinverse_diag_20260628/baseline_clip` | DDNM pseudoinverse diagnostic | 1 `ct_20` test slice | With the target image used as a perfect denoiser, the clipped LION DDNM correction `A^dagger y + x - A^dagger A(x)` reached 155.1 dB PSNR. This shows the target-consistent pseudoinverse terms cancel for noiseless data; VE-DDNM's remaining failure is not explained by `A^dagger A(target)` alone. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_local_cuda_validation_20260627/public_repo_pc_smoke` | `predictor_corrector` | 1 `ct_20` test slice, stopped after 2 outer steps | Public-repo compatibility smoke with literal README EDM schedule completed and saved trace images. Metrics record FDK init, norm-gradient data consistency, public adjoint schedule, linear PC step, and current-sigma corrector denoising. This is a dispatch/settings smoke, not a quality run. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_local_cuda_patch_assembly_20260627` | `patch_average`, `patch_stitch` | 1 `ct_20` test slice, stopped after 2 outer steps | Current fixed-overlap dispatch/output smoke with trace images enabled. It uses `--disable-data-consistency` because full fixed-overlap DPS exceeded 8GB GPU memory locally. Quality metrics are intentionally not meaningful for this truncated no-data-consistency run. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_local_cuda_patch_assembly_dc_probe_20260627` | `patch_average` | 1 `ct_20` test slice, stopped after 1 outer step | Earlier data-consistency probe failed with CUDA OOM on the local 8GB GPU even at `--patch-batch-size 1`, motivating checkpointed fixed-overlap denoising. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_patch_average_checkpoint_dc_probe_20260628` | `patch_average` | 1 `ct_20` test slice, stopped after 1 outer step | With `--patch-batch-size 1 --fixed-overlap-checkpoint-denoiser`, the full data-consistency fixed-overlap path completed on the local 8GB GPU. Quality metrics are intentionally not meaningful for a one-step truncated run. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_patch_stitch_checkpoint_dc_probe_20260628` | `patch_stitch` | 1 `ct_20` test slice, stopped after 1 outer step | With default checkpointed fixed-overlap denoising and `--patch-batch-size 1`, the full data-consistency stitch path completed on the local 8GB GPU. Quality metrics are intentionally not meaningful for a one-step truncated run. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_patch_average_checkpoint_dc_probe_5step_20260628` | `patch_average` | 1 `ct_20` test slice, stopped after 5 outer steps | Escaped-CUDA run with `--patch-batch-size 1 --fixed-overlap-checkpoint-denoiser`, data consistency, trace JSON, trace images, previews, and tensors completed in about 2.5 minutes on the local GTX 1070. Quality remains intentionally meaningless for a 5-step noise-initialized run. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_patch_stitch_checkpoint_dc_probe_5step_20260628` | `patch_stitch` | 1 `ct_20` test slice, stopped after 5 outer steps | Escaped-CUDA run with `--patch-batch-size 1 --fixed-overlap-checkpoint-denoiser`, data consistency, trace JSON, trace images, previews, and tensors completed in about 2.5 minutes on the local GTX 1070. Quality remains intentionally meaningless for a 5-step noise-initialized run. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_patch_average_checkpoint_dc_full_20260628` | `patch_average` | 1 `ct_20` test slice, full 100 outer / 10 inner sampler | Escaped-CUDA run with `--patch-batch-size 1 --fixed-overlap-checkpoint-denoiser`, data consistency, trace JSON, trace images, previews, and tensors completed in 48:54 on the local GTX 1070. The matrix verifier passed structurally, but quality was poor: mean PSNR 6.18 dB and SSIM -0.283 versus FDK 22.15 dB. This validates the full fixed-overlap reconstruction path, not paper-like CT reconstruction quality. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_patch_stitch_checkpoint_dc_full_20260628` | `patch_stitch` | 1 `ct_20` test slice, full 100 outer / 10 inner sampler | Escaped-CUDA run with `--patch-batch-size 1 --fixed-overlap-checkpoint-denoiser`, data consistency, trace JSON, trace images, previews, and tensors completed in 49:12 on the local GTX 1070. The matrix verifier passed structurally, but quality was poor: mean PSNR 6.63 dB and SSIM -0.241 versus FDK 22.15 dB. This validates the full fixed-overlap stitching reconstruction path, not paper-like CT reconstruction quality. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/local_public_patch_average_full_20260628` | `patch_average` | 1 `ct_20` test slice, full 100 outer / 10 inner sampler | Current public-helper default full run using `--implementation public_repo`, paper CT sigma schedule, public-overlap layout, and checkpointed denoising. It reached PSNR 33.30 dB, SSIM 0.801, edge SSIM 0.538, MAE 0.0156, and p95 absolute error 0.0428 versus FDK PSNR 25.43 dB and SSIM 0.489. This is the current evidence for the usable public-helper patch-average path. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/local_public_patch_stitch_full_20260628` | `patch_stitch` | 1 `ct_20` test slice, full 100 outer / 10 inner sampler | Current public-helper default full run using `--implementation public_repo`, paper CT sigma schedule, public-tile layout, and checkpointed denoising. It reached PSNR 32.89 dB, SSIM 0.787, edge SSIM 0.527, MAE 0.0162, and p95 absolute error 0.0451 versus FDK PSNR 25.43 dB and SSIM 0.489. This is the current evidence for the usable public-helper patch-stitch path. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_whole_image_training_smoke_20260628/whole_lidc_smoke` | `whole_image_diffusion` training | 1 training unit | Tiny CUDA training smoke produced `whole_image_lidc_256.pt` and sidecar metadata with `paper_preset=padis-paper-whole-ct-256`, `prior_mode=whole_image`, and `patch_sizes=[256]`. This validates the checkpoint path and metadata only, not paper-quality whole-image training. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_whole_image_reconstruction_smoke_20260628` | `whole_image_diffusion` | 1 `ct_20` test slice, stopped after 2 outer steps | CUDA reconstruction smoke loaded the tiny whole-image checkpoint and ran the LION CT sampler with `prior_mode=whole_image`, `model_patch_size=256`, 100 outer / 10 inner paper sampler settings, trace images, previews, and tensors. Quality metrics are intentionally not meaningful because the checkpoint is tiny and the run is truncated. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_local_cuda_pnp_smoke_20260627` | `pnp_admm` | 1 `ct_20` test slice | CUDA smoke with the tiny DRUNet denoiser completed and passed the matrix verifier with PnP-ADMM beating FDK on the sample, mean PSNR 26.90 dB versus FDK 22.15 dB. This is a wiring/runtime check, not a paper-quality PnP result. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_pnp_training_smoke_20260627` and `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_pnp_reconstruction_smoke_20260627` | `pnp_admm` | 1 `ct_20` test slice | Tiny DRUNet training-to-reconstruction integration smoke passed. This uses a deliberately tiny denoiser and is not a paper-quality PnP result. |

## Notes And Warnings

- The working reconstruction setup intentionally matches the public repo's
  executed README behavior where the paper and public code differ.
- The `ct_lion_fanbeam` path in `PaDIS_lion_recon` is a comparison shim, not a
  paper setting. It changes public-compatible geometry to LION geometry and uses
  `data_gradient_scale=0.09` to compensate for the ODL adjoint scale.
- PaDIS paper/public CT geometry is not implemented for LIDC-IDRI in LION. The
  processed LIDC tensors do not contain enough physical metadata to convert them
  into PaDIS's 40-unit object support and 80-unit detector span while respecting
  detector physics.
- LION's final reconstruction path keeps LION-native CT operations, fan-beam
  geometry, and FDK initialization as required for the LION implementation.
- `--implementation lion_physics` is the LION-native physical CT preset. It
  keeps the paper geometric sigma schedule but uses a least-squares data term
  with the data step normalized by the composed LION measurement Lipschitz
  constant. In normalized image coordinates the measurement map is
  `F(x)=A(measurement_scale*x + measurement_offset)`, so
  `L=||F||^2=(abs(measurement_scale) * ||A||)^2`; the affine offset is
  deliberately excluded because it does not change the operator norm. The
  operator norm `||A||` is estimated with LION's generic `power_method`,
  matching the convention already used by LION's FISTA implementation where the
  least-squares gradient Lipschitz constant is `||A||^2`. The sampler uses the
  summed residual objective, not a mean residual objective, so there is no
  division by the number of sinogram elements. New reconstruction outputs record
  this runtime scaling in `metrics.json` as `operator_norm_estimate`,
  `measurement_operator_norm`, and `data_lipschitz` when the norm is available
  from the sampler cache or an explicit `operator_norm` override. Its tuned
  default `zeta=3.0` is
  not the paper's `0.3 / L2Norm(y - A(x))` coefficient; this is an intentional
  divergence to avoid public-repo scale constants while keeping the update size
  appropriate for LION's operator normalization.
- The method-default matrix now uses `--implementation lion_physics` for the
  PaDIS diffusion CT rows: `whole_image_diffusion`, `padis_dps`, `langevin`,
  `predictor_corrector`, `ve_ddnm`, `patch_average`, and `patch_stitch`. DPS,
  predictor-corrector, and Langevin have been checked against 3-sample
  public-compatible references. Whole-image diffusion has 3-sample matrix-path
  validation for `ct_20`, `ct_8`, `ct_60`, and `ct_fanbeam_180`; VE-DDNM has
  3-sample `ct_20` matrix-path validation; patch averaging and patch stitching
  have one-slice full-run validation under the physical preset. A broader
  25-sample matrix run is still needed before treating the
  full paper result array as final; A100/CUDA CT validation remains the required
  evidence for final paper-matrix claims.
- The tuned LION-physics sampler settings are paper divergences where needed to
  match public-compatible reconstruction quality without public CT scale
  constants: PC uses `pc_snr=0.08`; Langevin uses
  `sampling_epsilon=0.5`; VE-DDNM uses `sampling_epsilon=0.1`, noise
  initialization, and corrected DDNM clipping for LION fan-beam stability; the
  whole-image `ct_fanbeam_180` row uses `dps_epsilon=0.5`; and fixed-overlap
  patch averaging/stitching use the public overlap/tile denoiser layouts plus
  checkpointed denoising, `patch_batch_size=1`, and `dps_epsilon=0.5`.
  Fixed-overlap denoising streams patch chunks of at most `patch_batch_size`
  through the denoiser and assembles them immediately; this is a memory
  control, and changing it should not change the CT data objective or physical
  update scale. The CT data objective and update scaling in these defaults
  remain LION-native least-squares with `operator_lipschitz` normalization.
- The normal PaDIS DPS patch path also supports streamed patch chunks. For the
  `ct_512_60` LION-physics row the driver defaults to `patch_batch_size=1` and
  `patch_checkpoint_denoiser=True`, because the full patch batch exceeded local
  8GB GPU memory before writing metrics. The older
  `fixed_overlap_checkpoint_denoiser` flag is still accepted as a compatibility
  alias in the reconstructor, but the matrix uses the generic flag for ordinary
  PaDIS DPS and keeps the fixed-overlap flag for patch averaging/stitching. This
  is a memory-only default and is not a public-repo CT scale constant.
- The `baseline` row uses FDK for LION fan-beam geometry. The paper names FBP;
  FDK is the corresponding analytic reconstruction for this final LION geometry.
- The `admm_tv` row is a LION-native TV substitute. It uses
  `LION.classical_algorithms.tv_min`, which is Chambolle-Pock TV minimization,
  not the paper's exact ADMM-TV implementation.
- The `pnp_admm` row is only comparable after a DRUNet denoiser has been trained
  or supplied. The paper says its denoising CNNs were trained following the
  PnP/RED baseline source, but does not give enough optimizer, architecture, or
  stopping-rule detail for a bit-exact denoiser reproduction. The provided
  `PaDIS_LIDC_PnP_denoiser.py` job is therefore a LION-native DRUNet surrogate;
  its quality depends on that denoiser checkpoint and is not determined by the
  PaDIS diffusion checkpoint. The PaDIS reconstruction driver now clips
  PnP-ADMM iterates and denoiser outputs to `[0, 1]`, because the LIDC
  reconstruction variable is a normalized image and the DRUNet denoiser was
  trained on that support. This is a LION-native stability constraint not
  specified by the PaDIS paper; it prevents sparse-view `ct_8` ADMM iterates
  from leaving the denoiser domain and producing NaNs.
- The `baseline`, `admm_tv`, and `pnp_admm` matrix rows are no-PaDIS-prior
  methods. They now run without a PaDIS diffusion checkpoint and use the LION
  CT experiment geometry plus `--image-scaling` defaults unless an explicit
  checkpoint is supplied for metadata only. The verifier records an empty
  checkpoint identity for these rows.
- The latest local six-hour W&B runs give different confidence levels for the
  two training-dependent rows. The PnP-ADMM DRUNet path now has a clean
  train-to-reconstruction run and beats FDK on one `ct_20` slice under both the
  earlier diagnostic path and the forced `lion_physics` matrix path, so the LION
  implementation wiring is credible; it is still a LION-native DRUNet surrogate
  because the PaDIS paper does not provide enough detail for an exact denoiser
  reproduction. The whole-image diffusion path also trains, loads, samples, and
  writes outputs under both the full paper CT preset and the forced
  `lion_physics` matrix path. Strict paper-mode reconstruction with the
  six-hour min-validation checkpoint was poor, but full LION-physics 3-sample
  matrix runs with the same min-validation checkpoint now beat FDK on PSNR for
  `ct_20`, `ct_8`, `ct_60`, and the promoted `ct_fanbeam_180` relaxation. The
  first fanbeam whole-image sample still has lower SSIM than FDK despite higher
  PSNR, so treat it as finite and competitive rather than a uniform metric win.
  The matrix therefore uses `whole_image_lidc_256_min_val.pt` for whole-image
  reconstruction. Treat this as small-sample physical-preset evidence, not final
  paper-array evidence, until the 25-sample A100 matrix has run.
- Strict paper-mode one-sample `predictor_corrector` and `ve_ddnm` runs
  completed but were below FDK on PSNR. The predictor-corrector implementation
  now follows the paper/public linear corrector step. The paper mode denoises
  the corrector at the next/lower sigma; `--implementation public_repo` uses the
  public code's current-sigma denoising behavior and reuses the predictor patch
  layout for the corrector denoising call, matching the public helper's
  `indices = getIndices(...)` placement. This produced much better local
  quality than strict paper mode. The older squared score-SDE step-size form is
  retained only behind `--pc-corrector-step-rule score_sde_squared` for
  diagnostics.
- The Slurm reconstruction scripts now default to
  `PADIS_RECON_IMPLEMENTATIONS=lion_physics` for the full reconstruction array.
  The Python matrix CLI still accepts `--implementations method_default` for
  method-specific diagnostics, but the A100 production path is explicitly
  LION-physics by default. Use `PADIS_RECON_IMPLEMENTATIONS=paper` for
  strict-paper diagnostic runs or `PADIS_RECON_IMPLEMENTATIONS=public_repo` for
  compatibility diagnostics against the original public helper behavior.
- The method-default `ve_ddnm` row is still not a literal Algorithm A.3 run.
  It keeps the paper `paper_1000x1` NFE layout, but uses
  `sampling_epsilon=0.1`, noise initialization, and corrected-estimate clipping
  for LION fan-beam stability. On one local `ct_20` slice the `lion_physics`
  default reached 33.06 dB PSNR versus 22.15 dB for FDK.
- The current full `paper_1000x1` VE-DDNM row completed locally but scored
  -23.07 dB PSNR on one `ct_20` slice, versus 22.15 dB for FDK. The
  non-paper `--ddnm-corrected-clip` diagnostic improved the same layout to
  15.64 dB, still below FDK. Reducing `sampling_epsilon` to 0.1 with the same
  corrected clipping produced the method-default LION-stabilized behavior.
- The VE-DDNM row now defaults to `--ve-ddnm-nfe-layout paper_1000x1`, because
  Algorithm A.3 shows one denoising call per noise level and the paper states
  1000 NFEs for diffusion experiments. Use
  `--ve-ddnm-nfe-layout public_inner` to reproduce the public helper's 100 outer
  noise levels with 10 repeated inner updates per level.
- The VE-DDNM row clips the LION FDK pseudoinverse terms for finite output under
  fan-beam geometry. The strict unclipped `A^dagger y` term produced non-finite
  reconstructions locally, and the closer public-helper diagnostic with
  unclipped `A^dagger A(D)` also produced non-finite metrics. This is a
  LION-geometry stability divergence from Algorithm A.3 in the paper. Use
  `--no-ddnm-pseudoinverse-clip` and/or
  `--no-ddnm-projected-pseudoinverse-clip` only for diagnostics.
- `--ddnm-corrected-clip` is an additional VE-DDNM diagnostic for LION fan-beam
  geometry. It clips the corrected clean estimate
  `A^dagger y + D - A^dagger A(D)`, which is not part of Algorithm A.3 or the
  public helper. It improved one-sample public-compatible VE-DDNM from about
  5.7 dB to 20.0 dB PSNR locally, but still stayed below FDK and is not used as
  evidence of a quality-matched VE-DDNM reproduction.
- Full DPS runs for `patch_average` and `patch_stitch` retain the fixed-overlap
  denoiser graph. These rows now enable activation checkpointing by default so
  the denoiser microbatches are recomputed during the DPS backward pass instead
  of retaining all activations. This made the full `patch_average` CT
  and `patch_stitch` CT reconstruction paths fit on the local 8GB GPU. Earlier
  fixed-overlap probes with older settings were far below FDK, but the current
  public-helper default full one-slice runs beat FDK clearly: patch averaging
  reached 33.30 dB PSNR and patch stitching reached 32.89 dB PSNR versus FDK
  25.43 dB on the same sample. The paper reports patch
  averaging and patch stitching as CT reconstruction comparison rows in the main
  inverse-problem table, and also shows unconditional CT generation examples in
  the appendix. Their inclusion in the LION reconstruction matrix is therefore
  paper-aligned. However, the PaDIS paper only gives a short adaptation of the
  cited methods: fixed patch locations, overlap 8, and average or overwrite
  overlap handling. Public-helper diagnostics remain available through
  `--implementation public_repo` and the `PaDIS_lion_recon` fork. The production
  method-default rows now use `--implementation lion_physics`, keeping the
  public overlap/tile denoiser layouts but replacing public CT scale constants
  with LION-native least-squares/operator-Lipschitz data updates. The only
  helper-level divergence in the public fork is that the public
  `denoisedOverlap(...)` formula is bounded to the last valid patch start,
  because the upstream helper overruns the default padded image and asserts.
  Treat the public-helper rows as compatibility/runtime validation rather than a
  quality-matched reproduction of the cited source papers.
- The public PaDIS repository only provides a directly comparable executable
  command-line path for the DPS reconstruction used by the README. It defines
  Langevin, predictor-corrector, DDNM, patch-overlap averaging, and
  patch-tiling helpers, but the public script's main reconstruction path invokes
  DPS. The LION-compatible `PaDIS_lion_recon` fork now exposes those helpers
  through `--sampler pc|langevin|ddnm|patch_average|patch_stitch` for diagnostic
  reference generation, with `--patch_batch_size` to fit them on smaller GPUs.
  LION still treats `--implementation public_repo` as formula-level
  compatibility unless an explicit helper-reference run is supplied through
  `--public-reference-reconstructions`. Full output-level matching is claimed
  for full `padis_dps` runs and for 100-step public-helper
  `predictor_corrector`, `langevin`, `patch_average`, and `patch_stitch`
  diagnostics on the local one-slice reference. The fixed-overlap
  `patch_average` and `patch_stitch` public-helper runs required one
  compatibility-fork runtime fix: detach the updated iterate between inner
  updates so PyTorch does not retain the whole previous denoiser/data-gradient
  graph. This preserves the update value but is a runtime divergence from the
  literal public helper code. LION and the fork also agree exactly at
  fixed-overlap denoiser-assembly level under a dummy denoiser. A one-step
  `predictor_corrector` comparison against the public-fork
  `pc_sampling` helper now matches closely when
  `--public-repo-helper-initialization` is used, because that flag reproduces
  the helper's central-noise-then-pad initial state. A one-step Langevin
  comparison also matches closely using the helper's padded-state noise
  convention. A one-step DDNM comparison also matches closely with the public
  helper's padded-state noise and `public_inner` loop layout.
  A 5-step predictor-corrector comparison with helper initialization,
  public-helper layout reuse, and the split direct-adjoint scale now matches
  the public fork closely (public-reference SSIM 0.9987, MAE 0.00459, p95
  absolute error 0.0133). A 5-step Langevin helper comparison also matches
  closely after routing Langevin through the same direct-adjoint correction
  helper (public-reference SSIM 0.9998, MAE 0.00165, p95 absolute error 0.0).
  A 5-step DDNM helper comparison also matches closely in `public_inner` layout
  (public-reference SSIM 0.9996, MAE 0.00217, p95 absolute error 0.0). The
  100-step public DDNM helper produced an all-NaN reconstruction in both the
  matched LION fan-beam setup and the LION-scale parallel-beam shim. The
  original public `ct_parbeam` geometry stayed finite over 100 steps, but with
  unusably low quality on the LIDC PNG diagnostic sample. The LION-stabilized
  VE-DDNM row therefore remains a deliberate non-public stability divergence
  for the LION geometry. The detach-fixed 100-step fixed-overlap helper checks
  then completed successfully: patch-average matched the public reference with
  SSIM 0.9973 and p95 absolute error 0.00387, and patch-stitch matched with
  SSIM 0.9966 and p95 absolute error 0.00500.
  The public repo does not provide runnable ADMM-TV, PnP-ADMM, or whole-image
  diffusion comparison rows to match against.
- The target images retain high-frequency CT texture/noise that both public
  PaDIS and LION PaDIS smooth. Matching the public repo and exactly preserving
  all target texture are not simultaneously achievable with this sampler setup.
