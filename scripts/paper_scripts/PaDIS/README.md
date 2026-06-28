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
| Noise schedule type | Geometrically spaced descending noise levels. | README command executes an EDM/Karras-style power schedule with `rho=7`; it is not geometric in code unless changed by command-line arguments. | Both `--implementation paper` and `--implementation public_repo` use the paper geometric schedule in the reconstruction array. | Matches the paper schedule for array runs; diverges from literal README defaults. |
| CT sigma range | For 20-view CT: `sigma_max=10`, `sigma_min=0.002`. For 8-view CT: `sigma_min=0.003`. 60-view and fan-beam CT use the main 20-view CT schedule. | README command shows `sigma_min=0.003` for 20 views, but this is command-line configurable. | Both `--implementation paper` and `--implementation public_repo` use the paper sigma range per experiment. | Matches the paper schedule for array runs; diverges from literal README defaults. |
| Sampler length | 100 outer steps and 10 inner steps, about 1000 neural function evaluations. | README command uses `steps=100`; public DPS performs 10 inner denoising steps per outer step. | Uses 100 outer steps and 10 inner steps. | Matches. |
| Initial sampler state | Paper Langevin pseudocode initializes from Gaussian noise. FBP is discussed as a baseline, not as the sampler initial state. | Public CT DPS computes a filtered backprojection-style reconstruction, clips it, pads it, and starts from that state. | Uses LION-native FDK initialization, clipped to `[0, 1]`, then padded. | Matches public behavior conceptually; differs from the paper. |
| FBP / FDK filter | Not applicable to paper sampler initialization. | Public parallel-beam path uses Hann FBP frequency scaling `0.9`; public fan-beam path uses `0.3`. | `--implementation public_repo --geometry lion` uses Hann FDK `0.3`. | Public-compatible for the validated LION geometry path; differs from the paper. |
| Reconstruction method | PaDIS with DPS or Langevin-style data consistency. | Public README reconstruction uses the `dps()` path in `inverse_nodist.py`. | Uses LION `dps_langevin`. | Matches the public DPS/Langevin method family. |
| Langevin / DPS epsilon | Paper states `epsilon=1` for Langevin and DDNM. | Public DPS code uses `alpha = 0.5 * sigma^2`. | `--implementation paper` uses `dps_epsilon=1`. `--implementation public_repo` uses `dps_epsilon=0.5`. | Toggle implemented. |
| Data consistency objective | Paper pseudocode applies an adjoint residual step. A strict paper-style LION preset corresponds to a squared-residual objective with residual-normalized step size. | Public DPS uses the gradient of the L2 norm of the residual, computed from `y - A(x0hat)`. | `--implementation paper` uses the squared-residual objective. `--implementation public_repo` uses the norm-gradient DPS objective. | Toggle implemented. |
| Data step size | Paper describes `zeta_i = 0.3 / L2Norm(y - A(x))` for Langevin and PC-style data steps. | Public DPS applies `x = x - zeta * grad(L2Norm(y - A(x0hat)))` with `zeta=0.3`; the norm-gradient already normalizes the gradient direction. | `--implementation paper` uses residual-normalized paper stepping. `--implementation public_repo --geometry lion` uses calibrated CT gradient scale `0.0405`. | Toggle implemented, with a LION-geometry public-repo calibration. |
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
| `--public-repo-sigma-schedule` | `paper` / `readme` | Select the sigma schedule for `--implementation public_repo`. The array default is `paper`; `readme` is only for reproducing literal public README/default runs. |
| `--public-repo-helper-initialization` | flag | Diagnostic output-comparison mode for `--implementation public_repo` with `predictor_corrector`, `langevin`, or `ve_ddnm`. It uses the public helper functions' Gaussian initial-state convention instead of the README DPS FDK initial-state convention: PC samples central-image noise and then pads it; Langevin/DDNM sample the already padded state. |
| `--implementation` | `lion_quality` | LION-native quality preset. The method-default matrix uses this for VE-DDNM because strict paper-mode VE-DDNM is unstable under the LION fan-beam pseudoinverse. |
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

## Paper Method Rows

`PaDIS_LIDC_reconstruction.py` exposes the paper comparison rows through
`--method`:

| Method | Training dependency | LION implementation | Paper agreement |
|---|---|---|---|
| `baseline` | None | LION FDK/FBP-style analytic reconstruction from the CT operator | Paper says CT baseline is FBP. In LION fan-beam geometry this is implemented with FDK, the corresponding fan-beam analytic baseline. |
| `admm_tv` | None | LION `tv_min` total-variation reconstruction | Similar method only: current LION TV uses Chambolle-Pock, not the paper's ADMM-TV solver. Uses paper CT `lambda=0.001` by default. |
| `pnp_admm` | DRUNet denoiser | LION `PnP(..., algorithm="ADMM")` with a DRUNet denoiser wrapper | Requires a trained denoising CNN. Agreement depends on the denoiser training run; this is not the PaDIS diffusion checkpoint. |
| `whole_image_diffusion` | Whole-image NCSN++ checkpoint | LION PaDIS sampler with `prior_mode=whole_image` | Matches the paper comparison design when the whole-image checkpoint is trained with the paper architecture/settings. |
| `langevin` | Patch PaDIS checkpoint | LION PaDIS Langevin sampler | Method default uses `--implementation public_repo` with the paper geometric CT sigma schedule, because the public helper mechanics reproduced reported-quality local behavior. Strict paper mode remains available explicitly and uses `epsilon=1`, noise initialization, and the paper residual step. |
| `predictor_corrector` | Patch PaDIS checkpoint | LION PaDIS predictor-corrector sampler | Method default uses `--implementation public_repo` with paper/public `r=0.16`, the linear corrector step `2*r*||z||/||score||`, and the public code's current-sigma corrector denoising. Strict paper mode remains available and denoises the corrector at the next/lower sigma. |
| `ve_ddnm` | Patch PaDIS checkpoint | LION PaDIS Langevin sampler with VE-DDNM correction | Strict `--implementation paper` follows Algorithm A.3 with 1000 sigma levels, one denoising update per level, and `epsilon=1`, but is unstable locally. The method-default matrix uses `--implementation lion_quality`, keeps the paper `paper_1000x1` NFE layout, clips LION FDK pseudoinverse terms and the corrected DDNM estimate, and uses `sampling_epsilon=0.1`. This is a LION fan-beam stability divergence from the paper. |
| `patch_average` | Patch PaDIS checkpoint | Fixed-overlap patch denoising with averaged overlap pixels | Method default uses `--implementation public_repo` and mirrors the public PaDIS `denoisedOverlap(...)` helper inside the DPS loop: patch size 56, overlap 8, start at `pad`, and average overlaps. LION clips the final overlap start to the last valid patch because the upstream helper overruns the README default padded image. `--implementation paper` keeps the earlier LION clipped layout. This is not a faithful implementation of the original conditional patch-DDPM paper cited as `[23]`. |
| `patch_stitch` | Patch PaDIS checkpoint | Fixed-overlap patch denoising with overwrite/stitching semantics | Method default uses `--implementation public_repo` and mirrors the public PaDIS `denoisedTile(...)` helper inside the DPS loop: patch size 56, overlap 8, hard-coded first start `4`, and overwrite/stitch overlaps. `--implementation paper` keeps the earlier LION clipped layout. This is not a faithful implementation of the original tile-and-stitch paper cited as `[66]`. |
| `padis_dps` | Patch PaDIS checkpoint | Main PaDIS patch sampler with DPS/Langevin-style data consistency | Method default uses `--implementation public_repo` with the paper geometric CT sigma schedule, matching the public README reconstruction mechanics while avoiding the README's EDM schedule. |

The Slurm reconstruction matrix defaults to `PADIS_RECON_METHODS=all`,
`PADIS_RECON_MODELS=method_default`, and
`PADIS_RECON_IMPLEMENTATIONS=method_default`. This means each method selects the
checkpoint family and implementation fallback it needs: patch PaDIS for patch
methods, whole-image PaDIS for `whole_image_diffusion`, a DRUNet denoiser for
`pnp_admm`, public-repo mechanics for `padis_dps`, `langevin`, and
`predictor_corrector`, and the LION-stabilized VE-DDNM settings for
`ve_ddnm`. The public-repo method defaults still use the paper geometric CT
sigma schedule. To force strict paper mechanics for a diagnostic run, set
`PADIS_RECON_IMPLEMENTATIONS=paper` or pass
`--implementations paper` to `PaDIS_run_reconstruction_matrix.py`.
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

Set `PADIS_SUBMIT_PNP_TRAINING=0` to disable the PnP denoiser job in either
submitter. To submit the reconstruction matrix automatically after real
training and PnP training finish, enable the opt-in reconstruction chain:

```bash
PADIS_SUBMIT_RECONSTRUCTION=1 \
PADIS_RECON_METHODS=all \
PADIS_RECON_MODELS=method_default \
PADIS_RECON_IMPLEMENTATIONS=method_default \
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
PADIS_RECON_IMPLEMENTATIONS=method_default \
PADIS_RECON_GEOMETRIES=lion \
PADIS_RECON_MAX_SAMPLES=25 \
scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_reconstruction.sh
```

The reconstruction scripts default to `--split test --max-samples 25`, matching
the paper's CT reconstruction evaluation budget. Lower
`PADIS_RECON_MAX_SAMPLES` only for pilot/debug runs.

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
PADIS_RECON_IMPLEMENTATIONS=method_default
PADIS_RECON_GEOMETRIES=lion
```

Raw reconstruction flags can be passed through the Slurm array with
`PADIS_RECON_EXTRA_ARGS`. Prefer method-specific defaults where possible:
`PADIS_RECON_METHODS=ve_ddnm PADIS_RECON_IMPLEMENTATIONS=method_default` runs
the LION-stabilized VE-DDNM row, while
`PADIS_RECON_METHODS=ve_ddnm PADIS_RECON_IMPLEMENTATIONS=paper` runs the
strict paper diagnostic row.

The matrix can also be inspected locally without submitting Slurm:

```bash
python scripts/paper_scripts/PaDIS/PaDIS_run_reconstruction_matrix.py \
  --training-root /path/to/a100_training_<stamp> \
  --output-root /tmp/padis_recon_matrix_preview \
  --models method_default \
  --methods all \
  --implementations method_default \
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
$TRAIN_ROOT/whole_lidc_default/whole_image_lidc_256.pt
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
  --implementations method_default \
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
  --implementations method_default \
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
  --implementations method_default \
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
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_cuda_training_dependent_rerun_20260628` | `whole_image_diffusion` training, `pnp_admm` denoiser training, `whole_image_diffusion`, `pnp_admm` | 1 tiny training smoke each; 1 `ct_20` reconstruction slice each | Current-code escaped-CUDA smoke for the two training-dependent rows. Whole-image diffusion trained one patch budget unit and produced `whole_lidc_default/whole_image_lidc_256.pt`; PnP trained a deliberately tiny DRUNet and produced `pnp_lidc_drunet/pnp_lidc_drunet.pt`. The verifier passed structurally for both rows and applied a quality gate to PnP-ADMM. PnP-ADMM reached 28.03 dB versus FDK 22.15 dB in this tiny smoke; whole-image diffusion was intentionally not quality-meaningful because the checkpoint is one-step trained and the reconstruction was stopped after 2 outer steps. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_sampler_dps_smoke_20260628` | public-fork `dps` helper | 1 LIDC PNG, stopped after 1 outer step | Escaped-CUDA smoke of the LION-compatible public fork's default DPS sampler after adding public helper sampler selection and patch-denoiser microbatch plumbing. The run completed with the default gradient-tracking DPS path and wrote `reconstructions.npz` plus `trace.json`. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_sampler_pc_trace_20260628` | public-fork `pc_sampling` helper | 1 LIDC PNG, stopped after 1 outer step | Escaped-CUDA smoke of the LION-compatible public fork's predictor-corrector helper via `--sampler pc --patch_batch_size 1 --trace_interval 1`. The helper path completed on the local 8GB GPU after disabling denoiser-gradient storage for non-DPS public helpers and wrote `reconstructions.npz` plus PC predictor/corrector trace statistics. This is a reference-generation smoke, not a quality run. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_sampler_langevin_smoke_20260628` | public-fork `langevin` helper | 1 LIDC PNG, stopped after 1 outer step | Escaped-CUDA smoke of the LION-compatible public fork's Langevin helper via `--sampler langevin --patch_batch_size 1`. The helper branch completed and wrote `reconstructions.npz`; this is a reference-generation smoke, not a quality run. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_sampler_ddnm_smoke_20260628` | public-fork DDNM helper | 1 LIDC PNG, stopped after 1 outer step | Escaped-CUDA smoke of the LION-compatible public fork's DDNM helper via `--sampler ddnm --patch_batch_size 1`. This executes the public `langevin(..., ddnm=True)` branch and wrote `reconstructions.npz`; it is a reference-generation smoke, not a quality run. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_patch_average_1step_checkpoint_20260628` | public-fork `patch_average` helper | 1 LIDC PNG, stopped after 1 outer step | Escaped-CUDA smoke of the LION-compatible public fork's `denoisedOverlap(...)` helper inside the DPS loop via `--sampler patch_average --patch_batch_size 1 --checkpoint_denoiser`. The upstream helper's default overrun is bounded to the last valid patch in this fork path. The run completed and wrote `reconstructions.npz`, intermediates, and trace statistics. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_public_helper_patch_stitch_1step_checkpoint_20260628` | public-fork `patch_stitch` helper | 1 LIDC PNG, stopped after 1 outer step | Escaped-CUDA smoke of the LION-compatible public fork's `denoisedTile(...)` helper inside the DPS loop via `--sampler patch_stitch --patch_batch_size 1 --checkpoint_denoiser`. This keeps the public helper's hard-coded start index `4`. The run completed and wrote `reconstructions.npz`, intermediates, and trace statistics. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_pc_1step_central_noise_20260628` | LION `predictor_corrector` versus public-fork `pc_sampling` helper | 1 `ct_20` test slice, stopped after 1 outer step | Escaped-CUDA comparison using `--implementation public_repo --public-repo-sigma-schedule readme --public-repo-helper-initialization` and `--public-reference-reconstructions` from the traced public-fork PC smoke. After matching the helper's central-noise-then-pad initialization, LION reached public-reference SSIM 0.991, edge SSIM 0.980, MAE 0.0120, and p95 absolute error 0.0255. This validates one-step output-level alignment for the public PC helper under LION geometry. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_langevin_1step_helper_init_fixed_20260628` | LION `langevin` versus public-fork `langevin` helper | 1 `ct_20` test slice, stopped after 1 outer step | Escaped-CUDA comparison using `--implementation public_repo --public-repo-sigma-schedule readme --public-repo-helper-initialization` and `--public-reference-reconstructions` from the public-fork Langevin smoke. With the public helper's padded-state noise convention, LION reached public-reference SSIM 0.980, edge SSIM 0.954, MAE 0.0196, and p95 absolute error 0.0612. This validates one-step output-level alignment for the public Langevin helper under LION geometry. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_ddnm_1step_helper_init_20260628` | LION `ve_ddnm` versus public-fork DDNM helper | 1 `ct_20` test slice, stopped after 1 outer step | Escaped-CUDA comparison using `--implementation public_repo --public-repo-sigma-schedule readme --public-repo-helper-initialization --ve-ddnm-nfe-layout public_inner` and `--public-reference-reconstructions` from the public-fork DDNM smoke. With padded-state noise and the public helper's 100x10 loop layout, LION reached public-reference SSIM 1.000, edge SSIM 0.999, MAE 0.00182, and p95 absolute error 0.0. This validates one-step output-level alignment for the public DDNM helper under LION geometry. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_patch_average_1step_20260628` | LION `patch_average` versus public-fork `patch_average` helper | 1 `ct_20` test slice, stopped after 1 outer step | Escaped-CUDA comparison using `--implementation public_repo --public-repo-sigma-schedule readme`, `fixed_overlap_layout=public_overlap`, checkpointed denoising, and `--public-reference-reconstructions` from the public-fork patch-average smoke. LION reached public-reference SSIM 1.000, edge SSIM 1.000, MAE 1.35e-5, and p95 absolute error 0.0. This validates one-step output-level alignment for the public overlap-averaging helper under LION geometry. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_lion_vs_public_helper_patch_stitch_1step_20260628` | LION `patch_stitch` versus public-fork `patch_stitch` helper | 1 `ct_20` test slice, stopped after 1 outer step | Escaped-CUDA comparison using `--implementation public_repo --public-repo-sigma-schedule readme`, `fixed_overlap_layout=public_tile`, checkpointed denoising, and `--public-reference-reconstructions` from the public-fork patch-stitch smoke. LION reached public-reference SSIM 1.000, edge SSIM 1.000, MAE 1.32e-5, and p95 absolute error 0.0. This validates one-step output-level alignment for the public tile/stitch helper under LION geometry. |
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
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_ve_ddnm_paper_layout_eps0p1_corrected_clip_full_20260628` | `ve_ddnm` | 1 `ct_20` test slice, full `paper_1000x1` layout | LION-stabilized VE-DDNM with `sampling_epsilon=0.1` and `--ddnm-corrected-clip` reached mean PSNR 32.94 dB and SSIM 0.766 versus FDK 22.15 dB. The method-default Slurm matrix uses these VE-DDNM stability settings through `--implementation lion_quality`. |
| `/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/codex_ve_ddnm_lion_quality_full_20260628` | `ve_ddnm` | 1 `ct_20` test slice, full `paper_1000x1` layout | Method-default `--implementation lion_quality` VE-DDNM completed with trace JSON, trace images, previews, and tensors. It reached mean PSNR 33.09 dB and SSIM 0.772 versus FDK 22.15 dB; the verifier passed `ve_ddnm >= 32 dB` and mean-better-than-FDK gates. This confirms the matrix default reproduces the stabilized VE-DDNM behavior without ad hoc extra flags. |
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
  PaDIS diffusion checkpoint.
- The `baseline`, `admm_tv`, and `pnp_admm` matrix rows are no-PaDIS-prior
  methods. They now run without a PaDIS diffusion checkpoint and use the LION
  CT experiment geometry plus `--image-scaling` defaults unless an explicit
  checkpoint is supplied for metadata only. The verifier records an empty
  checkpoint identity for these rows.
- The `whole_image_diffusion`, `pnp_admm`, and newly trained PaDIS comparison
  rows still require A100/CUDA CT validation after their checkpoints are
  available. Local CPU tests cover dispatch, metrics, and output writing, but
  not final CT reconstruction quality for these rows.
- Strict paper-mode one-sample `predictor_corrector` and `ve_ddnm` runs
  completed but were below FDK on PSNR. The predictor-corrector implementation
  now follows the paper/public linear corrector step. The paper mode denoises
  the corrector at the next/lower sigma; `--implementation public_repo` uses the
  public code's current-sigma denoising behavior and produced much better local
  quality. The older squared score-SDE step-size form is retained only behind
  `--pc-corrector-step-rule score_sde_squared` for diagnostics.
- The method-default matrix uses public-repo mechanics for `padis_dps`,
  `langevin`, and `predictor_corrector` because those settings are the only
  locally validated route to reported-like reconstruction behavior. This is an
  explicit divergence from the paper pseudocode. Use
  `PADIS_RECON_IMPLEMENTATIONS=paper` for strict-paper diagnostic runs.
- The method-default `ve_ddnm` row uses `--implementation lion_quality`, not
  strict paper mode. This keeps the paper `paper_1000x1` NFE layout, but uses
  `sampling_epsilon=0.1` and clips the corrected DDNM estimate. On one local
  `ct_20` slice this reached 33.09 dB PSNR versus 22.15 dB for FDK. Treat it as
  a LION fan-beam stability reproduction of VE-DDNM, not a literal Algorithm
  A.3 run.
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
  and `patch_stitch` CT reconstruction paths fit on the local 8GB GPU, but the
  one-sample full runs were far below FDK quality. The paper reports patch
  averaging and patch stitching as CT reconstruction comparison rows in the main
  inverse-problem table, and also shows unconditional CT generation examples in
  the appendix. Their inclusion in the LION reconstruction matrix is therefore
  paper-aligned. However, the PaDIS paper only gives a short adaptation of the
  cited methods: fixed patch locations, overlap 8, and average or overwrite
  overlap handling. The method-default rows now use `--implementation
  public_repo`, matching the public PaDIS helper semantics exposed through the
  `PaDIS_lion_recon` fork. The only helper-level divergence is that the public
  `denoisedOverlap(...)` formula is bounded to the last valid patch start,
  because the upstream helper overruns the default padded image and asserts.
  Treat these rows as public-helper compatibility and runtime validation rather
  than a quality-matched reproduction of the cited source papers.
- The public PaDIS repository only provides a directly comparable executable
  command-line path for the DPS reconstruction used by the README. It defines
  Langevin, predictor-corrector, DDNM, patch-overlap averaging, and
  patch-tiling helpers, but the public script's main reconstruction path invokes
  DPS. The LION-compatible `PaDIS_lion_recon` fork now exposes those helpers
  through `--sampler pc|langevin|ddnm|patch_average|patch_stitch` for diagnostic
  reference generation, with `--patch_batch_size` to fit them on smaller GPUs.
  LION still treats `--implementation public_repo` for `langevin`,
  `predictor_corrector`, and `ve_ddnm` as formula-level compatibility unless an
  explicit helper-reference run is supplied through
  `--public-reference-reconstructions`; full output-level matching is only
  claimed for full `padis_dps` runs. For `patch_average` and `patch_stitch`,
  LION and the fork now agree exactly at fixed-overlap denoiser-assembly level
  under a dummy denoiser. A one-step `predictor_corrector` comparison against
  the public-fork `pc_sampling` helper now matches closely when
  `--public-repo-helper-initialization` is used, because that flag reproduces
  the helper's central-noise-then-pad initial state. A one-step Langevin
  comparison also matches closely using the helper's padded-state noise
  convention. A one-step DDNM comparison also matches closely with the public
  helper's padded-state noise and `public_inner` loop layout.
  Full PC/Langevin/DDNM helper-output matching still needs longer reference runs
  before being claimed.
  The public repo does not provide runnable ADMM-TV, PnP-ADMM, whole-image
  diffusion, patch averaging, or patch stitching comparison rows to match
  against.
- The target images retain high-frequency CT texture/noise that both public
  PaDIS and LION PaDIS smooth. Matching the public repo and exactly preserving
  all target texture are not simultaneously achievable with this sampler setup.
