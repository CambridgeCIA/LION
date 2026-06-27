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

New runs that use `--experiment ct_20` write to:

```bash
/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/reconstruction_lion_public_preset_strict_visual_gates_3sample/ct_20/test/dps_langevin
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
| `--implementation` | `public_repo` | Public README/code reconstruction mechanics with paper CT sigma scheduling: FDK/FBP initialization, norm-gradient DPS objective, public RNG/patch-offset behavior, geometric schedule, and paper CT sigma range. |
| `--public-repo-sigma-schedule` | `paper` / `readme` | Select the sigma schedule for `--implementation public_repo`. The array default is `paper`; `readme` is only for reproducing literal public README/default runs. |
| `--implementation` | `lion_quality` | LION-native quality preset retained for diagnostics; not a paper or public-repo preset. |
| `--implementation` | `custom` | Use explicit low-level sampler flags. |
| `--geometry` | `lion` | LION fan-beam CT geometry: detector 900, full 360 degree angular range, LION-native FDK. This is the only executable LIDC-IDRI geometry. |
| `--geometry` | `padis` | Accepted only to raise an explicit error. The LIDC-IDRI tensors used here cannot be converted into the PaDIS public-repo geometry in a physically correct way. |
| `--geometry` | `padis_parallel` / `padis_fanbeam` | Also accepted only to raise the same explicit physical-correctness error. |
| `--experiment` | `ct_8`, `ct_20`, `ct_60`, `ct_fanbeam_180`, `ct_512_60` | Paper-facing CT experiment aliases. Old class names such as `PaDISFanBeam20CTRecon` are still accepted as aliases. |

The older flags `--paper-ct-sampling`, `--public-padis-ct-sampling`, and
`--lion-quality-ct-sampling` remain available as deprecated aliases.

The paper-matrix runner expands the trained-model tasks from
`slurm/submit_PaDIS_A100_training_only.sh` into reconstruction jobs:

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

With the default `paper,public_repo` implementations and `lion` geometry, this
is a 32-job Slurm array.

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
/home/thomas/DiS/Project/Data/experiments/PaDIS/debug_runs/padis_lidc_256_default_10h_local_20260624_232630/reconstruction_lion_public_preset_strict_visual_gates_3sample/ct_20/test/dps_langevin
```

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

After a training array submitted by
`scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_training_only.sh` has
finished, submit the matching reconstruction array from the LION root:

```bash
PADIS_TRAIN_ROOT=/path/to/a100_training_<stamp> \
PADIS_RECON_IMPLEMENTATIONS=paper,public_repo \
PADIS_RECON_GEOMETRIES=lion \
PADIS_RECON_MAX_SAMPLES=25 \
scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_reconstruction.sh
```

The reconstruction scripts default to `--split test --max-samples 25`, matching
the paper's CT reconstruction evaluation budget. Lower
`PADIS_RECON_MAX_SAMPLES` only for pilot/debug runs.

Useful selectors:

```bash
PADIS_RECON_MODELS=patch_lidc_default,whole_lidc_default
PADIS_RECON_EXPERIMENTS=ct_20,ct_8
PADIS_RECON_IMPLEMENTATIONS=paper
PADIS_RECON_GEOMETRIES=lion
```

The matrix can also be inspected locally without submitting Slurm:

```bash
python scripts/paper_scripts/PaDIS/PaDIS_run_reconstruction_matrix.py \
  --training-root /path/to/a100_training_<stamp> \
  --output-root /tmp/padis_recon_matrix_preview \
  --implementations paper,public_repo \
  --geometries lion \
  --count
```

The Slurm reconstruction array defaults to `LION_MAMBA_ENV=lion-dev` and
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
- The target images retain high-frequency CT texture/noise that both public
  PaDIS and LION PaDIS smooth. Matching the public repo and exactly preserving
  all target texture are not simultaneously achievable with this sampler setup.
