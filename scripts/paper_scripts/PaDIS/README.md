# PaDIS LIDC experiments

This directory contains the training, tuning, reconstruction, generation, and
verification pipeline used for the PaDIS experiments on LIDC-IDRI. The scripts
share one experiment definition and one repository-local hyperparameter
registry, so the GCP, Colab, direct Python, and Slurm paths construct equivalent
reconstruction jobs.

The former development diary is retained in [`LOG_README.md`](LOG_README.md).
It records pilots and historical decisions, but it is not the source of truth
for current defaults. Current behaviour is defined by the scripts and
[`config/reconstruction_hparam_defaults.json`](config/reconstruction_hparam_defaults.json).

## Workflow

The complete pipeline is:

1. Load processed LIDC-IDRI slices and, for training, stage matching compressed
   tensor caches into a RAM disk.
2. Train the required patch, whole-image, and PnP denoisers. Training is
   resumable from full-state checkpoints.
3. Continue each diffusion run through the validation-intensive phase and save
   the `min_intense_val` exponential-moving-average checkpoint. PnP models save
   their `min_val` checkpoint.
4. Resolve validation-selected reconstruction parameters from the repository
   JSON registry.
5. Build a deterministic reconstruction manifest, check every required
   checkpoint, and execute unfinished jobs.
6. Run unconditional generation where requested.
7. Verify saved metrics and produce JSON/CSV result tables with bootstrap
   uncertainty.

The GCP spot runner performs steps 1-5 automatically. The Colab/manual runner
checks for missing trainable inputs, runs the reconstruction matrix, and runs
generation before the expensive fixed-overlap and 512-resolution tail jobs.

## Main entry points

| File | Purpose |
|---|---|
| `PaDIS_LIDC_256.py` | Train 256-resolution patch or whole-image diffusion priors. |
| `PaDIS_LIDC_512.py` | Train the native-resolution patch prior. |
| `PaDIS_LIDC_PnP_denoiser.py` | Train unconditioned or noise-conditioned DRUNet models. |
| `PaDIS_LIDC_reconstruction.py` | Run one reconstruction configuration. |
| `PaDIS_run_reconstruction_matrix.py` | Construct, list, validate, or run the experiment matrix. |
| `PaDIS_LIDC_generation.py` | Generate unconditional whole-image or patch-assembled samples. |
| `PaDIS_tune_reconstruction_hyperparameters.py` | Run reconstruction tuning candidates. |
| `PaDIS_summarize_hparam_tuning.py` | Summarise completed tuning records. |
| `PaDIS_hparam_defaults.py` | Export tuning results to the repository JSON registry. |
| `PaDIS_reconcile_reconstruction_manifest.py` | Reconcile a rebuilt manifest with existing outputs. |
| `PaDIS_verify_reconstruction_matrix.py` | Verify outputs and write result/uncertainty tables. |
| `PaDIS_make_paper_figures.py` | Build figures from saved outputs. |
| `PaDIS_make_tables.py` | Build publication LaTeX tables and decoded table CSVs from the verification CSV. |
| `gcp/run_PaDIS_GCP_spot_training.sh` | Automatic GCP training, intensive validation, and inference. |
| `gcp/run_PaDIS_GCP_manual_reconstruction.sh` | Resumable GCP/Colab inference and generation. |
| `gcp/PaDIS_Colab_manual_reconstruction.ipynb` | Colab setup, authentication, environment installation, and launch cells. |
| `slurm/submit_PaDIS_A100_pipeline.sh` | Submit the equivalent training and reconstruction pipeline on Slurm. |

Run scripts from the LION repository root. Use each program's `--help` output
for diagnostic options not documented here.

## Data and persistent paths

Set `LION_DATA_PATH` to the data root. On the GCP/Colab path it defaults to
`/mnt/data/Datasets`, where `/mnt/data` is the mounted `padis-bucket`.

| Data | Default path below `LION_DATA_PATH` |
|---|---|
| Processed slices | `processed/LIDC-IDRI/LIDC-IDRI-*/` |
| 256 cache archives | `processed/LIDC-IDRI-cache/padis_256/archives/*.pt.zst` |
| 512 cache archives | `processed/LIDC-IDRI-cache/padis_512/archives/*.pt.zst` |
| PaDIS experiment root | `experiments/PaDIS/` |
| GCP training root | `experiments/PaDIS/final_real_runs/PaDIS-Reproduction-GCP/` |
| GCP reconstruction root | `experiments/PaDIS/final_real_runs/PaDIS-Reproduction-GCP_reconstruction/` |

The dataset split is deterministic by sorted patient identifier: 80% train,
10% validation, and 10% test. The default regime uses at most four slices per
patient; `--full-lidc` selects every processed slice. Cache identity includes
the split, resolution, and selection regime, preventing a cache for one setup
from being reused silently for another.

Training scripts can load `.pt.zst` archives directly into a RAM-backed cache.
Decompression uses `zstd -T0`, which uses all available CPU cores. The archive
and model outputs remain on persistent storage; the expanded RAM-disk files are
disposable. `run_prepare_lidc_cache.sh` prepares archives explicitly, while the
GCP runners stage the required `256-default`, `256-full`, and `512-default`
variants automatically.

## Training models and checkpoints

The GCP runner's default task order is:

| Task | Model/data change |
|---|---|
| `whole_lidc_full` | Whole-image prior, all processed training slices. |
| `whole_lidc_default` | Whole-image prior, four-slice regime. |
| `pnp_lidc_drunet` | Unconditioned DRUNet. |
| `pnp_lidc_drunet_noise_cond` | DRUNet with a noise-level input channel. |
| `patch_lidc_p96_default` | Patch prior with maximum patch size 96. |
| `patch_lidc_full` | Default patch prior, all processed training slices. |
| `patch_lidc_512` | Native-resolution patch prior. |
| `patch_lidc_default` | Main 256-resolution patch prior. |
| `patch_lidc_p32_default` | Maximum patch size 32. |
| `patch_lidc_p16_default` | Maximum patch size 16. |
| `patch_lidc_p8_default` | Fixed patch size 8. |
| `patch_lidc_no_pos_default` | Main patch prior without x/y coordinate channels. |

### Diffusion training defaults

Patch and whole-image priors use an EDM-preconditioned NCSN++ denoiser. Both
have four resolution levels, four residual blocks per level, 128 base channels,
dropout 0.05, and attention at resolution 16. Patch channel multipliers are
`[1, 2, 2, 2]`; whole-image multipliers are `[1, 2, 2, 4]`. Default models
receive the image plus normalised x/y coordinates. The no-position model
receives only the image channel.

Adam uses learning rate `2e-4`. Training noise is log-normal,
`log(sigma) ~ N(-1.2, 1.2^2)`, truncated to `[0.002, 40]`, with
`sigma_data=0.5`. Whole-image batches contain 8 slices. Patch effective batch
sizes are 128, 256, and 512 from largest to smallest patch. An EMA with a
500,000-example half-life supplies validation and inference weights.

| Training preset | Patch sizes | Probabilities | Padding |
|---|---|---|---:|
| `P=8` | 8 | 1.0 | 8 |
| `P=16` | 8, 16 | 0.3, 0.7 | 16 |
| `P=32` | 8, 16, 32 | 0.2, 0.3, 0.5 | 32 |
| Default `P=56` | 16, 32, 56 | 0.2, 0.3, 0.5 | 24 |
| `P=96` | 32, 64, 96 | 0.2, 0.3, 0.5 | 32 |
| Native 512 | 16, 32, 64 | 0.2, 0.3, 0.5 | 64 |

The GCP base phase budgets six hours for each patch task and eighteen hours for
each whole-image task. The following six-hour validation-intensive phase uses
4,000 validation patches every 20,000 seen patches, or all 328 validation
images every 2,500 seen images. The best EMA checkpoint in that phase is named
`<prefix>_min_intense_val.pt` and is the diffusion checkpoint used by the
reconstruction matrix.

### PnP defaults

Both DRUNets use 64 internal channels, four blocks per level, batch size 8,
Adam at `1e-4`, and 100 epochs. Gaussian training noise is sampled uniformly
from `[0, 0.05]`. The best validation files are:

```text
pnp_lidc_drunet/pnp_lidc_drunet_min_val.pt
pnp_lidc_drunet_noise_cond/pnp_lidc_drunet_noise_cond_min_val.pt
```

The conditioned reconstruction row supplies noise level `0.03`.

### Resuming and W&B

Periodic full-state files contain optimiser, counters, and EMA state. Training
scripts discover compatible periodic checkpoints automatically. The GCP runner
keeps per-task phase markers under:

```text
<training-root>/.gcp_spot/{done,running,failed,logs,runtime}/
```

Rerunning the same command skips valid completed phases and resumes incomplete
ones. W&B logging defaults to online mode and final artefacts are uploaded
unless `PADIS_NO_WANDB_ARTIFACT=1`; persistent checkpoints do not depend on W&B.

## Reconstruction methods

The matrix contains:

| Method | Summary |
|---|---|
| `baseline` | Hann-filtered LION FDK baseline. |
| `admm_tv` | TV reconstruction using LION's Chambolle-Pock solver. The historical matrix name is retained, but the solver is not ADMM. |
| `pnp_admm` | ADMM with conjugate-gradient data updates and DRUNet denoising. |
| `whole_image_diffusion` | Whole-image VE-DPS prior. |
| `langevin` | Data-consistency updates with patch or whole-image Langevin sampling. |
| `predictor_corrector` | VE predictor and Langevin corrector sampling. |
| `ve_ddnm` | VE-DDNM approximate-null-space correction. |
| `patch_average` | Fixed overlapping patches averaged in overlap regions. |
| `patch_stitch` | Fixed overlapping patches written in order, with later patches overwriting overlaps. |
| `padis_dps` | Randomly shifted PaDIS patch partitions inside VE-DPS. |

`admm_tv` is a historical internal matrix identifier retained to preserve
checkpoint, output-directory, manifest, and resume compatibility. The executed
solver is Chambolle--Pock, not ADMM. Presentation code therefore labels this
method **CP** in figures and reports it as sampler **CP** with prior **TV** in
tables. The inference CLI, saved method field, and existing output paths remain
`admm_tv`; consumers should translate that identifier only for display.

Three implementation tracks are used while keeping the LIDC images and LION
fan-beam geometry fixed:

| Aspect | `paper` | `public_repo` | `lion_physics` |
|---|---|---|---|
| Intent | Follow paper equations and sampler structure. | Follow released helper behaviour. | Use LION-native least-squares physics. |
| Data update | Paper residual-based rules. | Norm-gradient DPS or public direct-adjoint rules. | Least-squares gradient divided by the composed operator Lipschitz constant. |
| Schedule | Geometric. | Geometric in the final matrix; literal EDM schedule remains an ablation. | Geometric. |
| Initialisation | Gaussian for diffusion samplers. | FDK for DPS/fixed-overlap DPS; Gaussian for helper samplers. | Validation-selected by method. |
| Predictor-corrector | Corrector uses the next sigma and a fresh layout. | Corrector uses the current sigma and reuses the predictor layout. | Paper layout with LION-normalised data steps. |
| VE-DDNM | Paper `1000 x 1` update layout. | Public `100 x 10` layout. | Paper layout with LION FDK pseudoinverse. |

Public compatibility is formula-level because processed LIDC slices do not
contain the physical metadata needed to map them rigorously into the public
repository's unitless ODL coordinate system. Public rows therefore use the same
LION geometry as every other row and retain calibrated public-matching gradient
scales (`0.0405` for DPS and `0.1022` for direct-adjoint updates).

The `lion_physics` track removes those fixed scales. For a composed measurement
map `F`, it normalises diffusion data updates by `||F||_2^2`, estimated by power
iteration. This is applied only to diffusion data consistency, not to FDK, TV,
or PnP-ADMM. All final VE-DDNM rows clip FDK pseudoinverse terms because the
unbounded sparse-view fan-beam approximation otherwise becomes non-finite.

## Reconstruction experiments

| Experiment | Views | Angular span | Grid |
|---|---:|---:|---:|
| `ct_8` | 8 | 360 degrees | 256 x 256 |
| `ct_20` | 20 | 360 degrees | 256 x 256 |
| `ct_60` | 60 | 360 degrees | 256 x 256 |
| `ct_fanbeam_180` | 20 | 120 degrees | 256 x 256 |
| `ct_512_60` | 60 | 360 degrees | 512 x 512 |

`ct_fanbeam_180` is a legacy identifier. It is the 20-view, 120-degree
limited-angle experiment, not a 180-view acquisition.

With `method_default`, `paper_matrix`, all methods, and all trained ablations,
the current manifest contains 109 reconstruction jobs: 65 core rows, 42
ablation rows, and 2 noise-conditioned PnP rows. Standard jobs use test indices
0-24. `patch_average`, `patch_stitch`, and all `ct_512_60` jobs are capped at
indices 0-3 by default.

The `gcp_spot` order is:

1. Regular reconstruction rows.
2. Additional high-view VE-DDNM rows.
3. Noise-conditioned PnP rows.
4. Patch averaging, then patch stitching.
5. Native-resolution 512 rows.

The manual runner inserts unconditional generation after regular rows and
before the fixed-overlap/512 tail. Generation uses four samples, seed 33, 300
geometric levels from `sigma=10` to `0.002`, one inner step per level, and
epsilon 1. It compares whole-image, naive patch, PaDIS, fixed-average, and
fixed-stitch assembly.

## Tuned reconstruction defaults

The authoritative registry is
[`config/reconstruction_hparam_defaults.json`](config/reconstruction_hparam_defaults.json).
The matrix resolves an exact experiment/model record first, then any configured
high-view fallback, then a consensus record, and finally the implementation
default. Untuned 60-view and limited-angle rows therefore use an available
20-view record or the 8/20-view consensus. Native 512 PaDIS-DPS has dedicated
records.

`zeta` is the data-consistency coefficient, `epsilon` is the Langevin/DPS
coefficient multiplying the sigma-dependent step, and `r` is predictor-
corrector SNR. Values across columns are not directly comparable because each
track defines the data update differently.

| Method/prior | Paper-Style | Public-Compatible | LION-Physics |
|---|---|---|---|
| Whole-image DPS | `zeta=0.01`, `epsilon=0.5` | Not implemented | `zeta=4.0`, `epsilon=0.5` |
| PaDIS-DPS | `zeta=0.0075`, `epsilon=0.5`, noise init | `zeta=0.2`, `epsilon=0.5`, FDK init | `zeta=4.25`, `epsilon=0.5`, noise init |
| Patch Langevin | `zeta=0.03`, `epsilon=0.5` | `zeta=0.2`, `epsilon=0.5` | `zeta=4.0`, `epsilon=0.5` |
| Whole-image Langevin | Not scheduled | Not implemented | `zeta=3.5`, `epsilon=0.5` |
| Patch predictor-corrector | `zeta=0.02`, `r=0.04` | `zeta=0.5`, `r=0.16` | `zeta=4.25`, `r=0.01` |
| Whole-image predictor-corrector | Not scheduled | Not implemented | `zeta=4.25`, `r=0.02` |
| Patch VE-DDNM | `epsilon=0.1`, corrected-state clip | `epsilon=0.2` | `epsilon=0.1`, noise scale `0.5` |
| Whole-image VE-DDNM | Not scheduled | Not implemented | `epsilon=0.2` |
| Patch averaging | Not implemented | `zeta=0.3`, `epsilon=0.5` | `zeta=4.0`, `epsilon=0.5` |
| Patch stitching | Not implemented | `zeta=0.3`, `epsilon=0.5` | `zeta=3.0`, `epsilon=0.5` |
| TV | Not implemented | Not implemented | `lambda=0.001`, 1000 iterations |
| PnP-ADMM | Not implemented | Not implemented | `eta=3e-5`, 60 outer iterations, 100 CG iterations, tolerance `1e-7` |

Native 512 PaDIS-DPS uses `zeta=2.0`, `epsilon=0.5`, and Gaussian
initialisation for LION-Physics; Public-Compatible uses `zeta=1.2`,
`epsilon=0.5`, and FDK initialisation. Both process one patch at a time and use
activation checkpointing. FDK initialisation, where selected, uses a Hann
filter with frequency scaling 0.3 and clips the initial image to `[0,1]`.

### W&B pulled checkpoint tuning

On 2026-07-09 the final GCP W&B checkpoints were pulled with authenticated W&B
API access into:

```text
/home/thomas/DiS/Project/Data/experiments/PaDIS/wandb_checkpoints/PaDIS-Reproduction
```

The matrix-compatible staged root is:

```text
/home/thomas/DiS/Project/Data/experiments/PaDIS/final_real_runs/PaDIS-Reproduction-GCP-wandb-pulled
```

The full `patch_lidc_full:v1` artifact download first failed with a transient
`IncompleteRead`, so the final pull downloaded exact checkpoint entries from
the artifacts. All required checkpoint files were pulled successfully.

| Model row | W&B artifact | Checkpoint entry |
|---|---|---|
| Noise-conditioned PnP | `pnp_lidc_drunet_noise_cond:v0` | `pnp_lidc_drunet_noise_cond_min_val.pt` |
| Full-data PaDIS-DPS | `patch_lidc_full:v1` | `padis_lidc_256_min_intense_val.pt` |
| Full-data whole image | `whole_lidc_full:v1` | `whole_image_lidc_256_min_intense_val.pt` |
| Default-data PaDIS-DPS | `patch_lidc_default:v1` | `padis_lidc_256_min_intense_val.pt` |
| Default-data whole image | `whole_lidc_default:v1` | `whole_image_lidc_256_min_intense_val.pt` |

Completed local validation on the pulled checkpoints selected full-data
settings only. Default-data W&B checks are recorded as diagnostics; the
main/default-data reconstruction defaults remain the pre-existing defaults in
`reconstruction_hparam_defaults.json`.

| Row | Validation experiments | Selected setting | Mean PSNR | Mean SSIM | Mean MAE | Notes |
|---|---|---|---:|---:|---:|---|
| Noise-conditioned PnP | `ct_20`, `ct_8` | `eta=3e-5`, 60 iterations, noise level `0.03` | 24.62 | 0.630 | 0.03514 | Noise level was effectively flat from `0.01` to `0.05`; 100 iterations worsened both views. |
| Full-data whole image | `ct_20` | `zeta=4.0`, `epsilon=0.5` | 34.38 | 0.859 | 0.01279 | `zeta=5.0` diverged to non-finite output. |
| Default-data whole image | `ct_20`, `ct_8` | Diagnostic only; not promoted over existing defaults | 32.08 | 0.825 | 0.01529 | Tested `zeta=4.0`, `epsilon=0.5`; default-data config remains `current_defaults`. |
| Full-data PaDIS-DPS | `ct_20` | `zeta=4.5`, `epsilon=0.5`, noise init, no clipping, patch batch 8 | 28.25 | 0.578 | 0.02748 | `zeta=2.0` underfit; `4.75` exploded; `5.0` produced NaNs. |
| Default-data PaDIS-DPS | `ct_20`, `ct_8` | Diagnostic only; not promoted over existing defaults | 28.45 | 0.627 | 0.02555 | Tested `zeta=4.25`, `epsilon=0.5`, noise init, no clipping, patch batch 8; default-data config keeps the earlier consensus setting without the patch-batch runtime override. |

The PaDIS-DPS mechanics remain the same as the earlier LION-Physics patch-prior
setting: paper geometric sigma schedule, Gaussian/noisy initialization,
unclipped initial/output state, `epsilon=0.5`, LION-native CT operations, and
operator-Lipschitz data normalization. The W&B-pulled full-data patch
checkpoint changed the denoiser weights and the selected scalar `zeta`.
`patch_batch_size=8` is used only for the promoted full-data PaDIS-DPS row; it
is a runtime throughput setting and should not change the reconstruction values.

### Exporting defaults from a sweep

Tuning writes one `runs.jsonl` per sweep directory. Export consensus settings
from completed finite `ct_20` and `ct_8` records with:

```bash
python -u scripts/paper_scripts/PaDIS/PaDIS_hparam_defaults.py \
  --run-root "$PADIS_RUN_ROOT/hparam_tuning/runs" \
  --run-glob 'fixedval_*' \
  --selection-scope consensus \
  --expected-experiments ct_20,ct_8 \
  --require-records \
  --output /tmp/reconstruction_hparam_defaults.json
```

The exporter ranks finite completed candidates by mean PSNR, treats differences
below 0.1 dB as ties, then uses SSIM within 0.01 and finally MAE. The promoted
512 PaDIS-DPS defaults currently come from standalone validation runs under
`debug_runs`, not the `fixedval_*` sweep logs. They intentionally supersede the
older sweep-derived 512 candidate in the committed registry. Compare and merge
the staged JSON rather than overwriting the registry until those standalone
runs have been ingested into `runs.jsonl`.

## Running the pipeline

### GCP spot runner

After setting the data and environment paths, run:

```bash
bash scripts/paper_scripts/PaDIS/gcp/run_PaDIS_GCP_spot_training.sh
```

By default this stages caches, runs all missing base training, runs the
validation-intensive continuation, checks inputs, reconciles the matrix, and
runs inference. Reconstruction uses the JSON defaults and `min_intense_val`
diffusion checkpoints. Rerun the same command after pre-emption.

### Colab or manual GCP runner

The supported Colab entry point is:

```text
scripts/paper_scripts/PaDIS/gcp/PaDIS_Colab_manual_reconstruction.ipynb
```

It clones the repository to local disk, installs Miniforge and the LION
environment, authenticates GCP/W&B as required, mounts `padis-bucket` at
`/mnt/data`, and launches:

```bash
bash scripts/paper_scripts/PaDIS/gcp/run_PaDIS_GCP_manual_reconstruction.sh
```

The manual runner defaults to two concurrent reconstruction workers per GPU.
Set `PADIS_RECON_TASKS_PER_GPU` to override this. It checks all model inputs and
trains missing supported models before inference. Training archives are
decompressed into `/mnt/ram-disk`; the runner removes that temporary mount after
training and syncs persistent outputs after each job.

### Slurm

Submit the complete dependency chain with:

```bash
bash scripts/paper_scripts/PaDIS/slurm/submit_PaDIS_A100_pipeline.sh
```

Training-only and reconstruction-only wrappers are available in `slurm/`.
They use the same matrix builder, checkpoint policy, JSON defaults, and job
order as GCP. Cluster account, array width, time limits, and roots are
configured through the `PADIS_*` environment variables in those wrappers.

### Inspecting the matrix directly

List the resolved jobs without running them:

```bash
python -u scripts/paper_scripts/PaDIS/PaDIS_run_reconstruction_matrix.py \
  --training-root "$PADIS_TRAIN_ROOT" \
  --output-root "$PADIS_RECON_ROOT" \
  --checkpoint-policy min_intense_val \
  --hparam-defaults json \
  --models method_default \
  --methods all \
  --experiments paper_matrix \
  --ablations all \
  --implementations method_default \
  --geometries lion \
  --job-order gcp_spot \
  --pnp-cg-iterations 100 \
  --pnp-cg-tolerance 1e-7 \
  --list
```

Replace `--list` with `--count`, `--check-inputs`, or `--task-index N` to count,
validate, or execute one resolved job. The runners generate this list once,
store it as `reconstruction_matrix_jobs.json`, and reconcile rebuilt manifests
against valid existing outputs.

## Outputs, restart state, and verification

Each reconstruction result directory contains at least `metrics.json` and
`reconstructions.pt`; optional preview images and sampler traces are stored
alongside them. `metrics.json` records the resolved method, implementation,
experiment, checkpoint, sampler settings, per-image metrics, and summary.

Manual reconstruction state is stored under:

```text
<reconstruction-root>/.manual_gcp_reconstruction/
  done/
  running/
  failed/
  logs/
  runtime/
```

A `.done` marker is accepted only when the corresponding output validates.
Interrupted and failed jobs are therefore eligible for rerun, while valid
completed jobs survive task reordering or manifest rebuilding. Existing sample
files are also reusable when a resumed configuration requests fewer samples.

Generate the final verification tables with:

```bash
python -u scripts/paper_scripts/PaDIS/PaDIS_verify_reconstruction_matrix.py \
  --root "$PADIS_RECON_ROOT" \
  --expected-jobs-json "$PADIS_RECON_ROOT/reconstruction_matrix_jobs.json" \
  --output-json "$PADIS_RECON_ROOT/reconstruction_matrix_verification.json" \
  --output-csv "$PADIS_RECON_ROOT/reconstruction_matrix_verification.csv"
```

The default verifier performs 2,000 deterministic image-level bootstrap
resamples at 95% confidence using seed 33. The CSV includes the aggregate
metrics, bootstrap standard errors and confidence limits. Use the expected-jobs
manifest to detect missing outputs rather than relying only on marker counts.

Generate all publication tables using the standard experiment paths with:

```bash
python -u scripts/paper_scripts/PaDIS/PaDIS_make_tables.py
```

By default this reads
`$LION_EXPERIMENTS_PATH/PaDIS/final_real_runs/PaDIS-Reproduction-GCP_reconstruction/reconstruction_matrix_verification.csv`
and writes `reconstruction_tables.tex` plus seven decoded CSVs under
`$LION_EXPERIMENTS_PATH/PaDIS/paper_tables`. Use `--csv-path`, `--tex-path`,
and `--csv-output-dir` to override these locations.

The timing table is calculated from completed reconstruction progress logs; no
timings are stored in the table generator. `--timing-mode gcp` (the default)
and `--timing-mode colab` read the manual-runner log naming convention from the
standard GCP reconstruction root. For a Slurm reconstruction array, use:

```bash
python -u scripts/paper_scripts/PaDIS/PaDIS_make_tables.py \
  --timing-mode slurm \
  --timing-log-root /path/to/slurm/output/files \
  --timing-jobs-json /path/to/reconstruction_matrix_jobs.json
```

The parser uses the final completed `LIDC test run` progress record, whose
seconds-per-iteration value is already the mean wall time per reconstructed
slice. Timings are grouped by implementation and reconstruction method; where
multiple prior rows exist for a combination, their per-slice timings are
averaged.

Paper figures label the normalised-intensity colour scale as **NI**. CT panels
use the tightest centre-symmetric crop that contains the target foreground;
the default adds no border padding. This keeps anatomy centred consistently
between methods without retaining avoidable outer background. Pass
`--body-bbox-padding N` only when an explicit pixel border is required, or
`--no-body-crop` to retain the complete reconstruction field of view. PNG and
PDF exports additionally use a zero-padding page bounding box that is tight
vertically and has equal minimal margins outside the leftmost and rightmost
image edges. The **NI** label is placed close to its corresponding intensity
scale while remaining legible. All vertical scale labels, including **HU** and
**NI**, use one fixed horizontal coordinate so they align across figure rows.
