# PaDIS LIDC experiments

This directory contains the training, tuning, reconstruction, generation, and
verification pipeline used for the PaDIS experiments on LIDC-IDRI. The scripts
share one experiment definition and one repository-local hyperparameter
registry, so the GCP, Colab, direct Python, and Slurm paths construct equivalent
reconstruction jobs.

The validation-set search protocol, run sequence, selection rule and results
are documented in [`tuning/TUNING.md`](tuning/TUNING.md). Current runtime
behaviour is defined by the scripts, while
[`config/reconstruction_hparam_defaults.json`](config/reconstruction_hparam_defaults.json)
is the authoritative machine-readable reconstruction registry.

## Contents

- [Repository map](#repository-map)
- [Reproduction procedure](#reproduction-procedure)
  - [Install LION](#1-install-lion)
  - [Download and process LIDC-IDRI](#2-download-and-process-lidc-idri)
  - [Build training caches](#3-build-reusable-training-caches)
  - [Run the complete pipeline](#4-run-training-through-final-reporting)
  - [Check the outputs](#5-check-the-final-products)
- [Backend requirements](#backend-requirements)
- [Data layout and storage](#data-layout-and-storage)
- [Training and checkpoints](#training-models-and-checkpoints)
- [Methods and experiments](#reconstruction-methods)
- [Tuned defaults](#tuned-reconstruction-defaults)
- [Manual reconstruction and matrix inspection](#manual-reconstruction-and-matrix-inspection)
- [Expected reconstruction runtimes](#expected-reconstruction-runtimes)
- [Outputs, restart state, and reporting](#outputs-restart-state-and-reporting)

## Repository map

| File | Purpose |
|---|---|
| `LION/data_loaders/LIDC_IDRI/download_LIDC_IDRI.sh` | Download and resume the raw TCIA LIDC-IDRI collection. |
| `LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py` | Convert raw DICOM scans into LION's processed slice dataset. |
| `pipeline/run_prepare_lidc_cache.sh` | Build the default/full 256 and default 512 compressed training caches. |
| `pipeline/PaDIS_run_pipeline.sh` | Unified complete training, inference, generation, table, and figure entry point for GCP or Slurm. |
| `pipeline/PaDIS_finalise_pipeline.sh` | Re-run generation, verification, timing, table, and figure production. |
| `training/PaDIS_LIDC_256.py` | Train 256-resolution patch or whole-image diffusion priors. |
| `training/PaDIS_LIDC_512.py` | Train the native-resolution patch prior. |
| `training/PaDIS_LIDC_PnP_denoiser.py` | Train unconditioned or noise-conditioned DRUNet models. |
| `reconstruction/PaDIS_LIDC_reconstruction.py` | Run one reconstruction configuration. |
| `reconstruction/PaDIS_run_reconstruction_matrix.py` | Construct, list, validate, or run the experiment matrix. |
| `reconstruction/PaDIS_LIDC_generation.py` | Generate unconditional whole-image or patch-assembled samples. |
| `tuning/PaDIS_tune_reconstruction_hyperparameters.py` | Run reconstruction tuning candidates. |
| `tuning/PaDIS_run_reproduction_tuning.py` | Re-run the complete validation tuning grid on the final inference checkpoints. |
| `tuning/PaDIS_summarize_hparam_tuning.py` | Summarise completed tuning records. |
| `tuning/PaDIS_hparam_defaults.py` | Export tuning results to the repository JSON registry. |
| `tuning/TUNING.md` | Reproduce and audit the validation-set hyperparameter search. |
| `reconstruction/PaDIS_reconcile_reconstruction_manifest.py` | Reconcile a rebuilt manifest with existing outputs. |
| `reconstruction/PaDIS_verify_reconstruction_matrix.py` | Verify outputs and write result/uncertainty tables. |
| `reporting/PaDIS_make_paper_figures.py` | Build figures from saved outputs. |
| `reporting/PaDIS_make_tables.py` | Build publication LaTeX tables and decoded table CSVs from the verification CSV. |
| `platforms/gcp/run_PaDIS_GCP_spot_training.sh` | Automatic GCP training, intensive validation, and inference. |
| `platforms/gcp/run_PaDIS_GCP_manual_reconstruction.sh` | Resumable GCP/Colab inference and generation. |
| `platforms/gcp/PaDIS_Colab_manual_reconstruction.ipynb` | Colab setup, authentication, environment installation, and launch cells. |
| `platforms/slurm/submit_PaDIS_A100_pipeline.sh` | Submit the equivalent training and reconstruction pipeline on Slurm. |

The directory is organised by responsibility: `core/` contains shared preset
definitions, `pipeline/` contains end-to-end orchestration, `platforms/`
contains GCP and Slurm launchers, and `training/`, `reconstruction/`,
`tuning/`, and `reporting/` contain the corresponding experiment stages.
`config/` contains the checked-in inference registry. Experiment outputs
remain under `$LION_EXPERIMENTS_PATH/PaDIS`; the source layout does not rename
or migrate saved data.

Run scripts from the LION repository root. Use each program's `--help` output
for diagnostic options not documented here.

## Reproduction procedure

This is the shortest complete reproduction path. The later sections document
the models, matrix, restart behaviour, and platform controls in more detail.

### 1. Install LION

Clone the repository at the revision being reproduced and initialise its
submodules. The commands below follow LION's root installation instructions but
name the environment `lion-dev`, which is the name used by the PaDIS launchers:

```bash
git clone https://github.com/CambridgeCIA/LION.git
cd LION
git submodule update --init --recursive
conda env create --file env_base.yml --name lion-dev
conda activate lion-dev
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

Match the PyTorch wheel and `cuda-version` in `env_base.yml` to the installed
CUDA driver when CUDA 12.8 is unavailable. Record `git rev-parse HEAD` with the
outputs: code revision, dataset split, checkpoint policy, and registry contents
all affect reproducibility.

Set persistent paths before importing LION:

```bash
export LION_DATA_PATH=/path/to/Datasets
export LION_EXPERIMENTS_PATH="$LION_DATA_PATH/experiments"
mkdir -p "$LION_DATA_PATH" "$LION_EXPERIMENTS_PATH"
```

### 2. Download and process LIDC-IDRI

LION downloads the TCIA LIDC-IDRI collection using the repository's TCIA
manifest and the NBIA Data Retriever CLI. The helper installs the retriever
inside the data directory, downloads the diagnosis spreadsheet, verifies MD5
checksums, and resumes interrupted downloads by requesting missing series:

```bash
bash LION/data_loaders/LIDC_IDRI/download_LIDC_IDRI.sh
```

Raw DICOM data will be under `$LION_DATA_PATH/raw/LIDC-IDRI`. NBIA license
acceptance, Java/container alternatives, resume and redownload controls, and
recovery from incomplete series are documented in
[`LION/data_loaders/LIDC_IDRI/README.md`](../../../LION/data_loaders/LIDC_IDRI/README.md).

Preprocessing has a separate environment because `pylidc` has specialised
dependencies:

```bash
conda env create \
  --file LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri_environment.yml
conda run -n lidc_idri \
  python LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py
```

This writes 512-by-512 slice arrays, masks, and metadata under
`$LION_DATA_PATH/processed/LIDC-IDRI/LIDC-IDRI-*`. The preprocessor finishes
with a completeness check; a complete current download has at least 1,010
processed patient directories and 282,776 regular files. Do not start cache
creation or training if that check reports missing patients, scan errors, or a
short file count.

### 3. Build reusable training caches

Install `zstd`, activate `lion-dev`, and create the three cache archives used by
the training pipelines:

```bash
conda activate lion-dev
bash scripts/paper_scripts/PaDIS-Reproduction/pipeline/run_prepare_lidc_cache.sh
```

The default variants are `256-default`, `256-full`, and `512-default`. They are
written below:

```text
$LION_DATA_PATH/processed/LIDC-IDRI-cache/padis_256/archives/
$LION_DATA_PATH/processed/LIDC-IDRI-cache/padis_512/archives/
```

Cache construction is deterministic for the processed data and seed. It can be
rerun safely; set `PADIS_REBUILD_CACHE=1` only when the existing cache should be
replaced. On Slurm, the complete pipeline submits its equivalent cache job, so
this local command is optional if the compute nodes can read the processed
dataset and write the cache root.

### 4. Run training through final reporting

Check the backend dispatch first:

```bash
bash scripts/paper_scripts/PaDIS-Reproduction/pipeline/PaDIS_run_pipeline.sh \
  --backend gcp --dry-run
bash scripts/paper_scripts/PaDIS-Reproduction/pipeline/PaDIS_run_pipeline.sh \
  --backend slurm --dry-run
```

Then run one backend:

```bash
# Synchronous and resumable on the configured GCP machine:
bash scripts/paper_scripts/PaDIS-Reproduction/pipeline/PaDIS_run_pipeline.sh \
  --backend gcp

# Or submit the dependency-linked Slurm pipeline:
PADIS_RUN_STAMP=padis-reproduction \
  bash scripts/paper_scripts/PaDIS-Reproduction/pipeline/PaDIS_run_pipeline.sh \
  --backend slurm
```

For a short local integration check using existing staged checkpoints, run:

```bash
export PADIS_FAST_SMOKE_TRAINING_ROOT=/path/to/matrix-compatible/training-root
bash scripts/paper_scripts/PaDIS-Reproduction/pipeline/PaDIS_run_pipeline.sh \
  --fast-smoke
```

`--fast-smoke` executes one sample, one experiment per selected model, one
outer and one inner diffusion step, and one TV/PnP/CG iteration. It covers 12
representative tuning families, a compact reconstruction matrix, patch and
whole-image generation, verification, partial tables, and partial figures.
Outputs go to a new `debug_runs/fast_smoke_*` folder by default. It checks the
SHA256 of the committed hyperparameter registry before and after and fails if
the settings change. The tuning launcher's broader `--smoke` mode retains all
candidates while applying the same execution-depth limits.

After the processed dataset exists, these entry points cover cache staging,
diffusion and PnP training, intensive validation, checkpoint selection, the
109-job reconstruction matrix, unconditional generation, verification,
timings, LaTeX/CSV tables, and PNG/PDF figures. They are resumable; retain the
same GCP run name or Slurm run stamp when continuing an interrupted run.

Exact reproduction uses the committed
[`config/reconstruction_hparam_defaults.json`](config/reconstruction_hparam_defaults.json).
Re-running the validation search is an optional audit of how those settings
were selected, not a prerequisite for consuming the registry. After training,
the complete search can be repeated as described in
[`tuning/TUNING.md`](tuning/TUNING.md).

### 5. Check the final products

The default finaliser writes:

```text
$LION_EXPERIMENTS_PATH/PaDIS/reconstruction_presets/   unconditional samples
$LION_EXPERIMENTS_PATH/PaDIS/paper_tables/             LaTeX and decoded CSV tables
$LION_EXPERIMENTS_PATH/PaDIS/paper_figures/            PNG/PDF figures and manifest
<reconstruction-root>/reconstruction_matrix_verification.{json,csv}
```

Treat a run as complete only when verification succeeds against
`reconstruction_matrix_jobs.json`, rather than when scheduler or marker counts
alone look complete. The figure manifest records missing panels and whether
each generated figure is publication-ready.

## Backend requirements

Before a real launch, the following must be true:

- `LION_DATA_PATH` identifies the persistent dataset root and contains the
  processed LIDC-IDRI patient folders plus the `256-default`, `256-full`, and
  `512-default` cache archives listed below.
- The checkout contains the committed
  `config/reconstruction_hparam_defaults.json` used to build the 109-job
  reconstruction manifest.
- The `lion-dev` Conda/Mamba environment has been created from LION's current
  environment specification and includes PyTorch, ASTRA, and tomosipo.
- A CUDA GPU is visible to PyTorch and `nvidia-smi`.
- W&B authentication is available when `PADIS_WANDB_MODE=online`; set
  `PADIS_WANDB_MODE=offline` deliberately when network logging is not wanted.
- The persistent experiment root is writable and has the capacity described in
  [Data layout and storage](#data-layout-and-storage).

For **GCP**, use either the supported Colab notebook/startup hook, which mounts
`padis-bucket` at `/mnt/data` and provisions the environment, or prepare an
equivalent VM manually. The normal persistent layout is:

```bash
export LION_DATA_PATH=/mnt/data/Datasets
export LION_EXPERIMENTS_PATH=/mnt/data/Datasets/experiments
```

The GCP runner stages expanded training caches under `/mnt/ram-disk`; the VM
must have sufficient RAM and permission to create that temporary filesystem.
The bucket mount must be active before launch. Spot pre-emption is supported:
rerun the identical command with the same `PADIS_GCP_RUN_NAME` to resume.

For **Slurm**, run on a login node with `sbatch`, `squeue`, and `scancel`
available. The defaults target the Cambridge CSD3 Ampere setup and accounts
shown in the Slurm scripts. On another cluster, set at least
`PADIS_SLURM_ACCOUNT`, `PADIS_CACHE_SLURM_ACCOUNT`, relevant partitions, and
`LION_DATA_PATH`. The compute nodes must see the same repository, dataset,
cache, and experiment paths. The common Slurm helper activates `lion-dev`
(falling back to `padis-dev`) through the cluster module/Mamba setup. Set a
stable `PADIS_RUN_STAMP` when resubmitting or inspecting one run; otherwise a
new timestamped training and reconstruction root is created.

Common useful preflight commands are:

```bash
test -d "$LION_DATA_PATH/processed/LIDC-IDRI"
test -d "$LION_DATA_PATH/processed/LIDC-IDRI-cache/padis_256/archives"
test -d "$LION_DATA_PATH/processed/LIDC-IDRI-cache/padis_512/archives"
conda run -n lion-dev python -c 'import torch; print(torch.cuda.is_available())'
```

The unified wrapper delegates checkpointing and scheduling to the backend
scripts. GCP finalises synchronously; Slurm submits verification and finalisation
with scheduler dependencies. `PADIS_SUBMIT_FINALISE=0` deliberately omits
generation and publication outputs.

## Data layout and storage

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

The completed local `Data` tree occupied approximately **504 GiB of allocated
disk space** (`538,120,458,610` apparent bytes) when this README was updated.
Provision at least 504 GiB for an equivalent reproduction and preferably
600 GiB or more to accommodate temporary downloads, cache construction,
checkpoint replacement, and filesystem overhead. Check the actual requirement
and available capacity with:

```bash
du -sh "$LION_DATA_PATH"
df -h "$LION_DATA_PATH"
```

Raw DICOM, processed NumPy slices, compressed caches, expanded RAM-disk caches,
model checkpoints, and reconstruction tensors can coexist during the run. The
RAM-disk copy is temporary but requires separate system memory or scratch
capacity while training is active.

The archived split assigns sorted patient-ID slots without patient leakage.
The 1,012 slots include two empty slots retained to preserve that split, so
they do not imply 1,012 contributing patients. The completed dataset counts
are:

| Split | Patient-ID slots | Four-slice regime | All-slice regime |
|---|---:|---:|---:|
| Train | 809 | 2,713 | 189,725 |
| Validation | 101 | 328 | 27,426 |
| Test | 102 | 326 | 25,719 |
| Total | 1,012 | 3,367 | 242,870 |

The default regime uses at most four slices per patient, capped by the smaller
of the available nodule and non-nodule slice counts before balancing the two
classes. A patient without an annotated nodule can therefore contribute no
image. `--full-lidc` selects every processed slice. Cache identity includes the
split, resolution, and selection regime, preventing a cache for one setup from
being reused silently for another.

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
`sigma_data=0.5`; the learning rate is ramped over the first ten million
training images or patches. Whole-image batches contain 8 slices. Patch effective batch
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

Slurm follows the same two-phase schedule within each job. Patch-prior tasks
train normally for six hours and intensively validate for six hours; their
Slurm allocation is 12 hours plus a 30-minute setup/final-save buffer.
Whole-image tasks train normally for eighteen hours and intensively validate
for the final six hours; their allocation is 24 hours plus the same buffer.
The two task families are submitted as separate arrays so their scheduler time
limits are accurate. `P=96` remains capped at batch size 96 on A100 by
default, even if the general patch batch size is raised; lower it with
`PADIS_P96_BATCH_SIZE` or change the explicit safety ceiling with
`PADIS_P96_A100_BATCH_LIMIT` only after a memory check.

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

## Notes and limitations

The `cp_tv` implementation executes LION's Chambolle-Pock TV solver and is
not the exact ADMM-TV algorithm described by Hu et al. The `baseline` uses fan-beam FDK,
not parallel-beam FBP. `pnp_admm` uses a LION-native DRUNet surrogate as its
DRUNet denoiser;
the source description does not give enough optimizer, architecture, or
stopping-rule detail for exact identity. A no-PaDIS-prior or empty checkpoint
is never treated as checkpoint identity. The completed final training and
reconstruction used a Google-hosted NVIDIA RTX PRO 6000; fixed-validation
tuning also used a GTX 1070. The Slurm A100 path is an equivalent reproduction
backend, not the hardware on which the reported final results were obtained.

The CP baseline retains the isotropic TV objective, uses `lambda=0.001` for
1,000 iterations, and clips its output to `[0,1]` without imposing
non-negativity inside the solver. It is therefore reported as CP, not as an
exact reproduction of the original work's ADMM-TV baseline.

## Reconstruction methods

The matrix contains:

| Method | Summary |
|---|---|
| `baseline` | Hann-filtered LION FDK baseline. |
| `cp_tv` | TV reconstruction using LION's Chambolle-Pock solver. |
| `pnp_admm` | ADMM with conjugate-gradient data updates and DRUNet denoising. |
| `whole_image_diffusion` | Whole-image VE-DPS prior. |
| `langevin` | Data-consistency updates with patch or whole-image Langevin sampling. |
| `predictor_corrector` | VE predictor and Langevin corrector sampling. |
| `ve_ddnm` | VE-DDNM approximate-null-space correction. |
| `patch_average` | Fixed overlapping patches averaged in overlap regions. |
| `patch_stitch` | Fixed overlapping patches written in order, with later patches overwriting overlaps. |
| `padis_dps` | Randomly shifted PaDIS patch partitions inside VE-DPS. |

New commands, manifests, saved metrics, and output directories use `cp_tv`.
For backward compatibility, the inference and matrix CLIs also accept the old
`admm_tv` alias; hyperparameter selection and verification canonicalise old
saved method fields, and figure generation falls back to old `admm_tv` result
directories. The executed solver is Chambolle-Pock, so presentation code
labels it **CP** and tables report sampler **CP** with prior **TV**.

Unless stated otherwise, diffusion reconstruction uses 100 geometrically
spaced noise levels from `sigma_max=10` to `sigma_min=0.002`, with 10 inner
updates per level. The 8-view experiment uses `sigma_min=0.003`.
Predictor-corrector instead performs one predictor and one corrector update for
each adjacent noise-level pair. Paper-style and LION-physics VE-DDNM use 1,000
levels and one update per level; Public-compatible VE-DDNM follows the released
implementation with 100 levels and 10 inner updates.

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
| `ct_20_limited_angle_120` | 20 | 120 degrees | 256 x 256 |
| `ct_512_60` | 60 | 360 degrees | 512 x 512 |

Every experiment treats each axial slice independently using LION's 2D
tomosipo-based fan-beam operator and a 300 mm by 300 mm in-plane field of view.
The detector has 900 bins across 900 mm, the source-to-origin distance is
575 mm, and the source-to-detector distance is 1,050 mm. Measurements are
noise-free projections of normalised intensity, not calibrated attenuation.
Consequently these experiments isolate angular undersampling and limited
coverage; they do not model photon noise, scatter, beam hardening, motion, or
cone-beam effects.

New outputs use the descriptive `ct_20_limited_angle_120` identifier. The old
`ct_fanbeam_180` name remains an accepted CLI alias, and verification, tables,
and figures can consume existing artefacts bearing that legacy name.

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
geometric levels from `sigma=10` to `0.002`, one inner step per level, and a
separately tuned Langevin coefficient and noise scale for each primary prior:
whole-image uses `(epsilon, gamma)=(0.75, 0.75)`, randomly shifted PaDIS uses
`(0.8, 0.85)`, and fixed patch averaging and stitching use `(1, 1)`. The
pipeline also generates naive-patch and 300-evaluation Langevin diagnostic
presets needed by appendix figures. Both the GCP generation phase and the
cross-platform finaliser execute the authoritative named presets in
`core/PaDIS_experiments.py`; they do not duplicate these hyperparameters.
`PADIS_GENERATION_EPSILON` and `PADIS_GENERATION_NOISE_SCALE` may be set to
apply an explicit global diagnostic override, but are unset by default.

## Tuned reconstruction defaults

The authoritative registry is
[`config/reconstruction_hparam_defaults.json`](config/reconstruction_hparam_defaults.json).
The complete corrected-validation protocol, candidate matrix, inheritance
rules, and command for reproducing every tuning row are recorded in
[`tuning/TUNING.md`](tuning/TUNING.md).
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
| PnP-ADMM | Not implemented | Not implemented | `eta=3e-5`, 60 outer iterations, at most 50 CG iterations, tolerance `1e-7` |

Native 512 PaDIS-DPS uses `zeta=2.0`, `epsilon=0.5`, and Gaussian
initialisation for LION-Physics; Public-Compatible uses `zeta=1.2`,
`epsilon=0.5`, and FDK initialisation. Both process one patch at a time and use
activation checkpointing. FDK initialisation, where selected, uses a Hann
filter with frequency scaling 0.3 and clips the initial image to `[0,1]`.

### Re-running the selection procedure

The generic validation matrix and its inheritance rules are documented in
[`tuning/TUNING.md`](tuning/TUNING.md). Run it against the completed training
root used for inference:

```bash
conda run -n lion-dev python -u \
  scripts/paper_scripts/PaDIS-Reproduction/tuning/PaDIS_run_reproduction_tuning.py \
  --training-root "$PADIS_TRAIN_ROOT" \
  --output-root "$LION_EXPERIMENTS_PATH/PaDIS/hparam_tuning/reproduction"
```

The launcher forces the `min_intense_val` diffusion checkpoint policy, uses
the corrected validation split, and emits a manifest, exact command record,
summary JSON/CSV, and logs for every sweep. Existing metrics are reused unless
`--rerun-existing` is supplied. It is intentionally separate from the normal
pipeline: the pipeline consumes the committed registry so a complete inference
run cannot silently change merely because a tuning rerun produces a different
finite-sample ordering.

`PaDIS_hparam_defaults.py` remains available for inspecting or staging a
registry from completed tuner records. Do not overwrite the committed registry
without reviewing its exact-model and inheritance records; that JSON, rather
than an automatically exported partial sweep, is the inference source of truth.

## Manual reconstruction and matrix inspection

### Colab/manual GCP

The supported Colab entry point is:

```text
scripts/paper_scripts/PaDIS-Reproduction/platforms/gcp/PaDIS_Colab_manual_reconstruction.ipynb
```

It clones the repository to local disk, installs Miniforge and the LION
environment, authenticates GCP/W&B as required, mounts `padis-bucket` at
`/mnt/data`, and launches:

```bash
bash scripts/paper_scripts/PaDIS-Reproduction/platforms/gcp/run_PaDIS_GCP_manual_reconstruction.sh
```

The manual runner defaults to two concurrent reconstruction workers per GPU.
Set `PADIS_RECON_TASKS_PER_GPU` to override this. It checks all model inputs and
trains missing supported models before inference. Training archives are
decompressed into `/mnt/ram-disk`; the runner removes that temporary mount after
training and syncs persistent outputs after each job. Use this path to run
inference from existing checkpoints without resubmitting the complete training
pipeline.

### Inspecting the matrix directly

List the resolved jobs without running them:

```bash
python -u scripts/paper_scripts/PaDIS-Reproduction/reconstruction/PaDIS_run_reconstruction_matrix.py \
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
  --pnp-cg-iterations 50 \
  --pnp-cg-tolerance 1e-7 \
  --list
```

Replace `--list` with `--count`, `--check-inputs`, or `--task-index N` to count,
validate, or execute one resolved job. The runners generate this list once,
store it as `reconstruction_matrix_jobs.json`, and reconcile rebuilt manifests
against valid existing outputs.

## Expected reconstruction runtimes

The following estimates multiply the logged mean 20-view time per slice by the
planned evaluation size. Standard rows use 25 slices; patch averaging and
stitching use four. Each cell is the expected total followed by the measured
mean per slice.

| Reconstruction method | LION-physics | Paper | Public-compatible |
|---|---:|---:|---:|
| FDK | 2.0 min (4.71 s) | – | – |
| CP | 3.4 min (8.13 s) | – | – |
| Predictor-corrector | 5.9 min (14.11 s) | 5.9 min (14.16 s) | 5.8 min (14.02 s) |
| PnP-ADMM | 11.1 min (26.63 s) | – | – |
| Langevin | 21.5 min (51.49 s) | 23.5 min (56.42 s) | 23.6 min (56.55 s) |
| VE-DDNM | 25.9 min (62.08 s) | 25.6 min (61.34 s) | 24.4 min (58.54 s) |
| PaDIS-DPS | 49.5 min (118.85 s) | 50.2 min (120.45 s) | 49.7 min (119.22 s) |
| Whole-image VE-DPS | 1.09 h (156.44 s) | 1.09 h (156.89 s) | – |
| Patch stitching, four slices | 1.61 h (1,450.73 s) | – | 1.62 h (1,461.08 s) |
| Patch averaging, four slices | 1.63 h (1,468.76 s) | – | 1.63 h (1,468.75 s) |

These displayed 20-view rows sum to approximately 14.14 GPU-hours if executed
serially. This is not the wall-clock duration of the complete 109-job matrix:
jobs can share a GPU or run concurrently, other experiments have different
view counts, and native-512 work is substantially more expensive. The values
are planning estimates parsed from concurrent pipeline logs, not isolated
benchmarks. Training adds approximately 12 hours per patch prior and 24 hours
per whole-image prior, plus PnP training and scheduler/setup overhead; arrays
reduce elapsed wall time when enough GPUs are available.

## Outputs, restart state, and reporting

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

Generate the final verification JSON and CSV with:

```bash
python -u scripts/paper_scripts/PaDIS-Reproduction/reconstruction/PaDIS_verify_reconstruction_matrix.py \
  --root "$PADIS_RECON_ROOT" \
  --expected-jobs-json "$PADIS_RECON_ROOT/reconstruction_matrix_jobs.json" \
  --output-json "$PADIS_RECON_ROOT/reconstruction_matrix_verification.json" \
  --output-csv "$PADIS_RECON_ROOT/reconstruction_matrix_verification.csv"
```

The default verifier performs 2,000 deterministic image-level bootstrap
resamples at 95% confidence using seed 33. The CSV includes the aggregate
metrics, bootstrap standard errors and confidence limits. Use the expected-jobs
manifest to detect missing outputs rather than relying only on marker counts.
Metrics are calculated in the shared normalised-intensity domain with data
range one and without per-image rescaling. The bootstrap resamples images, not
training runs, so its intervals describe slice variation and do not measure
retraining uncertainty. Relative sinogram residual records measurement
consistency separately from MAE, PSNR, and SSIM.

To resume or repeat every post-inference stage in one command, set the roots
created by the selected backend and run the same finaliser used by both
pipelines:

```bash
export PADIS_RUN_ROOT="$LION_EXPERIMENTS_PATH/PaDIS"
export PADIS_TRAIN_ROOT=/path/to/completed/training-root
export PADIS_RECON_ROOT=/path/to/completed/reconstruction-root
export PADIS_TIMING_MODE=gcp             # use slurm for a Slurm array
export PADIS_TIMING_LOG_ROOT=/path/to/reconstruction/logs
conda run -n lion-dev bash \
  scripts/paper_scripts/PaDIS-Reproduction/pipeline/PaDIS_finalise_pipeline.sh
```

This creates any missing unconditional samples, verifies the matrix, derives
timings, and regenerates every table and figure. Existing `samples.pt` files
are reused.

Generate all publication tables using the standard experiment paths with:

```bash
python -u scripts/paper_scripts/PaDIS-Reproduction/reporting/PaDIS_make_tables.py
```

By default this reads
`$LION_EXPERIMENTS_PATH/PaDIS/final_real_runs/PaDIS-Reproduction-GCP_reconstruction/reconstruction_matrix_verification.csv`
and writes `reconstruction_tables.tex` plus seven decoded CSVs under
`$LION_EXPERIMENTS_PATH/PaDIS/paper_tables`. Use `--csv-path`, `--tex-path`,
and `--csv-output-dir` to override these locations. `--allow-missing` writes
only tables represented by a partial smoke-run CSV; do not use it for final
publication output.

The timing table is calculated from completed reconstruction progress logs; no
timings are stored in the table generator. `--timing-mode gcp` (the default)
and `--timing-mode colab` read the manual-runner log naming convention from the
standard GCP reconstruction root. For a Slurm reconstruction array, use:

```bash
python -u scripts/paper_scripts/PaDIS-Reproduction/reporting/PaDIS_make_tables.py \
  --timing-mode slurm \
  --timing-log-root /path/to/slurm/output/files \
  --timing-jobs-json /path/to/reconstruction_matrix_jobs.json
```

The parser uses the final completed `LIDC test run` progress record, whose
seconds-per-iteration value is already the mean wall time per reconstructed
slice. Timings are grouped by implementation and reconstruction method; where
multiple prior rows exist for a combination, their per-slice timings are
averaged.

Generate all figures independently with:

```bash
conda run -n lion-dev python -u \
  scripts/paper_scripts/PaDIS-Reproduction/reporting/PaDIS_make_paper_figures.py \
  --reconstruction-root "$PADIS_RECON_ROOT" \
  --generation-root "$PADIS_RUN_ROOT/reconstruction_presets" \
  --output-folder "$PADIS_RUN_ROOT/paper_figures" \
  --figures all
```

The command fails on missing required panels by default. Use `--allow-missing`
only for diagnostic partial output, not for a completed reproduction. Each
figure is written as PNG and PDF and described in the output manifest.

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
Display limits are the 15th and 95th percentiles of non-background target
pixels (`NI > 0.02`) and are shared by every reconstruction and target in a
row. HU panels use LION's normalised-intensity-to-HU conversion, while metric
calculation remains in NI. The 50 mm scale bars use the configured 300 mm field
of view rather than patient-native DICOM spacing.
