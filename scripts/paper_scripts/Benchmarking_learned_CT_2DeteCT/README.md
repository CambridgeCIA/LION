# Benchmarking learned algorithms for CT image reconstruction tasks

> **Untested: use with care.** LION has changed a lot since the paper's
> results were produced (2024): its models, solvers, operators, data loaders
> and evaluation have been rewritten, and the paper's code has been rebuilt
> here on the current LION from scattered scripts. These scripts have not
> been run on real data yet: they are checked only on a CPU with fake data
> and a toy operator, and their results are not yet known to match the
> paper's. "Differences" below lists what is known to differ, and what
> isn't known.

Scripts that reproduce part of the paper

> Kiss, Maximilian B., Ander Biguri, Zakhar Shumaylov, Ferdia Sherry, K.
> Joost Batenburg, Carola-Bibiane Schönlieb, and Felix Lucka. "Benchmarking
> learned algorithms for computed tomography image reconstruction tasks."
> *Applied Mathematics for Modern Challenges*, 2025.
> [doi:10.3934/ammc.2025001](https://doi.org/10.3934/ammc.2025001),
> [arXiv:2412.08350](https://arxiv.org/abs/2412.08350)

The paper benchmarks classical and learned reconstruction methods on the
measured data of 2DeteCT, on nine problems, by SSIM and PSNR (its Tables
5-7). The problems are LION's `deteCT_benchmark` experiments:

| Problem (paper) | `--experiment` | LION experiment |
|---|---|---|
| Full data (mode 2, 3600 angles) | `full` | `FullDataCTRecon` |
| Limited angle, 120° | `limited_120` | `LimitedAngle120CTRecon` |
| Limited angle, 90° | `limited_90` | `LimitedAngle90CTRecon` |
| Limited angle, 60° | `limited_60` | `LimitedAngle60CTRecon` |
| Sparse angle, 360 angles | `sparse_360` | `SparseAngle360CTRecon` |
| Sparse angle, 120 angles | `sparse_120` | `SparseAngle120CTRecon` |
| Sparse angle, 60 angles | `sparse_60` | `SparseAngle60CTRecon` |
| Low dose (mode 1) | `low_dose` | `LowDoseCTRecon` |
| Beam hardening (mode 3) | `beam_hardening` | `BeamHardeningCTRecon` |

The targets are the mode 2 reconstructions of the dataset.

## The methods

**These scripts cover the paper's 15 methods**, with the code that produced
its results:

- classical: **FBP**, **AGD** (Nesterov's accelerated gradient descent on the
  least squares), **ChP** (TV minimisation by Chambolle-Pock);
- post-processing, on FBP: **FBP+U-Net**, **FBP+MS-D Net**, **FBP+DnCNN**;
- unrolled: **LG** (learned gradient), **LGTV** (LG with a TV gradient),
  **LPD** (learned primal-dual);
- Plug-and-Play, with a Gaussian denoiser trained on the targets:
  **DnCNN-PnP**, **DRUNet-PnP**, **GS-PnP** (gradient-step DRUNet);
- learned regularisers: **AR** (adversarial regulariser), **ACR**
  (adversarial convex regulariser), and **TDV** (total deep variation, a
  variational network).

## Running everything

On a GPU machine with the processed 2DeteCT data (`LION_DATA_PATH`) and
`LION_RESULTS_PATH` set, from this folder:

```bash
cd scripts/paper_scripts/Benchmarking_learned_CT_2DeteCT

# 1. a short run, to check it works (minutes)
python train.py --experiment full --model all --epochs 1 --slices 20
python train_denoisers.py --denoiser all --noise_level 0.05 --epochs 1 --slices 20
python train_regularisers.py --experiment full --model all --epochs 1 --slices 20
python evaluate.py --experiment full --samples 10 --validation_samples 5 --final

# 2. all 54 trainings (6 learned methods on 9 problems; hours to days each)
python train.py --experiment all --model all

# 3. the 21 denoisers of the PnP methods (3 denoisers at 7 noise levels),
#    the same for every problem
python train_denoisers.py --denoiser all --noise_level all

# 4. AR, ACR and TDV on the 9 problems (27 trainings)
python train_regularisers.py --experiment all --model all

# 5. all nine evaluations (with the classical methods), and the tables
python evaluate.py --experiment all
```

Delete the short run's models (`$LION_RESULTS_PATH/paper_scripts/2detect_benchmark/`)
before steps 2 to 4: trained models are skipped. Steps 2, 3 and 4 are
independent and can run at the same time, on different GPUs (`--gpu`). Each
trains its models one after the other; a stopped run resumes from the last
checkpoint, skipping the models already trained. To use more GPUs, run one
problem (or one method, or one denoiser) per GPU:

```bash
python train.py --experiment full --model all --gpu 0 &
python train.py --experiment limited_120 --model all --gpu 1 &
# ...
python train.py --experiment all --model LPD --gpu 2 &
python train_denoisers.py --denoiser GS-PnP --noise_level all --gpu 3 &
python train_regularisers.py --experiment all --model TDV --gpu 4 &
```

Results go to `$LION_RESULTS_PATH/paper_scripts/2detect_benchmark/<experiment>/`:
the models (each `<model>` and `<model>_best`, with their records) and
`comparison/` (tables, per-slice scores, box plots, images). `evaluate.py
--experiment all` also writes `tables.md`, the nine tables together. The
evaluation leaves out, with a message, learned methods not trained yet, so
the classical methods can be evaluated first. For each PnP method, it first
chooses the noise level (below) on the validation set, and writes every
score and the choice to `comparison/pnp_noise_levels.json`. AR, ACR and TDV
are each tested three times (below): "AR (min val)", "AR (min train)" and
"AR (last)".

The denoisers go to `$LION_RESULTS_PATH/paper_scripts/2detect_benchmark/denoisers/`,
named by method and noise level (e.g. `DnCNN-PnP_sigma0_005` and
`DnCNN-PnP_sigma0_005_best`).

## Files

- `settings.py`: the problems, the learned methods with their parameters, the
  training, and the classical methods with their settings.
- `train.py`: trains one learned method on one problem, or all of them
  (`all`).
- `train_denoisers.py`: trains the PnP methods' denoisers, one or all, at
  one noise level or all.
- `train_regularisers.py`: trains AR, ACR and TDV on one problem, or all of
  them.
- `learned_regularisers.py`: ACR and the reconstruction with AR and ACR as
  the 2024 code had them, where they differ from LION's (`ACR2024`,
  `GradientDescent2024`).
- `evaluate.py`: tests every method on one problem's test set, or on all nine
  (`all`).

## The settings, from the code of 2024

From Max Kiss's training and evaluation scripts (PR #104).

- **Training** (every learned method): Adam (betas 0.9 and 0.99), learning
  rate 1e-4 (LG and LGTV: 1e-5), batch size 1, 100 epochs, the mean squared error to
  the target; validated every 10 epochs with it. The model with the lowest
  validation loss is the one tested.
- **FBP+U-Net, FBP+MS-D Net, FBP+DnCNN:** the network on the FDK of the
  measurements (negative values set to zero). DnCNN: 20 layers of 64
  channels, bias-free, batch normalisation, leaky ReLUs, learning the
  residual. MS-D Net: 100 layers of width 1, dilations 1 to 10.
- **LG:** 5 iterations, blocks of [7, 32, 32, 6] channels: the image, a
  memory of 5 channels and the data-fit gradient. **LGTV:** the same, with a
  TV gradient too (blocks of [8, 32, 32, 6], `LG.lgtv_parameters()`).
- **LPD:** 5 iterations of 3-layer blocks of 32 channels, no biases, fixed
  step sizes, the first primal channel projected in the dual step.
- **FBP:** FDK, negative values kept. **AGD:** 100 iterations. **ChP:** 500
  iterations, with the TV weight chosen for each problem (`TV_WEIGHTS`: from
  1e-10 for full data to 5e-6 for sparse angle and low dose).
- **Metrics:** SSIM and PSNR of each slice, with the target's own range
  (max - min), averaged over the test set.

### The PnP methods, from Ferdia Sherry's code of 2024

From his `train_gaussian_denoiser.py` and `eval_*_pnp.py`.

- **Denoisers:** trained on the targets (LION's `GroundTruthCT`), scaled to
  [0, 1] by the training targets' maximum (0.02366, the code's
  `normalisation.json`), with Gaussian noise of one standard deviation added
  after. One denoiser per noise level: 0.001, 0.005, 0.01, 0.02, 0.03, 0.05
  and 0.07 (the paper's).
  - **DnCNN:** 20 layers of 64 channels, with biases, batch normalisation and
    leaky ReLUs, learning the residual.
  - **DRUNet:** 64 channels, 4 blocks per scale, with biases and leaky ReLUs,
    without the noise level as an input.
  - **GS:** a gradient-step denoiser x - ∇g(x), g(x) = ‖x - N(x)‖²/2, N a
    DRUNet with ELUs, 32 channels and 2 blocks per scale.
- **Training:** Adam (betas 0.9 and 0.9), learning rate 1e-4, 25 epochs,
  one image at a time as 5 random patches of 256 × 256, each erased with
  probability 1/2 (a random rectangle set to 0, as kornia's `RandomErasing`);
  the mean squared error to the clean patch. The denoiser with the best
  validation is kept.
- **Reconstruction:** 100 iterations from zero, step size 1/‖A‖².
  - **DnCNN-PnP and DRUNet-PnP:** forward-backward splitting, x ← D(x -
    step Aᵀ(Ax - y)) (LION's `PnP`, "FBS").
  - **GS-PnP:** gradient descent on ‖Ax - y‖²/2 + g(x), then the denoiser
    applied once. In the descent, g is evaluated on the images as they are,
    not scaled to [0, 1], as the code did.
- **The noise level** of each method, for each problem: the one whose
  reconstructions have the best mean PSNR on the validation set (the paper:
  "selecting separately for each experiment the denoiser that performed best
  on the PnP reconstruction task").

### AR, ACR and TDV, from Zakhar Shumaylov's code of 2024

From his `ar_experiments.py`, `regulariser_experiments.py`,
`tdv_experiments.py` and `eval_method.py`, and LION's `AR`, `ACR`, `TDV` and
solvers as they were then (LION's history: commit `e4a57ba`, July 2024; the
scripts were removed in `8703693`).

- **Training:** Adam (betas 0.9 and 0.99), learning rate 1e-4, 25 epochs,
  validated and checkpointed every epoch.
  - **AR** (batch size 2) **and ACR** (batch size 5): the WGAN loss with
    gradient penalty (weight 0.1), the target images against the FDK of the
    batch's measurements (`ARSolver`). The images are scaled by the minimum
    and maximum of the first 100 training targets. Validated by the mean
    squared error of their reconstructions (below) of the first 50
    validation slices.
  - **TDV** (batch size 1): the mean squared error to the target
    (`SupervisedSolver`), validated with it on the whole validation set.
- **AR:** LION's (six 5×5 convolutions, two fully connected layers).
- **ACR:** 5 convex layers of 16 channels, 5×5, as LION's, but as the 2024
  code had it (`ACR2024`): its ‖·‖² term is of the last layer's output (not
  of the image), with weight softplus(-36); its non-negative weights start as
  torch's default, clipped at 0 (the 2024 code drew them in [0, 1e-3], but
  assigned them where it had no effect).
- **TDV:** LION's `TDV` (Kobler's regulariser, MIT licence, with the CT data
  term of 2024): 10 steps, 5 macro blocks of 5 scales, 16 channels, T and λ
  starting at 10 and 5; reflection padding, its transpose approximated by
  cropping, as in 2024.
- **AR's and ACR's reconstruction** (`GradientDescent2024`): 150 steps of
  gradient descent with momentum 0.5 on ½‖Ax - y‖² + α R(x), from FDK; each
  step's size reduced by 0.95 until the Armijo condition holds for a plain
  gradient step, and never grown back. The first step size is 0.2/‖A‖² for
  AR (which also stops once it is below 1e-3/‖A‖²) and 1/‖A‖² for ACR. α is
  the mean of ‖Aᵀ(Ax - y)‖ over the first 50 validation slices (Lunz et
  al.'s heuristic), divided by 10 for beam hardening.
- **Which model is tested:** the 2024 code kept a checkpoint per epoch and
  tested, for each problem, the models at the lowest validation loss, at
  the lowest training loss, and after the last epoch. The paper's tables
  show one of the three for each problem, chosen after testing; mostly the
  one at the lowest validation loss. So `evaluate.py` tests all three, and
  the tables list them as "AR (min val)", "AR (min train)", "AR (last)".
  In 2024, the tables used:
  - **AR:** min val, except the last for limited 60° and sparse 60, and min
    train for low dose and beam hardening;
  - **ACR:** min val, except the last for beam hardening;
  - **TDV:** min val, except min train for limited 120°, and for beam
    hardening the checkpoint of epoch 4 (none of the three). TDV's
    checkpoints in 2024 go up to epoch 12: its training seems to have
    stopped there, though the script says 25 epochs.

## Differences

**With the paper's text** (the scripts follow the code):

- **Not stated in the paper:** the batch size, the loss of the supervised
  methods, the number of iterations of LG and LPD, the architectures of the
  networks, how often it validated, and the classical methods' iterations and
  weights.
- **LPD's blocks** are 3-layer, as the code's; the paper describes "shallow
  four-layer CNN"s.
- **The denoisers' Adam:** betas 0.9 and 0.9, as the code; the paper says
  0.9 and 0.999.
- **GS-PnP:** gradient descent on both terms, then a denoising, as the code;
  the paper describes "a gradient step on the regularization functional and a
  proximal step on the data discrepancy functional".

**With the code of 2024:**

- **FBP+U-Net and FBP+MS-D Net:** the 2024 classes (`FBPUNet`, `FBPMS_D`)
  weren't kept. Here they are LION's `UNet` (the same U-Net) and `MSDNet` on
  FDK; the MS-D network's architecture may differ from 2024's.
- **The models** are LION's current classes, fixed against their papers since
  2024 (e.g. LPD); the parameters above set them as they were trained, where
  a parameter can.
- **Training and testing code:** LION's solver and `LION.evaluation` instead
  of the hand-written loops of 2024: the same optimisation and metrics.
- **Random numbers** (initial weights, shuffling) differ from 2024's.
- **The denoisers' validation:** on 5 patches of each validation image, by
  the mean squared error (LION's `GaussianDenoiserSolver`); the 2024 code
  validated on whole images, by PSNR. The noise level is then chosen, as in
  2024, by the PnP reconstructions' PSNR (here each slice's own range; in
  2024, the target's maximum: the same for targets whose minimum is 0).
- **AR's and ACR's validation** during training reconstructs each slice on
  its own; the 2024 code reconstructed each batch (2 or 5 slices) together,
  with one step size.
- **The 2024 training scripts** don't quite run with the LION of the time
  (e.g. the AR script calls `estimate_alpha`, which its AR named
  `estimate_lambda`): they were run from a slightly different version. The
  settings are theirs; where they disagree with the models of the time, the
  models' code is followed.

**Not known** (neither the 2024 code nor the paper says):

- **The denoisers' options are lost:** the training script took the
  channels, blocks, `bias_free` and `enforce_positivity` from its command
  line, which wasn't recorded, and the paper doesn't give them. These scripts
  use the script's defaults, except for GS: the paper says its DRUNet was
  scaled down "in number of blocks and width of the blocks", and 32 channels
  and 2 blocks is the 2024 GS class's default (the command line's default was
  64 and 4).
- **ChP's TV weights:** `TV_WEIGHTS` are those of #104. Max Kiss's `chp.py`
  (2024) tried three other sets, on three example slices
  (`CHP_PY_WEIGHTS`; the first: 1e-10 for full data, 1e-6, 1e-4, 1e-4 for
  limited 120°, 90°, 60°, 1e-5 for sparse angle and low dose, 1e-7 for beam
  hardening). Six of #104's nine weights are among those tried (not 5e-6, for
  sparse 120 and 60 and low dose), so #104's may be what was chosen after,
  but which gave the paper's tables isn't known.
  `evaluate.py --chp_weights` also tests ChP with each of chp.py's sets
  ("ChP (chp.py 1)", ...), to compare with the paper's ChP rows.
- **TDV's epochs:** 25 in the script, but its 2024 checkpoints go up to
  epoch 12.

## The paper's results, for reference

Selected rows (SSIM, PSNR in dB); the full tables are the paper's Tables 5-7.

| Problem | Method | SSIM | PSNR |
|---|---|---|---|
| Full data | FBP | 0.7463 | 35.03 |
| Full data | AGD | 0.7753 | 35.10 |
| Low dose | ChP | 0.6820 | 31.96 |
| Low dose | LPD | 0.8282 | 32.67 |
| Beam hardening | FBP+MS-D Net | 0.7991 | 31.44 |
| Beam hardening | LPD | 0.8352 | 33.14 |
| Limited angle 120° | LPD | 0.8296 | 31.17 |
| Limited angle 60° | FBP | 0.1557 | 16.91 |
| Limited angle 60° | LPD | 0.7724 | 28.07 |
