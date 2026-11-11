# Benchmarking learned algorithms for CT image reconstruction tasks

> **Unverified.** These scripts have not been run on real data yet: they are
> checked only on a CPU with fake data and a toy operator. Their results are
> not yet known to match the paper's.

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

**These scripts cover 8 of the paper's 15 methods**, those whose code
produced the paper's results and is in LION:

- classical: **FBP**, **AGD** (Nesterov's accelerated gradient descent on the
  least squares), **ChP** (TV minimisation by Chambolle-Pock);
- post-processing, on FBP: **FBP+U-Net**, **FBP+MS-D Net**, **FBP+DnCNN**;
- unrolled: **LG** (learned gradient), **LPD** (learned primal-dual).

Not yet scripted: LGTV, AR, ACR, TDV, DnCNN-PnP, DRUNet-PnP and GS-PnP.

## Running everything

On a GPU machine with the processed 2DeteCT data (`LION_DATA_PATH`) and
`LION_RESULTS_PATH` set, from this folder:

```bash
cd scripts/paper_scripts/Benchmarking_learned_CT_2DeteCT

# 1. a short run, to check it works (minutes)
python train.py --experiment full --model all --epochs 1 --slices 20
python evaluate.py --experiment full --samples 10 --final

# 2. all 45 trainings (5 learned methods on 9 problems; hours to days each)
python train.py --experiment all --model all

# 3. all nine evaluations (with the classical methods), and the tables
python evaluate.py --experiment all
```

Delete the short run's models (`$LION_RESULTS_PATH/paper_scripts/2detect_benchmark/`)
before step 2: trained models are skipped. `train.py --experiment all --model
all` trains one after the other; a stopped run resumes from the last
checkpoint, skipping the models already trained. To use several GPUs, run one
problem (or one method) per GPU:

```bash
python train.py --experiment full --model all --gpu 0 &
python train.py --experiment limited_120 --model all --gpu 1 &
# ...
python train.py --experiment all --model LPD --gpu 2 &
```

Results go to `$LION_RESULTS_PATH/paper_scripts/2detect_benchmark/<experiment>/`:
the models (each `<model>` and `<model>_best`, with their records) and
`comparison/` (tables, per-slice scores, box plots, images). `evaluate.py
--experiment all` also writes `tables.md`, the nine tables together. The
evaluation leaves out, with a message, learned methods not trained yet, so
the classical methods can be evaluated first.

## Files

- `settings.py`: the problems, the learned methods with their parameters, the
  training, and the classical methods with their settings.
- `train.py`: trains one learned method on one problem, or all of them
  (`all`).
- `evaluate.py`: tests every method on one problem's test set, or on all nine
  (`all`).

## The settings, from the code of 2024

From Max Kiss's training and evaluation scripts (PR #104).

- **Training** (every learned method): Adam (betas 0.9 and 0.99), learning
  rate 1e-4 (LG: 1e-5), batch size 1, 100 epochs, the mean squared error to
  the target; validated every 10 epochs with it. The model with the lowest
  validation loss is the one tested.
- **FBP+U-Net, FBP+MS-D Net, FBP+DnCNN:** the network on the FDK of the
  measurements (negative values set to zero). DnCNN: 20 layers of 64
  channels, bias-free, batch normalisation, leaky ReLUs, learning the
  residual. MS-D Net: 100 layers of width 1, dilations 1 to 10.
- **LG:** 5 iterations, blocks of [8, 32, 32, 6] channels; it includes a
  learned TV-gradient term (LION's LG).
- **LPD:** 5 iterations of 3-layer blocks of 32 channels, no biases, fixed
  step sizes, the first primal channel projected in the dual step.
- **FBP:** FDK, negative values kept. **AGD:** 100 iterations. **ChP:** 500
  iterations, with the TV weight chosen for each problem (`TV_WEIGHTS`: from
  1e-10 for full data to 5e-6 for sparse angle and low dose).
- **Metrics:** SSIM and PSNR of each slice, with the target's own range
  (max - min), averaged over the test set.

## Differences

**With the paper's text** (the scripts follow the code):

- **Not stated in the paper:** the batch size, the loss of the supervised
  methods, the number of iterations of LG and LPD, the architectures of the
  networks, how often it validated, and the classical methods' iterations and
  weights.
- **LG:** LION's LG has a TV-gradient term, as the code of 2024 did; the
  paper also lists an "LGTV", LG with TV. Which of the two the code's LG
  results are is to be checked.
- **LPD's blocks** are 3-layer; the paper describes "shallow four-layer CNN"s.

**With the code of 2024:**

- **FBP+U-Net and FBP+MS-D Net:** the 2024 classes (`FBPUNet`, `FBPMS_D`)
  weren't kept. Here they are LION's `UNet` and `MSDNet` on FDK, whose
  architectures may differ from 2024's (the U-Net's in particular).
- **The models** are LION's current classes, fixed against their papers since
  2024 (e.g. LPD); the parameters above set them as they were trained, where
  a parameter can.
- **Training and testing code:** LION's solver and `LION.evaluation` instead
  of the hand-written loops of 2024: the same optimisation and metrics.
- **Random numbers** (initial weights, shuffling) differ from 2024's.

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
