# Continuous Learned Primal Dual

> **Unverified.** These scripts have not been run on real data yet: they are
> checked only on a CPU with fake data and a toy operator. Their results are
> not yet known to match the paper's.

Scripts that reproduce the paper

> Runkel, Christina, Ander Biguri, and Carola-Bibiane Schönlieb. "Continuous
> Learned Primal Dual." *IEEE International Workshop on Machine Learning for
> Signal Processing (MLSP)*, 2024. [arXiv:2405.02478](https://arxiv.org/abs/2405.02478)

cLPD replaces the convolutional blocks of Learned Primal-Dual (LPD) by neural
ODEs. The paper trains cLPD and LPD on six CT problems from LIDC-IDRI and
compares them with each other and with FBP, by SSIM and PSNR (its Table 1).

The scripts follow **the code that trained the paper's models** (June 2024)
where it and the paper's text differ, as the code produced the results. All
the differences, with the paper and with 2024, are listed below.

| Problem (paper) | `--experiment` | LION experiment, with the settings of June 2024 |
|---|---|---|
| 1. Full angle, clinical dose | `full_low` | `LowDoseCTRecon` (I0 = 3500) |
| 2.a Full angle, extremely low dose | `full_extreme_low` | `ExtremeLowDoseCTRecon` (I0 = 1000) |
| 2.b Sparse angle, clinical dose | `sparse_low` | `SparseAngleLowDoseCTRecon` (I0 = 3500) |
| 2.c Sparse angle, extremely low dose | `sparse_extreme_low` | `SparseAngleExtremeLowDoseCTRecon` (I0 = 1000) |
| 3.a Limited angle, clinical dose | `limited_low` | `LimitedAngleLowDoseCTRecon` (I0 = 3500) |
| 3.b Limited angle, extremely low dose | `limited_extreme_low` | `LimitedAngleExtremeLowDoseCTRecon` (I0 = 1000) |

## Running everything

On a GPU machine with the processed LIDC-IDRI data (`LION_DATA_PATH`) and
`LION_RESULTS_PATH` set, from this folder:

```bash
cd scripts/paper_scripts/Continuous_Learned_Primal_Dual

# 1. a short run, to check it works (minutes)
python train.py --experiment full_low --model all --epochs 1 --slices 20
python evaluate.py --experiment full_low --samples 10

# 2. all twelve trainings (cLPD and LPD on six problems; days each)
python train.py --experiment all --model all

# 3. all six evaluations, and Table 1
python evaluate.py --experiment all
```

Delete the short run's models (`$LION_RESULTS_PATH/paper_scripts/cLPD/`)
before step 2: trained models are skipped. `train.py --experiment all --model
all` trains one after the other; a stopped run resumes from the last
checkpoint, skipping the models already trained. To use several GPUs, run one
problem per GPU:

```bash
python train.py --experiment full_low --model all --gpu 0 &
python train.py --experiment sparse_low --model all --gpu 1 &
# ...
```

Results go to `$LION_RESULTS_PATH/paper_scripts/cLPD/<experiment>/`: the models
(`cLPD`, `LPD`, and `*_best`, with their records) and `comparison/` (tables,
per-slice scores, box plots, images). `evaluate.py --experiment all` also
writes `table1.md`, the six tables together.

## Files

- `settings.py`: the problems (`settings.experiment(name)` makes each as
  LION's experiment with the settings of June 2024), the two models
  (`cLPD.continous_LPD_paper()`, `LPD.continous_LPD_paper()`) and the
  training.
- `train.py`: trains one model on one problem, or all of them (`all`).
- `evaluate.py`: tests cLPD, LPD and FDK on one problem's test set, or on all
  six (`all`).

## The settings

- **cLPD:** 10 iterations of first-order ODE blocks (4 layers, 32 channels,
  PReLU, no biases) with instance normalisation; learned, positive step
  sizes.
- **LPD:** 10 iterations of 4-layer blocks (32 channels, PReLU, no biases),
  without instance normalisation; learned, positive step sizes.
- **Training:** Adam (learning rate 1e-4, betas 0.9 and 0.99), batch size 2,
  100 epochs, the mean squared error to the target; validated every epoch
  with it. The final models are tested.
- **Problems:** LION's default geometry of 2024 (fan beam, 512x512 pixels over
  300 mm, a 900-pixel detector; 360 angles, 50 for sparse angle, 60 over 60
  degrees for limited angle); the relative noise model with I0 = 3500 or 1000
  photons, 5 photons of electronic noise and 5% cross-talk, fresh noise each
  time a slice is read; 5 slices per patient, half with nodules, patients
  split 80/10/10, chosen as LION did then.

## Differences

**With the paper's text** (the scripts follow the code of 2024):

- **"Clinical dose"** in the paper was LION's low dose then (I0 = 3500), not
  its clinical dose (I0 = 10000).
- **LPD's parameters:** the paper says LPD used the original LPD's (Adler and
  Öktem: 3-layer blocks with biases). The code used 4-layer blocks without
  biases and with learned steps, as cLPD.
- **Not stated in the paper:** the loss (the MSE), Adam's betas, the batch
  size of validation, how often it validated, and that the final models were
  tested.
- **"FBP"** in the paper is FDK here (the same for fan beam).

**With the code of 2024:**

- **LPD's iterations:** 10, as cLPD's, so the two compare at equal depth.
  The 2024 LPD script used 5 (`LPD_train.py`, June 2024).
- **The slices' pixel spacing:** in 2024 each scan kept its own pixel spacing
  (0.46 to 0.98 mm). LION's pre-processing now resamples every scan to 500 mm
  over 512 pixels, so the same slices show the patient smaller, and the 300 mm
  geometry of 2024 doesn't match the data (LION warns about it). With the
  relative noise model the noise doesn't depend on the patient's size. A
  copy of the data pre-processed before October 2026 gives the 2024 slices:
  `--data-folder <that folder>`.
- **The noise:** three errors of LION's noise in 2024 are fixed and not kept
  (cross-talk blurred across angles as well as detectors, the maximum of a
  whole batch, see `ct_utils.sinogram_add_noise`), so it is close to, not
  identical with, 2024's.
- **Training and testing code:** LION's solver and `LION.evaluation` instead
  of the hand-written loops of 2024: the same optimisation, but the metrics
  are LION's (PSNR, SSIM and HaarPSI on images normalised by the test set's
  range) rather than scikit-image's SSIM and PSNR with each target's own
  range.
- **Random numbers** (initial weights, noise, shuffling) differ from 2024's.

## The paper's Table 1, for reference

| Problem | cLPD SSIM | cLPD PSNR | LPD SSIM | LPD PSNR |
|---|---|---|---|---|
| 1. Full, clinical | 0.6140 ± 0.1263 | 34.18 ± 3.15 | 0.6157 ± 0.1245 | 34.12 ± 3.14 |
| 2.a Full, extremely low | 0.5773 ± 0.1287 | 32.37 ± 2.58 | 0.5790 ± 0.1251 | 32.23 ± 2.52 |
| 2.b Sparse, clinical | 0.5627 ± 0.1269 | 31.26 ± 2.30 | 0.5571 ± 0.1169 | 30.84 ± 2.18 |
| 2.c Sparse, extremely low | 0.5316 ± 0.1232 | 29.67 ± 1.95 | 0.5265 ± 0.1185 | 29.26 ± 1.89 |
| 3.a Limited, clinical | 0.4465 ± 0.1099 | 24.40 ± 1.69 | 0.3951 ± 0.0937 | 22.47 ± 1.71 |
| 3.b Limited, extremely low | 0.4371 ± 0.1081 | 24.02 ± 1.67 | 0.3823 ± 0.0933 | 22.25 ± 1.69 |
