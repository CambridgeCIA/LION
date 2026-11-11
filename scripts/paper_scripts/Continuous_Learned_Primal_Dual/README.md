# Continuous Learned Primal Dual

Scripts that reproduce, with LION's current experiments, the paper

> Runkel, Christina, Ander Biguri, and Carola-Bibiane Schönlieb. "Continuous
> Learned Primal Dual." *IEEE International Workshop on Machine Learning for
> Signal Processing (MLSP)*, 2024. [arXiv:2405.02478](https://arxiv.org/abs/2405.02478)

cLPD replaces the convolutional blocks of Learned Primal-Dual (LPD) by neural
ODEs. The paper trains cLPD and LPD on six CT problems from LIDC-IDRI and
compares them with each other and with FBP, by SSIM and PSNR (its Table 1).

These scripts follow **the code that trained the paper's models** (June
2024), where it and the paper's text differ: the code is what produced the
results.

| Problem (paper) | `--experiment` | LION experiment, with the settings of June 2024 |
|---|---|---|
| 1. Full angle, clinical dose | `full_low` | `LowDoseCTRecon` (I0 = 3500) |
| 2.a Full angle, extremely low dose | `full_extreme_low` | `ExtremeLowDoseCTRecon` (I0 = 1000) |
| 2.b Sparse angle, clinical dose | `sparse_low` | `SparseAngleLowDoseCTRecon` (I0 = 3500) |
| 2.c Sparse angle, extremely low dose | `sparse_extreme_low` | `SparseAngleExtremeLowDoseCTRecon` (I0 = 1000) |
| 3.a Limited angle, clinical dose | `limited_low` | `LimitedAngleLowDoseCTRecon` (I0 = 3500) |
| 3.b Limited angle, extremely low dose | `limited_extreme_low` | `LimitedAngleExtremeLowDoseCTRecon` (I0 = 1000) |

## Files

- `settings.py`: the problems, made as in June 2024
  (`settings.experiment(name)`: LION's default geometry then, the relative
  noise model with its doses, 5 slices per patient chosen and split as then),
  the two models (`cLPD.continous_LPD_paper()`, `LPD.continous_LPD_paper()`)
  and the training.
- `train.py`: trains one model on one problem.
- `evaluate.py`: tests both models and FDK on one problem's test set: the
  table (mean ± std of SSIM and PSNR), box plots and images.

## The settings, from the code of June 2024

- **Models.** cLPD: 10 iterations of first-order ODE blocks (4 layers, 32
  channels, PReLU, no biases) with instance normalisation, learned positive
  step sizes. LPD: 5 iterations of 4-layer blocks, no biases, no instance
  normalisation, learned positive step sizes. (The paper says LPD used the
  original LPD's parameters; the code didn't.)
- **Training.** Adam (learning rate 1e-4, betas 0.9 and 0.99), batch size 2,
  100 epochs, the MSE; validated every epoch with it. The models tested are
  the final ones.
- **Problems.** The paper's "clinical dose" was LION's low dose then (I0 =
  3500 photons), its "extremely low dose" I0 = 1000, with the relative noise
  model, 5 photons of electronic noise and 5% cross-talk. The geometry was
  LION's default then: fan beam, 512x512 pixels over 300 mm, a 900-pixel
  detector; 360 angles, 50 for sparse angle, 60 over 60 degrees for limited
  angle. Five slices per patient, half with nodules, patients split 80/10/10.

## Running

All twelve trainings, then the six evaluations (on a GPU machine with the
data; each training takes days):

```bash
cd scripts/paper_scripts/Continuous_Learned_Primal_Dual
for experiment in full_low full_extreme_low sparse_low sparse_extreme_low \
                  limited_low limited_extreme_low; do
    for model in cLPD LPD; do
        python train.py --experiment $experiment --model $model
    done
    python evaluate.py --experiment $experiment
done
```

A short run first, to check everything works:
`python train.py --experiment full_low --model cLPD --epochs 1 --slices 20`.

Results go to `$LION_RESULTS_PATH/paper_scripts/cLPD/<experiment>/`. A
stopped training resumes from its latest checkpoint when run again.

## What can't be reproduced exactly

- **The data's pixel spacing.** In 2024 the slices kept each scan's own pixel
  spacing; LION's pre-processing now resamples them all to 500 mm over 512
  pixels, and the 300 mm geometry of 2024 then doesn't match them (LION warns).
  With the relative noise model the noise doesn't depend on the patient's
  size, but the images are not the same. A copy of the data pre-processed
  before October 2026, if there is one, gives the 2024 slices: `--data-folder
  <that folder>`.
- **The noise's three old errors** (cross-talk across angles, the maximum of
  the whole batch, see `ct_utils.sinogram_add_noise`) were fixed and are not
  kept, so the noise is close to, not identical with, 2024's.
- **Random numbers** (initial weights, noise, shuffling) differ from 2024's.
- **Metrics** are LION's (`LION.evaluation`), on images normalised by the
  test set's range.

## The paper's Table 1, for reference

| Problem | cLPD SSIM | cLPD PSNR | LPD SSIM | LPD PSNR |
|---|---|---|---|---|
| 1. Full, clinical | 0.6140 ± 0.1263 | 34.18 ± 3.15 | 0.6157 ± 0.1245 | 34.12 ± 3.14 |
| 2.a Full, extremely low | 0.5773 ± 0.1287 | 32.37 ± 2.58 | 0.5790 ± 0.1251 | 32.23 ± 2.52 |
| 2.b Sparse, clinical | 0.5627 ± 0.1269 | 31.26 ± 2.30 | 0.5571 ± 0.1169 | 30.84 ± 2.18 |
| 2.c Sparse, extremely low | 0.5316 ± 0.1232 | 29.67 ± 1.95 | 0.5265 ± 0.1185 | 29.26 ± 1.89 |
| 3.a Limited, clinical | 0.4465 ± 0.1099 | 24.40 ± 1.69 | 0.3951 ± 0.0937 | 22.47 ± 1.71 |
| 3.b Limited, extremely low | 0.4371 ± 0.1081 | 24.02 ± 1.67 | 0.3823 ± 0.0933 | 22.25 ± 1.69 |
