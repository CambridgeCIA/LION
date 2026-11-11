# Continuous Learned Primal Dual

Scripts that reproduce, with LION's current experiments, the paper

> Runkel, Christina, Ander Biguri, and Carola-Bibiane Schönlieb. "Continuous
> Learned Primal Dual." *IEEE International Workshop on Machine Learning for
> Signal Processing (MLSP)*, 2024. [arXiv:2405.02478](https://arxiv.org/abs/2405.02478)

cLPD replaces the convolutional blocks of Learned Primal-Dual (LPD) by neural
ODEs. The paper trains cLPD and LPD on six CT problems from LIDC-IDRI and
compares them with each other and with FBP, by SSIM and PSNR (its Table 1).

| Problem (paper) | `--experiment` | LION experiment |
|---|---|---|
| 1. Full angle, clinical dose | `full_clinical` | `clinicalCTRecon` |
| 2.a Full angle, extremely low dose | `full_extreme_low_dose` | `ExtremeLowDoseCTRecon` |
| 2.b Sparse angle, clinical dose | `sparse_clinical` | `SparseAngleCTRecon` |
| 2.c Sparse angle, extremely low dose | `sparse_extreme_low_dose` | `SparseAngleExtremeLowDoseCTRecon` |
| 3.a Limited angle, clinical dose | `limited_clinical` | `LimitedAngleCTRecon` |
| 3.b Limited angle, extremely low dose | `limited_extreme_low_dose` | `LimitedAngleExtremeLowDoseCTRecon` |

## Files

- `settings.py`: the problems, the two models with the paper's parameters
  (`cLPD.continous_LPD_paper()`, `LPD.continous_LPD_paper()`), and the
  paper's training (Adam, learning rate 1e-4, batch size 2, 100 epochs, MSE).
- `train.py`: trains one model on one problem.
- `evaluate.py`: tests both models and FDK on one problem's test set: the
  table (mean ± std of SSIM and PSNR), box plots and images.

## Running

All twelve trainings, then the six evaluations (on a GPU machine with the
data; each training takes days):

```bash
cd scripts/paper_scripts/Continuous_Learned_Primal_Dual
for experiment in full_clinical full_extreme_low_dose sparse_clinical \
                  sparse_extreme_low_dose limited_clinical limited_extreme_low_dose; do
    for model in cLPD LPD; do
        python train.py --experiment $experiment --model $model
    done
    python evaluate.py --experiment $experiment
done
```

A short run first, to check everything works:
`python train.py --experiment full_clinical --model cLPD --epochs 1 --slices 20`.

Results go to `$LION_RESULTS_PATH/paper_scripts/cLPD/<experiment>/`. A
stopped training resumes from its latest checkpoint when run again.

## What differs from the paper

- **The data are simulated differently.** LION's CT experiments changed in
  October 2026 (see `LION/experiments/ct_experiments.py`): a physical noise
  model, doses chosen for it, and slices resampled to their real size with a
  500 mm geometry. The paper used the earlier relative noise model and
  geometry. Results are not comparable number for number with the paper's
  Table 1; the comparison between cLPD, LPD and FDK is what reproduces.
- **The paper doesn't give** the Adam betas (here 0.9, 0.99, as the original
  script) or how the final model was chosen: here, the model after 100 epochs
  (`--best` for the one with the best validation PSNR).
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
