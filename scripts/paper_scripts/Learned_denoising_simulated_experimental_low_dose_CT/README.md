# Learned denoising with simulated and experimental low-dose CT data

> **Unverified.** These scripts have not been run on real data yet: they are
> checked only on a CPU with fake data and a toy operator. Their results are
> not yet known to match the paper's.

Scripts that reproduce the paper

> Kiss, Maximilian B., Ander Biguri, Carola-Bibiane Schönlieb, K. Joost
> Batenburg, and Felix Lucka. "Learned denoising with simulated and
> experimental low-dose CT data." *Scientific Reports* 16 (2025).
> [doi:10.1038/s41598-025-30457-9](https://doi.org/10.1038/s41598-025-30457-9)
> ([arXiv:2408.08115](https://arxiv.org/abs/2408.08115))

The paper compares deep-learning denoisers trained on **simulated** low-dose
data with ones trained on **experimental** low-dose data, on the 2DeteCT
dataset, and tests each on both kinds of noise:

- **simulated:** the high-dose mode 2 sinograms with added noise (Poisson and
  5% detector cross-talk), I0 = 200 for every slice;
- **experimental:** the measured low-dose mode 1 sinograms (tube current
  33.3 µA instead of 1000 µA);
- **target:** mode 2: its sinograms, or its reconstructions.

Eight models: a **U-Net** and an **MS-D network**, each as a **sinogram
denoiser** (sinogram to sinogram) and **end to end** (on the FDK of the
sinogram: FBP+U-Net, FBP+MS-D Net), each trained on simulated and on
experimental data. They are tested in the sinogram domain and in the
reconstruction domain (the FDK of the denoised sinograms, and the end-to-end
outputs), on both test sets: the paper's Table 2.

## Running everything

On a GPU machine with the processed 2DeteCT data (`LION_DATA_PATH`) and
`LION_RESULTS_PATH` set, from this folder:

```bash
cd scripts/paper_scripts/Learned_denoising_simulated_experimental_low_dose_CT

# 1. a short run, to check it works (minutes)
python train.py --model all --noise all --epochs 1 --slices 20
python evaluate.py --samples 10 --final

# 2. all eight trainings (hours to days each)
python train.py --model all --noise all

# 3. the evaluation: both domains, both test sets (Table 2)
python evaluate.py
```

Delete the short run's models (`$LION_RESULTS_PATH/paper_scripts/noise_paper/`)
before step 2: trained models are skipped. `train.py --model all --noise all`
trains one after the other; a stopped run resumes from the last checkpoint,
skipping the models already trained. To use several GPUs, one model per GPU:

```bash
python train.py --model UNet --noise all --gpu 0 &
python train.py --model MSDNet --noise all --gpu 1 &
python train.py --model FBP+UNet --noise all --gpu 2 &
python train.py --model FBP+MSDNet --noise all --gpu 3 &
```

Results go to `$LION_RESULTS_PATH/paper_scripts/noise_paper/`: the models
(each `<model>_<noise>` and `<model>_<noise>_best`, with their records),
`comparisons/<domain>_<test noise>/` (tables, per-slice scores, box plots,
images) and `tables.md`, all the tables together. The evaluation leaves out,
with a message, models not trained yet.

## Files

- `settings.py`: the data of each setting (`settings.dataset(noise, task,
  split)`), the models, the training, and the evaluation's methods (the noisy
  sinograms, FDK, the FDK of a denoised sinogram).
- `unet.py`: the paper's U-Net (see below).
- `train.py`: trains one model on one kind of noise, or all eight (`all`).
- `evaluate.py`: tests the trained models in both domains, on both test sets.

## The settings, from the code of 2024

From Max Kiss's training and evaluation scripts (PR #103).

- **Data:** 2DeteCT, flat- and dark-field corrected and log-transformed,
  split 80/10/10. Simulated noise: `deteCT.noise_parameters(I0=200,
  cross_talk=0.05)`.
- **Training** (every model): Adam (learning rate 1e-4, betas 0.9 and 0.99),
  batch size 1, 100 epochs, the mean squared error to the target; validated
  every 10 epochs with it. The model with the lowest validation loss is the
  one tested.
- **U-Net:** the paper's (`unet.py`, from #103's `UNet_3` and `FBPUNet`): four
  downsamplings, 64 to 512 channels, batch normalisation, padding for any
  size. Not LION's `UNet`, which has group normalisation, 32 to 512 channels,
  and needs sizes divisible by 16 (2DeteCT's sinograms are 956 wide).
- **MS-D network:** 100 layers of width 1, dilations 1 to 10 (LION's
  `MSDNet`).
- **End to end:** the network on the FDK of the sinogram, negative values set
  to zero.
- **Evaluation:** SSIM and PSNR of each slice with the target's own range
  (max - min); the sinogram denoisers' reconstructions are the FDK of their
  outputs, negative values kept.

## Differences

**With the paper's text** (the scripts follow the code):

- **The domain:** the paper says the networks denoise the beam intensity loss
  images, before the logarithm. The code set this at the experiment's level,
  which the 2024 data loader never read, so its networks denoised
  log-transformed sinograms; so do these scripts.
- **Not stated in the paper:** the learning rate, batch size and loss, the
  architectures, and how often it validated.

**With the code of 2024:**

- **The simulated noise** is LION's 2DeteCT noise (`deteCT.add_sinogram_noise`,
  Poisson and cross-talk on the log-transformed sinogram). The 2024 data
  loader in #103 has no noise of its own: it was added by code not in the PR.
  Each slice's noise is drawn once (the same every epoch), as LION does; the
  2024 code's choice isn't known.
- **The MS-D network** was `msd_pytorch`'s in 2024; here it is LION's own
  implementation of the same architecture.
- **The models** are LION models (the U-Net: this folder's), trained by LION's
  solver and tested with `LION.evaluation` instead of the hand-written loops
  of 2024: the same optimisation and metrics.
- **Random numbers** (initial weights, noise, shuffling) differ from 2024's.

**Not scripted:** Table 1 (the noise level's choice, I0 from 200 to 350) and
the figures.

## The paper's Table 2, for reference

SSIM and PSNR (dB), on the experimental (Exp) and simulated (Sim) test sets.

| Domain | Model | Trained on | SSIM, Exp | PSNR, Exp | SSIM, Sim | PSNR, Sim |
|---|---|---|---|---|---|---|
| sinogram | U-Net | experimental | 0.8126 | 18.50 | 0.8167 | 19.32 |
| sinogram | U-Net | simulated | 0.8273 | 33.46 | 0.8795 | 36.60 |
| sinogram | MS-D Net | experimental | 0.8613 | 36.22 | 0.8239 | 20.47 |
| sinogram | MS-D Net | simulated | 0.7512 | 16.32 | 0.8835 | 38.03 |
| reconstruction (FDK of the denoised) | U-Net | experimental | 0.6134 | 26.71 | 0.6273 | 27.53 |
| reconstruction (FDK of the denoised) | U-Net | simulated | 0.5504 | 28.33 | 0.6351 | 32.56 |
| reconstruction (FDK of the denoised) | MS-D Net | experimental | 0.5984 | 30.92 | 0.6152 | 28.30 |
| reconstruction (FDK of the denoised) | MS-D Net | simulated | 0.3854 | 11.64 | 0.6372 | 32.66 |
| reconstruction (end to end) | FBP+U-Net | experimental | 0.8161 | 29.84 | 0.7466 | 28.04 |
| reconstruction (end to end) | FBP+U-Net | simulated | 0.5957 | 26.88 | 0.6693 | 28.81 |
| reconstruction (end to end) | FBP+MS-D Net | experimental | 0.7829 | 32.07 | 0.7892 | 32.07 |
| reconstruction (end to end) | FBP+MS-D Net | simulated | 0.7615 | 30.62 | 0.8204 | 33.11 |
