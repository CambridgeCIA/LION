# EXAMPLE SCRIPTS

This folder contains a series of full example scripts. Tutorial-like demos will instead be in /demos folder.
These scripts are full examples of using various algorithms. They are full programs, not modules:
run them from the repository root, e.g. `python scripts/example_scripts/LPD.py`.

All of them need a CUDA-capable machine, because LION's CT operators use tomosipo, which projects with
ASTRA's CUDA algorithms. Most of them also need a dataset (LIDC-IDRI or 2DeteCT, see the data loaders),
and all of them write their models and figures to a `savefolder` defined at the top of the file, which
points at the machine where they were written: change it before running them.

## Supervised training

### LPD.py

This file showcases how to train the Learned Primal Dual algorithm using LION.
You can however slightly modify this script (essentially just change the model) to use any supervised training model.

### Learned_FBP.py

Learned filtered back-projection: the same supervised training loop with the learned FBP networks
(`DeepFBPNetwork`, `DeepFusionBPNetwork` and `FusionFBPNetwork`), which learn the filter of an FBP instead
of an iterative solver.

## Self-supervised training

These methods train from the measurements alone, so they need no ground-truth images.

### Noise2Inverse.py

This implements the Noise2Inverse self-supervised solver. It uses the MSD-net as in its original paper,
but any other model should also work.

### SURE.py

Another self-supervised method: it trains with Stein's Unbiased Risk Estimate (`LION.losses.SURE`) as the
loss, which estimates the risk without clean data. It uses a UNet, but any other model should also work.

### Sparse2Inverse.py

Sparse2Inverse, which splits the measured angles into groups and reconstructs each group from the others.
It uses a UNet, and its solver accepts any number of splits of 3 or more.

### equivariant_imaging.py

Another self-supervised method. It uses the concept of equivariance to train a model: the reconstruction
must be the same if the image, or the measurements, are transformed in a way the physics commutes with.

## Denoisers, for plug-and-play

Denoisers trained on simulated Gaussian noise can be used as priors in iterative reconstruction, as in
`LION.reconstructors.PnP`.

### GaussianDenoiser.py

This example shows how to train a denoising network with Gaussian noise.
It will train a Gaussian denoiser on simulated Gaussian noise on the ground truth, rather than using the
noisy data from the data loader.
This exploits the idea that a simple Gaussian denoiser can be extrapolated to other applications, such as
CT reconstruction.
Ideally you want to use this denoiser as a prox operator or a regularizer in an iterative reconstruction
algorithm. It uses a DRUNet, and shows the trained denoiser inside a PnP reconstruction at the end.

### gradient_step_denoiser.py

The same, with a gradient-step denoiser (GSD) instead of a DRUNet: the network parameterises the gradient
of a regulariser, which makes the denoiser a proximal operator by construction. Also shown inside PnP at
the end.

## Plug-and-play reconstruction

### PnP_ADMM.py

Plug-and-play reconstruction with a pretrained denoising network: it takes a DRUNet pretrained on natural
images (downloaded by deepinv, so no CT dataset is needed for the prior) and uses it as the prior of an
ADMM reconstruction.

### photocurrent_mapping.py

Plug-and-play ADMM for Photocurrent Mapping (PCM) data, a different imaging modality that LION also
supports. It simulates subsampled PCM measurements from a high-resolution current map of a CIGS solar
cell, and compares a zero-filled pseudo-inverse, two compressed-sensing baselines (FISTA and SPGL1) and
PnP-ADMM with a pretrained DRUNet. It needs the PCM data in `data_dir`, at the top of the file.
