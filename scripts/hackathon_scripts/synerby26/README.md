# Hackathon scripts: synerby26

This repository contains a complete tutorial for generating synthetic foam data
and its subsequent reconstruction and denoising using the Noise2Inverse
algorithm with the LION library.

## Tutorial Structure

The workflow is divided into two main steps to ensure version compatibility:

- Creation of analytical bubble structures (`foam_env`): `python generate_foam_dataset.py`.
- Training a neural network for self-supervised denoising (`lion`): `python noise2inverse_foam.py`.

Create the environment with the specific version of Python for generating the
dataset, activate the environment and install the necessary stuff
(foam_ct_phantom is not on PyPI):

```sh
conda create -n foam_env python=3.9
conda activate foam_env
conda install -c conda-forge foam_ct_phantom h5py numpy
```

The data's geometry is `GEOMETRY` in `generate_foam_dataset.py`, as LION's
`Geometry` takes it: any 2D fan- or parallel-beam geometry with a square image
(default: fan beam over 360°, 900 detector pixels). The script saves it as
`geometry.json` next to the data, and `noise2inverse_foam.py` reconstructs with it.
Each sample is a slice of the foam: `gt_XXXX.h5` holds the image (`image`: 0 in
bubbles, 1 in the material, attenuation per mm) and `sino_XXXX.h5` its sinogram
(`sinogram`: angles × detector pixels, in LION's conventions).

For installing LION:

```sh
git clone https://github.com/CambridgeCIA/LION.git
cd LION
conda env create --file=env_base.yml --name=lion
conda activate lion
pip install .
```

You can change `lion` to a different environment name in the `conda env create`
line.
