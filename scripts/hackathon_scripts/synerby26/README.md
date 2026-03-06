This repository contains a complete tutorial for generating synthetic foam data and its subsequent reconstruction and denoising using the Noise2Inverse algorithm with the LION library.

Tutorial Structure
The workflow is divided into two main steps to ensure version compatibility:

  - Creation of analytical bubble structures (Foam_env).
  - Training a neural network for self-supervised denoising (lion_env).

Create the environment with the specific version of Python for generating the Dataset, activate the enviroment and instal the necessary stuff:
conda create -n foam_env python=3.9
conda activate foam_env
pip install foam-ct-phantom h5py numpy torch

For installing LION:
git clone https://github.com/CambridgeCIA/LION.git
cd LION
git submodule update --init --recursive            # Legacy: for MSD_pytorch_
conda env create --file=env_base.yml --name=lion   # You can change 'lion' to a different env name
conda activate lion
pip install .
