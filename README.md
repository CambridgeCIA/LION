# AI tools for learned tomographic reconstruction

University of Cambridge Computational Image Analysis (CIA) groups AI tools for tomographic reconstruction.

<<<<<<< HEAD
Building in progress.
=======

**WARNING** Building in progress. 
>>>>>>> 297b3d7d8821f31fbe0f025dc956fe45d8702aec

Install: (this is a temporary list, we will clean it up)

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install cudatoolkit=11.6 tomosipo tqdm pytorch matplotlib -c pytorch -c astra-toolbox -c aahendriksen -c defaults -c conda-forge
pip install git+https://github.com/ahendriksen/ts_algorithms.git
conda install scikit-image
conda install pip
/local/scratch/public/<your_username>/anaconda3/envs/<your_env_name>/bin/pip3 install pydicom
conda install natsort h5py
conda install -c simpleitk simpleitk
```


<<<<<<< HEAD
scripts: full fledged scripts that are reproducible, for e.g. data generation, or papers.
=======
# Developers:
Read `developers.md`
>>>>>>> 297b3d7d8821f31fbe0f025dc956fe45d8702aec
