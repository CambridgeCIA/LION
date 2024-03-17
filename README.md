# LION: AI tools for learned tomographic reconstruction
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

University of Cambridge Computational Image Analysis (CIA) groups AI tools for tomographic reconstruction, LION (Learned Iterative Optimization Networks)


**WARNING** Building in progress. This is a tool in development in very early stages. 
Many things are bound to fail and many more are bound to change. If you want to help with development, send an email to Ander Biguri and/or open and issue or a discussion. 

Install: 

```
git clone https://github.com/CambridgeCIA/LION.git
cd LION
git submodule update --init --recursive
conda env create --file=env.yml
conda activate LION
pip install .
```

Optional, if you want pre-commits. 
Install pre commits for auto-formating your commits.
Highly suggested if you want reproducibility, this will auto-save changes in your conda enviroments and will update your conda enviroment when pulling

```
conda activate LION
pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type post-merge
```

If you want the MS-D networks installed, you need to 
```
cd ./LION/models/CNNs/MS-D
pip install .
```
We are working on reimplementing MS-D due to the repo bein obsolete.

# Datasets

Currently there are several DataSets supported by LION. LION automatically knows where these are, but currently it only works for people working on the servers of CMS at University of Cambridge. To make it work in somewhere else, you just want to change LION/utils/paths.py line 7 `LION_DATA_PATH = pathlib.Path("/store/LION/datasets/")` to the actual location in your system. We are working on figuring out how to make LION flexible at install. 

NOTE: If you are part of CIA at DAMTP, you already have access to these datasets and you don't need to follow any instruction to dowload them.

The supported Datasets are:

## 2DeteCT

A dataset of uCT scans with different settings for sinograms. The data is made of composition of different materials. [Read more in the Nature paper](https://www.nature.com/articles/s41597-023-02484-6).
To install 2DeteCT, run `python ./LION/data_loaders/2DeteCT/dowload.py` and once finished (probably a few hours later) run `python ./LION/data_loaders/2DeteCT/pre_processs_2deteCT.py`. Once finished, you should have `LION_DATA_PATH/processed/2detect` folder in your system. You can now delete `LION_DATA_PATH/raw/2detect` to save space if you want. 

## LIDC-IDRI

A dataset of 1100 lung 3D CT scans with segmentations and diagnostics of lung nodules. To dowload the dataset, [follow the instructions in the dataset webpage](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254) (requires a third party software). Put this dataset in `LION_DATA_PATH/raw/LIDC-IDRI` such that it contains a folder structure as:

```
LION_DATA_PATH/raw/LIDC-IDRI/
├── LIDC-IDRI
├── LIDC-IDRI_MetaData.csv
├── metadata.csv
└── tcia-diagnosis-data-2012-04-20.xls
```

Then, create a new conda enviroment with `conda env create -f ./LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri_environment.yml`, activate it as `conda activate lidc_idri` and run the python file `python ./LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py`. 

You can now delete `LION_DATA_PATH/raw/LIDC-IDRI` to save space if you want. 

## Walnuts
A dataset of uCT scans of walnuts. [Read more in the Nature paper](https://www.nature.com/articles/s41597-023-02484-6).
You can dowload it using `python LION/data_loaders/walnuts/dowload_walnuts.py`
pre-processing and Data Loading WIP


# Developers:
Read `developers.md`


## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AnderBiguri"><img src="https://avatars.githubusercontent.com/u/11854388?v=4?s=100" width="100px;" alt="Biguri"/><br /><sub><b>Biguri</b></sub></a><br /><a href="https://github.com/CambridgeCIA/AItomotools/commits?author=AnderBiguri" title="Code">💻</a> <a href="#design-AnderBiguri" title="Design">🎨</a> <a href="#ideas-AnderBiguri" title="Ideas, Planning, & Feedback">🤔</a> <a href="#tutorial-AnderBiguri" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://emilienvalat.net/"><img src="https://avatars.githubusercontent.com/u/46785587?v=4?s=100" width="100px;" alt="Emilien Valat"/><br /><sub><b>Emilien Valat</b></sub></a><br /><a href="https://github.com/CambridgeCIA/AItomotools/commits?author=Emvlt" title="Code">💻</a> <a href="#design-Emvlt" title="Design">🎨</a> <a href="#ideas-Emvlt" title="Ideas, Planning, & Feedback">🤔</a> <a href="#data-Emvlt" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://ferdiasherry.com"><img src="https://avatars.githubusercontent.com/u/12610714?v=4?s=100" width="100px;" alt="Ferdia"/><br /><sub><b>Ferdia</b></sub></a><br /><a href="#design-fsherry" title="Design">🎨</a> <a href="#tool-fsherry" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/michi217"><img src="https://avatars.githubusercontent.com/u/62284237?v=4?s=100" width="100px;" alt="michi217"/><br /><sub><b>michi217</b></sub></a><br /><a href="https://github.com/CambridgeCIA/AItomotools/commits?author=michi217" title="Code">💻</a> <a href="#data-michi217" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/OliverCoughlan"><img src="https://avatars.githubusercontent.com/u/39098447?v=4?s=100" width="100px;" alt="Oliver Coughlan"/><br /><sub><b>Oliver Coughlan</b></sub></a><br /><a href="https://github.com/CambridgeCIA/AItomotools/commits?author=OliverCoughlan" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
