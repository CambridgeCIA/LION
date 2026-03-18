# LION: AI tools for learned tomographic and other image reconstruction tasks
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-12-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](#black-code-style-)

University of Cambridge Computational Image Analysis (CIA) groups AI tools for image reconstruction (e.g. tomographic reconstruction), LION (Learned Iterative Optimization Networks)

The library is a place to gather and provide resources for image reconstruction, such as Computed Tomography reconstruction, using machine learning tools implemented in PyTorch.
It contains data dowloading and pre-processing, torch `DataSet` classes for the existing data, and `Experiments` class with implementation of default realistic experiments, several data-driven methods and models, and features to enhance reproduciblity.

**WARNING** Building in progress. This is a tool in development in very early stages.
Many things are bound to fail and many more are bound to change. If you want to help with development, send an email to Ander Biguri and/or open and issue or a discussion.

Install:

```bash
git clone https://github.com/CambridgeCIA/LION.git
cd LION
git submodule update --init --recursive            # Legacy: for MSD_pytorch_
conda env create --file=env_base.yml --name=lion   # You can change 'lion' to a different env name
conda activate lion
pip install .                                      # NOTE: change to `pip install -e ".[dev]"` if
                                                   # you want to contribute to development, see below
```

If you would like to contribute to the development of LION, you can replace the last line of the above set of commands by

```
pip install -e ".[dev]"
```

to make the installation editable (i.e. changes you make to the source will be visible when you restart the REPL or start a new Python process) and include additional development dependencies like `pre-commit`.
Afterwards, install the pre-commit hooks for auto-formating your commits. We use Black for code formatting (see [.pre-commit-config.yaml](.pre-commit-config.yaml)):

```
pre-commit install --hook-type pre-commit --hook-type post-merge
```

## Datasets

Currently there are several DataSets supported by LION. LION automatically knows where these are, but currently it only works for people working on the servers of CMS at University of Cambridge. To make it work in somewhere else, you just want to change [LION/utils/paths.py](LION/utils/paths.py) line 7 `LION_DATA_PATH = pathlib.Path("/store/LION/datasets/")` to the actual location in your system. We are working on figuring out how to make LION flexible at install.

NOTE: If you are part of CIA at DAMTP, you already have access to these datasets and you don't need to follow any instruction to dowload them.

The supported Datasets are `2DeteCT`, `LIDC-IDRI`,

[Read more about them here](LION/data_loaders/README.md)

## Models/Methods

LION supports all types of data-driven methods for CT reconstructions. They can, as a general taxonomy, be described as:

- Post-Processing methods: a "denoising" network. Takes a noisy recon and cleans it.
- Iterative Unrolled methods: Uses the operator to imitate iterative recon algorithms, but has learned parts.
- Learned regularizer: Explicitly learned regularization functions.
- Plug-and-Play (PnP): Implicit learned regularization, a regularization optimization step is learned, rather than an explicit one.

Folders for each of these exist in `LION/models`. An extra folder for standard `CNNs` is also available.

[Read more about which models are available in each class here](LION/models/README.md)

[Read more about which papers are implemented in LION here](papers_in_LION.md)

## Operators

Originally designed for tomographic reconstruction tasks, specifically Computed Tomography (CT),
LION has been extended to work with more operators.
[Read more about operators here](LION/operators/_README.md)

## Developers

Read [`developers.md`](developers.md)

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AnderBiguri"><img src="https://avatars.githubusercontent.com/u/11854388?v=4?s=100" width="100px;" alt="Biguri"/><br /><sub><b>Biguri</b></sub></a><br /><a href="https://github.com/CambridgeCIA/LION/commits?author=AnderBiguri" title="Code">💻</a> <a href="#design-AnderBiguri" title="Design">🎨</a> <a href="#ideas-AnderBiguri" title="Ideas, Planning, & Feedback">🤔</a> <a href="#tutorial-AnderBiguri" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://emilienvalat.net/"><img src="https://avatars.githubusercontent.com/u/46785587?v=4?s=100" width="100px;" alt="Emilien Valat"/><br /><sub><b>Emilien Valat</b></sub></a><br /><a href="https://github.com/CambridgeCIA/LION/commits?author=Emvlt" title="Code">💻</a> <a href="#design-Emvlt" title="Design">🎨</a> <a href="#ideas-Emvlt" title="Ideas, Planning, & Feedback">🤔</a> <a href="#data-Emvlt" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://ferdiasherry.com"><img src="https://avatars.githubusercontent.com/u/12610714?v=4?s=100" width="100px;" alt="Ferdia"/><br /><sub><b>Ferdia</b></sub></a><br /><a href="#design-fsherry" title="Design">🎨</a> <a href="#tool-fsherry" title="Tools">🔧</a> <a href="https://github.com/CambridgeCIA/LION/commits?author=fsherry" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/michi217"><img src="https://avatars.githubusercontent.com/u/62284237?v=4?s=100" width="100px;" alt="michi217"/><br /><sub><b>michi217</b></sub></a><br /><a href="https://github.com/CambridgeCIA/LION/commits?author=michi217" title="Code">💻</a> <a href="#data-michi217" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/OliverCoughlan"><img src="https://avatars.githubusercontent.com/u/39098447?v=4?s=100" width="100px;" alt="Oliver Coughlan"/><br /><sub><b>Oliver Coughlan</b></sub></a><br /><a href="https://github.com/CambridgeCIA/LION/commits?author=OliverCoughlan" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mbkiss"><img src="https://avatars.githubusercontent.com/u/78095730?v=4?s=100" width="100px;" alt="mbkiss"/><br /><sub><b>mbkiss</b></sub></a><br /><a href="#design-mbkiss" title="Design">🎨</a> <a href="https://github.com/CambridgeCIA/LION/commits?author=mbkiss" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ChristinaRunkel"><img src="https://avatars.githubusercontent.com/u/20678760?v=4?s=100" width="100px;" alt="ChristinaRunkel"/><br /><sub><b>ChristinaRunkel</b></sub></a><br /><a href="https://github.com/CambridgeCIA/LION/commits?author=ChristinaRunkel" title="Code">💻</a> <a href="https://github.com/CambridgeCIA/LION/issues?q=author%3AChristinaRunkel" title="Bug reports">🐛</a> <a href="#design-ChristinaRunkel" title="Design">🎨</a> <a href="#example-ChristinaRunkel" title="Examples">💡</a> <a href="#ideas-ChristinaRunkel" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-ChristinaRunkel" title="Research">🔬</a> <a href="#userTesting-ChristinaRunkel" title="User Testing">📓</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Zakobian"><img src="https://avatars.githubusercontent.com/u/46059070?v=4?s=100" width="100px;" alt="Zak Shumaylov"/><br /><sub><b>Zak Shumaylov</b></sub></a><br /><a href="https://github.com/CambridgeCIA/LION/commits?author=Zakobian" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://hyt35.github.io"><img src="https://avatars.githubusercontent.com/u/56555137?v=4?s=100" width="100px;" alt="Hong Ye Tan"/><br /><sub><b>Hong Ye Tan</b></sub></a><br /><a href="https://github.com/CambridgeCIA/LION/commits?author=hyt35" title="Code">💻</a> <a href="#design-hyt35" title="Design">🎨</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cshoebridge"><img src="https://avatars.githubusercontent.com/u/74095041?v=4?s=100" width="100px;" alt="Charlie Shoebridge"/><br /><sub><b>Charlie Shoebridge</b></sub></a><br /><a href="https://github.com/CambridgeCIA/LION/commits?author=cshoebridge" title="Code">💻</a> <a href="#design-cshoebridge" title="Design">🎨</a> <a href="#ideas-cshoebridge" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AndreaSainz"><img src="https://avatars.githubusercontent.com/u/185353475?v=4?s=100" width="100px;" alt="Andrea Sainz Bear"/><br /><sub><b>Andrea Sainz Bear</b></sub></a><br /><a href="https://github.com/CambridgeCIA/LION/commits?author=AndreaSainz" title="Code">💻</a> <a href="https://github.com/CambridgeCIA/LION/issues?q=author%3AAndreaSainz" title="Bug reports">🐛</a> <a href="#research-AndreaSainz" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/trung-vt"><img src="https://avatars.githubusercontent.com/u/65380717?v=4?s=100" width="100px;" alt="Thanh Trung Vu"/><br /><sub><b>Thanh Trung Vu</b></sub></a><br /><a href="https://github.com/CambridgeCIA/LION/commits?author=trung-vt" title="Code">💻</a> <a href="https://github.com/CambridgeCIA/LION/issues?q=author%3Atrung-vt" title="Bug reports">🐛</a> <a href="#research-trung-vt" title="Research">🔬</a> <a href="#design-trung-vt" title="Design">🎨</a></td>
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
