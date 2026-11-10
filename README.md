# LION: Learned Iterative Optimization Networks
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-13-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://cambridgecia.github.io/LION/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

LION is a PyTorch library for learned reconstruction in tomography and other
inverse problems, made for research by the Cambridge Image Analysis (CIA) group
at the University of Cambridge. It stays close to plain torch: models are torch
modules, operators are functions of torch tensors, solvers are torch training
loops, and any piece can be your own. Around that, it keeps what research
needs to be repeated and compared:

- **experiments**: the geometry, noise and data of a problem, the same for
  every method;
- **records** saved with every model: the data it saw, how it was trained, the
  software it ran on, so it can be trained again or checked;
- **evaluation and comparison** of methods on the same test data, with checks
  that the comparison is fair, and the tables and figures a paper needs;
- **publishing** a trained model as a package anyone can load, test or train
  again.

**Documentation: https://cambridgecia.github.io/LION/** (tutorials, the
methods in LION, the API reference).

LION is approaching its first release (v1.0); until then its API may change.

## Install

```bash
git clone https://github.com/CambridgeCIA/LION.git
cd LION
conda env create --file=env_base.yml --name=lion   # or another name
conda activate lion
pip install .
```

With mamba, use `mamba env create --file=env_base.yml --name=lion` and `mamba activate lion` instead.

Everything compiled, torch included, comes from conda (`env_base.yml`), so all of it uses the same C++ runtime. Don't install torch with pip on top: a pip torch loads the system's C++ runtime, and conda's libraries (scipy, ...) then fail with `CXXABI_... not found` on machines with an older system. Create the environment fresh from `env_base.yml` rather than updating an old one.

- **CUDA version**: set by `cuda-version` in `env_base.yml` (12.9). Any CUDA 12 driver runs it (`nvidia-smi` shows the driver's CUDA version).
- **No GPU on the machine where you create the environment** (e.g. a login node): conda cannot see a CUDA driver and refuses `pytorch-gpu`. Tell it which CUDA to assume: `CONDA_OVERRIDE_CUDA=12.9 conda env create --file=env_base.yml --name=lion`.

To develop LION, install it editable with the development tools, and the pre-commit hooks (Black formatting):

```bash
pip install -e ".[dev]"
pre-commit install --hook-type pre-commit --hook-type post-merge
```

## A first training

```python
import torch
from torch.utils.data import DataLoader

from LION.experiments import ct_experiments
from LION.models.post_processing.FBPConvNet import FBPConvNet
from LION.optimizers.SupervisedSolver import SupervisedSolver

# the problem: low-dose CT of LIDC-IDRI slices (geometry, noise, data)
experiment = ct_experiments.LowDoseCTRecon()
training_loader = DataLoader(experiment.get_dataset("train"), batch_size=4, shuffle=True)

# a model, made from the problem's geometry, and a torch optimizer and loss
model = FBPConvNet(experiment.operator_params)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

solver = SupervisedSolver(model, optimizer, torch.nn.MSELoss())
solver.set_training(training_loader)
solver.set_saving("results/fbpconvnet", "FBPConvNet")
solver.train(100)
solver.save_final_results()  # the weights, and the records to train it again
```

Then evaluate and compare it (`LION.evaluation`), reconstruct with it (`LION.reconstructors`) or publish it (`LION.package`): the [tutorials](https://cambridgecia.github.io/LION/tutorials/index.html) go through each, and [`scripts/example_scripts`](scripts/example_scripts) has complete scripts, one feature each.

## Data and results

LION finds datasets in `LION_DATA_PATH`; set it, e.g. in your `.bashrc`: `export LION_DATA_PATH=/path/to/Datasets`. The supported datasets are LIDC-IDRI and 2DeteCT: [how to download and prepare them](LION/data_loaders/README.md).

`LION_RESULTS_PATH` (optional) is where results go when no folder is given. Paths inside either folder are written relative to it in LION's saved files, so the files mean the same on any machine.

NOTE: In CIA at DAMTP, the datasets are already there: `export LION_DATA_PATH=/store/cia-lion/datasets`, and set `LION_RESULTS_PATH` to your own user folder.

## Choosing a GPU

LION uses the GPU if there is one. Set another once, at the top of your script, and LION uses it wherever no device is given (solvers, loading models, datasets, noise simulation):

```python
import torch
torch.cuda.set_device(2)            # or: torch.set_default_device("cuda:2")
```

On machines with several GPUs where only one may be used, the most robust option is to make only that GPU visible to the process, so nothing (LION, PyTorch or ASTRA) can use another: `CUDA_VISIBLE_DEVICES=2 python my_script.py`.

## What is where

| Folder | What it has |
|---|---|
| `LION/operators/` | Forward operators: `ct/` (the CT geometry and projector, noise, FDK, SIRT, TV), `pcm/` (photocurrent mapping), wavelets, ... |
| `LION/data_loaders/` | Datasets (LIDC-IDRI, 2DeteCT) and their records |
| `LION/experiments/` | Experiments: a problem's geometry, noise and data |
| `LION/models/` | Models, by kind: post-processing, unrolled, learned FBP, CNNs and denoisers, PnP, learned regularisers |
| `LION/optimizers/` | Solvers: supervised, self-supervised, denoiser and adversarial training |
| `LION/reconstructors/` | Reconstructing with trained models: forward, PnP, learned regulariser |
| `LION/classical_algorithms/` | Conjugate gradient, FISTA, SPGL1, for any operator |
| `LION/evaluation/`, `LION/metrics/` | Evaluation and comparison; PSNR, SSIM, HaarPSI |
| `LION/losses/` | Losses and penalties on models |
| `LION/package.py` | Publishing and loading trained models |
| `demos/` | The tutorials |
| `scripts/` | Example scripts, and scripts that reproduce papers |
| `docs/` | The documentation site |

Which methods LION has, and their papers: [Methods in LION](https://cambridgecia.github.io/LION/methods.html). How models record their making, and how to make them again: [`metadata.md`](metadata.md).

## Not only CT

LION's general code uses general words, as an operator can be anything. For CT, they mean:

| LION says | In CT |
|---|---|
| `operator_params` (an `OperatorParameters`) | the geometry (`Geometry`) |
| the operator (`Operator`, `operator_params.make_operator()`) | the projector (`CTProjectionOp`) |
| domain (`op.domain_shape`) | the image (`geometry.image_shape`) |
| range (`op.range_shape`), measurements | the sinogram |
| the operator's default reconstruction | FDK |

So a CT model is made from its geometry, e.g. `LPD(geometry)`, and a solver takes `operator_params=geometry`. [More about operators](LION/operators/_README.rst).

## Contributing

Read the [contributor guide](contributor-guide.md) before changing code: it has LION's conventions. Run the tests with `pytest pytests/`; tests that need CUDA or tomosipo/ASTRA are skipped on machines without them, and the same tests run on every pull request.

## Citing LION

A paper is to come; until then, please cite the repository:

```bibtex
@software{LION,
  title  = {{LION}: Learned Iterative Optimization Networks},
  author = {Biguri, Ander and others},
  url    = {https://github.com/CambridgeCIA/LION},
}
```

and the papers of the methods you use (`Model.cite()`).

## Licence

LION is released under the GNU General Public License v3 (GPL-3), see [LICENSE.txt](LICENSE.txt). It has to be: its CT backend, [ASTRA](https://github.com/astra-toolbox/astra-toolbox) and [tomosipo](https://github.com/ahendriksen/tomosipo), are GPL-3.

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
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cslht"><img src="https://avatars.githubusercontent.com/u/263871389?v=4?s=100" width="100px;" alt="cslht"/><br /><sub><b>cslht</b></sub></a><br /><a href="https://github.com/CambridgeCIA/LION/commits?author=cslht" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
