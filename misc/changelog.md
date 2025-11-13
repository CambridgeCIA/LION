Add Plug-and-Play ADMM and update dependencies

Dependency update was tested on the following scripts:
- `equivariant_imaging.py`
- `GaussianDenoiser.py`
- `gradient_step_denoiser.py`
- `PnP_ADMM.py`
- `SURE.py` - up to `NotImplementedError`
- `ContinuousLPD.py`

Dependency update was not tested on `Noise2Inverse.py` as `msd_pytorch` is not installed due to conflicts with newer packages.
The latest `msd_pytorch` 0.10.1 only supports `python` up to 3.9 and `pytorch` up to 1.8.1 which lacks support for any CUDA capability beyond `sm_70` (any GPU released after 2019).
[TODO: Fork and update `msd_pytorch` or rewrite `MSD_pytorch.py` to remove this dependency.]

Main changes:
- `conjugate_gradient.py`, `admm_algorithm` for `PnP` reconstructor, `PnP_ADMM.py` script
- `env_base.yml` and `pyproject.toml`
- fix gradient step denoising: `grad` fails because of `torch.no_grad():`
- fix `torch.load` failing: allow setting `weights_only`

Minor changes:
-  add demo notebook to show usage of `to_autograd`
- `__init__.py` files with shortcuts for imports
-  add `from __future__ import annotations` line to support usage of ` | ` even with python<3.10
- some type-checking and small refactoring
- fix: `NumpyEncoder` fails when object is a numpy scalar
- add line to create `savefolder` if not existing
