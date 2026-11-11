# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Christina Runkel, Ander Biguri
# =============================================================================
"""Tests the cLPD paper's models on one problem: its Table 1 row and Figure 2.

cLPD and LPD (trained by ``train.py``) and FDK, the baseline, on the test set
of the problem, with SSIM and PSNR (mean ± standard deviation, as the paper's
Table 1), and the images of the best, median and worst slices:

    python evaluate.py --experiment full_clinical

Writes ``$LION_RESULTS_PATH/paper_scripts/cLPD/<experiment>/comparison``: the
table (markdown, LaTeX, CSV), every slice's scores, box plots and images, and
the test records. ``--samples`` tests on fewer slices, evenly spaced;
``--best`` uses the models with the best validation PSNR instead of the final
ones.

Needs a GPU, the processed LIDC-IDRI data in ``LION_DATA_PATH`` and
``LION_RESULTS_PATH``.
"""

# %% Imports
import argparse

import torch

from LION.evaluation import Comparison, Evaluation

import settings

# %% Settings
arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0])
arguments.add_argument("--experiment", choices=settings.EXPERIMENTS, required=True)
arguments.add_argument(
    "--samples", type=int, default=None, help="test slices (default: all)"
)
arguments.add_argument("--best", action="store_true", help="the best-validation models")
arguments.add_argument("--gpu", type=int, default=0)
arguments = arguments.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(arguments.gpu)
verbose = 1  # 0: nothing; 1: a progress bar per method

folder = settings.folder(arguments.experiment)
suffix = "_best" if arguments.best else ""

# %% The test set of the problem, and the methods
experiment = settings.EXPERIMENTS[arguments.experiment]()
evaluation = Evaluation(
    experiment, samples=arguments.samples, save_folder=folder / "evaluations"
)
# FDK, the baseline, is added by the comparison
methods = {name: folder / f"{name}{suffix}" for name in settings.MODELS}
comparison = Comparison(evaluation, methods)

# %% Run, and save the table and figures
comparison.run(verbose=verbose)
print(comparison.table())
comparison.save(folder / "comparison", differences=True)
