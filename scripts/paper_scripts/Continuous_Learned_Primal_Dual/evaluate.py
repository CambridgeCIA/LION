# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Christina Runkel, Ander Biguri
# =============================================================================
"""Tests the cLPD paper's models: its Table 1 and Figure 2, for one problem or
all six.

cLPD and LPD (trained by ``train.py``) and FDK, the baseline, on the test set
of each problem, with SSIM and PSNR (mean ± standard deviation, as the paper's
Table 1), and the images of the best, median and worst slices:

    python evaluate.py --experiment full_low
    python evaluate.py --experiment all

Writes ``$LION_RESULTS_PATH/paper_scripts/cLPD/<experiment>/comparison``: the
table (markdown, LaTeX, CSV), every slice's scores, box plots and images, and
the test records; with ``all``, also ``table1.md``, the six tables together.
``--samples`` tests on fewer slices, evenly spaced; ``--best`` uses the models
with the lowest validation loss instead of the final ones (the paper tested
the final ones).

Needs a GPU, the processed LIDC-IDRI data in ``LION_DATA_PATH`` and
``LION_RESULTS_PATH``.
"""

# %% Imports
import argparse

import torch

from LION.evaluation import Comparison, Evaluation
from LION.utils.paths import results_path

import settings

# %% Settings
arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0])
arguments.add_argument(
    "--experiment", choices=[*settings.EXPERIMENTS, "all"], required=True
)
arguments.add_argument(
    "--samples", type=int, default=None, help="test slices (default: all)"
)
arguments.add_argument("--best", action="store_true", help="the best-validation models")
arguments.add_argument(
    "--data-folder", default=None, help="another processed LIDC-IDRI folder"
)
arguments.add_argument("--gpu", type=int, default=0)
arguments = arguments.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(arguments.gpu)
verbose = 1  # 0: nothing; 1: a progress bar per method
suffix = "_best" if arguments.best else ""


# %% One problem: its test set, the methods, the comparison
def evaluate(experiment_name):
    folder = settings.folder(experiment_name)
    experiment = settings.experiment(experiment_name, arguments.data_folder)
    evaluation = Evaluation(
        experiment, samples=arguments.samples, save_folder=folder / "evaluations"
    )
    # FDK, the baseline, is added by the comparison
    methods = {name: folder / f"{name}{suffix}" for name in settings.MODELS}
    comparison = Comparison(evaluation, methods)
    comparison.run(verbose=verbose)
    comparison.save(folder / "comparison", differences=True)
    return comparison.table()


# %% Evaluate what was asked for
experiments = (
    list(settings.EXPERIMENTS)
    if arguments.experiment == "all"
    else [arguments.experiment]
)
tables = []
for experiment_name in experiments:
    table = evaluate(experiment_name)
    print(f"\n{experiment_name}\n{table}")
    tables.append(f"## {experiment_name}\n\n{table}\n")
if arguments.experiment == "all":
    summary = results_path("paper_scripts/cLPD") / "table1.md"
    summary.write_text("# cLPD paper, Table 1\n\n" + "\n".join(tables))
    print(f"All six: {summary}")
