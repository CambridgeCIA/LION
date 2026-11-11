# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Maximilian B. Kiss, Ander Biguri
# =============================================================================
"""Tests the 2DeteCT benchmark paper's methods: its Tables 5-7, for one
problem or all nine.

The classical methods (FBP, AGD, ChP) and the learned ones (trained by
``train.py``; the models with the lowest validation loss, as the paper) on the
test set of each problem, with SSIM and PSNR on each slice's own range
(mean ± standard deviation, as the paper's tables), and the images of the
best, median and worst slices:

    python evaluate.py --experiment full
    python evaluate.py --experiment all

Writes ``$LION_RESULTS_PATH/paper_scripts/2detect_benchmark/<experiment>/
comparison``: the table (markdown, LaTeX, CSV), every slice's scores, box
plots and images, and the test records; with ``all``, also ``tables.md``, the
nine tables together. Learned methods not trained yet are left out, with a
message. ``--samples`` tests on fewer slices, evenly spaced; ``--final`` uses
the models after the last epoch instead.

Needs a GPU, the processed 2DeteCT data in ``LION_DATA_PATH`` and
``LION_RESULTS_PATH``.
"""

# %% Imports
import argparse

import torch

from LION.evaluation import Comparison, Evaluation
from LION.metrics.psnr import PSNR
from LION.metrics.ssim import SSIM
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
arguments.add_argument("--final", action="store_true", help="the final models")
arguments.add_argument("--gpu", type=int, default=0)
arguments = arguments.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(arguments.gpu)
verbose = 1  # 0: nothing; 1: a progress bar per method
suffix = "" if arguments.final else "_best"


# %% One problem: its test set, the methods, the comparison
def evaluate(experiment_name):
    folder = settings.folder(experiment_name)
    experiment = settings.EXPERIMENTS[experiment_name]()
    # SSIM and PSNR, each on its target's own range (as the paper's)
    evaluation = Evaluation(
        experiment,
        metrics={"SSIM": SSIM(), "PSNR": PSNR()},
        normalisation="sample",
        samples=arguments.samples,
        save_folder=folder / "evaluations",
    )
    methods = settings.classical_methods(experiment_name)
    for name in settings.MODELS:
        file = folder / f"{name}{suffix}"
        if file.with_suffix(".json").exists():
            methods[name] = file
        else:
            print(f"{experiment_name}/{name}: not trained yet, left out")
    # every method in the table; none is the reference
    comparison = Comparison(evaluation, methods, baseline=False)
    comparison.run(verbose=verbose)
    comparison.save(folder / "comparison")
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
    summary = results_path("paper_scripts/2detect_benchmark") / "tables.md"
    summary.write_text("# 2DeteCT benchmark paper, Tables 5-7\n\n" + "\n".join(tables))
    print(f"All nine: {summary}")
