# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Evaluating and comparing methods on an experiment's test data.

- ``Evaluation``: the test data, operator and metrics (PSNR, SSIM, HaarPSI)
  of one problem. ``evaluation.run(method)`` scores a method on every test
  sample and returns a test record (per-sample scores, a summary), saved and
  reused: running it again reads it.
- ``Comparison``: several methods on one evaluation, against a baseline (FDK
  for CT). It first checks from the models' records that the comparison is
  fair: no model trained or validated on a test patient, the same problem,
  the same code. ``comparison.save(folder)`` writes what a paper needs: tables
  (markdown, LaTeX, CSV), per-sample scores, box plots and example images.

A method is a model, a model's file, a solver, a reconstructor (or its file)
or a function ``f(measurements, operator)``. This script compares the models
of 01, 02 and 04 (run those first) with SIRT, a classical method.

Needs: a GPU, the processed LIDC-IDRI data in ``LION_DATA_PATH``, and
``LION_RESULTS_PATH``.
"""

# %% Imports
import functools

from LION.evaluation import Comparison, Evaluation, summary
from LION.experiments import ct_experiments
from LION.operators.ct import sirt
from LION.utils.paths import results_path

# %% Settings
TEST_SLICES = 50  # evenly spaced over the test set; None: all of them

examples = results_path("examples")

# %% The evaluation
experiment = ct_experiments.LowDoseCTRecon()
evaluation = Evaluation(
    experiment, samples=TEST_SLICES, save_folder=examples / "evaluations"
)

# %% One method
record = evaluation.run(examples / "supervised" / "FBPConvNet")
for metric, values in summary(record).items():
    print(f"FBPConvNet {metric}: {values['mean']:.3f} ± {values['std']:.3f}")

# %% A comparison
methods = {
    "FBPConvNet": examples / "supervised" / "FBPConvNet",
    "LPD": examples / "lpd" / "LPD",
    "PnP-ADMM": examples / "pnp" / "PnP.json",
    # a function of (measurements, operator); a partial, so it can be recorded
    "SIRT": functools.partial(sirt, num_iterations=100),
}
comparison = Comparison(evaluation, methods)  # FDK is the baseline
comparison.run()
print(comparison.table())

# Tables, scores, box plots, the best, median and worst slices (by the
# baseline's PSNR), with a zoomed crop, and the differences to the targets.
comparison.save(examples / "comparison", zoom=[(200, 200, 100, 100)], differences=True)
# Made again, from its folder: Comparison.instantiate(examples / "comparison")
