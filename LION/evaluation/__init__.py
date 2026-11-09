# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Evaluating and comparing reconstruction methods.

- ``Evaluation``: the standard evaluation of one problem (test data, operator,
  metrics); ``evaluation.run(method)`` gives a test record (per-sample scores
  with sample IDs, a summary), saved and reused.
- ``method(x, operator)``: a model, a model's file, a solver, a reconstructor
  or a function as one call, measurements -> images.
- ``Comparison``: several methods on one evaluation, against a baseline,
  after checking from the models' records that the comparison is fair (no
  test patient trained on; the same problem; the same code).
  ``comparison.save(folder)`` writes it for a paper: tables (markdown, LaTeX,
  CSV), per-sample scores, box plots, differences to the baseline, example
  images (best/median/worst or chosen samples, zoom crops), and its spec
  (``Comparison.instantiate(folder)``).
- ``LION.evaluation.figures``: those figures as functions, in plain matplotlib
  (only imported there), and their data (``per_sample``,
  ``per_sample_differences``) for any other plot.
- ``baseline(operator)``: what methods are compared with, the operator's
  default reconstruction (FDK for CT, the inverse for an operator with one).
- ``scores``, ``summary``, ``table``: the same, as functions.
"""

from LION.evaluation.comparison import (
    Comparison,
    problem_differences,
    shared_samples,
)
from LION.evaluation.evaluation import (
    Evaluation,
    default_metrics,
    is_test_record,
    read_test_record,
    sample_ids,
    scores,
    summary,
    table,
)
from LION.evaluation.methods import baseline, method

__all__ = [
    "Comparison",
    "Evaluation",
    "baseline",
    "default_metrics",
    "is_test_record",
    "method",
    "problem_differences",
    "read_test_record",
    "sample_ids",
    "scores",
    "shared_samples",
    "summary",
    "table",
]
