"""Evaluating and comparing reconstruction methods.

- ``Evaluation``: the standard evaluation of one problem (test data, operator,
  metrics); ``evaluation.run(method)`` gives a test record (per-sample scores
  with sample IDs, a summary), saved and reused.
- ``method(x, operator)``: a model, a model's file, a solver, a reconstructor
  or a function as one call, measurements -> images.
- ``Comparison``: several methods on one evaluation, against a baseline,
  after checking from the models' records that the comparison is fair (no
  test patient trained on; the same problem; the same code).
- ``baseline(operator)``: the classical reconstruction of the modality (FDK
  for CT), which methods are compared with.
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
from LION.evaluation.methods import baseline, inverse, method

__all__ = [
    "Comparison",
    "Evaluation",
    "baseline",
    "default_metrics",
    "inverse",
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
