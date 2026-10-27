# =============================================================================
# This file is part of LION library
# License : GPL-3
# =============================================================================
"""Reconstruction methods as one call: measurements -> images.

``method(x, operator)`` turns anything that reconstructs into a function
``run(y)``, with ``run.record`` describing it (for test records and to know
when two evaluations ran the same method):

- a ``LIONmodel``: run as its solvers run it (``model_forward``: FDK first for
  image-domain models, and its normalisation);
- a model's file (``"model"``, ``.json``, ``.pt``): the model with its weights;
  if its training record names a solver with its own way of reconstructing
  (e.g. Noise2Inverse's), that solver's;
- a solver: its own reconstruction (``solver._forward``);
- a reconstructor (``PnP``, ``LearnedRegularizerReconstructor``): its
  ``reconstruct``;
- a function ``f(y, operator)``, e.g. ``fdk``, or ``functools.partial(tv_min,
  lam=...)``.

``baseline(operator)`` is the classical reconstruction methods are compared
with, which depends on the modality (one line each below).
"""

from __future__ import annotations

import hashlib
import inspect
import os
import pathlib

import torch

from LION.models.LIONmodel import LIONmodel
from LION.operators.Operator import Operator
from LION.optimizers.LIONsolver import LIONsolver, model_forward
from LION.optimizers.training_record import read_training_record, weights_sha256
from LION.reconstructors.LIONreconstructor import LIONReconstructor
from LION.utils import spec


def method(x, operator=None, *, allow=()):
    """``x`` as one call ``run(y) -> images`` (see the module docstring), with
    ``run.record`` describing it. ``operator``: the operator models and
    functions need (e.g. the CT operator for FDK)."""
    if isinstance(x, (str, os.PathLike)):
        return _from_file(pathlib.Path(x), operator, allow)
    if isinstance(x, LIONsolver):
        return _make(
            x._forward,
            {
                "kind": "solver",
                "solver": spec.import_path(type(x)),
                "model": _model_identity(x.model),
            },
            x.model,
        )
    if isinstance(x, LIONmodel):
        return _make(
            lambda y: model_forward(x, y, operator),
            {"kind": "model", "model": _model_identity(x)},
            x,
        )
    if isinstance(x, LIONReconstructor):
        modules = [
            v
            for v in vars(x).values()
            if isinstance(v, torch.nn.Module) and not isinstance(v, Operator)
        ]
        return _make(
            x.reconstruct,
            {
                "kind": "reconstructor",
                "reconstructor": spec.encode_or_name(x, "the reconstructor"),
                "models": [_model_identity(m) for m in modules],
            },
            *modules,
        )
    if callable(x):
        return _make(
            lambda y: x(y, operator),
            {
                "kind": "function",
                "function": spec.encode_or_name(x, "the function"),
                "source_sha256": _source_hash(x),
            },
        )
    raise TypeError(f"{x!r} is not a model, solver, reconstructor or function")


def _make(run, record, *modules):
    """``run`` without gradients, with its modules in evaluation mode."""

    def evaluate(y):
        for module in modules:
            module.eval()
        with torch.no_grad():
            return run(y)

    evaluate.record = record
    evaluate.modules = modules  # moved to the evaluation's device
    return evaluate


def _source_hash(function):
    """A hash of a function's source code (for a partial, its function's), so
    a saved result isn't reused after the code changed. None if unavailable."""
    function = getattr(function, "func", function)
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _model_identity(model) -> dict:
    return {
        "spec": spec.encode_or_name(model, "the model"),
        "weights_sha256": weights_sha256(model),
    }


def _from_file(path, operator, allow):
    model, _, _ = LIONmodel.load(path, allow=allow, supress_warnings=True)
    try:
        solver_class = spec.resolve(
            read_training_record(path).solver["_target_"], allow
        )
    except (ValueError, spec.SpecError, KeyError, AttributeError):
        solver_class = None
    if solver_class is not None and solver_class._forward is not LIONsolver._forward:
        solver = LIONsolver.instantiate(path, model, allow=allow, verbose=False)
        made = method(solver, operator)
    else:
        made = method(model, operator)
    made.record["file"] = spec.encode(path.with_suffix(""))
    return made


# ---------------------------------------------------------------------------
# The baseline: the classical reconstruction of each modality


def inverse(y, operator):
    """The operator's own inverse (or pseudo-inverse), e.g. for the
    Walsh-Hadamard transform."""
    return operator.inverse(y)


def baseline(operator):
    """The classical reconstruction methods are compared with, for
    ``operator``'s modality, as a function ``f(y, operator)``:

    - CT (``CTProjectionOp``): FDK (FBP for parallel beam);
    - an operator with its own ``inverse()``: that;
    - otherwise there is no standard one: give it yourself.
    """
    from LION.classical_algorithms.fdk import fdk
    from LION.operators.CTProjectionOp import CTProjectionOp

    if isinstance(operator, CTProjectionOp):
        return fdk
    if (
        isinstance(operator, Operator)
        and type(operator).inverse is not Operator.inverse
    ):
        return inverse
    raise ValueError(
        f"no standard baseline for {type(operator).__name__}: give one, e.g. "
        "Comparison(..., baseline=my_reconstruction)"
    )
