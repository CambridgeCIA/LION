# =============================================================================
# This file is part of LION library
# License : GPL-3
# =============================================================================
"""Comparing reconstruction methods on one evaluation.

``Comparison(evaluation, methods)`` runs each method, and a baseline, on the
same ``Evaluation``, after checking from the models' records that the
comparison is fair:

- **no leakage**: a model trained or validated on test data is an error. The
  check is by group where the data have groups (LIDC-IDRI's patients, whose
  slices are near-copies; 2DeteCT's physical samples), otherwise by sample;
- **the same problem**: a model trained on other data or noise (another
  experiment, dose, geometry) gives a warning naming the differences;
- **the same code**: a model trained with other LION code is listed.

Methods without a record (a model or reconstructor made in the script) can't
be checked, which is said; classical methods (functions) need no check. The
baseline is the experiment's (``Experiment.baseline()``) or the modality's
(``LION.evaluation.baseline``: FDK for CT).
"""

from __future__ import annotations

import json
import os
import pathlib
import warnings

from LION.data_loaders.data_record import read_record
from LION.evaluation.evaluation import _default_name, table
from LION.evaluation.methods import baseline as modality_baseline
from LION.evaluation.methods import method
from LION.models.LIONmodel import LIONmodel
from LION.optimizers.LIONsolver import LIONsolver
from LION.optimizers.training_record import read_training_record
from LION.reconstructors.LIONreconstructor import LIONReconstructor
from LION.utils import environment, spec
from LION.utils.parameter import LIONParameter

# Where the data were read from doesn't make another problem.
_NOT_THE_PROBLEM = ("folder", "datafolder")


class Comparison:
    """Several methods on one ``Evaluation``, against a baseline (see the
    module docstring).

    ``methods``: ``{name: method}``, or a list (a file is named by its file
    name, anything else as ``Evaluation`` names it); a method is anything
    ``LION.evaluation.method`` takes. ``baseline``: a method; None: the
    experiment's, else the modality's; False: no baseline. ``allow``: modules
    outside LION that model files may name.
    """

    def __init__(self, evaluation, methods, baseline=None, allow=()):
        self.evaluation = evaluation
        self.methods = methods
        self.baseline = baseline
        self.allow = allow
        self.records = None

    def named_methods(self) -> dict:
        """``{name: method}``, the baseline first."""
        named = {}
        found = self.get_baseline()
        if found is not None:
            named[_name_of(found, self.evaluation.get_operator())] = found
        methods = self.methods
        if not isinstance(methods, dict):
            methods = {
                _name_of(m, self.evaluation.get_operator(), self.allow): m
                for m in methods
            }
            if len(methods) < len(self.methods):
                raise ValueError("two methods have the same name: give them names")
        for name, m in methods.items():
            if name in named:
                raise ValueError(f"{name!r} is the baseline's name: name it otherwise")
            named[name] = m
        return named

    def get_baseline(self):
        """The baseline: the one given, the experiment's, or the modality's."""
        if self.baseline is False:
            return None
        if self.baseline is not None:
            return self.baseline
        experiment = self.evaluation.experiment
        found = experiment.baseline() if experiment is not None else None
        if found is not None:
            return found
        return modality_baseline(self.evaluation.get_operator())

    def check(self) -> list:
        """Checks the methods against the test data (see the module
        docstring): raises if a model was trained or validated on test data;
        returns what else differs, as sentences."""
        test = _as_dict(self.evaluation.data_record())
        errors, notes = [], []
        for name, m in self.named_methods().items():
            found = _records_of(m, self.allow)
            if found is None:
                if isinstance(m, (LIONmodel, LIONReconstructor)):
                    notes.append(
                        f"{name}: no record of its training, so it can't be "
                        "checked (give its file instead)"
                    )
                continue
            data, validation, code = found
            for role, record in (("trained", data), ("validated", validation)):
                if record is None:
                    continue
                shared, unit, note = shared_samples(test, record)
                if shared:
                    examples = ", ".join(shared[:3]) + (", ..." if shared[3:] else "")
                    errors.append(
                        f"{name} was {role} on {len(shared)} of the test data's "
                        f"{unit}: {examples}"
                    )
                if note:
                    notes.append(f"{name}: {note}")
            if data is None:
                notes.append(f"{name}: its file has no record of its training data")
            else:
                differences = problem_differences(test, data)
                if differences:
                    notes.append(
                        f"{name} was trained on another problem: "
                        + "; ".join(differences)
                    )
            if code is not None:
                notes.append(f"{name} was trained with other LION code: {code}")
        if errors:
            raise ValueError(
                "models tested on data they were trained on (by group: a "
                "patient for LIDC-IDRI, a physical sample for 2DeteCT):\n- "
                + "\n- ".join(errors)
                + "\nTheir scores wouldn't be fair: test on data they didn't see."
            )
        return notes

    def run(self, *, save: bool = True) -> dict:
        """Checks the methods (warning what differs), then runs each on the
        evaluation (reusing saved results). Returns ``{name: test record}``,
        the baseline first, also kept as ``self.records``."""
        for note in self.check():
            warnings.warn(note, stacklevel=2)
        self.records = {
            name: self.evaluation.run(m, name, save=save, allow=self.allow)
            for name, m in self.named_methods().items()
        }
        return self.records

    def table(self, format: str = "markdown", digits: dict | None = None) -> str:
        """The records as a table (``LION.evaluation.table``), the baseline
        first."""
        if self.records is None:
            self.run()
        return table(list(self.records.values()), format=format, digits=digits)


# ---------------------------------------------------------------------------
# The checks, on data records (usable on their own)


def shared_samples(test, training):
    """The test data's samples (or groups) that are also in ``training``,
    both data records. Returns ``(shared, what they are, a note or None)``:
    by group when both records have them, otherwise by sample; data from
    another dataset share none."""
    test, training = _as_dict(test), _as_dict(training)
    if _target(test.get("dataset")) != _target(training.get("dataset")):
        return [], "samples", None
    test_samples = test.get("samples") or {}
    training_samples = training.get("samples") or {}
    if "ids" not in test_samples or "ids" not in training_samples:
        return [], "samples", "its records don't list the samples, so not checked"
    if "groups" in test_samples and "groups" in training_samples:
        shared = set(test_samples["groups"]) & set(training_samples["groups"])
        return sorted(shared), "groups", None
    note = None
    if "groups" in test_samples:
        note = "its records don't say the groups (e.g. patients): checked by sample"
    shared = set(test_samples["ids"]) & set(training_samples["ids"])
    return sorted(shared), "samples", note


def problem_differences(test, training) -> list:
    """How the problem of ``training`` differs from that of ``test`` (data
    records): the experiment, if both have one; otherwise the dataset (but its
    split) and its noise. Where the data were read from isn't compared."""
    test, training = _as_dict(test), _as_dict(training)
    if "experiment" in test and "experiment" in training:
        return _differences(test["experiment"], training["experiment"], "experiment")
    return _differences(_problem(test), _problem(training), "")


def _problem(record) -> dict:
    dataset = dict(record.get("dataset") or {})
    dataset.pop("mode", None)
    return {
        "dataset": dataset,
        "noise": record.get("sinogram_transform"),
        "noise_seed": record.get("sinogram_transform_seed"),
    }


def _differences(a, b, path, limit=5) -> list:
    found = []
    _collect(a, b, path, found)
    if len(found) > limit:
        found = found[:limit] + [f"and {len(found) - limit} more"]
    return found


def _collect(a, b, path, found):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            if key in _NOT_THE_PROBLEM:
                continue
            where = f"{path}.{key}" if path else str(key)
            if key not in a or key not in b:
                found.append(
                    f"{where} only in the {'test' if key in a else 'training'} data"
                )
            else:
                _collect(a[key], b[key], where, found)
    elif a != b:
        found.append(f"{path}: {_short(b)} (test: {_short(a)})")


def _short(value, size=60) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= size else text[: size - 3] + "..."


def _target(value):
    return value.get(spec.TARGET) if isinstance(value, dict) else None


def _as_dict(record):
    if record is None:
        return None
    if isinstance(record, LIONParameter):
        return record.serialize()
    return record


# ---------------------------------------------------------------------------
# What a method's records say


def _records_of(x, allow):
    """``(training data, validation data, LION code difference)`` of a model
    file or a solver; None for methods without records."""
    if isinstance(x, (str, os.PathLike)):
        path = pathlib.Path(x)
        saved = spec.read(path)
        try:
            data = _as_dict(read_record(saved))
        except ValueError:
            data = None
        try:
            training = _as_dict(read_training_record(saved))
        except ValueError:
            training = None
        validation = ((training or {}).get("validation") or {}).get("data")
        code = None
        if "environment" in saved:
            code = environment.code_differences(saved["environment"])
        return data, validation, code
    if isinstance(x, LIONsolver):
        data = _as_dict(x._training_data_record())
        validation = None
        if x.validation_loader is not None:
            validation = _as_dict(x._data_record(x.validation_loader))
        return data, validation, None
    return None


def _name_of(x, operator, allow=()) -> str:
    if isinstance(x, (str, os.PathLike)):
        return pathlib.Path(x).with_suffix("").name
    return _default_name(method(x, operator, allow=allow).record)
