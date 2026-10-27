# =============================================================================
# This file is part of LION library
# License : GPL-3
# =============================================================================
"""Evaluating a reconstruction method on a test set.

``Evaluation`` is the standard evaluation of one problem, like an
``Experiment``: the test data (an experiment's split, or a dataset), the
operator, the metrics (PSNR, SSIM, HaarPSI) and how their range is set. Its
``run(method)`` scores every test sample and gives a **test record**:

- ``name``, ``method`` (what was run, e.g. a model's spec and weights hash);
- ``evaluation`` (this evaluation's spec), ``data`` (the test data's record,
  ``LION.data_loaders.data_record``), ``metrics`` (their specs; and whether
  ``higher_is_better``), ``data_range``;
- ``scores``: the sample ``ids`` and, per metric, the score of each sample;
- ``summary``: per metric ``mean``, ``std``, ``median``, ``q25``, ``q75``,
  ``min``, ``max``; ``seconds_per_sample``; ``environment``.

Records are ``LIONParameter``s, saved as plain JSON in ``save_folder`` (by
default ``$LION_RESULTS_PATH/evaluations``). Running the same method on the
same evaluation again reads the saved record instead of recomputing it.

The functions work on their own too: ``scores`` (one method on some data),
``summary``, ``table`` (records as markdown, LaTeX or CSV).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from LION.data_loaders.data_record import data_record
from LION.evaluation.methods import method
from LION.metrics.haarpsi import HAARPsi
from LION.metrics.psnr import PSNR
from LION.metrics.ssim import SSIM
from LION.utils import environment, spec
from LION.utils.device import default_device
from LION.utils.normaliser import Normalisation
from LION.utils.parameter import LIONParameter

RECORD_KEY = "lion_test_record"
RECORD_VERSION = 1


def default_metrics() -> dict:
    """PSNR, SSIM and HaarPSI."""
    return {"PSNR": PSNR(), "SSIM": SSIM(), "HaarPSI": HAARPsi()}


def is_test_record(value) -> bool:
    if isinstance(value, LIONParameter):
        return hasattr(value, RECORD_KEY)
    return isinstance(value, dict) and RECORD_KEY in value


def read_test_record(source) -> LIONParameter:
    """A test record: a record, or the path of its ``.json``."""
    saved = spec.read(source)
    if not is_test_record(saved):
        raise ValueError("not a test record")
    return LIONParameter(**saved)


class Evaluation:
    """The standard evaluation of one problem (see the module docstring).

    ``experiment``: its ``split`` is the test data, and its geometry gives the
    operator; or ``dataset`` (and ``operator``) for data set up without an
    experiment. ``metrics``: ``{name: metric}``, by default PSNR, SSIM and
    HaarPSI. ``normalisation``: the range the metrics use: ``"dataset"`` (the
    test samples' minimum and maximum, the same for every method; default),
    ``"sample"`` (each target's own range), ``(min, max)``, or a LION
    ``Normalisation``. ``samples``: all (None), ``n`` (evenly spaced over the
    test set) or a list of sample IDs. ``save_folder``: where records are
    saved and reused (default ``$LION_RESULTS_PATH/evaluations``; None
    without it). ``device``: where methods run (default LION's).
    """

    def __init__(
        self,
        experiment=None,
        dataset=None,
        operator=None,
        split: str = "test",
        metrics: dict | None = None,
        normalisation="dataset",
        samples=None,
        batch_size: int = 1,
        save_folder=None,
        device=None,
    ):
        if experiment is None and dataset is None:
            raise ValueError("give an experiment or a dataset")
        self.experiment = experiment
        self.dataset = dataset
        self.operator = operator
        self.split = split
        self.metrics = default_metrics() if metrics is None else metrics
        self.normalisation = normalisation
        self.samples = samples
        self.batch_size = batch_size
        if save_folder is None and os.environ.get("LION_RESULTS_PATH"):
            save_folder = pathlib.Path(os.environ["LION_RESULTS_PATH"]) / "evaluations"
        self.save_folder = None if save_folder is None else pathlib.Path(save_folder)
        self.device = default_device() if device is None else torch.device(device)
        self._data = None
        self._data_record = None
        self._operator = None

    # -- what is evaluated -------------------------------------------------

    def data(self):
        """The test samples (the dataset, or the chosen ones of it)."""
        if self._data is None:
            data = self.dataset
            if data is None:
                data = self.experiment.get_dataset(self.split)
            self._data = _select(data, self.samples)
        return self._data

    def get_operator(self):
        """The operator: the one given, or the experiment's."""
        if self._operator is None:
            if self.operator is not None:
                self._operator = self.operator
            elif self.experiment is not None:
                from LION.CTtools.ct_utils import make_operator

                self._operator = make_operator(self.experiment.geometry)
        return self._operator

    # -- running -----------------------------------------------------------

    def run(
        self, x, name: str | None = None, *, save: bool = True, allow=()
    ) -> LIONParameter:
        """Score ``x`` (a model, a model's file, a solver, a reconstructor or a
        function ``f(y, operator)``; see ``LION.evaluation.method``) on every
        test sample. Returns its test record, saved in ``save_folder``; a
        record saved before for the same method and evaluation is read
        instead. ``allow``: modules outside LION a model file may name (e.g.
        your own solver)."""
        run = method(x, self.get_operator(), allow=allow)
        for module in run.modules:
            module.to(self.device)
        name = name or _default_name(run.record)
        evaluation_key = _key({"evaluation": self.spec(), "data": self.data_record()})
        method_key = _key({"method": run.record, "evaluation": evaluation_key})
        # only what is fully described can be recognised later
        cacheable = "unsaved" not in json.dumps(
            [run.record, self.spec(), self.data_record()]
        )

        if cacheable and self.save_folder is not None:
            saved = self._saved(f"*-{method_key[:16]}.json")
            if saved:
                record = read_test_record(saved[0])
                difference = environment.code_differences(record.environment)
                if difference is not None:
                    warnings.warn(
                        f"{name}: reusing {saved[0].name}, but {difference}",
                        stacklevel=2,
                    )
                return record

        data = self.data()
        low_high = self.data_range(evaluation_key)
        ids, values, seconds = scores(
            run, data, self.metrics, low_high, self.batch_size, self.device
        )
        record = {
            RECORD_KEY: RECORD_VERSION,
            "name": name,
            "method": run.record,
            "evaluation": self.spec(),
            "evaluation_key": evaluation_key,
            "data": self.data_record(),
            "metrics": {k: spec.encode_or_name(m, k) for k, m in self.metrics.items()},
            "higher_is_better": {
                k: bool(getattr(m, "higher_is_better", False))
                for k, m in self.metrics.items()
            },
            "data_range": None if low_high is None else list(low_high),
            "scores": {"ids": ids, **{k: v.tolist() for k, v in values.items()}},
            "summary": {k: summary_of(v) for k, v in values.items()},
            "seconds_per_sample": seconds,
            "environment": environment.environment_record(self.device).serialize(),
        }
        if save and self.save_folder is not None:
            if cacheable:
                self.save_folder.mkdir(parents=True, exist_ok=True)
                path = self.save_folder / f"{_file_name(name)}-{method_key[:16]}.json"
                path.write_text(json.dumps(record, indent=1))
            else:
                warnings.warn(
                    f"{name}: not saved, as part of the method or the data can't be "
                    "recorded (e.g. a lambda, or a dataset that isn't a LION one), "
                    "so it couldn't be recognised later",
                    stacklevel=2,
                )
        return LIONParameter(**record)

    def data_range(self, evaluation_key=None):
        """``(min, max)`` the metrics use, or None (each target's own)."""
        normalisation = self.normalisation
        if isinstance(normalisation, Normalisation):
            if normalisation.type == "sample":
                return None
            if normalisation.type in ("dataset", "custom"):
                return float(normalisation.xmin), float(normalisation.xmax)
            raise ValueError("a 'none' Normalisation sets no range")
        if normalisation == "sample":
            return None
        if normalisation == "dataset":
            known = self._known_range(evaluation_key)
            return known if known is not None else self._dataset_range()
        low, high = normalisation
        return float(low), float(high)

    def _dataset_range(self):
        """The test targets' minimum and maximum (LION's dataset normalisation)."""
        dataset = Normalisation("dataset", dataset=self.data())
        return float(dataset.xmin), float(dataset.xmax)

    # -- records -----------------------------------------------------------

    def spec(self) -> dict:
        """This evaluation's spec (its arguments; not where it saves or runs)."""
        written = {spec.TARGET: spec.import_path(type(self))}
        for name in (
            "experiment",
            "dataset",
            "operator",
            "split",
            "metrics",
            "normalisation",
            "samples",
            "batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, Normalisation):
                value = _normalisation_value(value)
            written[name] = spec.encode_or_name(value, name)
        return written

    def data_record(self) -> dict:
        """The test data's record (made once)."""
        if self._data_record is None:
            self._data_record = data_record(self.data()).serialize()
        return self._data_record

    @classmethod
    def instantiate(cls, source, *, allow=(), **arguments):
        """The evaluation a test record (or its spec) describes;
        ``arguments`` replace recorded ones (e.g. ``save_folder``)."""
        saved = spec.read(source)
        written = saved["evaluation"] if is_test_record(saved) else saved
        written = {k: v for k, v in written.items() if k not in arguments}
        path, kwargs = spec.build_arguments(written, allow)
        target = spec.target_class(path, cls, Evaluation, allow)
        return target(**kwargs, **arguments)

    def _saved(self, pattern):
        if self.save_folder is None or not self.save_folder.is_dir():
            return []
        return sorted(self.save_folder.glob(pattern))

    def _known_range(self, evaluation_key):
        """The range a saved record of this evaluation already computed."""
        for path in self._saved("*.json"):
            try:
                saved = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if saved.get("evaluation_key") == evaluation_key and saved.get(
                "data_range"
            ):
                return tuple(saved["data_range"])
        return None


# ---------------------------------------------------------------------------
# Functions


def scores(run, data, metrics, data_range=None, batch_size=1, device=None):
    """Every sample of ``data`` (a dataset of ``(measurement, target)``)
    through ``run``, in order, scored by every metric. ``data_range``:
    ``(min, max)`` (outputs and targets are shifted by ``min`` and the metrics
    get ``max - min``), or None for each target's own range.

    Returns ``(ids, {name: numpy array of per-sample scores}, seconds per
    sample)``."""
    device = default_device() if device is None else torch.device(device)
    values = {name: [] for name in metrics}
    seconds = 0.0
    for y, target in DataLoader(data, batch_size=batch_size, shuffle=False):
        y, target = y.to(device), target.to(device)
        _synchronise(device)
        start = time.perf_counter()
        output = run(y)
        _synchronise(device)
        seconds += time.perf_counter() - start
        output = output.to(target.dtype)
        if data_range is not None:
            low, high = data_range
            output, target = output - low, target - low
        for name, metric in metrics.items():
            if data_range is None:
                score = metric(output, target)
            else:
                score = metric(output, target, data_range=high - low)
            values[name].append(torch.as_tensor(score).detach().cpu().reshape(-1))
    values = {k: torch.cat(v).double().numpy() for k, v in values.items()}
    n = len(next(iter(values.values()))) if values else 0
    return sample_ids(data), values, seconds / max(n, 1)


def sample_ids(data) -> list:
    """The IDs of a dataset's samples (their indices if it has none)."""
    if isinstance(data, Subset):
        ids = sample_ids(data.dataset)
        return [ids[i] for i in data.indices]
    if hasattr(data, "sample_ids"):
        return list(data.sample_ids())
    return [str(i) for i in range(len(data))]


def summary_of(values) -> dict:
    """Mean, std, median, quartiles, min and max (NaNs left out)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {k: None for k in ("mean", "std", "median", "q25", "q75", "min", "max")}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summary(record) -> dict:
    """The summary of a test record, ``{metric: {mean, std, ...}}``."""
    return spec.read(record)["summary"]


def table(records, format: str = "markdown", digits: dict | None = None) -> str:
    """Records as a table: one row per method (its ``name``), one column per
    metric, ``mean ± std``, the best of each column in bold. ``format``:
    ``"markdown"``, ``"latex"`` (booktabs) or ``"csv"`` (mean and std columns).
    ``digits``: decimals per metric (default: 2 for values of 10 or more, else
    4)."""
    records = [spec.read(r) for r in records]
    if not records:
        return ""
    metrics = list(records[0]["summary"])
    names = [r["name"] for r in records]
    means = {m: [r["summary"][m]["mean"] for r in records] for m in metrics}
    stds = {m: [r["summary"][m]["std"] for r in records] for m in metrics}
    best = {m: _best(means[m], _higher_is_better(records[0], m)) for m in metrics}

    places = {}
    for m in metrics:
        given = (digits or {}).get(m)
        if given is None:
            known = [abs(v) for v in means[m] if v is not None and not math.isnan(v)]
            given = 2 if known and max(known) >= 10 else 4
        places[m] = given

    def number(value, metric):
        if value is None or math.isnan(value):
            return "–"
        return f"{value:.{places[metric]}f}"

    if format == "csv":
        header = ["method"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
        lines = [",".join(header)]
        for i, name in enumerate(names):
            cells = [_csv(name)]
            for m in metrics:
                cells += [_plain(means[m][i]), _plain(stds[m][i])]
            lines.append(",".join(cells))
        return "\n".join(lines) + "\n"

    if format == "markdown":
        lines = [
            "| Method | " + " | ".join(metrics) + " |",
            "|---|" + "---|" * len(metrics),
        ]
        for i, name in enumerate(names):
            cells = []
            for m in metrics:
                cell = f"{number(means[m][i], m)} ± {number(stds[m][i], m)}"
                cells.append(f"**{cell}**" if i == best[m] else cell)
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    if format == "latex":
        lines = [
            r"\begin{tabular}{l" + "c" * len(metrics) + "}",
            r"\toprule",
            "Method & " + " & ".join(_latex(m) for m in metrics) + r" \\",
            r"\midrule",
        ]
        for i, name in enumerate(names):
            cells = []
            for m in metrics:
                cell = f"{number(means[m][i], m)} $\\pm$ {number(stds[m][i], m)}"
                cells.append(f"\\textbf{{{cell}}}" if i == best[m] else cell)
            lines.append(f"{_latex(name)} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines) + "\n"

    raise ValueError(f"format must be 'markdown', 'latex' or 'csv', not {format!r}")


# ---------------------------------------------------------------------------
# Helpers


def _select(data, samples):
    if samples is None:
        return data
    if isinstance(samples, int):
        if samples >= len(data):
            return data
        indices = np.linspace(0, len(data) - 1, samples).round().astype(int)
        return Subset(data, sorted(set(int(i) for i in indices)))
    ids = sample_ids(data)
    where = {sample: i for i, sample in enumerate(ids)}
    missing = [s for s in samples if s not in where]
    if missing:
        raise ValueError(f"samples not in the test data: {missing[:5]}")
    return Subset(data, [where[s] for s in samples])


def _normalisation_value(normalisation):
    if normalisation.type in ("dataset", "custom"):
        return (float(normalisation.xmin), float(normalisation.xmax))
    return normalisation.type


def _key(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _default_name(record) -> str:
    kind = record["kind"]
    if kind in ("model", "solver"):
        target = record["model"]["spec"].get(spec.TARGET, kind)
        return target.rsplit(".", 1)[-1]
    if kind == "function":
        function = record["function"]
        if function.get(spec.TARGET) == "functools.partial":
            function = function[spec.ARGS][0]
        return str(function.get(spec.TARGET, "function")).rsplit(".", 1)[-1]
    if kind == "reconstructor":
        return str(record["reconstructor"].get(spec.TARGET, "reconstructor")).rsplit(
            ".", 1
        )[-1]
    return kind


def _file_name(name) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "method"


def _synchronise(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _higher_is_better(record, metric) -> bool:
    return bool(record.get("higher_is_better", {}).get(metric, False))


def _best(values, higher) -> int | None:
    candidates = [
        (v, i) for i, v in enumerate(values) if v is not None and not math.isnan(v)
    ]
    if len(candidates) < 2:
        return None
    return (max if higher else min)(candidates)[1]


def _plain(value) -> str:
    return "" if value is None else repr(float(value))


def _csv(text) -> str:
    return f'"{text}"' if "," in text or '"' in text else text


def _latex(text) -> str:
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
    ):
        text = text.replace(old, new)
    return text
