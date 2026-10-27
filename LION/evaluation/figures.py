# =============================================================================
# This file is part of LION library
# License : GPL-3
# =============================================================================
"""Figures of test records: score plots and example images, in plain
matplotlib (your style sheet and rcParams apply).

- ``per_sample(records, metric)``: each method's per-sample scores, and
  ``per_sample_differences(records, metric, baseline)``: each method's minus
  the baseline's, sample by sample: the data, for any plot;
- ``box_plot`` and ``difference_plot``: those as ``ax.boxplot``s;
- ``chosen_samples(record)``: the samples where a method (the baseline) does
  best, median and worst;
- ``reconstructions(evaluation, methods, ids)`` and ``example_images(...)``:
  the targets and each method's images of some samples, in one grey window,
  with zoomed crops and differences to the target if asked.

Plots take ``ax=`` (draw into your figure) and return the ``Figure``. matplotlib
is imported only here, so ``LION.evaluation`` imports without it.
"""

from __future__ import annotations

import numpy as np
import torch

from LION.evaluation.evaluation import _higher_is_better, sample_ids
from LION.evaluation.methods import method
from LION.utils import spec


def per_sample(records, metric) -> dict:
    """``{name: [score of each sample]}`` of each record, in order."""
    records = [spec.read(r) for r in records]
    return {r["name"]: _scores(r, metric) for r in records}


def per_sample_differences(records, metric, baseline) -> dict:
    """``{name: [score - baseline's score, per sample]}`` of each record but
    the baseline's (the record named ``baseline``), matched by sample ID."""
    records = [spec.read(r) for r in records]
    by_name = {r["name"]: r for r in records}
    if baseline not in by_name:
        raise ValueError(f"no record named {baseline!r}")
    reference = dict(
        zip(by_name[baseline]["scores"]["ids"], _scores(by_name[baseline], metric))
    )
    found = {}
    for record in records:
        if record["name"] == baseline:
            continue
        scores = dict(zip(record["scores"]["ids"], _scores(record, metric)))
        found[record["name"]] = [
            scores[i] - reference[i] for i in scores if i in reference
        ]
    return found


def box_plot(records, metric, ax=None):
    """Each method's per-sample ``metric`` (``per_sample``), one box each."""
    fig, ax = _figure(ax)
    _boxes(ax, per_sample(records, metric))
    ax.set_ylabel(f"{metric}{_direction(records[0], metric)}")
    return fig


def difference_plot(records, metric, baseline, ax=None):
    """Each method's per-sample ``metric`` minus the baseline's
    (``per_sample_differences``), one box each, with a line at 0."""
    fig, ax = _figure(ax)
    ax.axhline(0, linestyle="--")
    _boxes(ax, per_sample_differences(records, metric, baseline))
    ax.set_ylabel(f"{metric} − {baseline}{_direction(records[0], metric)}")
    return fig


def chosen_samples(record, metric=None) -> dict:
    """``{"best": id, "median": id, "worst": id}``: the samples where
    ``record``'s method scores best, median and worst on ``metric`` (default:
    the first)."""
    record = spec.read(record)
    metric = metric or next(iter(record["summary"]))
    pairs = [
        (v, i)
        for i, v in zip(record["scores"]["ids"], _scores(record, metric))
        if v == v  # not NaN
    ]
    if not pairs:
        raise ValueError(f"no {metric} scores in {record['name']!r}")
    pairs.sort(reverse=_higher_is_better(record, metric))
    return {
        "best": pairs[0][1],
        "median": pairs[len(pairs) // 2][1],
        "worst": pairs[-1][1],
    }


def reconstructions(evaluation, methods: dict, ids, allow=()):
    """The targets and each method's images of the samples ``ids`` (of
    ``evaluation``'s test data), as 2D CPU tensors: ``(targets, outputs)``,
    ``targets[id]`` and ``outputs[name][id]``."""
    data = evaluation.data()
    where = {sample: i for i, sample in enumerate(sample_ids(data))}
    missing = [i for i in ids if i not in where]
    if missing:
        raise ValueError(f"samples not in the test data: {missing[:5]}")
    targets, inputs = {}, {}
    for sample in ids:
        y, target = data[where[sample]]
        inputs[sample] = y.unsqueeze(0).to(evaluation.device)
        targets[sample] = _image(target)
    outputs = {}
    for name, x in methods.items():
        run = method(x, evaluation.get_operator(), allow=allow)
        for module in run.modules:
            module.to(evaluation.device)
        outputs[name] = {sample: _image(run(inputs[sample])) for sample in ids}
    return targets, outputs


def example_images(
    targets,
    outputs,
    data_range=None,
    zoom=None,
    differences=False,
    labels=None,
    scores=None,
    zoom_color="C1",
):
    """The targets and each method's images, one row per sample (the target
    first, then the methods), in one grey window: ``data_range`` (``(min,
    max)``; None: each target's own).

    ``zoom``: ``[(row, col, height, width), ...]`` in pixels: each region is
    outlined and shown enlarged in a row below. ``differences``: a row of each
    method's image minus the target. ``labels``: ``{id: row label}``.
    ``scores``: ``{name: {id: text}}`` written under the images.
    ``zoom_color``: of the outlines (default: the second colour of your
    colour cycle)."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    ids = list(targets)
    names = list(outputs)
    zoom = list(zoom or [])
    rows_per_sample = 1 + len(zoom) + (1 if differences else 0)
    columns = 1 + len(names)
    # about 2 inches per image
    fig = Figure(figsize=(2 * columns, 2 * rows_per_sample * len(ids)))
    axes = fig.subplots(rows_per_sample * len(ids), columns, squeeze=False)

    for k, sample in enumerate(ids):
        target = targets[sample]
        low, high = data_range if data_range is not None else _range(target)
        images = [target] + [outputs[n][sample] for n in names]
        titles = ["target"] + names
        row = rows_per_sample * k
        for c, image in enumerate(images):
            ax = axes[row, c]
            ax.imshow(_array(image), cmap="gray", vmin=low, vmax=high)
            for r0, c0, h, w in zoom:
                ax.add_patch(
                    Rectangle(
                        (c0 - 0.5, r0 - 0.5),
                        w,
                        h,
                        fill=False,
                        edgecolor=zoom_color,
                    )
                )
            if k == 0:
                ax.set_title(titles[c])
            if c > 0 and scores is not None:
                ax.set_xlabel(scores.get(names[c - 1], {}).get(sample, ""))
        axes[row, 0].set_ylabel((labels or {}).get(sample, sample))
        for z, (r0, c0, h, w) in enumerate(zoom):
            axes[row + 1 + z, 0].set_ylabel(
                f"zoom: rows {r0}–{r0 + h},\ncolumns {c0}–{c0 + w}"
            )
            for c, image in enumerate(images):
                ax = axes[row + 1 + z, c]
                ax.imshow(
                    _array(image[r0 : r0 + h, c0 : c0 + w]),
                    cmap="gray",
                    vmin=low,
                    vmax=high,
                )
        if differences:
            ax_row = row + 1 + len(zoom)
            errors = [outputs[n][sample] - target for n in names]
            bound = max([float(e.abs().max()) for e in errors] + [1e-12])
            # on the first method's panel: the target's is empty
            axes[ax_row, 1].set_ylabel(f"image − target\n(±{bound:.3g})")
            for c, error in enumerate(errors, start=1):
                axes[ax_row, c].imshow(
                    _array(error), cmap="RdBu_r", vmin=-bound, vmax=bound
                )
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        if not ax.images:
            ax.set_axis_off()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Helpers


def _scores(record, metric):
    values = record["scores"].get(metric)
    if values is None:
        raise ValueError(f"{record['name']!r} has no {metric} scores")
    return [float("nan") if v is None else float(v) for v in values]


def _direction(record, metric) -> str:
    record = spec.read(record)
    return (
        " (higher is better)"
        if _higher_is_better(record, metric)
        else " (lower is better)"
    )


def _figure(ax):
    if ax is not None:
        return ax.figure, ax
    from matplotlib.figure import Figure

    fig = Figure()
    return fig, fig.subplots()


def _boxes(ax, values: dict):
    """``ax.boxplot`` of ``{name: values}`` (NaNs left out), labelled."""
    ax.boxplot([[v for v in vs if v == v] for vs in values.values()])
    ax.set_xticks(range(1, len(values) + 1), list(values))


def _image(x) -> torch.Tensor:
    """A 2D CPU image of a sample (its first channel)."""
    x = torch.as_tensor(x).detach().float().cpu()
    while x.dim() > 2:
        x = x[0]
    return x


def _array(image) -> np.ndarray:
    return torch.as_tensor(image).detach().float().cpu().numpy()


def _range(image):
    return float(image.min()), float(image.max())
