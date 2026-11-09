# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Figures of comparisons (LION.evaluation.figures, Comparison.figures/images/
save/instantiate): score plots, the chosen samples, example images with zoom
crops and differences, the saved folder and making it again from it."""

import csv
import json
import os
import tempfile
import warnings

os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import pytest  # noqa: E402
import torch  # noqa: E402

import LION.evaluation.evaluation as evaluation_module  # noqa: E402
from LION.evaluation import Comparison  # noqa: E402
from LION.evaluation.figures import (  # noqa: E402
    example_images,
    per_sample,
    per_sample_differences,
    reconstructions,
)
from pytests.test_comparison import Slices, doubling, evaluation  # noqa: E402
from pytests.test_evaluation import as_is  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Expected '(dataset|training|geometry)'"),
    pytest.mark.filterwarnings("ignore:.*can't be written as a spec"),
]


def halving(y, operator):
    return y / 2


def comparison(folder, **kwargs):
    kwargs.setdefault("baseline", doubling)
    return Comparison(evaluation(folder), [as_is, halving], **kwargs)


def record(name, values, higher=True):
    ids = [f"s{i}" for i in range(len(values))]
    return {
        "name": name,
        "scores": {"ids": ids, "M": values},
        "summary": {"M": {}},
        "higher_is_better": {"M": higher},
    }


# --------------------------------------------------------------------------
# Score plots and the chosen samples


def test_per_sample_scores_and_differences():
    base = record("base", [1.0, 2.0, 3.0])
    net = record("net", [3.0, 2.0, 4.0])
    assert per_sample([base, net], "M") == {
        "base": [1.0, 2.0, 3.0],
        "net": [3.0, 2.0, 4.0],
    }
    assert per_sample_differences([base, net], "M", "base") == {"net": [2.0, 0.0, 1.0]}
    # matched by sample ID, not position
    net["scores"]["ids"] = ["s2", "s0", "s1"]
    assert per_sample_differences([base, net], "M", "base") == {"net": [0.0, 1.0, 2.0]}
    with pytest.raises(ValueError, match="no record named"):
        per_sample_differences([net], "M", "base")


# --------------------------------------------------------------------------
# Example images


def test_example_images_layout_window_zoom_and_differences(tmp_path):
    targets = {"a": torch.zeros(8, 8), "b": torch.ones(8, 8)}
    outputs = {
        "m1": {k: v + 0.5 for k, v in targets.items()},
        "m2": {k: v * 2 for k, v in targets.items()},
    }
    zoom = [(2, 3, 4, 2)]
    fig = example_images(
        targets, outputs, data_range=(0.0, 2.0), zoom=zoom, differences=True
    )
    rows, columns = fig.axes[0].get_subplotspec().get_gridspec().get_geometry()
    assert (rows, columns) == (2 * 3, 3)  # per sample: images, zoom, differences
    first = fig.axes[0].images[0]
    assert first.get_clim() == (0.0, 2.0)  # one window for all
    crop = fig.axes[columns].images[0].get_array()  # the target's zoomed region
    assert crop.shape == (4, 2)
    difference = fig.axes[2 * columns + 1].images[0]  # m1 - target, sample a
    assert float(difference.get_array().max()) == 0.5
    assert difference.get_clim() == (-0.5, 0.5)  # symmetric, over the methods
    assert not fig.axes[2 * columns].images  # no difference for the target
    assert fig.axes[2 * columns + 1].get_ylabel() == "image − target\n(±0.5)"

    test = evaluation(tmp_path)
    targets, outputs = reconstructions(
        test, {"as_is": as_is, "half": halving}, ["p6/slice_1"]
    )
    noisy, clean = Slices("test")[3]
    torch.testing.assert_close(targets["p6/slice_1"], clean[0])
    torch.testing.assert_close(outputs["half"]["p6/slice_1"], noisy[0] / 2)
    with pytest.raises(ValueError, match="not in the test data"):
        reconstructions(test, {"as_is": as_is}, ["p0/slice_0"])


# --------------------------------------------------------------------------
# Comparison: figures, images, the saved folder, made again


def test_comparison_figures(tmp_path):
    figures = comparison(tmp_path).figures()
    metrics = ["PSNR", "SSIM", "HaarPSI"]
    assert set(figures) == {f"{k}_{m}" for k in ("box", "difference") for m in metrics}
    assert set(comparison(tmp_path, baseline=False).figures()) == {
        f"box_{m}" for m in metrics
    }


def test_save_writes_the_comparison(tmp_path):
    made = comparison(tmp_path)
    folder = made.save(tmp_path / "paper", zoom=[(1, 1, 4, 4)], differences=True)
    for name in ["comparison.json", "table.md", "table.tex", "table.csv", "scores.csv"]:
        assert (folder / name).is_file()
    for name in ["box_PSNR", "difference_SSIM", "images"]:
        assert (folder / f"{name}.pdf").is_file() and (folder / f"{name}.png").is_file()
    assert sorted(p.name for p in (folder / "records").iterdir()) == [
        "as_is.json",
        "doubling.json",
        "halving.json",
    ]
    with open(folder / "scores.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0][:3] == ["sample", "doubling PSNR", "doubling SSIM"]
    assert [r[0] for r in rows[1:]] == made.records["as_is"].scores.ids
    assert float(rows[1][rows[0].index("as_is SSIM")]) == pytest.approx(
        made.records["as_is"].scores.SSIM[0]
    )
    saved = json.loads((folder / "comparison.json").read_text())
    assert saved["images"]["zoom"] == [[1, 1, 4, 4]]
    assert saved["comparison"]["methods"][0]["_target_"].endswith("as_is")


def test_made_again_from_its_folder(tmp_path, monkeypatch):
    made = comparison(tmp_path)
    folder = made.save(tmp_path / "paper")
    again = Comparison.instantiate(
        folder,
        allow=["pytests."],
        evaluation_arguments={"save_folder": tmp_path / "results", "device": "cpu"},
    )
    without_allow = lambda c: {k: v for k, v in c.spec().items() if k != "allow"}
    assert without_allow(again) == without_allow(made)
    # the same evaluation: its results are read, not computed again
    monkeypatch.setattr(
        evaluation_module, "scores", lambda *a, **k: pytest.fail("recomputed")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert again.table() == made.table()
