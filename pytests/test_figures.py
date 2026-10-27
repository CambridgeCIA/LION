"""Figures of comparisons (LION.evaluation.figures, Comparison.figures/images/
save/instantiate): score plots, the chosen samples, example images with zoom
crops and differences, the saved folder and making it again from it."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import warnings

os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import pytest  # noqa: E402
import torch  # noqa: E402

import LION.evaluation.evaluation as evaluation_module  # noqa: E402
from LION.evaluation import Comparison  # noqa: E402
from LION.evaluation.figures import (  # noqa: E402
    box_plot,
    chosen_samples,
    difference_plot,
    example_images,
    per_sample,
    per_sample_differences,
    reconstructions,
)
from pytests.test_comparison import Slices, doubling, evaluation  # noqa: E402
from pytests.test_evaluation import as_is, model  # noqa: E402

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


def test_plots_are_boxes_of_that_data():
    base = record("base", [1.0, 2.0, 3.0])
    net = record("net", [3.0, 2.0, 4.0])
    fig = box_plot([base, net], "M")
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["base", "net"]
    assert "higher is better" in fig.axes[0].get_ylabel()
    fig = difference_plot([base, net], "M", "base")
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["net"]
    assert fig.axes[0].get_ylabel().startswith("M − base")


def test_plots_draw_into_a_given_axes():
    from matplotlib.figure import Figure

    mine = Figure()
    left, right = mine.subplots(1, 2)
    records = [record("base", [1.0, 2.0]), record("net", [2.0, 3.0])]
    assert box_plot(records, "M", ax=left) is mine
    assert difference_plot(records, "M", "base", ax=right) is mine
    assert len(mine.axes) == 2


def test_chosen_samples():
    assert chosen_samples(record("a", [3.0, 1.0, 2.0])) == {
        "best": "s0",
        "median": "s2",
        "worst": "s1",
    }
    # lower is better: the lowest is best
    assert chosen_samples(record("a", [3.0, 1.0, 2.0], higher=False))["best"] == "s1"


# --------------------------------------------------------------------------
# Example images


def test_reconstructions_are_the_methods_images(tmp_path):
    test = evaluation(tmp_path)
    targets, outputs = reconstructions(
        test, {"as_is": as_is, "half": halving}, ["p6/slice_1"]
    )
    noisy, clean = Slices("test")[3]
    torch.testing.assert_close(targets["p6/slice_1"], clean[0])
    torch.testing.assert_close(outputs["half"]["p6/slice_1"], noisy[0] / 2)
    with pytest.raises(ValueError, match="not in the test data"):
        reconstructions(test, {"as_is": as_is}, ["p0/slice_0"])


def test_example_images_layout_window_zoom_and_differences():
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


# --------------------------------------------------------------------------
# Comparison: figures, images, the saved folder, made again


def test_comparison_figures(tmp_path):
    figures = comparison(tmp_path).figures()
    metrics = ["PSNR", "SSIM", "HaarPSI"]
    assert set(figures) == {f"{k}_{m}" for k in ("box", "difference") for m in metrics}
    assert set(comparison(tmp_path, baseline=False).figures()) == {
        f"box_{m}" for m in metrics
    }


def test_comparison_images_of_the_chosen_samples(tmp_path):
    made = comparison(tmp_path)
    chosen = made.chosen_samples()
    # by the baseline's first metric
    from LION.evaluation.figures import chosen_samples as by_record

    expected = by_record(made.records["doubling"])
    assert chosen == {f"{label}: {i}": i for label, i in expected.items()}
    given = made.chosen_samples(ids=["p5/slice_0"])
    assert given == {"p5/slice_0": "p5/slice_0"}
    fig = made.images(ids=["p5/slice_0"], zoom=[(0, 0, 4, 4)])
    rows, columns = fig.axes[0].get_subplotspec().get_gridspec().get_geometry()
    assert (rows, columns) == (2, 4)  # target, doubling, as_is, halving
    assert fig.axes[1].get_xlabel().startswith("PSNR ")


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


def test_the_file_cant_widen_what_is_built(tmp_path):
    folder = comparison(tmp_path, allow=["pytests."]).save(tmp_path / "paper")
    assert json.loads((folder / "comparison.json").read_text())["comparison"][
        "allow"
    ] == ["pytests."]
    with pytest.raises(Exception, match="not in the modules a spec may use"):
        Comparison.instantiate(folder)


def test_methods_made_in_the_script_cant_be_made_again(tmp_path):
    made = Comparison(evaluation(tmp_path), {"net": model()}, baseline=as_is)
    with pytest.warns(UserWarning):
        written = made.spec()
    with pytest.raises(ValueError, match="net was made in the script"):
        Comparison.instantiate(
            written, allow=["pytests."], evaluation_arguments={"device": "cpu"}
        )


def test_evaluation_imports_without_matplotlib():
    code = (
        "import sys; sys.modules['matplotlib'] = None\n"
        "import LION.evaluation, LION.evaluation.figures\n"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.stdout.strip() == "ok", result.stderr
