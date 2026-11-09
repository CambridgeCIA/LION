# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Evaluating reconstruction methods (LION.evaluation): methods as one call,
the baseline, scores, the Evaluation component and its test records (saved,
reused), tables. On CPU without tomosipo: models take their input directly
(sinogram-domain), or FDK is replaced."""

import functools
import os
import tempfile
import warnings

# the datasets' default folders are in LION_DATA_PATH
os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import LION.evaluation.evaluation as evaluation_module  # noqa: E402
from LION.operators.ct.ct_geometry import Geometry  # noqa: E402
from LION.evaluation import (  # noqa: E402
    Evaluation,
    baseline,
    method,
    read_test_record,
    table,
)
from LION.data_loaders.LIONDataset import LIONDataset  # noqa: E402
from LION.models.LIONmodel import ModelInputType  # noqa: E402
from LION.optimizers.SupervisedSolver import SupervisedSolver  # noqa: E402
from LION.optimizers.training_record import weights_sha256  # noqa: E402
from LION.reconstructors.LIONreconstructor import LIONReconstructor  # noqa: E402
from LION.utils.normaliser import Normalisation  # noqa: E402
from LION.utils.parameter import LIONParameter  # noqa: E402
from pytests.helpers import ImageMatrixOp, ScaleOp, small_dncnn  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Expected '(dataset|training)'"),
    # TensorDatasets (and lambdas) can't be specs: records name them
    pytest.mark.filterwarnings("ignore:.*can't be written as a spec"),
]


class Pairs(LIONDataset):
    """Noisy images (measurements) and clean ones (targets): a LION dataset,
    so records can describe it."""

    def __init__(self, n=6, size=16, seed=0):
        self.n, self.size, self.seed = n, size, seed
        g = torch.Generator().manual_seed(seed)
        self.clean = torch.rand(n, 1, size, size, generator=g)
        self.noisy = self.clean + 0.1 * torch.randn(n, 1, size, size, generator=g)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.noisy[i], self.clean[i]

    def sample_ids(self):
        return [f"pair_{i}" for i in range(self.n)]


def data(n=6, size=16, seed=0):
    return Pairs(n, size, seed)


def model(seed=0):
    torch.manual_seed(seed)
    return small_dncnn(ModelInputType.SINOGRAM)  # takes its input as it is


def as_is(y, operator):
    """The measurement itself: a stand-in for a classical reconstruction."""
    return y


def evaluation(folder, **kwargs):
    return Evaluation(dataset=data(), save_folder=folder, device="cpu", **kwargs)


# --------------------------------------------------------------------------
# Methods as one call


def test_a_model(no_ct, tmp_path):
    m = model()
    run = method(m)
    y = data()[0][0].unsqueeze(0)
    with torch.no_grad():
        expected = m.eval()(y)
    torch.testing.assert_close(run(y), expected)
    assert run.record["kind"] == "model"
    assert run.record["model"]["weights_sha256"] == weights_sha256(m)
    assert run.modules == (m,)

    seen = []
    run = method(lambda y, op: seen.append(op) or y, operator="the operator")
    run(torch.zeros(1))
    assert seen == ["the operator"]
    named = method(functools.partial(as_is))
    assert named.record["function"]["_target_"] == "functools.partial"
    assert named.record["source_sha256"] is not None

    net = small_dncnn(ModelInputType.SINOGRAM)
    reconstructor = Doubling(ScaleOp(1.0, (1, 2, 2)), net)
    run = method(reconstructor)
    torch.testing.assert_close(run(torch.ones(1, 2, 2)), 2 * torch.ones(1, 2, 2))
    assert run.record["kind"] == "reconstructor"
    assert run.record["model"]["weights_sha256"] == weights_sha256(net)
    assert run.modules == (net,)
    # recorded by its spec: operator and parameters, not the model
    assert set(run.record["reconstructor"]) == {"_target_", "operator", "parameters"}

    solver = make_solver(Halving)
    run = method(solver)
    y = data()[0][0].unsqueeze(0)
    with torch.no_grad():
        expected = solver.model.eval()(y) / 2
    torch.testing.assert_close(run(y), expected)
    assert run.record["kind"] == "solver" and run.record["solver"].endswith("Halving")

    solver = make_solver()
    path = train_and_save(solver, tmp_path)
    run = method(path)
    assert run.record["kind"] == "model"
    assert run.record["model"]["weights_sha256"] == weights_sha256(solver.model)
    assert run.record["file"]["_args_"][0].endswith("final")


class Doubling(LIONReconstructor):
    def reconstruct_sample(self, y, x0=None, verbose=0):
        return 2 * y


@pytest.fixture
def no_ct(monkeypatch):
    monkeypatch.setattr(Geometry, "make_operator", lambda self: None)


def make_solver(cls=SupervisedSolver, m=None):
    m = model() if m is None else m
    return cls(
        m,
        torch.optim.SGD(m.parameters(), lr=0.01),
        torch.nn.MSELoss(),
        operator_params=Geometry.default_parameters(),
        device=torch.device("cpu"),
    )


class Halving(SupervisedSolver):
    """A solver with its own reconstruction."""

    def _forward(self, data):
        return self.model(data) / 2


def train_and_save(solver, folder):
    solver.set_training(DataLoader(data(), batch_size=2))
    solver.set_saving(folder, "final")
    solver.train(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.save_final_results()
    return folder / "final"


# --------------------------------------------------------------------------
# The baseline


def test_baseline():
    from LION.operators.ct.fdk import fdk
    from LION.operators.ct.CTProjectionOp import CTProjectionOp
    from LION.operators.pcm.WalshHadamard2D import WalshHadamard2D

    from LION.operators.Operator import Operator, inverse_reconstruction
    from pytests.helpers import adjoint_reconstruction

    # the operator's default reconstruction: FDK for CT, its inverse if it has
    # one, else what it says (the stand-in's adjoint)
    assert baseline(object.__new__(CTProjectionOp)) is fdk
    assert baseline(object.__new__(WalshHadamard2D)) is inverse_reconstruction
    assert baseline(ImageMatrixOp(torch.eye(4), (1, 2, 2))) is adjoint_reconstruction

    class NoDefault(ImageMatrixOp):  # neither an inverse nor a default
        default_reconstruction_fn = Operator.default_reconstruction_fn

    with pytest.raises(ValueError, match="no standard baseline"):
        baseline(NoDefault(torch.eye(4), (1, 2, 2)))


# --------------------------------------------------------------------------
# Evaluation


def test_run_gives_a_test_record(tmp_path):
    record = evaluation(tmp_path).run(model(), "DnCNN")
    assert isinstance(record, LIONParameter) and record.name == "DnCNN"
    assert record.scores.ids == [f"pair_{i}" for i in range(6)]
    assert set(record.summary.serialize()) == {"PSNR", "SSIM", "HaarPSI"}
    assert len(record.scores.PSNR) == 6
    assert record.higher_is_better.PSNR is True
    assert record.environment.lion.version
    assert record.method.kind == "model"
    assert record.data.dataset["_target_"].endswith("Pairs")
    saved = list(tmp_path.glob("DnCNN-*.json"))
    assert len(saved) == 1
    assert read_test_record(saved[0]).serialize() == record.serialize()


def test_saved_results_are_reused(tmp_path, monkeypatch):
    first = evaluation(tmp_path).run(model(), "DnCNN")
    monkeypatch.setattr(
        evaluation_module, "scores", lambda *a, **k: pytest.fail("recomputed")
    )
    again = evaluation(tmp_path).run(model(), "DnCNN")
    assert again.serialize() == first.serialize()

    # other weights or other data are new results
    monkeypatch.undo()
    other = tmp_path / "other"
    evaluation(other).run(model(0), "DnCNN")
    evaluation(other).run(model(1), "DnCNN")
    Evaluation(dataset=data(seed=1), save_folder=other, device="cpu").run(
        model(0), "DnCNN"
    )
    assert len(list(other.glob("DnCNN-*.json"))) == 3


def test_normalisations(tmp_path):
    clean = data().clean
    by_dataset = evaluation(tmp_path).run(as_is)
    assert by_dataset.data_range == pytest.approx(
        [float(clean.min()), float(clean.max())]
    )
    assert evaluation(tmp_path, normalisation="sample").run(as_is).data_range is None
    fixed = evaluation(tmp_path, normalisation=(0.0, 2.0)).run(as_is)
    assert fixed.data_range == [0.0, 2.0]
    custom = Normalisation("custom", data_range=(0.0, 2.0))
    assert evaluation(tmp_path, normalisation=custom).run(as_is).data_range == [
        0.0,
        2.0,
    ]
    # PSNR with a wider range is higher
    assert fixed.summary.PSNR.mean > by_dataset.summary.PSNR.mean


# --------------------------------------------------------------------------
# Tables


def records(folder):
    return [
        evaluation(folder).run(as_is, "noisy input"),
        evaluation(folder).run(model(), "DnCNN_1"),
    ]


def test_markdown_table(tmp_path):
    lines = table(records(tmp_path)).splitlines()
    assert lines[0] == "| Method | PSNR | SSIM | HaarPSI |"
    assert lines[2].startswith("| noisy input | **")  # the better here
    psnr = lines[2].split("|")[2]
    mean, std = psnr.strip(" *").split(" ± ")
    assert len(mean.split(".")[1]) == len(std.split(".")[1]) == 2

    latex = table(records(tmp_path), "latex")
    assert latex.startswith(r"\begin{tabular}{lccc}") and r"\toprule" in latex
    assert r"DnCNN\_1" in latex and r"$\pm$" in latex and r"\textbf{" in latex


def test_lower_is_better_metrics(tmp_path):
    class PerSample(torch.nn.Module):
        def forward(self, x, target, data_range=None):
            return ((x - target) ** 2).flatten(1).mean(1)

    per_sample = {"MSE": PerSample()}
    better = Evaluation(dataset=data(), metrics=per_sample, device="cpu").run(
        as_is, "as is"
    )
    worse = Evaluation(dataset=data(), metrics=per_sample, device="cpu").run(
        lambda y, op: 2 * y, "doubled"
    )
    assert better.higher_is_better.MSE is False  # a loss: lower is better
    lines = table([worse, better]).splitlines()
    assert "**" in lines[3] and "**" not in lines[2]


# --------------------------------------------------------------------------
# Made again from a record


def test_instantiate_from_a_record(tmp_path):
    record = evaluation(tmp_path, normalisation=(0.0, 2.0), samples=3).run(as_is)
    made = Evaluation.instantiate(record, device="cpu", allow=["pytests."])
    assert isinstance(made.dataset, Pairs) and made.dataset.seed == 0
    assert made.normalisation == (0.0, 2.0) and made.samples == 3
    assert set(made.metrics) == {"PSNR", "SSIM", "HaarPSI"}
    assert made.run(as_is, save=False).summary.serialize() == record.summary.serialize()


# --------------------------------------------------------------------------
# With an experiment (fake LIDC-IDRI data, a stand-in projection)


operators_seen = []


def zeros(y, operator):
    """512x512 zeros (LIDC's size), noting the operator it was given."""
    operators_seen.append(operator)
    return torch.zeros(y.shape[0], 1, 512, 512, device=y.device)


def test_an_experiment(tmp_path, monkeypatch):
    import LION.operators.ct.ct_utils as ct_utils
    import LION.data_loaders.LIDC_IDRI as lidc
    import LION.experiments.ct_experiments as ct_experiments
    from pytests.test_data_record import stand_in_projection
    from pytests.test_lidc_resampling import fake_lidc

    # the loader projects with it, and the evaluation gives it to methods
    assert lidc.ct is ct_utils
    monkeypatch.setattr(Geometry, "make_operator", lambda self: stand_in_projection)
    folder = tmp_path / "LIDC-IDRI"
    folder.mkdir()
    fake_lidc(folder, range(1, 11))
    for k, path in enumerate(sorted(folder.glob("LIDC-IDRI-*/slice_*.npy"))):
        np.save(path, np.full((512, 512), -500 + k, np.int16))
    experiment = ct_experiments.LowDoseCTRecon(datafolder=folder)

    operators_seen.clear()
    evaluation_ = Evaluation(experiment, save_folder=tmp_path / "results", samples=2)
    record = evaluation_.run(zeros, "zeros")
    assert operators_seen[0] is stand_in_projection
    assert record.data.split == "test"
    assert record.data.experiment["_target_"].endswith("LowDoseCTRecon")
    assert record.evaluation["experiment"]["_target_"].endswith("LowDoseCTRecon")
    assert len(record.scores.ids) == 2
    assert record.scores.ids[0].startswith("LIDC-IDRI-")
    assert len(list((tmp_path / "results").glob("zeros-*.json"))) == 1
    # run again (a new evaluation, as in a new script): read, not recomputed
    monkeypatch.setattr(
        evaluation_module, "scores", lambda *a, **k: pytest.fail("recomputed")
    )
    again = Evaluation(
        ct_experiments.LowDoseCTRecon(datafolder=folder),
        save_folder=tmp_path / "results",
        samples=2,
    ).run(zeros, "zeros")
    assert again.scores.ids == record.scores.ids
    # and again in the same evaluation
    evaluation_.run(zeros, "zeros")
