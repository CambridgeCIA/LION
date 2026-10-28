"""Comparing reconstruction methods (LION.evaluation.Comparison): the baseline
row, names, the checks from the models' records (leakage by patient, another
problem, other code, methods that can't be checked), tables. On CPU: models
take their input directly (sinogram-domain)."""

import os
import tempfile
import warnings

os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

import LION.evaluation.comparison as comparison_module  # noqa: E402
import LION.optimizers.LIONsolver as lion_solver  # noqa: E402
from LION.data_loaders.LIONDataset import LIONDataset  # noqa: E402
from LION.evaluation import (  # noqa: E402
    Comparison,
    Evaluation,
    problem_differences,
    shared_samples,
)
from LION.experiments import ct_experiments  # noqa: E402
from pytests.helpers import ImageMatrixOp  # noqa: E402
from pytests.test_evaluation import as_is, make_solver, model  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Expected '(dataset|training|geometry)'"),
    pytest.mark.filterwarnings("ignore:.*can't be written as a spec"),
]

PATIENTS = {"train": [0, 1, 2, 3], "validation": [4], "test": [5, 6]}


class Slices(LIONDataset):
    """Two slices per patient, the patients split by ``mode`` (as LIDC-IDRI's
    are): noisy slices (measurements) and clean ones (targets)."""

    def __init__(self, mode="test", noise=0.1, size=8):
        self.mode, self.noise, self.size = mode, noise, size
        self.patients = PATIENTS[mode]
        self.noisy, self.clean = [], []
        for patient in self.patients:
            g = torch.Generator().manual_seed(patient)
            for _ in range(2):
                clean = torch.rand(1, size, size, generator=g)
                self.clean.append(clean)
                self.noisy.append(clean + noise * torch.randn(clean.shape, generator=g))

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, i):
        return self.noisy[i], self.clean[i]

    def sample_ids(self):
        return [f"p{p}/slice_{s}" for p in self.patients for s in range(2)]

    def sample_groups(self):
        return [f"p{p}" for p in self.patients for _ in range(2)]


def doubling(y, operator):
    return 2 * y


@pytest.fixture
def no_ct(monkeypatch):
    monkeypatch.setattr(lion_solver, "make_operator", lambda geometry: None)


def evaluation(folder, **kwargs):
    kwargs.setdefault("dataset", Slices("test"))
    return Evaluation(save_folder=folder / "results", device="cpu", **kwargs)


def trained(folder, data, validation=None, name="model", seed=0):
    """A model trained (1 epoch) on ``data``, saved as ``folder/name``."""
    solver = make_solver(m=model(seed))
    solver.set_training(DataLoader(data, batch_size=2))
    if validation is not None:
        solver.set_validation(DataLoader(validation, batch_size=2), 1)
    solver.set_saving(folder, name)
    solver.train(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.save_final_results()
    return folder / name


# --------------------------------------------------------------------------
# Running


def test_the_baseline_comes_first(no_ct, tmp_path):
    path = trained(tmp_path, Slices("train"), Slices("validation"))
    comparison = Comparison(evaluation(tmp_path), [path, as_is], baseline=doubling)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing to warn about
        records = comparison.run()
    assert list(records) == ["doubling", "model", "as_is"]
    assert comparison.records is records
    lines = comparison.table().splitlines()
    assert [line.split("|")[1].strip() for line in lines[2:]] == list(records)
    assert "\\toprule" in comparison.table("latex")


def test_names_given_or_made(tmp_path):
    comparison = Comparison(evaluation(tmp_path), {"mine": as_is}, baseline=doubling)
    assert list(comparison.named_methods()) == ["doubling", "mine"]
    with pytest.raises(ValueError, match="baseline's name"):
        Comparison(evaluation(tmp_path), {"doubling": as_is}, baseline=doubling).run()
    with pytest.raises(ValueError, match="same name"):
        Comparison(evaluation(tmp_path), [as_is, as_is], baseline=False).run()


class Invertible(ImageMatrixOp):
    def inverse(self, y):
        return y


class Problem:
    """A stand-in experiment with its own baseline."""

    def get_dataset(self, split):
        return Slices(split)

    def baseline(self):
        return doubling


def test_the_default_baseline(tmp_path):
    op = Invertible(torch.eye(64), (1, 8, 8))
    by_operator = Comparison(evaluation(tmp_path, operator=op), [as_is])
    assert list(by_operator.named_methods()) == ["inverse", "as_is"]
    by_experiment = Comparison(
        Evaluation(Problem(), operator=op, save_folder=None, device="cpu"), [as_is]
    )
    assert list(by_experiment.named_methods()) == ["doubling", "as_is"]
    none = Comparison(evaluation(tmp_path, operator=op), [as_is], baseline=False)
    assert list(none.named_methods()) == ["as_is"]
    assert ct_experiments.Experiment.baseline(None) is None  # FDK, the modality's


# --------------------------------------------------------------------------
# Leakage


def test_a_model_trained_on_the_test_data_is_an_error(no_ct, tmp_path):
    path = trained(tmp_path, Slices("test"))
    with pytest.raises(
        ValueError, match=r"model was trained on 2 of the test data's groups: p5, p6"
    ):
        Comparison(evaluation(tmp_path), [path], baseline=False).run()


def test_leakage_is_by_patient(no_ct, tmp_path):
    """Trained on one slice of a patient, tested on another: an error."""
    path = trained(tmp_path, Subset(Slices("test"), [0]))
    test = evaluation(tmp_path, samples=["p5/slice_1"])
    with pytest.raises(
        ValueError, match=r"trained on 1 of the test data's groups: p5\n"
    ):
        Comparison(test, [path], baseline=False).check()


def test_a_model_validated_on_the_test_data_is_an_error(no_ct, tmp_path):
    path = trained(tmp_path, Slices("train"), validation=Slices("test"))
    with pytest.raises(ValueError, match="model was validated on"):
        Comparison(evaluation(tmp_path), [path], baseline=False).check()


def test_a_solver_is_checked_too(no_ct, tmp_path):
    solver = make_solver()
    solver.set_training(DataLoader(Slices("test"), batch_size=2))
    with pytest.raises(ValueError, match="was trained on"):
        Comparison(evaluation(tmp_path), {"solver": solver}, baseline=False).check()


def test_records_without_groups_are_checked_by_sample():
    test = {
        "dataset": {"_target_": "Slices"},
        "samples": {"ids": ["p5/slice_0", "p5/slice_1"], "groups": ["p5"]},
    }
    old = {"dataset": {"_target_": "Slices"}, "samples": {"ids": ["p5/slice_1"]}}
    shared, unit, note = shared_samples(test, old)
    assert shared == ["p5/slice_1"] and unit == "samples" and "by sample" in note
    other = {"dataset": {"_target_": "Other"}, "samples": {"ids": ["p5/slice_1"]}}
    assert shared_samples(test, other)[0] == []


# --------------------------------------------------------------------------
# Other differences: warnings


def test_another_problem_warns(no_ct, tmp_path):
    path = trained(tmp_path, Slices("train", noise=0.3))
    comparison = Comparison(evaluation(tmp_path), [path], baseline=False)
    (note,) = comparison.check()
    assert "another problem" in note and "dataset.noise: 0.3 (test: 0.1)" in note
    with pytest.warns(UserWarning, match="another problem"):
        comparison.run()


def test_where_the_data_are_and_the_split_are_not_the_problem():
    test = {
        "dataset": {"_target_": "D", "mode": "test", "parameters": {"folder": "a"}},
        "sinogram_transform": "noise",
    }
    training = {
        "dataset": {"_target_": "D", "mode": "train", "parameters": {"folder": "b"}},
        "sinogram_transform": "noise",
    }
    assert problem_differences(test, training) == []
    training["sinogram_transform"] = "more noise"
    assert problem_differences(test, training) == [
        'noise: "more noise" (test: "noise")'
    ]
    # with experiments, the experiments are compared
    test["experiment"] = {"_target_": "LowDose", "datafolder": "a"}
    training["experiment"] = {"_target_": "ExtremeLowDose", "datafolder": "b"}
    assert problem_differences(test, training) == [
        'experiment._target_: "ExtremeLowDose" (test: "LowDose")'
    ]


def test_other_lion_code_is_listed(no_ct, tmp_path, monkeypatch):
    path = trained(tmp_path, Slices("train"))
    monkeypatch.setattr(
        comparison_module.environment,
        "code_differences",
        lambda saved: "LION commit abc, now def",
    )
    (note,) = Comparison(evaluation(tmp_path), [path], baseline=False).check()
    assert note == "model was trained with other LION code: LION commit abc, now def"


def test_models_without_records_are_said(tmp_path):
    comparison = Comparison(evaluation(tmp_path), {"net": model()}, baseline=as_is)
    assert comparison.check() == [
        "net: no record of its training, so it can't be checked (give its file "
        "instead)"
    ]


# --------------------------------------------------------------------------
# Reconstructors: checked and recorded through their model's file


def fbs(model):
    from LION.reconstructors import PnP
    from pytests.helpers import ScaleOp

    parameters = PnP.default_parameters("FBS")
    parameters.max_iter = 2
    return PnP(ScaleOp(1.0, (1, 8, 8)), model, parameters)


def loaded(path):
    from LION.models.LIONmodel import LIONmodel

    return LIONmodel.load(path, supress_warnings=True)[0]


def test_a_reconstructors_model_file_is_checked(no_ct, tmp_path):
    path = trained(tmp_path, Slices("test"))
    model = loaded(path)
    assert model.source_file == path
    with pytest.raises(ValueError, match="pnp was trained on"):
        Comparison(evaluation(tmp_path), {"pnp": fbs(model)}, baseline=False).check()


def test_a_reconstructor_is_saved_and_made_again(no_ct, tmp_path):
    path = trained(tmp_path, Slices("train"))
    made = Comparison(
        evaluation(tmp_path),
        {"pnp": fbs(loaded(path)), "pnp_as_is": fbs(as_is_image)},
        baseline=False,
    )
    folder = made.save(tmp_path / "paper")
    again = Comparison.instantiate(
        folder,
        allow=["pytests."],
        evaluation_arguments={"save_folder": tmp_path / "results", "device": "cpu"},
    )
    assert again.methods["pnp"].model.source_file == path
    assert again.methods["pnp_as_is"].model is as_is_image
    assert again.table() == made.table()  # the same methods: results read again


def test_a_changed_model_file_warns(no_ct, tmp_path):
    path = trained(tmp_path, Slices("train"))
    folder = Comparison(
        evaluation(tmp_path), {"pnp": fbs(loaded(path))}, baseline=False
    ).save(tmp_path / "paper")
    trained(tmp_path, Slices("train"), seed=1)  # other weights, same file
    with pytest.warns(UserWarning, match="other weights"):
        Comparison.instantiate(
            folder, allow=["pytests."], evaluation_arguments={"device": "cpu"}
        )


def as_is_image(x):
    """A prior that changes nothing."""
    return x
