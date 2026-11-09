# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Publishing a trained model (LION.package.publish): what a package holds,
that its files alone train the model again (the model card's code), what
publish refuses, and the notes. On CPU, on fake LIDC-IDRI slices, with a
stand-in CT operator (an identity), and LION's code taken as committed and
pushed (it is checked in its own test)."""

import json
import os
import re
import shutil
import tempfile
import warnings

os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from LION.operators.ct.ct_geometry import Geometry
from LION import package  # noqa: E402
from LION.evaluation import Evaluation  # noqa: E402
from LION.experiments import ct_experiments  # noqa: E402
from LION.models.LIONmodel import LIONmodel, ModelInputType  # noqa: E402
from LION.optimizers.SupervisedSolver import SupervisedSolver  # noqa: E402
from LION.optimizers.training_record import weights_sha256  # noqa: E402
from LION.utils import environment  # noqa: E402
from pytests.helpers import ScaleOp, small_dncnn  # noqa: E402
from pytests.test_lidc_resampling import fake_lidc  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Expected '(dataset|training)'"),
]


def identity_operator(operator_params):
    return ScaleOp(1.0, (1, 512, 512))


def patch(monkeypatch):

    monkeypatch.setattr(Geometry, "make_operator", identity_operator)
    monkeypatch.setattr(environment, "publish_problems", lambda record: [])


@pytest.fixture(autouse=True)
def stand_ins(monkeypatch):
    patch(monkeypatch)


def train(
    folder, data_folder, name="final", start=None, start_file=None, scheduled=False
):
    """A small sinogram-domain DnCNN trained 2 epochs on 4 fake LIDC slices,
    validated every epoch, saved as ``folder/name``; ``start``: weights to
    start from (else new ones), or ``start_file``: a model file to fine-tune.
    ``scheduled``: with two parameter groups and a learning-rate scheduler."""
    experiment = ct_experiments.LowDoseCTRecon(datafolder=data_folder)
    model = small_dncnn(ModelInputType.SINOGRAM)
    if start is not None:
        model.load_state_dict(start)
    if start_file is not None:
        model = LIONmodel.load(start_file, supress_warnings=True)[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = None
    if scheduled:
        first, *rest = list(model.parameters())
        optimizer = torch.optim.Adam(
            [{"params": [first], "lr": 1e-4}, {"params": rest}], lr=1e-3
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
    solver = SupervisedSolver(
        model,
        optimizer,
        torch.nn.MSELoss(),
        operator_params=experiment.operator_params,
        scheduler=scheduler,
        device=torch.device("cpu"),
        verbose=False,
    )
    training = Subset(experiment.get_dataset("train"), [0, 2, 3, 5])
    solver.set_training(DataLoader(training, batch_size=2, shuffle=True))
    solver.set_validation(DataLoader(experiment.get_dataset("validation")), 1)
    folder.mkdir(exist_ok=True)
    solver.set_saving(folder, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.train(2)
        solver.save_final_results()
    return folder / name


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """``(model file, data folder)``, made once for the module. The data are
    in ``LION_DATA_PATH`` (as LIDC-IDRI is, by default), for the module."""
    root = tmp_path_factory.mktemp("package")
    data_folder = root / "data" / "LIDC-IDRI"
    data_folder.mkdir(parents=True)
    fake_lidc(data_folder, range(1, 11))
    for k, path in enumerate(sorted(data_folder.glob("LIDC-IDRI-*/slice_*.npy"))):
        values = np.random.default_rng(k).integers(-1000, 400, (512, 512))
        np.save(path, values.astype(np.int16))
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("LION_DATA_PATH", str(root / "data"))
        patch(monkeypatch)
        yield train(root / "results", data_folder), data_folder


def copy(file, folder):
    """The model file copied to ``folder``, to change it."""
    folder.mkdir()
    for suffix in (".json", ".pt"):
        shutil.copyfile(file.with_suffix(suffix), folder / f"model{suffix}")
    return folder / "model"


def edit(file, change):
    path = file.with_suffix(".json")
    saved = json.loads(path.read_text())
    change(saved)
    path.write_text(json.dumps(saved))


def published(file, folder, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return package.publish(file, folder, **kwargs)


# ---------------------------------------------------------------------------
# What a package holds


def test_a_package_holds_the_model_and_its_records(trained, tmp_path):
    file, _ = trained
    folder = published(file, tmp_path / "package")
    assert sorted(
        p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()
    ) == [
        "README.md",
        "data/training.json",
        "data/validation.json",
        "environment.txt",
        "experiment.json",
        "model.json",
        "model.pt",
        "package.json",
        "solver.json",
    ]
    # the model file as LION saved it
    for suffix in (".json", ".pt"):
        assert (folder / f"model{suffix}").read_bytes() == file.with_suffix(
            suffix
        ).read_bytes()
    model = LIONmodel.load(folder / "model", supress_warnings=True)[0]
    written = json.loads((folder / "package.json").read_text())
    assert written["weights_sha256"] == weights_sha256(model)
    # every file but the card and package.json, by its sha256
    assert sorted(written["sha256"]) == [
        "data/training.json",
        "data/validation.json",
        "environment.txt",
        "experiment.json",
        "model.json",
        "model.pt",
        "solver.json",
    ]
    for name, digest in written["sha256"].items():
        assert package.file_sha256(folder / name) == digest
    saved = json.loads(file.with_suffix(".json").read_text())
    assert written["lion"]["commit"] == saved["environment"]["lion"]["commit"]
    assert [c["class"] for c in written["citations"]] == [
        "LION.models.CNNs.dncnn.DnCNN",
        "LION.data_loaders.LIDC_IDRI.LIDC_IDRI",
    ]
    assert "@article{zhang2017beyond" in written["citations"][0]["bibtex"]
    # the records, as they are in the model file
    records = {
        "data/training.json": saved["dataset_params"],
        "data/validation.json": saved["training"]["validation"]["data"],
        "solver.json": saved["training"],
        "experiment.json": saved["dataset_params"]["experiment"],
    }
    for name, record in records.items():
        assert json.loads((folder / name).read_text()) == record
    assert written["notes"] == []

    card = (folder / "README.md").read_text()
    assert card.startswith(
        "# DnCNN trained on LIDC-IDRI: Low dose full angular sampling experiment\n"
    )
    for line in (
        "- Geometry: fan beam, 360 angles, detector 1x1024, image 1x512x512.",
        "- Data: the train data, 4 samples from 3 groups; batch size 2, shuffled.",
        "- Validation: 2 samples from 1 group, every 1 epoch, with `MSELoss`.",
        "- Seed 0; initial weights made again from the random state saved with "
        "the model.",
        'model, _, _ = LIONmodel.load("model")',
        "@article{armato2011lidc,",
    ):
        assert line in card
    assert "## Notes" not in card and "## Test" not in card


def test_each_file_is_made_again_by_its_own_instantiate(trained, tmp_path):
    from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
    from LION.data_loaders.LIONDataset import LIONDataset
    from LION.optimizers.LIONsolver import LIONsolver

    file, _ = trained
    folder = published(file, tmp_path / "package")
    experiment = ct_experiments.Experiment.instantiate(folder / "experiment.json")
    assert isinstance(experiment, ct_experiments.LowDoseCTRecon)
    assert experiment.param.name == "Low dose full angular sampling experiment"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the data's noise isn't the dataset's
        training = LIONDataset.instantiate(folder / "data" / "training.json")
    assert isinstance(training, Subset) and isinstance(training.dataset, LIDC_IDRI)
    assert training.indices == [0, 2, 3, 5]
    model = LIONmodel.instantiate(folder / "model", initial_weights=True)
    solver = LIONsolver.instantiate(
        folder / "solver.json", model, device=torch.device("cpu")
    )
    assert isinstance(solver, SupervisedSolver)


def test_the_package_trains_the_model_again(trained, tmp_path, monkeypatch):
    """The model card's code, run in the package, gives the same weights."""
    file, _ = trained
    folder = published(file, tmp_path / "package")
    card = (folder / "README.md").read_text()
    code = re.search(r"### Step by step.*?```python\n(.*?)```", card, re.S)
    monkeypatch.chdir(folder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(code.group(1), {"__name__": "__main__"})
    again = LIONmodel.load(folder / "retrained" / "model", supress_warnings=True)[0]
    written = json.loads((folder / "package.json").read_text())
    assert weights_sha256(again) == written["weights_sha256"]

    model = package.train_again(folder, tmp_path / "again")
    written = json.loads((folder / "package.json").read_text())
    assert weights_sha256(model) == written["weights_sha256"]
    saved = LIONmodel.load(tmp_path / "again" / "model", supress_warnings=True)[0]
    assert weights_sha256(saved) == written["weights_sha256"]


# ---------------------------------------------------------------------------
# With a test


def test_with_a_test_record(trained, tmp_path):
    file, data_folder = trained
    evaluation = Evaluation(
        ct_experiments.LowDoseCTRecon(datafolder=data_folder),
        samples=2,
        save_folder=tmp_path / "evaluations",
        device="cpu",
    )
    folder = published(file, tmp_path / "package", evaluation=evaluation)
    record = json.loads((folder / "test.json").read_text())
    assert record["lion_test_record"] == 1
    assert record["data"]["split"] == "test"
    written = json.loads((folder / "package.json").read_text())
    assert written["files"]["test"] == "test.json"
    assert "test.json" in written["sha256"]
    card = (folder / "README.md").read_text()
    assert "## Test" in card and "| PSNR" in card
    # or its record's file
    saved = next((tmp_path / "evaluations").glob("*.json"))
    again = published(file, tmp_path / "again", evaluation=saved)
    assert (again / "test.json").read_text() == (folder / "test.json").read_text()


def test_refusals(trained, tmp_path, monkeypatch):
    file, data_folder = trained
    record = Evaluation(
        ct_experiments.LowDoseCTRecon(datafolder=data_folder),
        samples=2,
        save_folder=None,
        device="cpu",
    ).run(file)
    record.method.model.weights_sha256 = "0" * 64
    with pytest.raises(package.PublishError, match="other weights"):
        published(file, tmp_path / "package", evaluation=record)
    assert not (tmp_path / "package").exists()

    file, data_folder = trained
    evaluation = Evaluation(
        ct_experiments.LowDoseCTRecon(datafolder=data_folder),
        split="train",
        samples=2,
        save_folder=None,
        device="cpu",
    )
    with pytest.raises(package.PublishError, match="trained on 1 of the test"):
        published(file, tmp_path / "package", evaluation=evaluation)

    file, _ = trained
    monkeypatch.setattr(
        environment,
        "publish_problems",
        lambda record: ["LION has uncommitted changes: commit them first (x.py)"],
    )
    with pytest.raises(package.PublishError) as error:
        published(file, tmp_path / "package")
    message = str(error.value)
    assert "(now)" in message and "(when the model was saved)" in message
    assert not (tmp_path / "package").exists()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # saved without its data and training
        small_dncnn().save(tmp_path / "model")
    with pytest.raises(package.PublishError) as error:
        published(tmp_path / "model", tmp_path / "package")
    assert "no training record" in str(error.value)
    assert "no data record" in str(error.value)

    file = copy(trained[0], tmp_path / "copy")
    edit(
        file,
        lambda saved: saved["training"]["optimizer"].update(
            {"_target_": "my_package.optimizers.Fast"}
        ),
    )
    with pytest.raises(package.PublishError, match="my_package.optimizers.Fast"):
        published(file, tmp_path / "package")


# ---------------------------------------------------------------------------
# Initial weights, and notes


def test_initial_weights_saved_by_the_solver_are_included(trained, tmp_path):
    file, data_folder = trained
    start = LIONmodel.load(file, supress_warnings=True)[0].state_dict()
    tuned = train(tmp_path / "tuned", data_folder, "tuned", start=start)
    assert (tmp_path / "tuned" / "tuned_initial.pt").is_file()
    folder = published(tuned, tmp_path / "package")
    written = json.loads((folder / "package.json").read_text())
    assert written["files"]["initial_weights"] == "tuned_initial.pt"
    assert "tuned_initial.pt" in written["sha256"]
    model = LIONmodel.instantiate(folder / "model", initial_weights=True)
    assert all(
        torch.equal(a.cpu(), b.cpu())
        for a, b in zip(model.state_dict().values(), start.values())
    )
    # and without that file, it can't be published
    (tmp_path / "tuned" / "tuned_initial.pt").unlink()
    with pytest.raises(package.PublishError, match="tuned_initial.pt"):
        published(tuned, tmp_path / "again")


# ---------------------------------------------------------------------------
# Using a package: load, build, train_again


def published_package(trained, tmp_path, **kwargs):
    return published(trained[0], tmp_path / "package", **kwargs)


def test_load_gives_the_recorded_arguments(trained, tmp_path):
    folder = published_package(trained, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing changed
        p = package.load(folder)
    assert p.model.model_parameters.blocks == 3
    assert p.model.model_parameters.model_input_type == ModelInputType.SINOGRAM
    assert p.experiment.experiment_params.noise_params.I0 == 11000
    assert (p.training_data.split, p.training_data.subset) == ("train", [0, 2, 3, 5])
    assert (p.validation_data.split, p.validation_data.subset) == ("validation", None)
    assert p.solver.seed == 0 and isinstance(p.solver.loss_fn, torch.nn.MSELoss)
    assert p.optimizer.lr == 1e-3
    t = p.training
    assert (t.batch_size, t.shuffle, t.drop_last, t.epochs) == (2, True, False, 2)
    assert p.validation.every == 1 and p.validation.reconstructor is None
    assert p.evaluation is None and p.records.test is None
    assert p.device == "cpu"
    assert p.records.training.epochs.done == 2


def test_load_and_build_as_published(trained, tmp_path):
    folder = published_package(trained, tmp_path)
    built = package.load_and_build(folder)
    written = json.loads((folder / "package.json").read_text())
    assert weights_sha256(built.model) == written["weights_sha256"]
    assert type(built.experiment) is ct_experiments.LowDoseCTRecon
    assert isinstance(built.training_data, Subset)
    assert built.training_data.indices == [0, 2, 3, 5]
    assert len(built.validation_data) == 2
    assert type(built.solver) is SupervisedSolver and built.solver.model is built.model
    assert built.solver.train_loader.batch_size == 2
    assert built.solver.validation_freq == 1
    assert built.evaluation is None
    assert built.device == torch.device("cpu")

    # Trained, initial (the recorded ones), or new for a changed model.
    p = package.load(folder)
    record = p.records.training
    trained_weights = json.loads((folder / "package.json").read_text())
    initial = package.build(p, initial_weights=True).model
    assert weights_sha256(initial) == record.initial_weights_sha256
    assert weights_sha256(package.build(p).model) == trained_weights["weights_sha256"]
    p.model.model_parameters.bias_free = True
    changed = package.build(p, initial_weights=True).model
    assert changed.model_parameters.bias_free
    assert weights_sha256(changed) != record.initial_weights_sha256


def test_a_package_makes_the_scheduler_and_groups_again(trained, tmp_path):
    file = train(tmp_path / "scheduled", trained[1], scheduled=True)
    folder = published(file, tmp_path / "package")
    p = package.load(folder)
    assert [g["lr"] for g in p.parameter_groups] == [1e-4, 1e-3]
    assert (p.scheduler.step_size, p.scheduler.gamma) == (1, 0.5)
    p.scheduler.gamma = 0.1
    solver = package.build(p).solver
    assert [g["lr"] for g in solver.optimizer.param_groups] == [1e-4, 1e-3]
    assert type(solver.scheduler) is torch.optim.lr_scheduler.StepLR
    assert solver.scheduler.optimizer is solver.optimizer
    assert solver.scheduler.gamma == 0.1


def test_train_again_with_changes(trained, tmp_path):
    folder = published_package(trained, tmp_path)
    p = package.load(folder)
    p.model.model_parameters.blocks = 2
    p.experiment.experiment_params.noise_params.I0 = 5000
    p.optimizer.lr = 1e-2
    p.training.epochs = 1
    p.training_data.subset = [0, 1]
    built = package.build(p)
    assert built.model.model_parameters.blocks == 2
    model = package.train_again(p, tmp_path / "changed")
    assert model.model_parameters.blocks == 2
    saved = json.loads((tmp_path / "changed" / "model.json").read_text())
    assert saved["model"]["model_parameters"]["blocks"] == 2
    experiment = saved["dataset_params"]["experiment"]["experiment_params"]
    assert experiment["noise_params"]["I0"] == 5000
    assert saved["dataset_params"]["subset"]["indices"] == [0, 1]
    assert saved["training"]["optimizer"]["lr"] == 1e-2
    assert saved["training"]["epochs"]["done"] == 1


# ---------------------------------------------------------------------------
# A reconstructor, published with its model


def saved_reconstructor(trained, tmp_path, kind="PnP"):
    """A reconstructor with the trained model (loaded from its file), saved."""
    from LION.operators.ct import fdk
    from LION.reconstructors import ForwardReconstructor, PnP

    file, data_folder = trained
    operator_params = ct_experiments.LowDoseCTRecon(
        datafolder=data_folder
    ).operator_params
    model = LIONmodel.load(file, supress_warnings=True)[0]
    if kind == "PnP":
        parameters = PnP.default_parameters("FBS")
        parameters.max_iter = 2
        # CT's default (the stand-in operator's is a test function, which
        # can't be published)
        parameters.initial_reconstruction = fdk
        reconstructor = PnP(operator_params, model, parameters)
    else:
        reconstructor = ForwardReconstructor(operator_params, model)
    reconstructor.save(tmp_path / kind)
    return tmp_path / kind


def test_a_reconstructor_is_published_with_its_model(trained, tmp_path):
    from LION.reconstructors import LIONReconstructor

    folder = published(saved_reconstructor(trained, tmp_path), tmp_path / "package")
    written = json.loads((folder / "package.json").read_text())
    assert written["files"]["reconstructor"] == "reconstructor.json"
    assert "reconstructor.json" in written["sha256"]
    assert "LION.reconstructors.PnP.PnP" in [c["class"] for c in written["citations"]]
    again = LIONReconstructor.load(folder / "reconstructor")
    assert again.parameters.max_iter == 2
    assert weights_sha256(again.model) == written["weights_sha256"]
    card = (folder / "README.md").read_text()
    assert card.startswith("# PnP with a DnCNN trained on LIDC-IDRI")
    assert "## The reconstruction" in card
    assert 'LIONReconstructor.load("reconstructor")' in card
    p = package.load(folder)
    assert p.reconstructor.parameters.max_iter == 2
    built = package.build(p)
    assert built.reconstructor.model is built.model

    file, data_folder = trained
    source = saved_reconstructor(trained, tmp_path, kind="forward")
    evaluation = Evaluation(
        ct_experiments.LowDoseCTRecon(datafolder=data_folder),
        samples=2,
        save_folder=None,
        device="cpu",
    )
    folder = published(source, tmp_path / "tested", evaluation=evaluation)
    test = json.loads((folder / "test.json").read_text())
    assert test["method"]["kind"] == "reconstructor"
    # its test isn't the model's
    record = evaluation.run(source)
    with pytest.raises(package.PublishError, match="publish the reconstructor's"):
        published(file, tmp_path / "again", evaluation=record)


# ---------------------------------------------------------------------------
# A fine-tuned model, published with its original's records


def fine_tuned(trained, tmp_path):
    """A copy of the trained model, and a model fine-tuned from the copy."""
    file, data_folder = trained
    original = copy(file, tmp_path / "original")
    return original, train(tmp_path / "tuned", data_folder, start_file=original)


def test_a_fine_tuned_model_is_published_with_its_original(trained, tmp_path):
    original, tuned = fine_tuned(trained, tmp_path)
    folder = published(tuned, tmp_path / "package")
    written = json.loads((folder / "package.json").read_text())
    assert written["files"]["origins"] == ["origin/model.json"]
    assert "origin/model.json" in written["sha256"]
    assert json.loads((folder / "origin" / "model.json").read_text()) == json.loads(
        original.with_suffix(".json").read_text()
    )
    card = (folder / "README.md").read_text()
    assert "## Fine-tuned from" in card and str(original) in card
    p = package.load(folder)
    assert p.records.origins[0].training.epochs.done == 2
    assert p.records.training.initial_weights.origin.weights_sha256
