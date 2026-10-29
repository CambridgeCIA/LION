"""Publishing a trained model (LION.package.publish): what a package holds,
that its files alone train the model again (the model card's code), what
publish refuses, and the notes. On CPU, on fake LIDC-IDRI slices, with a
stand-in CT operator (an identity), and LION's code taken as committed and
pushed (it is checked in its own test)."""

import json
import os
import re
import shutil
import sys
import tempfile
import warnings

os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

import LION.CTtools.ct_utils as ct_utils  # noqa: E402
import LION.optimizers.LIONsolver as lion_solver  # noqa: E402
from LION import package  # noqa: E402
from LION.evaluation import Evaluation  # noqa: E402
from LION.experiments import ct_experiments  # noqa: E402
from LION.models.LIONmodel import LIONmodel, ModelInputType  # noqa: E402
from LION.optimizers.SupervisedSolver import SupervisedSolver  # noqa: E402
from LION.optimizers.training_record import weights_sha256  # noqa: E402
from LION.utils import environment, paths  # noqa: E402
from pytests.helpers import ScaleOp, small_dncnn  # noqa: E402
from pytests.test_lidc_resampling import fake_lidc  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Expected '(dataset|training|geometry)'"),
]


def identity_operator(geometry):
    return ScaleOp(1.0, (1, 512, 512))


def patch(monkeypatch):
    monkeypatch.setattr(ct_utils, "make_operator", identity_operator)
    monkeypatch.setattr(lion_solver, "make_operator", lambda geometry: None)
    monkeypatch.setattr(environment, "publish_problems", lambda record: [])


@pytest.fixture(autouse=True)
def stand_ins(monkeypatch):
    patch(monkeypatch)


def train(folder, data_folder, name="final", start=None):
    """A small sinogram-domain DnCNN trained 2 epochs on 4 fake LIDC slices,
    validated every epoch, saved as ``folder/name``; ``start``: weights to
    start from (else new ones)."""
    experiment = ct_experiments.LowDoseCTRecon(datafolder=data_folder)
    model = small_dncnn(ModelInputType.SINOGRAM)
    if start is not None:
        model.load_state_dict(start)
    solver = SupervisedSolver(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        torch.nn.MSELoss(),
        geometry=experiment.geometry,
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
        # saving reads the variable; data_path() what it was at import
        monkeypatch.setenv("LION_DATA_PATH", str(root / "data"))
        monkeypatch.setattr(paths, "LION_DATA_PATH", root / "data")
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
    code = re.search(r"## Training it again.*?```python\n(.*?)```", card, re.S)
    monkeypatch.chdir(folder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(code.group(1), {"__name__": "__main__"})
    again = LIONmodel.load(folder / "retrained" / "model", supress_warnings=True)[0]
    written = json.loads((folder / "package.json").read_text())
    assert weights_sha256(again) == written["weights_sha256"]


def test_the_model_card(trained, tmp_path):
    file, _ = trained
    card = (published(file, tmp_path / "package") / "README.md").read_text()
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


def test_the_environment_list():
    listed = package.environment_list()
    lines = [line for line in listed.splitlines() if not line.startswith("#")]
    names = {re.split("==|=", line)[0].lower() for line in lines}
    assert {"torch", "numpy"} <= names
    assert all(sys.prefix not in line for line in lines)


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


def test_a_test_of_other_weights_is_refused(trained, tmp_path):
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


def test_a_test_on_the_training_data_is_refused(trained, tmp_path):
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


# ---------------------------------------------------------------------------
# Refusals


def test_lion_code_that_isnt_committed_and_pushed_is_refused(
    trained, tmp_path, monkeypatch
):
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


def test_a_model_not_trained_by_a_solver_is_refused(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # saved without its data and training
        small_dncnn().save(tmp_path / "model")
    with pytest.raises(package.PublishError) as error:
        published(tmp_path / "model", tmp_path / "package")
    assert "no training record" in str(error.value)
    assert "no data record" in str(error.value)


def test_code_outside_lion_is_refused(trained, tmp_path):
    file = copy(trained[0], tmp_path / "copy")
    edit(
        file,
        lambda saved: saved["training"]["optimizer"].update(
            {"_target_": "my_package.optimizers.Fast"}
        ),
    )
    with pytest.raises(package.PublishError, match="my_package.optimizers.Fast"):
        published(file, tmp_path / "package")


def test_what_could_only_be_named_is_refused(trained, tmp_path):
    file = copy(trained[0], tmp_path / "copy")
    edit(
        file,
        lambda saved: saved["training"]["validation"].update(
            {"function": {"unsaved": "<function <lambda>>", "reason": "a lambda"}}
        ),
    )
    with pytest.raises(package.PublishError, match="lambda"):
        published(file, tmp_path / "package")


def test_a_folder_that_isnt_empty_is_refused(trained, tmp_path):
    (tmp_path / "package").mkdir()
    (tmp_path / "package" / "notes.txt").write_text("mine")
    with pytest.raises(package.PublishError, match="isn't an empty folder"):
        published(trained[0], tmp_path / "package")


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


def test_what_cant_be_repeated_exactly_is_a_note(trained, tmp_path):
    file = copy(trained[0], tmp_path / "copy")
    edit(file, lambda saved: saved["training"].update({"initial_weights": None}))
    with pytest.warns(UserWarning, match="can't be made again"):
        folder = package.publish(file, tmp_path / "package")
    written = json.loads((folder / "package.json").read_text())
    assert written["notes"][0].startswith("the weights training started from")
    card = (folder / "README.md").read_text()
    assert "## Notes" in card and "(but for the initial weights" in card
    assert 'LIONmodel.instantiate("model", initial_weights=False)' in card


def test_data_outside_lion_data_path_is_a_note(trained, tmp_path):
    """Saved where LION_DATA_PATH wasn't set, or another folder: the data's
    folder is this machine's."""
    file, data_folder = trained
    file = copy(file, tmp_path / "copy")

    def absolute(value):
        if isinstance(value, list):
            return [absolute(v) for v in value]
        if not isinstance(value, dict):
            return value
        if value.get("_target_") == "LION.utils.paths.data_path":
            return {"_target_": "pathlib.Path", "_args_": [str(data_folder)]}
        return {k: absolute(v) for k, v in value.items()}

    edit(file, lambda saved: saved.update(absolute(saved)))
    with pytest.warns(UserWarning, match="outside LION_DATA_PATH"):
        folder = package.publish(file, tmp_path / "package")
    written = json.loads((folder / "package.json").read_text())
    assert any(str(data_folder) in note for note in written["notes"])
