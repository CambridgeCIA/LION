"""The record of the environment a model was saved in (LION.utils.environment):
LION's version and code state, packages, determinism settings; the warning on
loading a file saved with other code; what keeps a result from being published;
the solvers' deterministic switch."""

import importlib
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys
import warnings

import pytest
import torch

import LION
import LION.optimizers.LIONsolver as lion_solver
from LION.CTtools.ct_geometry import Geometry
from LION.models.CNNs.dncnn import DnCNN
from LION.models.LIONmodel import LIONmodel
from LION.optimizers.SupervisedSolver import SupervisedSolver
from LION.utils import environment
from LION.utils.parameter import LIONParameter
from LION.utils.spec import to_spec
from pytests.helpers import small_dncnn


def save(model, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # no dataset/training given
        model.save(path)


# --------------------------------------------------------------------------
# The version


def test_version():
    assert LION.__version__ == "0.2.0"
    root = pathlib.Path(LION.__file__).parent
    assert (root / "VERSION").read_text().strip() == LION.__version__
    pyproject = (root.parent / "pyproject.toml").read_text()
    assert 'version_file = "LION/VERSION"' in pyproject


# --------------------------------------------------------------------------
# The record


def test_record():
    record = environment.environment_record("cpu")
    assert isinstance(record, LIONParameter)
    assert record.lion.version == LION.__version__
    assert record.packages.torch == importlib.metadata.version("torch")
    assert record.torch.deterministic_algorithms is False
    assert record.device.type == "cpu"
    assert not hasattr(record, "hostname")
    json.dumps(record.serialize())  # plain JSON


def test_record_in_every_model_file(tmp_path):
    save(small_dncnn(), tmp_path / "model")
    saved = json.loads((tmp_path / "model.json").read_text())
    assert saved["environment"]["lion"]["version"] == LION.__version__
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the same code: no warning
        _, options, _ = DnCNN.load(tmp_path / "model", supress_warnings=True)
    assert isinstance(options.environment, LIONParameter)
    assert options.environment.serialize() == saved["environment"]


def test_loading_a_file_saved_with_other_code_warns(tmp_path):
    save(small_dncnn(), tmp_path / "model")
    path = tmp_path / "model.json"
    saved = json.loads(path.read_text())
    if saved["environment"]["lion"]["commit"] is None:
        pytest.skip("LION is not a git checkout here")
    saved["environment"]["lion"]["commit"] = "0" * 40
    path.write_text(json.dumps(saved))
    with pytest.warns(UserWarning, match="saved with other LION code.*0000000000"):
        DnCNN.load(tmp_path / "model", supress_warnings=True)


def test_other_version_without_git_warns():
    record = {"lion": {"version": "0.1.0", "commit": None}}
    assert "LION 0.1.0" in environment.code_differences(record)
    record["lion"]["version"] = LION.__version__
    if environment.code_state()["commit"] is None:
        assert environment.code_differences(record) is None


# --------------------------------------------------------------------------
# LION's code: a package in a temporary git repository


def git(folder, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=folder,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repository(tmp_path):
    """A git repository with a package ``pkg`` in it, committed."""
    root = tmp_path / "repo"
    package = root / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "a.py").write_text("A = 1\n")
    (root / "notes.txt").write_text("outside the package\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "first")
    environment._git_root.cache_clear()
    environment._pushed.cache_clear()
    return root


def test_clean_checkout(repository):
    state = environment.code_state(repository / "pkg")
    assert len(state["commit"]) == 40
    assert state["uncommitted"] == [] and state["diff_sha256"] is None
    assert state["pushed"] is False  # no remote


def test_changed_files(repository):
    package = repository / "pkg"
    (package / "a.py").write_text("A = 2\n")
    (repository / "notes.txt").write_text("changed, but not in the package\n")
    first = environment.code_state(package)
    assert first["uncommitted"] == ["pkg/a.py"]
    (package / "a.py").write_text("A = 3\n")
    second = environment.code_state(package)
    assert second["diff_sha256"] not in (None, first["diff_sha256"])


def test_untracked_files_count_if_imported(repository, monkeypatch):
    package = repository / "pkg"
    (package / "leftover.py").write_text("X = 1\n")  # e.g. an old file
    (package / "new.py").write_text("Y = 1\n")
    assert environment.code_state(package)["uncommitted"] == []
    monkeypatch.syspath_prepend(str(repository))
    for name in [m for m in sys.modules if m == "pkg" or m.startswith("pkg.")]:
        monkeypatch.delitem(sys.modules, name)
    importlib.import_module("pkg.new")
    assert environment.code_state(package)["uncommitted"] == ["pkg/new.py"]


def test_pushed(repository, tmp_path):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(repository, "remote", "add", "origin", str(remote))
    git(repository, "push", "-q", "origin", "main")
    assert environment.code_state(repository / "pkg")["pushed"] is True
    (repository / "pkg" / "a.py").write_text("A = 2\n")
    git(repository, "commit", "-q", "-am", "second")  # not pushed
    assert environment.code_state(repository / "pkg")["pushed"] is False


def test_not_a_checkout(tmp_path):
    environment._git_root.cache_clear()
    state = environment.code_state(tmp_path)
    assert state == {
        "commit": None,
        "uncommitted": None,
        "diff_sha256": None,
        "pushed": None,
    }


# --------------------------------------------------------------------------
# What keeps a result from being published


def record_of(**lion):
    return LIONParameter(lion=LIONParameter(**lion))


def test_publish_problems():
    clean = dict(commit="a" * 40, uncommitted=[], pushed=True)
    assert environment.publish_problems(record_of(**clean)) == []
    dirty = dict(clean, uncommitted=["LION/a.py"])
    assert "uncommitted changes" in environment.publish_problems(record_of(**dirty))[0]
    unpushed = dict(clean, pushed=False)
    assert "not pushed" in environment.publish_problems(record_of(**unpushed))[0]
    assert (
        environment.publish_problems(record_of(commit=None, distribution="0.2.0")) == []
    )
    for distribution in ("0.2.0+dirtyabc", None):
        problems = environment.publish_problems(
            record_of(commit=None, distribution=distribution)
        )
        assert "released version" in problems[0]


# --------------------------------------------------------------------------
# The solvers' deterministic switch


@pytest.fixture
def restore_determinism():
    yield
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def make_solver(**kwargs):
    model = small_dncnn()
    return SupervisedSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.nn.MSELoss(),
        geometry=Geometry.default_parameters(),
        device=torch.device("cpu"),
        **kwargs,
    )


def test_not_deterministic_by_default(monkeypatch):
    monkeypatch.setattr(lion_solver, "make_operator", lambda geometry: None)
    solver = make_solver()
    assert solver.deterministic is False
    assert not torch.are_deterministic_algorithms_enabled()


def test_deterministic(monkeypatch, restore_determinism):
    monkeypatch.setattr(lion_solver, "make_operator", lambda geometry: None)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    solver = make_solver(deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert solver.spec_kwargs()["deterministic"] is True
    assert environment.environment_record("cpu").torch.deterministic_algorithms
