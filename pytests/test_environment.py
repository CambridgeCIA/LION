# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""The record of the environment a model was saved in (LION.utils.environment):
LION's version and code state, packages, determinism settings; the warning on
loading a file saved with other code; what keeps a result from being published;
the solvers' deterministic switch."""

import importlib
import importlib.metadata
import json
import subprocess
import sys
import warnings

import pytest

import LION
from LION.models.CNNs.dncnn import DnCNN
from LION.utils import environment
from LION.utils.parameter import LIONParameter
from pytests.helpers import small_dncnn


def save(model, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # no dataset/training given
        model.save(path)


# --------------------------------------------------------------------------
# The record


def test_record(tmp_path):
    record = environment.environment_record("cpu")
    assert isinstance(record, LIONParameter)
    assert record.lion.version == LION.__version__
    assert record.packages.torch == importlib.metadata.version("torch")
    assert record.torch.deterministic_algorithms is False
    assert record.device.type == "cpu"
    assert not hasattr(record, "hostname")
    json.dumps(record.serialize())  # plain JSON

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
    return root


def test_git_state(repository, monkeypatch, tmp_path):
    state = environment.code_state(repository / "pkg")
    assert len(state["commit"]) == 40
    assert state["uncommitted"] == [] and state["diff_sha256"] is None
    assert state["pushed"] is False  # no remote

    package = repository / "pkg"
    (package / "a.py").write_text("A = 2\n")
    (repository / "notes.txt").write_text("changed, but not in the package\n")
    first = environment.code_state(package)
    assert first["uncommitted"] == ["pkg/a.py"]
    (package / "a.py").write_text("A = 3\n")
    second = environment.code_state(package)
    assert second["diff_sha256"] not in (None, first["diff_sha256"])
    git(repository, "checkout", "-q", "--", "pkg/a.py")

    (package / "leftover.py").write_text("X = 1\n")  # e.g. an old file
    (package / "new.py").write_text("Y = 1\n")
    assert environment.code_state(package)["uncommitted"] == []
    monkeypatch.syspath_prepend(str(repository))
    for name in [m for m in sys.modules if m == "pkg" or m.startswith("pkg.")]:
        monkeypatch.delitem(sys.modules, name)
    importlib.import_module("pkg.new")
    assert environment.code_state(package)["uncommitted"] == ["pkg/new.py"]

    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(repository, "remote", "add", "origin", str(remote))
    git(repository, "push", "-q", "origin", "main")
    assert environment.code_state(repository / "pkg")["pushed"] is True
    (repository / "pkg" / "a.py").write_text("A = 2\n")
    git(repository, "commit", "-q", "-am", "second")  # not pushed
    assert environment.code_state(repository / "pkg")["pushed"] is False
    git(repository, "push", "-q", "origin", "main")  # pushed while it runs
    assert environment.code_state(repository / "pkg")["pushed"] is True

    environment._git_root.cache_clear()
    state = environment.code_state(tmp_path)
    assert state == {
        "commit": None,
        "uncommitted": None,
        "diff_sha256": None,
        "pushed": None,
    }

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
# What keeps a result from being published


def record_of(**lion):
    return LIONParameter(lion=LIONParameter(**lion))
