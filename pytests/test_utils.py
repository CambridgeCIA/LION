"""Utilities: the default device, LIONParameter (de)serialisation and the
download/unzip command lines."""

import pathlib
import shlex


import pytest
import torch

import LION.optimizers.SupervisedSolver as supervised_solver
import LION.utils.utils as utils
from LION.operators.ct.ct_geometry import Geometry
from LION.utils.device import default_device
from LION.utils.parameter import LIONParameter
from pytests.helpers import small_dncnn

# --------------------------------------------------------------------------
# LION's default device follows torch.set_default_device / torch.cuda.set_device


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def restore_default_device():
    yield
    torch.set_default_device("cpu")


def test_follows_torch_set_default_device(no_cuda, restore_default_device):
    torch.set_default_device("meta")  # any non-CPU device; meta needs no GPU
    assert default_device() == torch.device("meta")


def test_follows_torch_cuda_set_device(monkeypatch):
    # what torch.cuda.set_device(2) does, without needing three GPUs
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert default_device() == torch.device("cuda", 2)


def test_solver_uses_default_device(monkeypatch, restore_default_device):
    monkeypatch.setattr(Geometry, "make_operator", lambda self: None)
    torch.set_default_device("meta")
    model = small_dncnn()
    solver = supervised_solver.SupervisedSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.nn.MSELoss(),
        operator_params=Geometry.default_parameters(),
    )
    assert solver.device == torch.device("meta")
    assert next(solver.model.parameters()).device == torch.device("meta")


def test_explicit_device_wins(monkeypatch):
    monkeypatch.setattr(Geometry, "make_operator", lambda self: None)
    model = small_dncnn()
    solver = supervised_solver.SupervisedSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.nn.MSELoss(),
        operator_params=Geometry.default_parameters(),
        device=torch.device("cpu"),
    )
    assert solver.device == torch.device("cpu")


# --------------------------------------------------------------------------
# LIONParameter and Geometry (de)serialisation


def test_serialize_does_not_modify_the_parameter():
    p = LIONParameter(a=1)
    p.inner = LIONParameter(b=2)
    p.path = pathlib.Path("some/folder")

    d = p.serialize()

    assert d == {"a": 1, "inner": {"b": 2}, "path": "some/folder"}
    assert isinstance(p.inner, LIONParameter)
    assert isinstance(p.path, pathlib.Path)


# --------------------------------------------------------------------------
# Command lines built by the download/unzip helpers


@pytest.fixture
def commands(monkeypatch):
    calls = []

    def fake_run_cmd(cmd, verbose=True):
        calls.append(shlex.split(cmd))
        if calls[-1][0] in fake_run_cmd.fail:
            raise RuntimeError("failed")
        return ""

    fake_run_cmd.fail = set()
    monkeypatch.setattr(utils, "run_cmd", fake_run_cmd)
    return calls, fake_run_cmd.fail


def test_wget_writes_to_the_requested_file(commands, tmp_path):
    calls, _ = commands
    target = tmp_path / "my data" / "file.zip"
    utils.download_file("https://example.org/file.zip", target)
    assert calls == [["wget", "https://example.org/file.zip", "-O", str(target)]]


def test_curl_fallback(commands, tmp_path):
    calls, fail = commands
    fail.add("wget")
    target = tmp_path / "file.zip"
    utils.download_file("https://example.org/file.zip", target)
    assert calls[-1] == [
        "curl",
        "-fL",
        "https://example.org/file.zip",
        "-o",
        str(target),
    ]


def test_7z_fallback_output_folder(commands, tmp_path):
    calls, fail = commands
    fail.add("unzip")
    archive = tmp_path / "a b.zip"
    archive.write_bytes(b"")
    out = tmp_path / "out dir"
    utils.unzip_file(archive, out)
    assert calls[0] == ["unzip", str(archive), "-d", str(out)]
    assert calls[1] == ["7z", "x", str(archive), f"-o{out}", "-y"]


def test_lion_parameter_keeps_specs_as_dicts():
    """A dict with "_target_" describes an object (LION.utils.spec): it stays a
    dict; other dicts become LIONParameters, as before."""
    from LION.utils.parameter import LIONParameter

    spec = {"_target_": "torch.nn.modules.loss.MSELoss", "reduction": "mean"}
    p = LIONParameter(loss=spec, inner={"a": 1, "spec": spec})
    assert p.loss == spec and isinstance(p.loss, dict)
    assert isinstance(p.inner, LIONParameter) and p.inner.spec == spec
    assert p.serialize() == {"loss": spec, "inner": {"a": 1, "spec": spec}}


# --------------------------------------------------------------------------
# LION's folders


def test_data_folders_are_read_when_used(monkeypatch, tmp_path):
    """LION_DATA_PATH and LION_RESULTS_PATH are read when a path is used, as
    saving reads them, so a saved path means the same folder when read."""
    from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
    from LION.utils import paths, spec

    monkeypatch.setenv("LION_DATA_PATH", str(tmp_path / "a"))
    assert paths.data_path("x") == (tmp_path / "a" / "x").resolve()
    monkeypatch.setenv("LION_DATA_PATH", str(tmp_path / "b"))
    folder = (tmp_path / "b" / "processed" / "LIDC-IDRI").resolve()
    assert paths.LIDC_IDRI_PROCESSED_DATASET_PATH == folder
    parameters = LIDC_IDRI.default_parameters(
        operator_params=Geometry.default_parameters()
    )
    assert parameters.folder == folder
    written = spec.encode(parameters.folder)
    assert written == {
        "_target_": "LION.utils.paths.data_path",
        "_args_": ["processed/LIDC-IDRI"],
    }
    assert spec.build(written) == folder
    monkeypatch.delenv("LION_DATA_PATH")
    with pytest.raises(RuntimeError, match="LION_DATA_PATH is not set"):
        paths.data_path("x")
    monkeypatch.delenv("LION_RESULTS_PATH", raising=False)
    assert paths.LION_RESULTS_PATH is None
    monkeypatch.setenv("LION_RESULTS_PATH", str(tmp_path / "r"))
    assert paths.LION_RESULTS_PATH == (tmp_path / "r").resolve()
