# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Utilities: the default device, LIONParameter (de)serialisation and the
download/unzip command lines."""

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


def test_follows_torch_set_default_device(no_cuda, restore_default_device, monkeypatch):
    torch.set_default_device("meta")  # any non-CPU device; meta needs no GPU
    assert default_device() == torch.device("meta")

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


def test_download_and_unzip_fallbacks(commands, tmp_path):
    calls, fail = commands
    target = tmp_path / "my data" / "file.zip"
    utils.download_file("https://example.org/file.zip", target)
    assert calls == [["wget", "https://example.org/file.zip", "-O", str(target)]]

    calls, fail = commands
    fail.add("wget")
    target = tmp_path / "other" / "file.zip"
    utils.download_file("https://example.org/file.zip", target)
    assert calls[-1] == [
        "curl",
        "-fL",
        "https://example.org/file.zip",
        "-o",
        str(target),
    ]

    calls.clear()
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

    spec = {"_target_": "torch.nn.modules.loss.MSELoss", "reduction": "mean"}
    p = LIONParameter(loss=spec, inner={"a": 1, "spec": spec})
    assert p.loss == spec and isinstance(p.loss, dict)
    assert isinstance(p.inner, LIONParameter) and p.inner.spec == spec
    assert p.serialize() == {"loss": spec, "inner": {"a": 1, "spec": spec}}
