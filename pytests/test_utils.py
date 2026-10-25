"""Utilities: the default device, LIONParameter (de)serialisation and the
download/unzip command lines."""

import pathlib
import shlex
import subprocess

import numpy as np
import pytest
import torch

import LION.optimizers.LIONsolver as lion_solver
import LION.optimizers.SupervisedSolver as supervised_solver
import LION.utils.utils as utils
from LION.CTtools.ct_geometry import Geometry
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


def test_cpu_without_cuda(no_cuda):
    assert default_device() == torch.device("cpu")


def test_follows_torch_set_default_device(no_cuda, restore_default_device):
    torch.set_default_device("meta")  # any non-CPU device; meta needs no GPU
    assert default_device() == torch.device("meta")


def test_follows_torch_cuda_set_device(monkeypatch):
    # what torch.cuda.set_device(2) does, without needing three GPUs
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert default_device() == torch.device("cuda", 2)


def test_solver_uses_default_device(monkeypatch, restore_default_device):
    monkeypatch.setattr(lion_solver, "make_operator", lambda geometry: None)
    monkeypatch.setattr(supervised_solver, "make_operator", lambda geometry: None)
    torch.set_default_device("meta")
    model = small_dncnn()
    solver = supervised_solver.SupervisedSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.nn.MSELoss(),
        geometry=Geometry.default_parameters(),
    )
    assert solver.device == torch.device("meta")
    assert next(solver.model.parameters()).device == torch.device("meta")


def test_explicit_device_wins(monkeypatch):
    monkeypatch.setattr(lion_solver, "make_operator", lambda geometry: None)
    monkeypatch.setattr(supervised_solver, "make_operator", lambda geometry: None)
    model = small_dncnn()
    solver = supervised_solver.SupervisedSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.nn.MSELoss(),
        geometry=Geometry.default_parameters(),
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


def test_save_load_round_trip(tmp_path):
    p = LIONParameter(a=1, inner={"b": [1, 2]})
    p.save(tmp_path / "params.json")
    q = LIONParameter()
    q.load(tmp_path / "params.json")
    assert q.a == 1
    assert isinstance(q.inner, LIONParameter) and q.inner.b == [1, 2]


def test_geometry_round_trip_keeps_image_pos(tmp_path):
    geo = Geometry.default_parameters()
    geo.image_pos = np.array([0.0, 10.0, -5.0])
    geo.save(tmp_path / "geo.json")

    loaded = LIONParameter()
    loaded.load(tmp_path / "geo.json")
    restored = Geometry.init_from_parameter(loaded)

    np.testing.assert_allclose(restored.image_pos, geo.image_pos)
    np.testing.assert_allclose(restored.angles, geo.angles)
    assert restored.mode == geo.mode


def test_git_hash_is_lions_not_the_working_directory(tmp_path, monkeypatch):
    lion_folder = pathlib.Path(utils.__file__).resolve().parent
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=lion_folder, capture_output=True, text=True
    )
    if expected.returncode != 0:
        pytest.skip("LION is not a git checkout here")
    monkeypatch.chdir(tmp_path)  # not a git repository
    assert utils.get_git_revision_hash() == expected.stdout.strip()


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


# --------------------------------------------------------------------------
# Testing a saved model on an experiment


def test_test_with_experiment_evaluates_in_eval_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("LION_DATA_PATH", str(tmp_path))  # needed to import experiments
    from torch.utils.data import TensorDataset

    from LION.models.CNNs.dncnn import DnCNN
    from LION.utils.test_model import test_with_experiment

    torch.manual_seed(0)
    model = small_dncnn().train()  # DnCNN has BatchNorm layers

    class Saved(DnCNN):
        @classmethod
        def load(cls, fname):
            return model, None, None

    class Experiment:
        def get_testing_dataset(self):
            return TensorDataset(torch.rand(4, 1, 8, 8), torch.rand(4, 1, 8, 8))

    pred, _ = test_with_experiment(Experiment(), Saved, "model.pt", n=2)
    assert not model.training and not pred.requires_grad


# --------------------------------------------------------------------------
# Plain-torch image transforms (replacing torchvision's)


def test_gaussian_kernel():
    from LION.utils.image_transforms import gaussian_kernel1d

    k = gaussian_kernel1d(5, 0.8)
    torch.testing.assert_close(k.sum(), torch.tensor(1.0))
    torch.testing.assert_close(k, k.flip(0))
    x = torch.arange(-2.0, 3.0)
    torch.testing.assert_close(k[0] / k[2], torch.exp(-0.5 * (x[0] / 0.8) ** 2))
    with pytest.raises(ValueError):
        gaussian_kernel1d(4, 1.0)


def test_gaussian_blur_dims():
    from LION.utils.image_transforms import gaussian_blur

    torch.manual_seed(0)
    x = torch.rand(2, 1, 6, 10)
    torch.testing.assert_close(
        gaussian_blur(torch.ones(1, 6, 10), 5, 1.0), torch.ones(1, 6, 10)
    )
    along_last = gaussian_blur(x, 3, 0.5, dims=(-1,))
    # rows are blurred independently
    torch.testing.assert_close(
        along_last[:, :, :1], gaussian_blur(x[:, :, :1], 3, 0.5, dims=(-1,))
    )
    torch.testing.assert_close(
        gaussian_blur(x, 3, 0.5),
        gaussian_blur(gaussian_blur(x, 3, 0.5, dims=(-1,)), 3, 0.5, dims=(-2,)),
    )


@pytest.mark.parametrize("shape", [(2, 1, 16, 16), (3, 15, 17)])
def test_rotate(shape):
    from LION.utils.image_transforms import rotate

    torch.manual_seed(0)
    x = torch.rand(*shape)
    torch.testing.assert_close(rotate(x, 0.0), x)
    # 90 degrees counter-clockwise, for square images exactly a rot90
    if shape[-1] == shape[-2]:
        torch.testing.assert_close(rotate(x, 90.0), torch.rot90(x, 1, dims=(-2, -1)))
    disk = torch.zeros(1, 1, 32, 32)
    yy, xx = torch.meshgrid(
        torch.arange(32.0) - 15.5, torch.arange(32.0) - 15.5, indexing="ij"
    )
    disk[..., (xx**2 + yy**2) < 100] = 1.0
    torch.testing.assert_close(rotate(disk, 37.0, "bilinear"), disk, atol=0.6, rtol=0)
    with pytest.raises(ValueError):
        rotate(x, 10.0, "bicubic")


@pytest.mark.parametrize("angle", [30.0, 137.5, -45.0])
def test_transforms_match_torchvision(angle):
    """Same results as torchvision, which LION used before."""
    TF = pytest.importorskip("torchvision.transforms.functional")
    from LION.utils.image_transforms import gaussian_blur, rotate

    torch.manual_seed(0)
    x = torch.rand(2, 1, 20, 23)
    bilinear = TF.InterpolationMode.BILINEAR
    torch.testing.assert_close(
        rotate(x, angle, "bilinear"), TF.rotate(x, angle, interpolation=bilinear)
    )
    # nearest: equal up to rounding at exact ties between two pixels
    same = rotate(x, angle) == TF.rotate(x, angle)
    assert same.float().mean() > 0.99
    # ct_utils's detector cross-talk, and Noisier2Inverse's 2D blur
    torch.testing.assert_close(
        gaussian_blur(x, 3, 0.3015, dims=(-1,)),
        TF.gaussian_blur(x, kernel_size=[3, 1], sigma=[0.3015, 1.0]),
    )
    torch.testing.assert_close(
        gaussian_blur(x, 7, 1.2),
        TF.gaussian_blur(x, kernel_size=[7, 7], sigma=[1.2, 1.2]),
    )
