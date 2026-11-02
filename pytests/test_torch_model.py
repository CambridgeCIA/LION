"""TorchModel: a plain torch module as a LIONmodel."""

import pytest
import torch

from LION.models.LIONmodel import LIONmodel
from LION.models.torch_model import TorchModel
from LION.utils.parameter import LIONParameter
from LION.utils.spec import SpecError


def _save(model, path):
    model.save(path, dataset=LIONParameter(name="test"), training=LIONParameter())


class TwoConvolutions(torch.nn.Module):
    """A plain torch module, as a user would write one."""

    def __init__(self, channels, kernel_size=(3, 3)):
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)
        self.first = torch.nn.Conv2d(1, channels, kernel_size, padding=padding)
        self.second = torch.nn.Conv2d(channels, 1, kernel_size, padding=padding)

    def forward(self, x):
        return self.second(torch.relu(self.first(x)))


def test_torch_model_saves_and_loads_a_torch_module(tmp_path):
    torch.manual_seed(0)
    model = TorchModel(
        torch.nn.Conv2d,
        {"in_channels": 1, "out_channels": 1, "kernel_size": (3, 3), "padding": 1},
    ).eval()
    _save(model, tmp_path / "model")
    loaded, _, _ = LIONmodel.load(tmp_path / "model", supress_warnings=True)
    loaded = loaded.cpu().eval()
    assert isinstance(loaded, TorchModel) and loaded.module.kernel_size == (3, 3)
    x = torch.rand(1, 1, 8, 8)
    torch.testing.assert_close(loaded(x), model(x))


def test_torch_model_loads_your_own_module_when_allowed(tmp_path):
    torch.manual_seed(0)
    model = TorchModel(TwoConvolutions, {"channels": 4}).eval()
    _save(model, tmp_path / "model")
    with pytest.raises(SpecError, match="allow"):
        TorchModel.load(tmp_path / "model", supress_warnings=True)
    loaded, _, _ = TorchModel.load(
        tmp_path / "model", supress_warnings=True, allow=["pytests."]
    )
    x = torch.rand(1, 1, 8, 8)
    torch.testing.assert_close(loaded.cpu().eval()(x), model(x))


class SigmaKeyword(torch.nn.Module):
    """A denoiser taking the noise level as ``sigma`` (as deepinv's do)."""

    def forward(self, x, sigma=None):
        return x * sigma


class TwoChannels(torch.nn.Module):
    """A denoiser taking the noise level as a second input channel."""

    def forward(self, x):
        return x[:, :1] * x[:, 1:]


def test_torch_model_finds_the_noise_level_keyword():
    from pytests.helpers import NoiseLevelConvolution

    assert TorchModel(NoiseLevelConvolution).takes_noise_level()
    model = TorchModel(SigmaKeyword)
    assert model.takes_noise_level()
    x = torch.ones(1, 1, 4, 4)
    torch.testing.assert_close(model(x, noise_level=0.5), x * 0.5)
    assert not TorchModel(TwoConvolutions, {"channels": 2}).takes_noise_level()


def test_torch_model_gives_the_noise_level_as_a_channel(tmp_path, monkeypatch):
    from LION.models import torch_model

    x = torch.ones(1, 1, 4, 4)
    assert not TorchModel(TwoChannels).takes_noise_level()
    model = TorchModel(TwoChannels, noise_level="channel")
    torch.testing.assert_close(model(x, noise_level=0.5), x * 0.5)
    _save(model, tmp_path / "model")  # how it takes it is saved with it
    loaded, _, _ = TorchModel.load(
        tmp_path / "model", supress_warnings=True, allow=["pytests."]
    )
    torch.testing.assert_close(loaded.cpu()(x, noise_level=0.5), x * 0.5)

    path = f"{TwoChannels.__module__}.TwoChannels"
    monkeypatch.setitem(torch_model.NOISE_LEVEL, path, "channel")
    assert TorchModel(TwoChannels).takes_noise_level()
