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
