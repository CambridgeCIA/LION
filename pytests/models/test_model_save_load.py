"""Save/load round-trip tests for LIONmodel.

Regression tests for two bugs that together broke loading saved models:

- ``LIONmodel.load()``/``load_checkpoint()`` passed the loaded parameters to the
  model as a plain ``LIONParameter`` (that is how nested parameters come back from
  JSON), which the ``LIONModelParameter`` type check in ``LIONmodel.__init__``
  rejects.
- ``LIONModelParameter.__init__`` applied its defaults *after* the keyword
  arguments, so values passed in (e.g. ``model_input_type`` when loading) were
  overwritten with ``None``.

DnCNN is used because it needs no CT geometry, so these tests run on CPU.
"""

import pytest
import torch

from LION.models.CNNs.dncnn import DnCNN
from LION.models.LIONmodel import LIONModelParameter, ModelInputType
from LION.utils.parameter import LIONParameter

# Saving a model without geometry warns about it; that is expected for DnCNN.
pytestmark = pytest.mark.filterwarnings("ignore:Expected 'geometry' parameter")


def _save(model, path):
    model.save(path, dataset=LIONParameter(name="test"), training=LIONParameter())


def _json_normalised(params):
    # JSON has no tuple type, so tuples come back as lists.
    return {k: list(v) if isinstance(v, tuple) else v for k, v in vars(params).items()}


def test_model_parameter_defaults():
    params = LIONModelParameter()
    assert params.model_input_type is None
    assert params.normalisator is None


def test_model_parameter_kwargs_are_not_overwritten():
    params = LIONModelParameter(model_input_type=ModelInputType.IMAGE, depth=3)
    assert params.model_input_type == ModelInputType.IMAGE
    assert params.depth == 3


def test_model_parameter_file_round_trip(tmp_path):
    params = LIONModelParameter()
    params.model_input_type = ModelInputType.IMAGE
    params.save(tmp_path / "params.json")

    loaded = LIONModelParameter()
    loaded.load(tmp_path / "params.json")
    assert loaded.model_input_type == ModelInputType.IMAGE


def test_load_round_trip(tmp_path):
    torch.manual_seed(0)
    model = DnCNN()
    _save(model, tmp_path / "model")

    loaded, options, _ = DnCNN.load(tmp_path / "model", supress_warnings=True)

    assert isinstance(loaded, DnCNN)
    assert isinstance(loaded.model_parameters, LIONModelParameter)
    assert loaded.get_input_type() is ModelInputType.IMAGE
    assert _json_normalised(loaded.model_parameters) == _json_normalised(
        model.model_parameters
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key].cpu(), value.cpu())


def test_load_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = DnCNN()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
    model.save_checkpoint(
        tmp_path / "checkpoint",
        epoch=3,
        loss=0.5,
        optimizer=optimiser,
        training_param=LIONParameter(),
        dataset=LIONParameter(name="test"),
    )

    loaded, _, data = DnCNN.load_checkpoint(tmp_path / "checkpoint")

    assert isinstance(loaded.model_parameters, LIONModelParameter)
    assert loaded.get_input_type() is ModelInputType.IMAGE
    assert data["epoch"] == 3
    assert data["loss"] == 0.5
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key].cpu(), value.cpu())


@pytest.mark.filterwarnings("ignore")
def test_save_without_optional_metadata(tmp_path):
    # dataset/training/geometry are optional (save() only warns)
    DnCNN().save(tmp_path / "model")
    loaded, _, _ = DnCNN.load(tmp_path / "model", supress_warnings=True)
    assert isinstance(loaded, DnCNN)


def test_save_does_not_change_model_parameters(tmp_path):
    params = DnCNN.default_parameters()
    params.extra = LIONParameter(depth=2)
    model = DnCNN(params)
    _save(model, tmp_path / "model")
    assert isinstance(model.model_parameters.extra, LIONParameter)
    assert model.model_parameters.extra.depth == 2
