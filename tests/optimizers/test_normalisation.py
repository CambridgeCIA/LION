"""model.set_normalisation() must actually be used by the solvers."""

import pytest
import torch

from tests.optimizers._helpers import loader, make_solver, small_dncnn

pytestmark = pytest.mark.filterwarnings("ignore")


def _capture_inputs(model):
    seen = []
    model.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
    return seen


def test_do_normalise_follows_model():
    solver = make_solver()
    assert not solver.do_normalise
    solver.model.set_normalisation("sample")
    assert solver.do_normalise


def test_training_step_normalises_input_and_returns_target_units():
    solver = make_solver()
    solver.model.set_normalisation("sample")
    seen = _capture_inputs(solver.model)

    data = 100 + 50 * torch.rand(2, 1, 8, 8)  # far from [0, 1]
    target = data.clone()
    loss = solver.mini_batch_step(data, target)

    assert torch.isfinite(loss)
    model_input = seen[-1]
    assert model_input.min() >= 0 and model_input.max() <= 1
    # output is mapped back to the data range before the loss
    output = solver._forward(data)
    assert output.mean() > 50


def test_validation_uses_normalisation(tmp_path):
    solver = make_solver()
    solver.model.set_normalisation("sample")
    solver.set_saving(tmp_path, "model")
    solver.set_validation(loader(), validation_freq=1)
    seen = _capture_inputs(solver.model)
    solver._validation_loop()
    assert all(x.min() >= 0 and x.max() <= 1 for x in seen)


def test_model_normalise_is_identity_without_normaliser():
    model = small_dncnn()
    x = torch.rand(2, 1, 8, 8) * 10
    assert model.normalise(x) is x
    assert model.unnormalise(x) is x


def test_model_normalise_does_not_hide_errors():
    model = small_dncnn()
    model.set_normalisation("sample")
    with pytest.raises(Exception):
        model.normalise(torch.rand(8))  # 'sample' needs (B, C, H, W)
