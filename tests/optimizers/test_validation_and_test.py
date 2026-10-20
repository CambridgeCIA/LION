"""Validation and testing must run for every model type, with the same
(output, target) argument order as the training loss."""

import pytest
import torch

from LION.models.LIONmodel import ModelInputType
from tests.optimizers._helpers import loader, make_solver, small_dncnn

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.mark.parametrize("input_type", [ModelInputType.IMAGE, ModelInputType.SINOGRAM])
def test_validation_runs_for_all_input_types(tmp_path, input_type):
    solver = make_solver(small_dncnn(input_type))
    solver.set_saving(tmp_path, "model")
    solver.set_validation(loader(n=6), validation_freq=1)
    losses = solver._validation_loop()
    assert losses.shape == (3,)  # one value per batch
    assert solver.validate() > 0


def _order_check_fn(targets):
    seen = []

    def fn(first, second):
        # targets come straight from the loader, so they are leaf tensors with the
        # exact values of the dataset; the model output is not.
        seen.append(any(torch.equal(second, t) for t in targets))
        return torch.mean((first - second) ** 2)

    return fn, seen


@pytest.mark.parametrize("input_type", [ModelInputType.IMAGE, ModelInputType.SINOGRAM])
def test_validation_and_test_take_output_then_target(tmp_path, input_type):
    data = loader(n=4, batch_size=2)
    targets = [y for _, y in data]
    solver = make_solver(small_dncnn(input_type))
    solver.set_saving(tmp_path, "model")

    fn, seen = _order_check_fn(targets)
    solver.set_validation(data, validation_freq=1, validation_fn=fn)
    solver._validation_loop()
    assert seen == [True, True]

    fn, seen = _order_check_fn(targets)
    solver.set_testing(data, testing_fn=fn)
    solver.test()
    assert seen == [True, True]
