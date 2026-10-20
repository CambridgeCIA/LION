"""Solvers validate and test under torch.no_grad(); gradient-step denoisers
enable autograd themselves, so they work too."""

import pytest

from pytests.models.test_gradient_step_denoisers import small_gsd, small_gsdrunet
from pytests.optimizers._helpers import loader, make_solver

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.mark.parametrize("make_model", [small_gsd, small_gsdrunet])
def test_gradient_step_denoisers_validate_and_test(tmp_path, make_model):
    solver = make_solver(make_model())
    solver.set_saving(tmp_path, "model")
    data = loader(n=2, batch_size=2)
    solver.set_validation(data, validation_freq=1)
    solver.set_testing(data)
    assert solver.validate() > 0
    assert solver.test().shape == (1,)
