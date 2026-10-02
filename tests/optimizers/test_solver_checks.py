"""Tests for the LIONsolver readiness checks.

DnCNN is used because it needs no CT geometry of its own, so these run on CPU.
"""

import sys
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# tomosipo needs a CUDA-enabled astra build, unavailable on some platforms.
try:
    import tomosipo  # noqa: F401
except ImportError:
    sys.modules["tomosipo"] = MagicMock()
    sys.modules["tomosipo.torch_support"] = MagicMock()

# Other test modules mock LION modules in sys.modules for the whole process, so
# drop those fakes to keep the imports below order-independent.
for _name, _module in list(sys.modules.items()):
    if _name.startswith("LION") and isinstance(_module, MagicMock):
        del sys.modules[_name]

from LION.CTtools.ct_geometry import Geometry
from LION.models.CNNs.dncnn import DnCNN
from LION.optimizers.SupervisedSolver import SupervisedSolver

# The checks warn about whatever is not set; that is the behaviour under test.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _loader():
    data = torch.zeros(2, 1, 4, 4)
    return DataLoader(TensorDataset(data, data), batch_size=1)


def _solver(**kwargs):
    model = DnCNN()
    return SupervisedSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.nn.MSELoss(),
        geometry=Geometry.default_parameters(),
        device=torch.device("cpu"),
        **kwargs,
    )


def _ready_solver(tmp_path):
    """A solver with training, validation, testing and saving all configured."""
    solver = _solver()
    solver.set_training(_loader())
    solver.set_saving(tmp_path, "final_result.pt")
    solver.set_validation(_loader(), 1)
    solver.set_testing(_loader())
    return solver


def test_fully_configured_solver_reports_ready(tmp_path):
    solver = _ready_solver(tmp_path)
    assert solver.check_training_ready() == 0
    assert solver.check_validation_ready() == 0
    assert solver.check_testing_ready() == 0
    assert solver.check_checkpointing_ready() == 0
    assert solver.check_complete() == 0


def test_check_validation_ready_reports_missing_loader(tmp_path):
    solver = _ready_solver(tmp_path)
    solver.validation_loader = None

    # callers run validation when this returns 0
    assert solver.check_validation_ready() != 0


def test_check_training_ready_reports_failure_from_an_earlier_check(tmp_path):
    solver = _ready_solver(tmp_path)
    solver.train_loader = None  # fails an early check; later checks still pass

    assert solver.check_training_ready(error=False) != 0


def test_check_testing_ready_reports_failure_from_an_earlier_check(tmp_path):
    solver = _ready_solver(tmp_path)
    solver.test_loader = None  # the testing_fn check after it still passes

    assert solver.check_testing_ready(error=False) != 0


def test_check_complete_reports_failure_from_an_earlier_check(tmp_path):
    solver = _ready_solver(tmp_path)
    solver.test_loader = None  # checkpointing is checked last and autofills

    assert solver.check_complete(error=False) != 0


def test_check_complete_reports_bad_validation_fn_when_not_autofilling(tmp_path):
    solver = _ready_solver(tmp_path)
    solver.validation_fn = 42  # not callable
    solver.verbose = True

    # check_complete must pass its own `verbose` to check_validation_ready
    with pytest.warns(UserWarning, match="validation_fn"):
        assert solver.check_complete(error=False, autofill=False) != 0


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="checks the fallback when CUDA is absent"
)
def test_solver_defaults_to_cpu_without_cuda():
    model = DnCNN()
    solver = SupervisedSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.nn.MSELoss(),
        geometry=Geometry.default_parameters(),
    )
    assert solver.device == torch.device("cpu")
