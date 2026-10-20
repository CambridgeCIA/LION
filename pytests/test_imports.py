"""LION's core (operators, classical algorithms, models, solvers base) must import
without tomosipo/astra, so the pure-torch parts can be used on any machine.
Importing must also not touch CUDA (checked by making every CUDA query fail)."""

import subprocess
import sys

import pytest

MODULES = [
    "LION.operators",
    "LION.classical_algorithms",
    "LION.CTtools.ct_utils",
    "LION.models.LIONmodel",
    "LION.models.CNNs.dncnn",
    "LION.models.CNNs.drunet",
    "LION.models.CNNs.UNets.Unet",
    "LION.reconstructors",
    "LION.optimizers.LIONsolver",
    "LION.optimizers.SupervisedSolver",
    "LION.optimizers.EquivariantSolver",
    "LION.optimizers.AdversarialRegularizerSolver",
]


@pytest.mark.parametrize("module", MODULES)
def test_imports_without_tomosipo(module):
    # Setting a sys.modules entry to None makes any later import of it fail.
    code = (
        "import sys\n"
        "for name in ('tomosipo', 'tomosipo.torch_support', 'ts_algorithms', 'astra'):\n"
        "    sys.modules[name] = None\n"
        "import torch\n"
        "def _no_cuda(*args, **kwargs):\n"
        "    raise RuntimeError('CUDA queried at import time')\n"
        "torch.cuda.current_device = _no_cuda\n"
        f"import {module}\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "call",
    [
        "from LION.classical_algorithms import fdk; fdk(torch.zeros(1, 2, 3), Geometry.default_parameters())",
        "from LION.classical_algorithms import sirt; sirt(torch.zeros(1, 2, 3), Geometry.default_parameters(), 2)",
        "from LION.operators import CTProjectionOp; CTProjectionOp(Geometry.default_parameters())",
    ],
)
def test_ct_functions_say_what_is_missing(call):
    code = (
        "import sys\n"
        "for name in ('tomosipo', 'tomosipo.torch_support', 'ts_algorithms', 'astra'):\n"
        "    sys.modules[name] = None\n"
        "import torch\n"
        "from LION.CTtools.ct_geometry import Geometry\n"
        "try:\n"
        f"    {call}\n"
        "except ImportError as e:\n"
        "    print(e)\n"
        "    sys.exit(3)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 3, result.stderr
    assert "CT backend" in result.stdout and "env_base.yml" in result.stdout
