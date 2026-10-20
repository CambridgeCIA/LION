"""LION's core (operators, classical algorithms, models, solvers base) must import
without tomosipo/astra, so the pure-torch parts can be used on any machine."""

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
]


@pytest.mark.parametrize("module", MODULES)
def test_imports_without_tomosipo(module):
    # Setting a sys.modules entry to None makes any later import of it fail.
    code = (
        "import sys\n"
        "for name in ('tomosipo', 'tomosipo.torch_support', 'ts_algorithms', 'astra'):\n"
        "    sys.modules[name] = None\n"
        f"import {module}\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
