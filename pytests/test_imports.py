"""LION's core (operators, classical algorithms, models, solvers base) must import
without tomosipo/astra, so the pure-torch parts can be used on any machine.
Importing must also not touch CUDA (checked by making every CUDA query fail).

Everything runs in one fresh interpreter per check (starting Python and torch
is the slow part), with the results reported per module.
"""

import json
import subprocess
import sys
import textwrap

import pytest

MODULES = [
    "LION.operators",
    "LION.classical_algorithms",
    "LION.CTtools.ct_utils",
    "LION.models.LIONmodel",
    "LION.models.CNNs.dncnn",
    "LION.models.CNNs.drunet",
    "LION.models.CNNs.UNets.Unet",
    "LION.models.post_processing.cFBPConvNet",
    "LION.reconstructors",
    "LION.optimizers.LIONsolver",
    "LION.optimizers.SupervisedSolver",
    "LION.optimizers.EquivariantSolver",
    "LION.optimizers.AdversarialRegularizerSolver",
]

CALLS = [
    "from LION.classical_algorithms import fdk; fdk(torch.zeros(1, 2, 3), Geometry.default_parameters())",
    "from LION.classical_algorithms import sirt; sirt(torch.zeros(1, 2, 3), Geometry.default_parameters(), 2)",
    "from LION.operators import CTProjectionOp; CTProjectionOp(Geometry.default_parameters())",
]

# Setting a sys.modules entry to None makes any later import of it fail.
NO_TOMOSIPO = """
import json, sys, traceback
for name in ("tomosipo", "tomosipo.torch_support", "ts_algorithms", "astra"):
    sys.modules[name] = None
import torch
"""


def run_isolated(code):
    """Run code in a fresh interpreter without tomosipo; return what it prints as JSON."""
    result = subprocess.run(
        [sys.executable, "-c", NO_TOMOSIPO + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.fixture(scope="module")
def import_errors():
    """{module: traceback or None}, imported one after another in one interpreter."""
    return run_isolated(f"""
        def _no_cuda(*args, **kwargs):
            raise RuntimeError("CUDA queried at import time")
        torch.cuda.current_device = _no_cuda
        errors = {{}}
        for module in {MODULES!r}:
            try:
                __import__(module)
                errors[module] = None
            except Exception:
                errors[module] = traceback.format_exc()
        print(json.dumps(errors))
        """)


@pytest.fixture(scope="module")
def missing_backend_errors():
    """{call: ImportError message or None} for CT functions called without tomosipo."""
    return run_isolated(f"""
        from LION.CTtools.ct_geometry import Geometry
        errors = {{}}
        for call in {CALLS!r}:
            try:
                exec(call)
                errors[call] = None
            except ImportError as e:
                errors[call] = str(e)
        print(json.dumps(errors))
        """)


@pytest.mark.parametrize("module", MODULES)
def test_imports_without_tomosipo(import_errors, module):
    assert import_errors[module] is None, import_errors[module]


@pytest.mark.parametrize("call", CALLS)
def test_ct_functions_say_what_is_missing(missing_backend_errors, call):
    message = missing_backend_errors[call]
    assert message is not None, "no ImportError raised"
    assert "CT backend" in message and "env_base.yml" in message
