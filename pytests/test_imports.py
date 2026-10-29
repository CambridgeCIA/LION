"""LION's core (operators, classical algorithms, models, solvers base) must import
without tomosipo/astra, so the pure-torch parts can be used on any machine.
Importing must also not touch CUDA (checked by making every CUDA query fail),
nor need LION_DATA_PATH (it is unset), nor load matplotlib or the optional
backends; a missing backend says what is missing when it is needed.

Everything runs in one fresh interpreter per check (starting Python and torch
is the slow part), with the results reported per module.
"""

import json
import os
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
    "LION.models.CNNs.MSD_pytorch",  # msd_pytorch only when the model is made
    "LION.models.post_processing.cFBPConvNet",
    "LION.reconstructors",
    "LION.optimizers.LIONsolver",
    "LION.optimizers.SupervisedSolver",
    "LION.optimizers.EquivariantSolver",
    "LION.optimizers.AdversarialRegularizerSolver",
    "LION.optimizers.Noisier2InverseSolver",
    "LION.optimizers.Proj2ProjSolver",
    "LION.models.iterative_unrolled.LG",
    "LION.models.iterative_unrolled.LPD",
    "LION.models.iterative_unrolled.cLPD",
    "LION.models.learned_fbp.DeepFBP",
    "LION.models.learned_fbp.DeepFusionBP",
    "LION.models.learned_fbp.FusionFBP",
    "LION.experiments.ct_experiments",
    "LION.evaluation",
    "LION.package",
]

# a call that needs a backend, and what its error must say
CALLS = {
    "from LION.classical_algorithms import fdk; fdk(torch.zeros(1, 2, 3), Geometry.default_parameters())": "CT backend",
    "from LION.classical_algorithms import sirt; sirt(torch.zeros(1, 2, 3), Geometry.default_parameters(), 2)": "CT backend",
    "from LION.operators import CTProjectionOp; CTProjectionOp(Geometry.default_parameters())": "CT backend",
    "from LION.utils.mrpro_import import import_mrpro; import_mrpro('mrpro', 'MrproLinearOp')": "MRI backend (mrpro)",
    "from LION.operators.WalshHadamard2D import fwht; fwht(torch.zeros(4))": "spyrit",
}

# Setting a sys.modules entry to None makes any later import of it fail.
NO_TOMOSIPO = """
import json, sys, traceback
for name in ("tomosipo", "tomosipo.torch_support", "ts_algorithms", "astra",
             "mrpro", "spyrit"):
    sys.modules[name] = None
import torch
"""


def run_isolated(code):
    """Run code in a fresh interpreter without tomosipo, the optional backends
    and LION_DATA_PATH; return what it prints as JSON."""
    environment = {k: v for k, v in os.environ.items() if k != "LION_DATA_PATH"}
    result = subprocess.run(
        [sys.executable, "-c", NO_TOMOSIPO + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.fixture(scope="module")
def import_errors():
    """{module: traceback or None}, imported one after another in one interpreter."""
    return run_isolated(
        f"""
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
        """
    )


@pytest.fixture(scope="module")
def missing_backend_errors():
    """{call: ImportError message or None} for CT functions called without tomosipo."""
    return run_isolated(
        f"""
        from LION.CTtools.ct_geometry import Geometry
        errors = {{}}
        for call in {list(CALLS)!r}:
            try:
                exec(call)
                errors[call] = None
            except ImportError as e:
                errors[call] = str(e)
        print(json.dumps(errors))
        """
    )


@pytest.mark.parametrize("module", MODULES)
def test_imports_without_tomosipo(import_errors, module):
    assert import_errors[module] is None, import_errors[module]


@pytest.mark.parametrize("call, says", CALLS.items(), ids=lambda v: v[:40])
def test_backends_say_what_is_missing(missing_backend_errors, call, says):
    message = missing_backend_errors[call]
    assert message is not None, "no ImportError raised"
    assert says in message


# torchvision is not used by LION; mrpro (MRI) and spyrit (Walsh-Hadamard) are
# backends, imported on first use like tomosipo. Optimizers that need tomosipo
# at import time are left out (they fail at import without it).
LIGHT_MODULES = MODULES + [
    "LION.optimizers.Noisier2InverseSolver",
    "LION.optimizers.Noise2InverseSolver",
    "LION.metrics.haarpsi",
]


def test_import_does_not_load_heavy_backends():
    loaded = run_isolated(
        f"""
        for module in {LIGHT_MODULES!r}:
            try:
                __import__(module)
            except ImportError:
                pass  # e.g. tomosipo missing: not what this test checks
        print(json.dumps([m for m in ("torchvision", "sympy", "matplotlib")
                          if m in sys.modules]))
        """
    )
    assert loaded == []


def test_mrpro_wrapper_is_defined_on_first_use():
    pytest.importorskip("mrpro")
    from LION.operators.Operator import MrproLinearOp
    from mrpro.operators.LinearOperator import LinearOperator

    assert issubclass(MrproLinearOp, LinearOperator)
    assert MrproLinearOp.__qualname__ == "MrproLinearOp"
    module = sys.modules["LION.operators.Operator"]
    assert module.MrproLinearOp is MrproLinearOp  # defined once
