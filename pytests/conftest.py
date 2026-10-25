"""Shared pytest configuration.

- Tests that need tomosipo (and therefore astra with CUDA) are marked with
  ``@pytest.mark.tomosipo``. They are skipped when tomosipo cannot be imported
  or no CUDA device is available: tomosipo projects with astra's CUDA
  algorithms, and pip's astra-toolbox imports fine on machines without a GPU.
- Tests marked ``@pytest.mark.cuda`` are skipped when no CUDA device is
  available, so ``pytest`` works everywhere without ``-m "not cuda"``.
"""

import pytest
import torch


def _tomosipo_available() -> bool:
    try:
        import tomosipo  # noqa: F401
    except Exception:  # astra can fail with errors other than ImportError
        return False
    return True


CUDA_AVAILABLE = torch.cuda.is_available()
TOMOSIPO_AVAILABLE = CUDA_AVAILABLE and _tomosipo_available()


def pytest_collection_modifyitems(config, items):
    skip_tomosipo = pytest.mark.skip(reason="tomosipo/astra with CUDA is not available")
    skip_cuda = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "tomosipo" in item.keywords and not TOMOSIPO_AVAILABLE:
            item.add_marker(skip_tomosipo)
        if "cuda" in item.keywords and not CUDA_AVAILABLE:
            item.add_marker(skip_cuda)


# Known to fail: they replace LION modules in sys.modules with mocks, which breaks
# other tests in the same process. Their batching checks are covered in
# test_classical_algorithms.py.
collect_ignore = [
    "test_sirt.py",
    "test_tv_min.py",
]


@pytest.fixture
def import_without_tomosipo(monkeypatch):
    """Import a module that imports tomosipo/ts_algorithms at the top, with those
    stubbed, e.g. to test a model's torch code on CPU with a stand-in operator.
    The stubbed import is removed afterwards."""
    import importlib
    import sys
    import types

    stubs = {
        "tomosipo": types.ModuleType("tomosipo"),
        "tomosipo.torch_support": types.ModuleType("tomosipo.torch_support"),
        "ts_algorithms": types.ModuleType("ts_algorithms"),
    }
    stubs["tomosipo.torch_support"].to_autograd = lambda op, **kw: op
    stubs["ts_algorithms"].fdk = None
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    imported = []

    def import_(name):
        monkeypatch.delitem(sys.modules, name, raising=False)
        imported.append(name)
        return importlib.import_module(name)

    yield import_
    for name in imported:
        sys.modules.pop(name, None)
