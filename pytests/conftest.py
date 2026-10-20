"""Shared pytest configuration.

- Tests that need tomosipo (and therefore astra with CUDA) are marked with
  ``@pytest.mark.tomosipo``. They are skipped when tomosipo cannot be imported
  or no CUDA device is available: tomosipo projects with astra's CUDA
  algorithms, and pip's astra-toolbox imports fine on machines without a GPU.
- Tests marked ``@pytest.mark.cuda`` are skipped when no CUDA device is
  available, so ``pytest tests`` works everywhere without ``-m "not cuda"``.
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
