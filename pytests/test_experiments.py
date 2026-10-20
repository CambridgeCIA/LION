"""Experiment definitions (no data needed)."""

import os
import tempfile

# LION.utils.paths requires LION_DATA_PATH at import time
os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import pytest  # noqa: E402

import LION.experiments.ct_experiments as ct_experiments  # noqa: E402

EXPERIMENTS = [
    cls
    for cls in vars(ct_experiments).values()
    if isinstance(cls, type)
    and issubclass(cls, ct_experiments.Experiment)
    and cls is not ct_experiments.Experiment
]


def test_experiment_names_are_unique():
    names = [cls.default_parameters().name for cls in EXPERIMENTS]
    assert len(EXPERIMENTS) == 9
    assert len(set(names)) == len(names), names


@pytest.mark.parametrize("cls", EXPERIMENTS, ids=lambda c: c.__name__)
def test_noise_function_is_always_defined(cls):
    params = cls.default_parameters()
    assert cls(params).sino_fun is not None
    del params.noise_params
    assert cls(params).sino_fun is None
