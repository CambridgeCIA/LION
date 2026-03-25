"""Tests for CT experiments."""

from LION.experiments.ct_experiments import Experiment


def test_experiment_raise_not_implemented() -> None:
    try:
        Experiment()  # attempt to construct using abstract class
    except TypeError as error:
        print(f"TypeError raised as expected: {error}")
    else:
        raise AssertionError("Expected TypeError was not raised.")
