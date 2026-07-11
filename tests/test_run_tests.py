"""Tests for the hardware-aware local test launcher."""

from scripts.run_tests import pytest_args


def test_pytest_args_runs_full_suite_when_cuda_is_available():
    assert pytest_args(["-q"], cuda_available=True) == ["-q"]


def test_pytest_args_excludes_cuda_without_a_gpu():
    assert pytest_args(["-q"], cuda_available=False) == [
        "-q",
        "-m",
        "not cuda",
    ]


def test_pytest_args_preserves_an_explicit_marker_expression():
    assert pytest_args(["-q", "-m", "cuda"], cuda_available=False) == [
        "-q",
        "-m",
        "cuda",
    ]
