"""Tests for local development and test tooling."""

import re
from pathlib import Path

from scripts.run_tests import pytest_args


ROOT = Path(__file__).resolve().parents[1]


def test_black_dependency_matches_pre_commit_hook_revision():
    """Keep fresh development installs aligned with the formatting hook."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pre_commit = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    dependency = re.search(r'"black==([^"]+)"', pyproject)
    hook = re.search(
        r"repo: https://github\.com/psf/black\s+.*?rev: (\S+)",
        pre_commit,
        flags=re.DOTALL,
    )

    assert dependency is not None
    assert hook is not None
    assert dependency.group(1) == hook.group(1)


def test_pytest_args_runs_full_suite_when_cuda_is_available():
    """Verify that pytest args runs full suite when cuda is available."""
    assert pytest_args(["-q"], cuda_available=True) == ["-q"]


def test_pytest_args_excludes_cuda_without_a_gpu():
    """Verify that pytest args excludes cuda without a gpu."""
    assert pytest_args(["-q"], cuda_available=False) == [
        "-q",
        "-m",
        "not cuda",
    ]


def test_pytest_args_preserves_an_explicit_marker_expression():
    """Verify that pytest args preserves an explicit marker expression."""
    assert pytest_args(["-q", "-m", "cuda"], cuda_available=False) == [
        "-q",
        "-m",
        "cuda",
    ]
