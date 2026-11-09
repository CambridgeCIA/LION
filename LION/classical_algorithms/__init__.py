# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  :
# =============================================================================
"""LION classical algorithms that work with any operator. Those for one kind
of operator are in its folder (e.g. FDK, SIRT and TV-min in
``LION.operators.ct``)."""

from LION.classical_algorithms.conjugate_gradient import conjugate_gradient
from LION.classical_algorithms.fista import fista_l1
from LION.classical_algorithms.spgl1_torch import spgl1_torch

__all__ = ["conjugate_gradient", "fista_l1", "spgl1_torch"]
