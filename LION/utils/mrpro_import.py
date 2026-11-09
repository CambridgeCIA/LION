# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Import mrpro only where MRI functionality needs it.

mrpro is LION's MRI backend, as tomosipo is the CT one (see
`LION.utils.tomosipo_import`): LION imports without it, and MRI functions
import it through `import_mrpro`, so that a missing installation gives a clear
error at the point of use and `import LION` does not load mrpro and its
dependencies.
"""

import importlib
from types import ModuleType


def import_mrpro(
    module: str = "mrpro", feature: str = "This MRI function"
) -> ModuleType:
    """The module ``module`` (``mrpro`` or one of its submodules, e.g.
    ``mrpro.operators``), imported, or an ImportError that says what is missing
    and how to install it. ``feature``: what needs it, for the message (e.g.
    "MrproLinearOp")."""
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as error:
        raise ImportError(
            f"{feature} needs `{module}`, part of the MRI backend (mrpro), which "
            f"is not available: {error}. Install it with `pip install mrpro`."
        ) from error
