# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Import TIGRE only where CT functionality needs it.

TIGRE is one of LION's CT backends (``LION.operators.ct.backend``). It is
compiled with CUDA, so it is optional: CT functions import it through
``import_tigre``, so that a missing installation gives a clear error at the
point of use.
"""

import importlib
from types import ModuleType

#: The TIGRE release LION is tested with: the first whose ``Ax``/``Atb`` keep
#: PyTorch CUDA tensors on their GPU.
TIGRE_VERSION = "3.3.0"


def import_tigre(
    module: str = "tigre", feature: str = "This CT function"
) -> ModuleType:
    """The module ``module`` (``tigre`` or one of its submodules), imported, or
    an ImportError that says what is missing and how to install it.
    ``feature``: what needs it, for the message (e.g. "CTProjectionOp")."""
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as error:  # its CUDA libraries can fail to load
        raise ImportError(
            f"{feature} needs `{module}`, part of the TIGRE CT backend, which is "
            f"not available: {error}. Install TIGRE {TIGRE_VERSION} or later "
            "(it needs the CUDA toolkit): pip install "
            f'"tigre @ git+https://github.com/CERN/TIGRE.git@v{TIGRE_VERSION}" '
            "(TIGRE isn't on PyPI: `pip install tigre` installs another package), "
            'or use the tomosipo backend: LION.operators.ct.set_ct_backend("tomosipo").'
        ) from error
