# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Import spyrit only where its operators are needed.

spyrit (single-pixel imaging) gives LION the fast Walsh-Hadamard transform of
the Walsh-Hadamard and photocurrent mapping operators. Like tomosipo (CT) and
mrpro (MRI), it is imported through `import_spyrit` where it is used, so that
`import LION` does not load it, nor torchvision and sympy, which it imports,
and a missing installation gives a clear error at the point of use.
"""

import importlib
from types import ModuleType


def import_spyrit(module: str = "spyrit", feature: str = "This operator") -> ModuleType:
    """The module ``module`` (``spyrit`` or one of its submodules, e.g.
    ``spyrit.core.torch``), imported, or an ImportError that says what is
    missing and how to install it. ``feature``: what needs it, for the message
    (e.g. "WalshHadamard2D")."""
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as error:
        raise ImportError(
            f"{feature} needs `{module}`, part of spyrit, which is not available: "
            f"{error}. Install it with `pip install spyrit`."
        ) from error
