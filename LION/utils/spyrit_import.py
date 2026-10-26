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
    """Import ``module`` (``spyrit`` or one of its submodules), raising an
    ImportError that says what is missing.

    Args:
        module: Name of the module to import, e.g. ``spyrit.core.torch``.
        feature: What needs it, for the error message (e.g. "WalshHadamard2D").

    Returns:
        The imported module.
    """
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as error:
        raise ImportError(
            f"{feature} needs `{module}`, part of spyrit, which is not available: "
            f"{error}. Install it with `pip install spyrit`."
        ) from error
