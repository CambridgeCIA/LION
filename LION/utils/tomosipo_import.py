# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Import tomosipo / ts_algorithms only where CT functionality needs them.

LION's non-CT parts (operators, models, FISTA, PnP, ...) work without tomosipo
and astra. CT functions import them through `import_tomosipo`, so that a missing
installation gives a clear error at the point of use.
"""

import importlib
from types import ModuleType


def import_tomosipo(
    module: str = "tomosipo", feature: str = "This CT function"
) -> ModuleType:
    """The module ``module`` (``tomosipo``, ``tomosipo.torch_support`` or
    ``ts_algorithms``), imported, or an ImportError that says what is missing
    and how to install it (astra can also fail to load its libraries).
    ``feature``: what needs it, for the message (e.g. "fdk")."""
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as error:  # astra can fail to load its libraries
        raise ImportError(
            f"{feature} needs `{module}`, part of the CT backend (tomosipo, "
            "ts_algorithms and astra-toolbox), which is not available: "
            f"{error}. Install the conda environment (env_base.yml) as described "
            "in LION's README, or use LION's non-CT operators and algorithms."
        ) from error
