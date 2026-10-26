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
    """Import ``module`` (``tomosipo``, ``tomosipo.torch_support`` or
    ``ts_algorithms``), raising an ImportError that says what is missing.

    Args:
        module: Name of the module to import.
        feature: What needs it, for the error message (e.g. "fdk").

    Returns:
        The imported module.
    """
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as error:  # astra can fail to load its libraries
        raise ImportError(
            f"{feature} needs `{module}`, part of the CT backend (tomosipo, "
            "ts_algorithms and astra-toolbox), which is not available: "
            f"{error}. Install the conda environment (env_base.yml) as described "
            "in LION's README, or use LION's non-CT operators and algorithms."
        ) from error
