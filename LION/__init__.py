"""LION: learned iterative optimisation networks, for tomographic (and other
inverse-problem) reconstruction in PyTorch."""

import pathlib

#: LION's version (the file ``LION/VERSION``, also used by the package build).
__version__ = (pathlib.Path(__file__).parent / "VERSION").read_text().strip()
