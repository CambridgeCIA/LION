Testing
=======

Hardware-aware test launcher
----------------------------

Run the same launcher used by local commit hooks and GitHub Actions::

   python scripts/run_tests.py -q

When :func:`torch.cuda.is_available` returns true, the launcher runs the full
suite.  Otherwise it adds ``-m "not cuda"`` and excludes tests that require a
CUDA CT projector.

Explicit selections
-------------------

Run only CPU-safe tests::

   pytest -q -m "not cuda"

Run only CUDA-marked tests::

   pytest -q -m cuda

Run a focused module while developing::

   pytest -q tests/models/test_padis_reconstructor.py

Continuous integration
----------------------

The ``Tests`` GitHub Actions workflow invokes the hardware-aware launcher.  A
standard ``ubuntu-latest`` worker runs the CPU suite; routing the job to a
properly configured GPU runner automatically includes CUDA tests.  The status
is reported in the README but is not configured as a mandatory merge check.
