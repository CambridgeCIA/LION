LION
====

**LION** (Learned Iterative Optimization Networks) is a PyTorch library for
learned reconstruction in tomography and other inverse problems, for research.
It stays close to plain torch: its models are torch modules, its
operators torch functions, its solvers torch training loops, and any piece can
be your own.

Around that, it keeps what research needs to be repeated:

- **experiments** that fix the geometry, noise and data of a problem;
- **records** saved with every model: the data it saw, how it was trained, the
  software it ran on;
- **evaluation and comparison** of methods on the same test data, with
  checks that the comparison is fair;
- **publishing** a trained model as a package anyone can load, test or train
  again.

Where to start
--------------

- **Install it:** see the `README <https://github.com/CambridgeCIA/LION#readme>`_.
- **Learn it:** the :doc:`tutorials <tutorials/index>` build LION up one piece
  at a time, from the CT operator to publishing a model.
- **See it run:** the :doc:`example scripts <examples>` are complete trainings,
  one feature each.
- **See what's there:** :doc:`methods`, the kinds of learned reconstruction
  and every method LION has, with its paper.
- **Look something up:** the :doc:`API reference <api>`, from the docstrings.
- **Change LION:** the :doc:`developer guide <developers>`: how it is built,
  one training followed through the code, and what a new piece needs.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Using LION

   tutorials/index
   examples

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Reference

   methods
   api

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Project

   developers
   citing
