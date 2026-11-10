API reference
=============

Every module of LION, from its docstrings. The tutorials show how the pieces
fit together; this is where each one is described.

Operators
---------

The forward operators: CT (``LION.operators.ct``: the geometry, the
projector, FDK, noise), photocurrent mapping (``LION.operators.pcm``), and
generic ones.

.. autosummary::
   :toctree: generated
   :recursive:

   LION.operators

Data and experiments
--------------------

.. autosummary::
   :toctree: generated
   :recursive:

   LION.data_loaders
   LION.experiments

Models
------

.. autosummary::
   :toctree: generated
   :recursive:

   LION.models

Training
--------

Solvers, losses and model regularisations.

.. autosummary::
   :toctree: generated
   :recursive:

   LION.optimizers
   LION.losses

Reconstruction
--------------

Reconstructing with trained models, and classical algorithms.

.. autosummary::
   :toctree: generated
   :recursive:

   LION.reconstructors
   LION.classical_algorithms

Evaluation
----------

.. autosummary::
   :toctree: generated
   :recursive:

   LION.evaluation
   LION.metrics

Publishing
----------

.. autosummary::
   :toctree: generated

   LION.package

Utilities
---------

.. autosummary::
   :toctree: generated
   :recursive:

   LION.utils
