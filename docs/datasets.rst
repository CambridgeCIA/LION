Datasets
========

LION trains and tests on public CT datasets, downloaded and prepared once on
your machine. Each has its own page: what it is, how to get it and prepare
it, and how LION uses it. The tutorials and example scripts use LIDC-IDRI.

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Dataset
     - What it is
     - In LION
   * - :doc:`datasets/lidc_idri`
     - Clinical lung CT of 1010 patients, with annotated nodules. LION
       simulates the measurements from the images.
     - ``LIDC_IDRI``; ``ct_experiments``
   * - :doc:`datasets/2detect`
     - Measured fan-beam CT of 5,000 slices of 109 objects, in three modes
       (low dose, high dose, beam hardening).
     - ``deteCT``; ``ct_experiments``, ``deteCT_benchmark``
   * - :doc:`datasets/walnuts`
     - Cone-beam micro-CT of 42 walnuts.
     - download only, for now

.. toctree::
   :hidden:

   datasets/lidc_idri
   datasets/2detect
   datasets/walnuts

Where LION looks for data
-------------------------

LION finds the data in the folder named by ``LION_DATA_PATH``: set it once,
e.g. in your ``~/.bashrc``,

.. code-block:: bash

   export LION_DATA_PATH=/path/to/datasets

Downloads go to ``raw/`` inside it, and the prepared data that LION reads to
``processed/``, one folder per dataset:

.. code-block:: text

   LION_DATA_PATH/
   ├── raw/
   │   ├── LIDC-IDRI/
   │   ├── 2detect/
   │   └── walnuts/
   └── processed/
       ├── LIDC-IDRI/
       └── 2detect/

Saved models record their data's paths relative to ``LION_DATA_PATH``, so a
model trained on one machine finds its data on another.

.. note::

   At CIA (DAMTP, Cambridge), the datasets are ready:
   ``export LION_DATA_PATH=/store/cia-lion/datasets``.

Using the data
--------------

Mostly through an experiment, which fixes the geometry, the noise and the
splits of a problem and makes the datasets (:doc:`tutorials/d03_data`):

.. code-block:: python

   from LION.experiments import ct_experiments

   experiment = ct_experiments.LowDoseCTRecon()             # LIDC-IDRI
   # experiment = ct_experiments.clinicalCTRecon(dataset="2DeteCT")  # measured
   training = experiment.get_dataset("train")              # (sinogram, image) pairs

The loaders also work on their own, with their own parameters (each dataset's
page). Any torch ``Dataset`` of (measurements, target) pairs trains with
LION; a ``LIONDataset`` is also recorded with the model.

A new dataset
-------------

A dataset joins LION with a loader deriving from ``LIONDataset`` (in
``LION/data_loaders``), its download and preparation scripts (in
``LION/data_loaders/<dataset>/``), and a page here, in ``docs/datasets/``,
listed above. The :doc:`developer guide <developers>` says what the loader
needs; :doc:`tutorials/d13_extending_lion` writes one.
