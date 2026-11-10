Datasets
========

LION trains and tests on public CT datasets, downloaded and prepared once on
your machine. This page says what each dataset is, how to get it, and where
LION looks for it. The tutorials and example scripts use LIDC-IDRI.

.. list-table::
   :header-rows: 1
   :widths: 16 44 20 20

   * - Dataset
     - What it is
     - Loader
     - Experiments
   * - LIDC-IDRI
     - Clinical lung CT of 1010 patients, with annotated nodules. LION
       simulates the measurements (sinograms) from the images, with the noise
       and geometry of an experiment.
     - ``LION.data_loaders.LIDC_IDRI``
     - ``ct_experiments``
   * - 2DeteCT
     - Measured fan-beam sinograms of 5,000 slices of 109 objects (mixtures
       of materials), each in three modes: low dose, high dose, beam
       hardening; with reference reconstructions and segmentations.
     - ``LION.data_loaders.deteCT``
     - ``ct_experiments`` (where the measured data allow),
       ``deteCT_benchmark``
   * - Walnuts
     - Cone-beam micro-CT of 42 walnuts. A download script only for now: its
       loader is being reworked.
     -
     -

Where LION looks for data
-------------------------

LION finds the data in the folder named by ``LION_DATA_PATH``: set it once,
e.g. in your ``~/.bashrc``,

.. code-block:: bash

   export LION_DATA_PATH=/path/to/datasets

Downloads go to ``raw/`` inside it, and the prepared data that LION reads to
``processed/``:

.. code-block:: text

   LION_DATA_PATH/
   ├── raw/
   │   ├── LIDC-IDRI/      the DICOM files (can be deleted once processed)
   │   ├── 2detect/        the Zenodo archives, unzipped
   │   └── walnuts/
   └── processed/
       ├── LIDC-IDRI/      one folder per patient: slices as .npy, masks
       └── 2detect/        sinograms and reconstructions per slice and mode

Saved models record their data's paths relative to ``LION_DATA_PATH``, so a
model trained on one machine finds its data on another.

.. note::

   At CIA (DAMTP, Cambridge), the datasets are ready:
   ``export LION_DATA_PATH=/store/cia-lion/datasets``.

LIDC-IDRI
---------

The Lung Image Database Consortium image collection (Armato et al., Medical
Physics, 2011), from The Cancer Imaging Archive (TCIA). The download is about
133 GB; downloading and preparing it take hours.

**Download** with the NBIA Data Retriever, which the script installs in the
data folder (it needs Java 17, or Apptainer, Singularity or Docker to run it
in a container):

.. code-block:: bash

   bash LION/data_loaders/LIDC_IDRI/download_LIDC_IDRI.sh

It writes ``LION_DATA_PATH/raw/LIDC-IDRI`` and the diagnosis file. If it is
interrupted, run it again: it downloads only what is missing. Its options
(another folder, the manifest, the container) are in
`its README <https://github.com/CambridgeCIA/LION/blob/main/LION/data_loaders/LIDC_IDRI/README.md>`_.

**Prepare** it in its own environment (``pylidc`` needs older packages):

.. code-block:: bash

   conda env create -f LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri_environment.yml
   conda activate lidc_idri
   python LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py

This saves every slice (in HU), resampled to the same pixel spacing (500 mm
over 512×512 pixels, so no scan is cropped), with the nodule masks and the
diagnoses, and records what it did in
``processed/LIDC-IDRI/pre_processing.json``. The loader converts the slices to
densities (g/cm³), and ``LIDC_IDRI.default_geometry()`` is the CT geometry
that matches them.

.. note::

   Datasets prepared before October 2026 kept each scan's own pixel spacing.
   Resample one in place, without the raw files:
   ``python LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py --resample-only``
   (``--folder`` for another folder than ``LION_DATA_PATH/processed/LIDC-IDRI``).

Patients whose files are damaged are listed in ``pre_processing.json``; the
loader leaves them out with a warning. The patients are split into training,
validation and test sets, so no patient is in two of them.

2DeteCT
-------

A dataset of measured CT data for machine learning (Kiss et al., Scientific
Data, 2023), from Zenodo. Downloading it takes a few hours.

.. code-block:: bash

   python LION/data_loaders/2deteCT/download.py
   python LION/data_loaders/2deteCT/pre_process_2deteCT.py

The first writes ``LION_DATA_PATH/raw/2detect``, the second
``LION_DATA_PATH/processed/2detect``; the raw folder can then be deleted. The
``deteCT`` loader's docstring (:doc:`api`) describes the modes, the two
detectors (samples 1–63 and 64–109 were measured with different ones), the
splits and the noise options.

Walnuts
-------

Cone-beam micro-CT scans of 42 walnuts (Der Sarkissian et al., Scientific
Data, 2019), from Zenodo:

.. code-block:: bash

   python LION/data_loaders/walnuts/download.py

It writes ``LION_DATA_PATH/raw/walnuts``. LION has no loader for them in its
datasets' form yet.

Using the data
--------------

Mostly through an experiment, which fixes the geometry, the noise and the
splits of a problem and makes the datasets
(:doc:`tutorials/d03_data`):

.. code-block:: python

   from LION.experiments import ct_experiments

   experiment = ct_experiments.LowDoseCTRecon()             # LIDC-IDRI
   # experiment = ct_experiments.LowDoseCTRecon(dataset="2DeteCT")
   training = experiment.get_dataset("train")              # (sinogram, image) pairs

The loaders also work on their own, with their own parameters (e.g.
LIDC-IDRI's segmentation task). Any torch ``Dataset`` of (measurements,
target) pairs trains with LION; a ``LIONDataset`` is also recorded with the
model (:doc:`tutorials/d13_extending_lion`).
