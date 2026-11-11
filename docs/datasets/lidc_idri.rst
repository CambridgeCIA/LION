LIDC-IDRI
=========

The Lung Image Database Consortium image collection: clinical lung CT scans of
1010 patients, with nodules annotated by up to four radiologists, and their
diagnoses (Armato et al., "The Lung Image Database Consortium (LIDC) and Image
Database Resource Initiative (IDRI)", Medical Physics, 2011). It is
distributed by The Cancer Imaging Archive (TCIA).

LION uses the reconstructed images as ground truth and **simulates** the
measurements from them, with an experiment's geometry and noise. It is the
dataset of LION's experiments (``ct_experiments``), its tutorials and its
example scripts.

Download
--------

About 133 GB; downloading takes hours. The NBIA Data Retriever, which the
script installs in the data folder, needs Java 17, or Apptainer, Singularity
or Docker to run it in a container:

.. code-block:: bash

   bash LION/data_loaders/LIDC_IDRI/download_LIDC_IDRI.sh

It writes ``LION_DATA_PATH/raw/LIDC-IDRI`` and the diagnosis file. If it is
interrupted, run it again: it downloads only what is missing. Its options
(another folder, the manifest, the container) are in
`its README <https://github.com/CambridgeCIA/LION/blob/main/LION/data_loaders/LIDC_IDRI/README.md>`_.

Prepare
-------

In its own environment (``pylidc`` needs older packages):

.. code-block:: bash

   conda env create -f LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri_environment.yml
   conda activate lidc_idri
   python LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py

This saves every slice (in HU), resampled to the same pixel spacing (500 mm
over 512×512 pixels, so no scan is cropped), with the nodule masks and the
diagnoses, in ``LION_DATA_PATH/processed/LIDC-IDRI`` (one folder per patient),
and records what it did in ``pre_processing.json``. The raw files can then be
deleted.

.. note::

   Datasets prepared before October 2026 kept each scan's own pixel spacing.
   Resample one in place, without the raw files:
   ``python LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py --resample-only``
   (``--folder`` for another folder than ``LION_DATA_PATH/processed/LIDC-IDRI``).

In LION
-------

The loader is ``LION.data_loaders.LIDC_IDRI.LIDC_IDRI``. Its images are
densities (g/cm³), and ``LIDC_IDRI.default_geometry()`` is the CT geometry
that matches the slices. Its parameters (``LIDC_IDRI.default_parameters()``)
choose:

- **the split**: patients are split into training, validation and test (80%,
  10%, 10%), so no patient is in two of them;
- **the slices**: 5 per patient by default (``max_num_slices_per_patient``;
  -1 for all), half of them with a nodule (``pcg_slices_nodule``);
- **the task**: "reconstruction" gives (sinogram, image) pairs, the sinogram
  simulated with the geometry and the experiment's noise; "segmentation" (and
  "joint", "end_to_end") gives (image, nodule mask), the mask from the
  radiologists' consensus or one annotation; "diagnostic" the patient's
  diagnosis.

Patients whose files are damaged are listed in ``pre_processing.json``; the
loader leaves them out with a warning. ``legacy_slice_selection`` and
``legacy_patient_list`` reproduce LION's selection before October 2026.

.. code-block:: python

   from LION.experiments import ct_experiments

   experiment = ct_experiments.LowDoseCTRecon()   # geometry, noise, splits
   training = experiment.get_dataset("train")     # (sinogram, image) pairs

   # the loader alone, e.g. for segmentation
   from LION.data_loaders.LIDC_IDRI import LIDC_IDRI

   parameters = LIDC_IDRI.default_parameters()
   parameters.task = "segmentation"
   segmentation = LIDC_IDRI("train", parameters=parameters)

The experiments on it are in ``LION.experiments.ct_experiments`` (clinical,
low and extremely low dose; full, sparse and limited angle):
:doc:`../tutorials/d03_data`, and their noise, doses and geometries in the
module's documentation (:doc:`../api`).
