2DeteCT
=======

A dataset of **measured** CT data for machine learning: fan-beam sinograms of
5,000 slices of 109 objects (fruit, nuts and other materials), each measured
in three modes, with reference reconstructions and segmentations (Kiss et al.,
"2DeteCT - A large 2D expandable, trainable, experimental Computed Tomography
dataset for machine learning", Scientific Data, 2023). It is distributed on
Zenodo.

============  ==========  ==========  ===============
              mode1       mode2       mode3
============  ==========  ==========  ===============
Tube voltage  90 kV       90 kV       60 kV
Tube power    3 W         90 W        60 W
Filter        Thoraeus    Thoraeus    none
Meaning       low dose    high dose   beam hardening
============  ==========  ==========  ===============

Download and prepare
--------------------

Downloading takes a few hours:

.. code-block:: bash

   python LION/data_loaders/2deteCT/download.py
   python LION/data_loaders/2deteCT/pre_process_2deteCT.py

The first writes ``LION_DATA_PATH/raw/2detect``, the second
``LION_DATA_PATH/processed/2detect``; the raw folder can then be deleted.

In LION
-------

The loader is ``LION.data_loaders.deteCT.deteCT``. Its parameters
(``deteCT.default_parameters()``) choose:

- **the modes**: ``input_mode`` (the measurements) and ``target_mode`` (the
  reference reconstruction), mode2 for both by default;
- **the task**: "sino2recon" (sinogram, reconstruction) and others, e.g. for
  segmentation;
- **the split**: samples in order, 80%, 10%, 10%;
- **corrections**: flat- and dark-field correction and the log transform, on
  by default;
- **added noise** (``add_noise``, ``noise_params``): Poisson noise and
  cross-talk on top of the measured sinograms, as Kiss et al. (2024) did.

``deteCT.get_default_geometry()`` is the scan's geometry: fan beam, 3600
angles over 360 degrees, a 956-pixel detector (2× subsampled), 1024×1024
images.

.. note::

   The data were acquired with **two detectors**: samples 1–63 with one,
   64–109 with the other, which gets about a third of the photons at the same
   settings. Split in order, the validation and test sets are all from the
   second detector and the training set mostly from the first. The data
   record lists each sample's detector, and ``noise_params.I0`` can be one
   value per detector (``deteCT.noise_parameters()``).

.. code-block:: python

   from LION.experiments import ct_experiments, deteCT_benchmark

   # LION's experiments, on the measured data (only the angles the scan has)
   experiment = ct_experiments.clinicalCTRecon(dataset="2DeteCT")

   # the 2DeteCT benchmark paper's experiments
   experiment = deteCT_benchmark.LimitedAngle90CTRecon()

   training = experiment.get_dataset("train")

Two sets of experiments use it: LION's (``ct_experiments``, with
``dataset="2DeteCT"``: clinical dose is the measured mode2, extremely low dose
mode1; the low-dose experiments have no measured equivalent) and
those of the 2DeteCT benchmark paper (``deteCT_benchmark``: Kiss et al.,
"Benchmarking learned algorithms for computed tomography image reconstruction
tasks", 2025), which reproduce it and are not changed.
