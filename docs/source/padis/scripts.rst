Script Reference
================

The reproduction scripts are intentionally thin orchestration layers around
the documented LION APIs.  Their filenames are stable command-line entry
points; output manifests record the exact expanded commands.

Pipeline
--------

``pipeline/PaDIS_run_pipeline.sh``
   Top-level dispatcher for GCP, Slurm, smoke, and fast-smoke modes.

``pipeline/PaDIS_finalise_pipeline.sh``
   Runs unconditional generation, verification, timing extraction, tables,
   and figures after reconstruction.

``pipeline/PaDIS_run_fast_smoke.sh``
   Hardware-bounded end-to-end validation using representative models and
   methods without changing final settings.

Training
--------

``training/PaDIS_LIDC_256.py``
   Patch/whole-image 256-square prior training with caching, W&B resumption,
   timed checkpoints, and validation-intensive selection.

``training/PaDIS_LIDC_512.py``
   Native-512 patch-prior training.

``training/PaDIS_LIDC_PnP_denoiser.py``
   Fixed-noise and noise-conditioned DRUNet training for PnP-ADMM.

Reconstruction
--------------

``reconstruction/PaDIS_LIDC_reconstruction.py``
   Single-method reconstruction driver shared by matrix and tuning jobs.

``reconstruction/PaDIS_run_reconstruction_matrix.py``
   Declarative 109-job matrix expansion, ordering, tuned-default application,
   and command generation.

``reconstruction/PaDIS_verify_reconstruction_matrix.py``
   Completeness, settings, finite-metric, bootstrap, and optional quality-gate
   verification.

``reconstruction/PaDIS_reconcile_reconstruction_manifest.py``
   Semantic migration of runner markers when a manifest is rebuilt.

``reconstruction/PaDIS_LIDC_generation.py``
   Seeded unconditional prior generation.

Tuning and reporting
--------------------

``tuning/PaDIS_run_reproduction_tuning.py`` and ``tuning/PaDIS_tune_reconstruction_hyperparameters.py``
   Reproduce the documented validation-only candidate design.

``tuning/PaDIS_hparam_defaults.py``
   Select and export model-safe final hyperparameter defaults.

``reporting/PaDIS_make_tables.py``
   Decode verification metrics, extract GCP/Colab or Slurm timings, and write
   CSV/LaTeX tables.

``reporting/PaDIS_make_paper_figures.py``
   Render the complete figure set with canonical labels and legacy path
   compatibility.

Platform launchers
------------------

``platforms/gcp`` contains restartable spot-instance training and manual
multi-GPU reconstruction runners.  ``platforms/slurm`` contains matching A100
array jobs, submission helpers, preflight checks, and finalisation jobs.  Both
backends call the same Python training/reconstruction programs and use the same
hyperparameter registry; only scheduling, cache staging, and hardware-safe
batch limits differ.

Developer diagnostics
---------------------

The scripts under ``scripts/dev`` check public-repository equivalence, short-
run determinism, login-node cache visibility, machine readiness, training
checkpoint resumption, and reconstruction quality.  They emit structured JSON
where possible so failures can be archived alongside run logs.
