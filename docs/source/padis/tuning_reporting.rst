Tuning and Reporting
====================

Fixed-validation tuning
-----------------------

The reproducible tuning launcher runs the candidate table documented in
``tuning/TUNING.md``.  Candidates use validation data only, fixed sample
indices, and the final checkpoint policy used by reconstruction.  The launcher
does not rewrite the checked-in default registry unless export is requested
explicitly.

Reporting
---------

``PaDIS_make_tables.py`` consumes the verification CSV and can calculate timing
rows from GCP/Colab progress logs or Slurm output.  It writes decoded CSV tables
and a LaTeX fragment.

``PaDIS_make_paper_figures.py`` reads saved reconstruction and generation
payloads.  It uses LION's normalised-intensity-to-HU conversion, symmetric
content crops, aligned colour-scale labels, and legacy-path fallbacks.

Reproducibility checks
----------------------

Before accepting final artefacts, confirm that:

- the expected-jobs manifest matches the intended commit;
- verification reports no missing, unexpected, or settings-mismatch rows;
- timing mode matches the platform logs;
- the hyperparameter registry checksum is unchanged by smoke runs; and
- generated tables use CP/TV and FDK terminology consistently.
