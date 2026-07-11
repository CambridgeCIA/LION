End-to-end Pipeline
===================

The canonical entry point is::

   bash scripts/paper_scripts/PaDIS-Reproduction/pipeline/PaDIS_run_pipeline.sh MODE

where ``MODE`` is ``gcp``, ``slurm``, ``smoke``, or ``fast-smoke``.  The GCP
and Slurm paths implement equivalent training, validation-intensive final
phases, reconstruction, unconditional generation, verification, and reporting.

Before starting
---------------

Complete the LIDC-IDRI download and preprocessing instructions in the
PaDIS-Reproduction README.  Configure ``LION_DATA_PATH``, verify available disk
space, and install the platform-specific dependencies.  Production runs also
need W&B credentials unless logging is explicitly disabled.

Smoke modes
-----------

``fast-smoke`` selects representative training/tuning configurations and a
small reconstruction set.  ``smoke`` exercises one experiment per trained
model.  Both use one sample and deliberately truncated sampler work; neither
changes the checked-in hyperparameter registry.

Resumption
----------

Platform runners persist phase markers and manifests beneath their output
roots.  Jobs are identified from semantic configuration rather than only task
position, allowing completed work to survive compatible manifest rebuilding.
Always inspect reconciliation reports when switching commits around an active
production run.
