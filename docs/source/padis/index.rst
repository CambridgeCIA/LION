PaDIS Reproduction
==================

The PaDIS reproduction package provides the complete study workflow around the
LION-native implementation: dataset preparation, prior training, fixed-
validation tuning, reconstruction and unconditional generation, verification,
and final table/figure generation. The pages below divide the maintained
PaDIS-Reproduction README into task-focused guides, preserving its complete
commands, tables, defaults, runtime estimates, and operational cautions.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Setup and data
      :link: setup
      :link-type: doc

      Install LION, obtain and process LIDC-IDRI, build caches, and provision a backend.

   .. grid-item-card:: Training and checkpoints
      :link: training
      :link-type: doc

      Reproduce diffusion and PnP training, checkpoint selection, W&B logging, and resumption.

   .. grid-item-card:: Methods and experiments
      :link: methods
      :link-type: doc

      Understand every reconstruction method, implementation track, CT experiment, and generation preset.

   .. grid-item-card:: Hyperparameter tuning
      :link: tuning
      :link-type: doc

      Inspect final defaults, inheritance, and the validation-only selection procedure.

   .. grid-item-card:: Running and reporting
      :link: running_reporting
      :link-type: doc

      Run manual or matrix inference, estimate runtimes, resume work, verify outputs, and generate artefacts.

   .. grid-item-card:: Concepts and conventions
      :link: concepts
      :link-type: doc

      Understand priors, schedules, implementations, and corrected result identifiers.

   .. grid-item-card:: Script reference
      :link: scripts
      :link-type: doc

      Locate each pipeline, platform, training, reconstruction, tuning, verification, and reporting entry point.

.. toctree::
   :maxdepth: 2
   :hidden:

   setup
   training
   methods
   tuning
   running_reporting
   concepts
   scripts
