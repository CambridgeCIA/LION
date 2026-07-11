PaDIS Reproduction
==================

The PaDIS reproduction package provides a complete study workflow around the
LION-native implementation: dataset preparation, prior training, fixed-
validation tuning, reconstruction and unconditional generation, verification,
and final table/figure generation.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Concepts
      :link: concepts
      :link-type: doc

      Understand priors, schedules, implementations, and corrected result identifiers.

   .. grid-item-card:: End-to-end pipeline
      :link: pipeline
      :link-type: doc

      Run the same logical workflow on GCP or Slurm.

   .. grid-item-card:: Reconstruction matrix
      :link: reconstruction
      :link-type: doc

      Inspect the 109-job experiment registry and backward-compatible outputs.

   .. grid-item-card:: Tuning and reporting
      :link: tuning_reporting
      :link-type: doc

      Reproduce validation tuning and regenerate metrics, tables, and figures.

.. toctree::
   :maxdepth: 2

   concepts
   pipeline
   reconstruction
   tuning_reporting
   scripts
