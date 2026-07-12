LION Documentation
==================

.. rst-class:: lead

LION is a PyTorch framework for learned tomographic and computational-imaging
reconstruction.  It combines CT geometry and operators, reusable datasets and
experiments, learned reconstruction models, classical baselines, and
reproducible research pipelines.

This first documentation release gives full attention to the PaDIS diffusion
workflows, the LIDC-IDRI extensions used by them, and the associated CT
reconstruction tools.  Older package areas are indexed so that their public
surface is discoverable, but are explicitly marked where narrative
documentation is still incomplete.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: getting_started
      :link-type: doc

      Install LION, configure data paths, and verify the environment.

   .. grid-item-card:: Architecture
      :link: architecture
      :link-type: doc

      Understand the package layers and how data, models, solvers, and reconstructors interact.

   .. grid-item-card:: PaDIS Reproduction
      :link: padis/index
      :link-type: doc

      Train priors, tune reconstruction, run the experiment matrix, and produce paper artefacts.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Browse the documented Python interfaces and provisional package stubs.

Quick start
-----------

Install LION in an isolated Conda environment and point it at a data root::

   conda env create --file env_base.yml --name lion
   conda activate lion
   pip install -e ".[dev]"
   export LION_DATA_PATH=/path/to/Data

Build this site locally with::

   cd docs
   make html

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   architecture
   testing
   documentation_status
   padis/index
   readmes/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
