Getting Started
===============

Prerequisites
-------------

LION supports Python 3.9--3.12.  CT operators require ASTRA and tomosipo; a
CUDA-capable installation is strongly recommended for reconstruction and
training.  CPU-only environments can build the documentation and run the
non-CUDA test suite.

Installation
------------

Clone the repository and initialise its legacy submodule::

   git clone https://github.com/CambridgeCIA/LION.git
   cd LION
   git submodule update --init --recursive

Create the supplied Conda environment, install a compatible PyTorch build, and
install LION in editable mode::

   conda env create --file env_base.yml --name lion
   conda activate lion
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   python -m pip install -e ".[dev]"

Data and experiment roots
-------------------------

LION intentionally has no implicit machine-specific data location.  Define at
least ``LION_DATA_PATH`` before importing modules that resolve dataset paths::

   export LION_DATA_PATH=/path/to/Data

Experiment outputs default to ``$LION_DATA_PATH/experiments``.  Override that
location independently when needed::

   export LION_EXPERIMENTS_PATH=/scratch/lion-experiments

Local development hooks
-----------------------

Install formatting and test hooks with::

   pre-commit install --hook-type pre-commit --hook-type post-merge

When commits are initiated by a GUI that does not inherit the active Conda
environment, configure the interpreter for that checkout::

   git config --local lion.testPython "$CONDA_PREFIX/bin/python"

The hardware-aware test launcher includes CUDA tests when CUDA is available
and selects the CPU-safe suite otherwise.

Building the documentation
--------------------------

The source lives under ``docs/source``.  Build the HTML site with::

   cd docs
   make html

The result is written to ``docs/_build/html/index.html``.

Hosted documentation
--------------------

The repository's ``.readthedocs.yaml`` builds the same strict Sphinx target
with Python 3.12 and the ``docs`` dependency extra. After importing the GitHub
repository into Read the Docs, select the branch to publish as ``latest`` and
optionally enable pull-request previews. The GitHub documentation workflow
also builds and uploads the HTML site on pushes and pull requests.
