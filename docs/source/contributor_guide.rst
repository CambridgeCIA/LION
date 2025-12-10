=================
Contributor Guide
=================

Repo structure
==============
Except for ``astra-toolbox`` and ``pip`` which are specified in the conda yaml setup file,
this repository uses a *pyproject.toml* file to specify all the requirements.

**.github/workflows**
    Definitions of GitHub action workflows to carry out formatting checks.

**.docs**
    Files to create this documentation.

**examples**
    Python scripts showcasing how LION can be used.
    The translation from Python script to Jupyter notebook is done
    using `jupytext <https://jupytext.readthedocs.io/en/latest/>`_ .
    See its documentation for more details.

    After translating the scripts to notebooks, the notebooks are run and their output is converted to html and added
    to this documentation in the *Examples* section.

    All output cells in the notebooks are automatically cleared, and only cleared notebooks should be added to the repository.

**LION/classical_algorithms**
    Implementation of classical algorithms for tomographic reconstruction tasks.

**LION/CTtools**
    Tools for CT reconstruction tasks.

**LION/data_loaders**
    TODO: Add description about dataloaders.

**TODO: Add other LION subfolders**

**tests**
    Tests which are run by pytest.
    The subfolder structure should follow the same structure as in *LION/*.


Linting
=======
We use Black for linting.

In CI, our linting is driven by `pre-commit <https://pre-commit.com/>`_.
If you install LION via ``pip install -e .[dev]``, pre-commit will be installed in your Python environment.
You can either add pre-commit to your git pre-commit hooks, requiring it to pass before each commit (``pre-commit install``),
or run it manually using ``pre-commit run --all-files`` after making your changes, before requesting a PR review.

Naming convention
=================
TODO: Add naming convention.

Testing
=======
We use pytest for testing. All required packages will be installed if you install LION via ``pip install -e .[dev]`` or ``pip install -e .[tests]``.
You can use VSCode's test panel to discover and run tests. All tests must pass before a PR can be merged. By default, we skip running CUDA tests.  You can use ``pytest -m cuda`` to run the CUDA tests if your development machine has a GPU available.

Building the Documentation
==========================
You can build the documentation locally via running ``make html`` in the ``docs`` folder.

By default, ``make html`` with look for notebooks in ``examples/notebooks`` and execute them.
If you want to build without running the notebooks for a quick check, you can use ``NORUN=1 make html``.
Note that this will replace all executed notebooks in the documentation with cleared notebooks
so you will need to run ``make html`` again to check the executed notebooks.
Since all notebooks are run, this can take some time, so it is recommended to keep notebooks small and fast.

Please check how your new additions render in the documentation before requesting a PR review.


Adding new examples
===================
New exciting examples of LION can be added in ``scripts/example_scripts`` as ``.py`` files.

Release Strategy
================
TODO: Add release strategy.

Compatibility
=============
TODO: Add compatibility information.
