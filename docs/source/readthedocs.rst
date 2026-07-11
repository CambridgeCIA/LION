Publishing on Read the Docs
===========================

LION includes a version-2 Read the Docs configuration at
``.readthedocs.yaml``. It builds this Sphinx site with Python 3.12, installs
the ``docs`` dependency extra, checks out repository submodules, and treats
documentation warnings as build failures.

One-time project setup
----------------------

The repository configuration cannot create the hosted Read the Docs project.
A repository administrator must perform these steps once:

1. Sign in to `Read the Docs <https://readthedocs.org/>`_ using GitHub.
2. Choose **Add project** and import ``THartigan/LION``.
3. Leave the configuration-file path at ``.readthedocs.yaml``.
4. In **Admin > Advanced settings**, set the default branch to the branch that
   should publish as ``latest`` (normally ``main``).
5. Enable builds for pull requests if documentation previews are wanted.
6. Trigger the first build and confirm that the configuration step reports
   ``docs/source/conf.py`` and Python 3.12.

The GitHub integration installs a webhook during import. Subsequent pushes to
an active version build automatically; enabled pull requests receive isolated
preview builds. Read the Docs version settings control which branches and tags
remain public.

Local equivalent
----------------

Run the same strict Sphinx build before pushing::

   python -m pip install -e ".[docs]"
   make -C docs clean html

The result is written to ``docs/_build/html``. The separate GitHub Actions
documentation workflow performs this check as an additional pre-publication
signal; Read the Docs remains responsible for hosting and versioned builds.

Badge setup
-----------

After importing the project, copy its exact Read the Docs *project slug* from
**Admin > General settings** and add the badge offered by **Admin > Badges** to
the root README. The slug is assigned by Read the Docs and should not be
guessed in repository configuration.
