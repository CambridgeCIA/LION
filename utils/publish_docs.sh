#!/bin/bash
# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
# Build the documentation site with the demos run (their figures and output),
# and publish it to GitHub Pages (the gh-pages branch).
#
# Run from the repository root, on a machine with a GPU and the data
# (LION_DATA_PATH), in an environment with LION and its docs extra
# (pip install -e ".[docs]"):
#
#   bash utils/publish_docs.sh          # build and publish
#   bash utils/publish_docs.sh --build  # build only (docs/_build/html)
set -euo pipefail

rm -rf docs/_build docs/generated docs/tutorials
LION_DOCS_RUN_DEMOS=1 python -m sphinx -b html docs docs/_build/html

if [ "${1:-}" != "--build" ]; then
    # one commit on gh-pages, replacing what was there; .nojekyll so GitHub
    # serves Sphinx's _static folders
    ghp-import --no-jekyll --push --force \
        --message "Docs from $(git rev-parse --short HEAD)" docs/_build/html
fi
