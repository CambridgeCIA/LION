#!/usr/bin/env bash
# Select the project Python explicitly because GUI Git clients usually do not
# inherit an activated Conda environment.
set -euo pipefail

LION_ROOT="$(git rev-parse --show-toplevel)"
cd "$LION_ROOT"

test_python="${LION_TEST_PYTHON:-}"
if [ -z "$test_python" ]; then
        test_python="$(git config --local --get lion.testPython || true)"
fi
if [ -z "$test_python" ] && [ -n "${CONDA_PREFIX:-}" ]; then
        test_python="$CONDA_PREFIX/bin/python"
fi
if [ -z "$test_python" ]; then
        for candidate in python3 python; do
                if command -v "$candidate" >/dev/null 2>&1 \
                        && "$candidate" -c 'import torch' >/dev/null 2>&1; then
                        test_python="$(command -v "$candidate")"
                        break
                fi
        done
fi

if [ -z "$test_python" ] || [ ! -x "$test_python" ]; then
        cat >&2 <<'EOF'
No LION test interpreter with PyTorch is configured.
Configure this checkout once with:

  git config --local lion.testPython /absolute/path/to/lion-env/bin/python

Alternatively, set LION_TEST_PYTHON in the environment used by Git.
EOF
        exit 1
fi

# Selecting an interpreter directly does not activate its Conda environment.
# CuPy uses CONDA_PREFIX to locate the bundled CUDA toolkit, so reconstruct the
# prefix for configured Conda interpreters (notably when Git is launched by a
# desktop application such as VS Code).
test_prefix="$(cd "$(dirname "$test_python")/.." && pwd -P)"
if [ -d "$test_prefix/conda-meta" ]; then
        export CONDA_PREFIX="$test_prefix"
        export PATH="$test_prefix/bin:$PATH"
fi

exec "$test_python" scripts/run_tests.py "$@"
