# LION

LION is a research library for learned tomographic (and other inverse-problem)
reconstruction in PyTorch. Keep it modular, light and close to plain torch.

Read `contributor-guide.md` before changing code: it has the conventions
(tests in flat `pytests/`, backends imported only where used, a missing
parameter means the old behaviour, models follow their papers, conda for
anything compiled, `(output, target)` order, the metric interface).

# Project Instructions

## Project Structure

LION is a modular Python/PyTorch research library for learned tomographic reconstruction and inverse problems.

### Main Package: `LION/`

- `LION/data_loaders/`: Dataset implementations, downloading and preprocessing.
- `LION/models/`: Neural network architectures and learned reconstruction methods.
- `LION/operators/`: Forward operators, adjoint operators and mathematical imaging operators.
- `LION/solvers/`: Iterative reconstruction algorithms and optimization methods.
- `LION/experiments/`: Experiment configurations, geometries, noise models and acquisition settings.
- `LION/utils/`: Shared utilities used by the main package.

### Supporting Directories

- `pytests/`: Automated tests. Keep tests in this directory rather than creating a new `tests/` directory.
- `demos/`: Demonstrations, examples and usage notebooks.
- `scripts/`: Standalone execution and development scripts.
- `utils/`: Repository-level supporting utilities.
- `misc/`: Miscellaneous and experimental code.
- `.github/workflows/`: CI workflows and automated testing.

### Documentation and Configuration

- `README.md`: Project overview and installation.
- `CLAUDE.md`: AI development instructions.
- `contributor-guide.md`: Coding conventions and contribution guidelines.
- `developers.md`: Developer documentation.
- `pyproject.toml`: Python project and package installation configuration.
- `env_base.yml`: Conda environment configuration.

## Development efficiency

- Inspect only files relevant to the requested change and their direct dependencies.
- Avoid exploring unrelated modules or the entire repository.
- Follow existing implementations and patterns.
- Avoid unrelated refactoring.
- Run targeted tests rather than the full test suite.
- Keep responses concise and avoid repeating established context.
