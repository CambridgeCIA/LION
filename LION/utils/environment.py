# =============================================================================
# This file is part of LION library
# License : GPL-3
# =============================================================================
"""The record of the environment a model was saved in.

``environment_record()`` is written by ``LIONmodel.save`` in every model file,
as its ``environment`` (a ``LIONParameter``, saved as plain JSON):

- ``lion``: LION's ``version``; for a git checkout its ``commit``, the LION
  files with ``uncommitted`` changes (files git tracks that changed, and
  untracked files that were imported), a ``diff_sha256`` of those changes, and
  whether the commit is ``pushed`` (in a remote branch git knows of); for an
  installed LION, its ``distribution`` version;
- ``python``, and ``packages``: the versions of the packages that change
  LION's numbers (torch, numpy, the CT backends; ``None`` if not installed);
- ``torch``: the CUDA and cuDNN it was built with, and its determinism
  settings (``deterministic_algorithms``, ``cudnn_deterministic``,
  ``cudnn_benchmark``): same-seed GPU runs are only identical with them on;
- ``device``: the device type the model was on, the GPUs and their driver;
- ``platform``: the operating system.

Nothing personal is recorded (no hostname, user or paths). A full list of the
installed packages belongs to ``publish()``, not to every file.

``code_differences(record)`` says how the running LION code differs from the
code a file was saved with (``LIONmodel`` warns on load);
``publish_problems(record)`` lists what keeps a result from being published
(uncommitted or unpushed LION code).
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import pathlib
import platform
import subprocess
import sys

import torch

import LION
from LION.utils.parameter import LIONParameter

RECORD_KEY = "lion_environment_record"
RECORD_VERSION = 1

#: The LION package folder.
LION_FOLDER = pathlib.Path(LION.__file__).resolve().parent
#: Packages whose versions change LION's numbers, and the modules they install.
PACKAGES = {
    "torch": "torch",
    "numpy": "numpy",
    "astra-toolbox": "astra",
    "tomosipo": "tomosipo",
    "ts_algorithms": "ts_algorithms",
}


def is_environment_record(value) -> bool:
    if isinstance(value, LIONParameter):
        return hasattr(value, RECORD_KEY)
    return isinstance(value, dict) and RECORD_KEY in value


def environment_record(device=None) -> LIONParameter:
    """The environment LION runs in now (see the module docstring); ``device``:
    the device of the model being saved (default: LION's default device)."""
    if device is None:
        from LION.utils.device import default_device

        device = default_device()
    device = torch.device(device)
    static = _static()
    record = {
        RECORD_KEY: RECORD_VERSION,
        "lion": {"version": LION.__version__, **code_state()},
        "python": static["python"],
        "packages": static["packages"],
        "torch": {
            "cuda": torch.version.cuda,
            "cudnn": static["cudnn"],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "device": {"type": device.type, **static["gpus"]},
        "platform": static["platform"],
    }
    if record["lion"]["commit"] is None:
        record["lion"]["distribution"] = _version("LION")
    return LIONParameter(**record)


# ---------------------------------------------------------------------------
# LION's code


def _git(folder, *args):
    """``git args`` in ``folder``, its output, or None (no git, not a checkout)."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=folder,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def code_state(folder=LION_FOLDER) -> dict:
    """The state of the code in ``folder`` (a package in a git checkout):
    ``commit``, ``uncommitted`` (files changed since it: tracked files, and
    untracked ones that were imported), ``diff_sha256`` (of those changes; None
    if there are none) and ``pushed`` (the commit is in a remote branch).
    All None if ``folder`` is not in a git checkout."""
    folder = pathlib.Path(folder).resolve()
    root = _git_root(folder)
    if root is None:
        return {
            "commit": None,
            "uncommitted": None,
            "diff_sha256": None,
            "pushed": None,
        }
    inside = folder.relative_to(root).as_posix()
    # one call: the commit (branch.oid) and the changed files
    status = _git(
        root,
        "status",
        "--porcelain=v2",
        "--branch",
        "--untracked-files=all",
        "--",
        inside,
    )
    commit, changed, untracked = None, [], []
    for line in (status or "").splitlines():
        if line.startswith("# branch.oid "):
            commit = line.split()[-1]
        elif line.startswith("? "):
            untracked.append(line[2:])
        elif line[:2] in ("1 ", "2 ", "u "):
            # ordinary, renamed/copied and unmerged entries: the path is last
            fields = {"1": 8, "2": 9, "u": 10}[line[0]]
            changed.append(line.split(" ", fields)[-1].split("\t")[0])
    if untracked:  # only those that were imported are LION's code
        loaded = _loaded_files(folder)
        untracked = [p for p in untracked if (root / p).resolve() in loaded]

    uncommitted = sorted(changed + untracked)
    digest = None
    if uncommitted:
        hasher = hashlib.sha256(
            (_git(root, "diff", "HEAD", "--", inside) or "").encode()
        )
        for path in sorted(untracked):
            hasher.update(path.encode())
            hasher.update((root / path).read_bytes())
        digest = hasher.hexdigest()
    return {
        "commit": commit,
        "uncommitted": uncommitted,
        "diff_sha256": digest,
        "pushed": _pushed(root, commit),
    }


@functools.lru_cache(maxsize=None)
def _git_root(folder):
    top = _git(folder, "rev-parse", "--show-toplevel")
    return None if top is None else pathlib.Path(top.strip()).resolve()


def _pushed(root, commit) -> bool:
    # not cached: the commit may be pushed while a program (a notebook) runs
    branches = _git(root, "branch", "-r", "--contains", commit) if commit else None
    return bool(branches and branches.strip())


def _loaded_files(folder) -> set:
    """The files of the modules imported from ``folder``."""
    prefix = str(folder)
    files = set()
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file and file.startswith(prefix):
            files.add(pathlib.Path(file).resolve())
    return files


def _field(record, *names):
    for name in names:
        record = (
            record.get(name)
            if isinstance(record, dict)
            else getattr(record, name, None)
        )
    return record


def code_differences(record, folder=LION_FOLDER):
    """How the LION code running now differs from the code ``record`` (an
    environment record) was made with, as a sentence; None if it doesn't (or
    it can't be told)."""
    saved = {
        name: _field(record, "lion", name)
        for name in ("version", "commit", "diff_sha256", "uncommitted")
    }
    now = {"version": LION.__version__, **code_state(folder)}
    if saved["commit"] is not None and now["commit"] is not None:
        if (saved["commit"], saved["diff_sha256"]) == (
            now["commit"],
            now["diff_sha256"],
        ):
            return None
        return (
            f"it was saved with LION {saved['version']} at commit {saved['commit'][:10]}"
            + (
                f" with uncommitted changes to {', '.join(saved['uncommitted'])}"
                if saved["uncommitted"]
                else ""
            )
            + f"; this is commit {now['commit'][:10]}"
            + (
                f" with uncommitted changes to {', '.join(now['uncommitted'])}"
                if now["uncommitted"]
                else ""
            )
        )
    if saved["version"] is not None and saved["version"] != now["version"]:
        return (
            f"it was saved with LION {saved['version']}; this is LION {now['version']}"
        )
    return None


def publish_problems(record) -> list:
    """What keeps a result made in environment ``record`` from being
    published: LION code with uncommitted changes, or a commit that isn't
    pushed. Empty if nothing."""
    lion = _field(record, "lion")
    problems = []
    if _field(lion, "commit") is None:
        distribution = _field(lion, "distribution")
        if distribution is None or "+" in distribution:
            problems.append(
                f"LION {distribution} is neither a git checkout nor a released version"
            )
        return problems
    if _field(lion, "uncommitted"):
        problems.append(
            "LION has uncommitted changes: commit them first ("
            + ", ".join(_field(lion, "uncommitted"))
            + ")"
        )
    if not _field(lion, "pushed"):
        problems.append(
            f"LION's commit {_field(lion, 'commit')[:10]} is not pushed: push it first"
        )
    return problems


# ---------------------------------------------------------------------------
# What doesn't change while a program runs (looked up once)


def _version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


@functools.lru_cache(maxsize=None)
def _static() -> dict:
    packages = {}
    for distribution, module in PACKAGES.items():
        version = _version(distribution)
        if version is None and module in sys.modules:  # e.g. built by conda
            version = getattr(sys.modules[module], "__version__", None)
        packages[distribution] = version
    gpus = {"gpus": [], "driver": None}
    if torch.cuda.is_available():
        gpus["gpus"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        driver = _run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )
        gpus["driver"] = driver.splitlines()[0].strip() if driver else None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "cudnn": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
        "gpus": gpus,
        "platform": platform.platform(terse=True),
    }


def _run(command):
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
