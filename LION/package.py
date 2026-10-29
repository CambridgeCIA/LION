# =============================================================================
# This file is part of LION library
# License : GPL-3
# =============================================================================
"""Publishing a trained model: everything needed to make it, train it again
and test it, in one folder (a package), from the records LION saved.

    from LION.package import publish
    publish("results/lpd/final", "lpd_package", evaluation="results/evaluations/lpd-1234.json")

``publish(model_file, folder, evaluation=None)``: ``model_file`` is a model a
LION solver saved (``"final"``, ``.json`` or ``.pt``); ``evaluation`` is
optional: its test record (``Evaluation.run``), the record's file, or an
``Evaluation``, which is then run on the model. The package:

    model.json, model.pt   the model file, exactly as LION saved it: LIONmodel.load("model")
    <name>_initial.pt      the weights training started from, when the solver saved them
    experiment.json        the experiment: geometry, noise, dataset (Experiment.instantiate)
    data/training.json     the training data's record: split, samples, noise seed
    data/validation.json   the validation data's record, if it was validated
    solver.json            the training record: solver, optimizer, seed, settings, history
    test.json              the test record, if an evaluation was given
    environment.txt        the packages installed where it was published (conda's and Python's)
    package.json           the contents and their sha256, the citations, LION's version and commit, notes
    README.md              a model card, generated from the records (edit it: it isn't checksummed)

Each file is made again by its own ``instantiate`` (``LIONmodel``,
``Experiment``, ``LIONDataset``, ``LIONsolver``, ``Evaluation``). A package
is used whole with:

    package = load("lpd_package")      # the recorded arguments, to change; nothing made
    package.model.model_parameters.n_iters = 5
    package.experiment.experiment_params.noise_params.I0 = 5000
    built = build(package)             # the model, experiment, data, solver, evaluation
    built = load_and_build("lpd_package")       # as published
    model = train_again(package, "retrained")   # trained again, changed or not

``load`` warns about files changed since publishing and LION code other than
the model's. Things run where the model was trained unless ``device=`` says
otherwise. The model card shows the same, and the code step by step.

``publish`` refuses, with an error and before writing anything:

- LION code that isn't committed and pushed (``environment.publish_problems``):
  the code running now, the code that saved the model and the code that tested it;
- a model file without a training record or a data record (one not saved by a
  LION solver from a LION dataset);
- what training again would miss: initial weights that weren't saved, an
  unseeded training (``seed=None``), data not made by an ``Experiment``, data
  outside ``LION_DATA_PATH`` (another machine can't find them);
- records naming code outside LION (other than torch's layers, losses and
  optimizers): a package is made again with LION alone, without ``allow=``;
- records that could only name something (e.g. a lambda loss), which can't be
  made again;
- a test record of other weights than the model's, or test data the model was
  trained or validated on;
- a folder that isn't empty.

What a reader should know of a complete package (it was published in another
environment than it was trained in, or tested on another problem) is a note,
in ``package.json`` and the model card, and a warning.
``citations`` are each component's ``cite()``: where it comes from, not the
work being published.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import pathlib
import shutil
import sys
import warnings


from LION.utils import environment, spec
from LION.utils.parameter import LIONParameter

PACKAGE_KEY = "lion_package"
PACKAGE_VERSION = 1

#: The files of a package, and what each is.
FILES = {
    "model": "model",
    "experiment": "experiment.json",
    "training_data": "data/training.json",
    "validation_data": "data/validation.json",
    "solver": "solver.json",
    "test": "test.json",
    "environment": "environment.txt",
}


class PublishError(ValueError):
    """What keeps a model from being published."""


def publish(model_file, folder, evaluation=None) -> pathlib.Path:
    """Write the package of ``model_file`` in ``folder`` (see the module
    docstring); returns the folder."""
    from LION.models.LIONmodel import LIONmodel
    from LION.optimizers.training_record import weights_sha256

    model_file = pathlib.Path(model_file).with_suffix("")
    folder = pathlib.Path(folder)
    for suffix in (".json", ".pt"):
        if not model_file.with_suffix(suffix).is_file():
            raise PublishError(f"{model_file.with_suffix(suffix)} doesn't exist")
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise PublishError(f"{folder} isn't an empty folder")

    saved = spec.read(model_file)
    training, data = saved.get("training"), saved.get("dataset_params")
    problems = []
    if training is None:
        problems.append(
            "the model file has no training record: publish a model a LION solver saved"
        )
    if data is None:
        problems.append(
            "the model file has no data record: train on a LION dataset "
            "(e.g. an Experiment's), so the data can be made again"
        )
    if problems:
        raise PublishError(_sentence(problems))

    test = _test_record(evaluation, model_file)
    problems += _code_problems(saved, test)
    problems += _not_made_again(saved, test)
    problems += _not_repeatable(saved, test)
    if problems:  # before loading: the model may name code outside LION
        raise PublishError(_sentence(problems))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _, _ = LIONmodel.load(model_file, supress_warnings=True)
    weights = weights_sha256(model)
    initial = _initial_weights_file(model_file, training)
    validation = (training.get("validation") or {}).get("data")
    if test is not None:
        problems += _test_problems(test, weights, data, validation)
    if problems:
        raise PublishError(_sentence(problems))
    notes = _notes(saved, test)
    for note in notes:
        warnings.warn(note, stacklevel=2)

    # -- writing ---------------------------------------------------------------
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data").mkdir()
    files = {}
    for suffix in (".json", ".pt"):
        shutil.copyfile(model_file.with_suffix(suffix), folder / f"model{suffix}")
    files["model"] = "model"
    if initial is not None:
        shutil.copyfile(initial, folder / initial.name)
        files["initial_weights"] = initial.name
    _write_json(folder / FILES["experiment"], data["experiment"])
    files["experiment"] = FILES["experiment"]
    _write_json(folder / FILES["training_data"], data)
    files["training_data"] = FILES["training_data"]
    if validation is not None:
        _write_json(folder / FILES["validation_data"], validation)
        files["validation_data"] = FILES["validation_data"]
    _write_json(folder / FILES["solver"], training)
    files["solver"] = FILES["solver"]
    if test is not None:
        _write_json(folder / FILES["test"], test)
        files["test"] = FILES["test"]
    (folder / FILES["environment"]).write_text(environment_list(), encoding="utf-8")
    files["environment"] = FILES["environment"]

    written = {
        PACKAGE_KEY: PACKAGE_VERSION,
        "files": files,
        "sha256": {
            path.relative_to(folder).as_posix(): file_sha256(path)
            for path in sorted(folder.rglob("*"))
            if path.is_file()
        },
        "weights_sha256": weights,
        "lion": _lion(saved.get("environment")),
        "published_with": _lion(environment.environment_record().serialize()),
        "published": datetime.date.today().isoformat(),
        "citations": citations(saved, test),
        "notes": notes,
    }
    _write_json(folder / "package.json", written)
    (folder / "README.md").write_text(
        model_card(saved, written, model, test), encoding="utf-8"
    )
    return folder


# ---------------------------------------------------------------------------
# Using a package


def load(folder) -> LIONParameter:
    """A package's components as their recorded arguments, to change before
    ``build`` makes them (nothing is made here):

    - ``model``: the model's arguments (``model_parameters``, and e.g.
      ``geometry``);
    - ``experiment``: the experiment's (``experiment_params``, ``dataset``);
    - ``training_data``, ``validation_data``: which data: ``split`` and
      ``subset`` (indices, or None); ``validation_data`` is None if it wasn't
      validated;
    - ``solver``: the solver's arguments (``loss_fn``, ``seed``,
      ``solver_params``, ...), and ``optimizer``: its hyperparameters (``lr``);
    - ``training``: ``batch_size``, ``shuffle``, ``drop_last`` and ``epochs``
      (those done);
    - ``validation``: ``batch_size``, ``every``, ``function`` and
      ``reconstructor`` (its arguments, or None); None if not validated;
    - ``evaluation``: the test's evaluation's arguments; None without a test;
    - ``device``: where it was trained; ``folder``; ``records``: the
      ``training``, ``data`` and ``test`` records and ``package.json``, as read.

    The classes are the recorded ones (another class's parameters come from
    its ``parameters_from_file``). Files changed since the package was
    published, and LION code other than the model was saved with, give a
    warning.
    """
    from LION.optimizers.training_record import read_training_record

    folder = pathlib.Path(folder)
    written = json.loads((folder / "package.json").read_text(encoding="utf-8"))
    _warn_changed_files(folder, written)
    saved = spec.read(folder / written["files"]["model"])
    difference = environment.code_differences(saved.get("environment") or {})
    if difference is not None:
        warnings.warn(f"{folder}: {difference}", stacklevel=2)
    training, data = saved["training"], saved["dataset_params"]
    test = None
    if "test" in written["files"]:
        test = spec.read(folder / written["files"]["test"])

    package = LIONParameter()
    package.folder = folder
    package.device = (saved.get("environment") or {}).get("device", {}).get("type")
    package.model = _arguments(saved["model"])
    package.experiment = _arguments(data["experiment"])
    package.training_data = _which(data, data)
    package.solver = _arguments(training["solver"])
    package.optimizer = _arguments(training["optimizer"])
    feed = training["train"]
    package.training = LIONParameter()
    package.training.batch_size = feed["batch_size"]
    package.training.shuffle = feed["shuffle"]
    package.training.drop_last = feed["drop_last"]
    package.training.epochs = training["epochs"]["done"]
    package.validation_data = package.validation = None
    validation = training.get("validation")
    if validation is not None:
        package.validation_data = _which(validation["data"], data)
        package.validation = LIONParameter()
        package.validation.batch_size = validation["batch_size"]
        package.validation.every = validation["every"]
        package.validation.function = spec.build(validation["function"])
        package.validation.reconstructor = (
            None
            if validation.get("reconstructor") is None
            else _arguments(validation["reconstructor"])
        )
    package.evaluation = None if test is None else _arguments(test["evaluation"])
    package.records = LIONParameter()
    package.records.training = read_training_record(saved)
    package.records.data = LIONParameter(**data)
    package.records.test = None if test is None else LIONParameter(**test)
    package.records.package = LIONParameter(**written)
    return package


def build(package, *, device=None, initial_weights=False) -> LIONParameter:
    """The components a package (``load``) describes, made, changed or not:
    ``model``, ``experiment``, ``training_data``, ``validation_data``,
    ``solver`` (with ``model``, its training and validation data set; not its
    saving), ``evaluation`` (None without a test) and ``device``.

    The model has its trained weights, or, with ``initial_weights=True``, the
    weights its training started from; a model whose arguments were changed
    has new, random ones (it is to be trained). ``device``: where the
    solver and the evaluation run; by default where it was trained (the CPU
    if that was a GPU and there is none).
    """
    from torch.utils.data import DataLoader, Subset

    from LION.data_loaders.LIONDataset import _check_samples
    from LION.evaluation.evaluation import Evaluation
    from LION.models.LIONmodel import LIONmodel

    folder = package.folder
    written = package.records.package.files
    saved = spec.read(folder / written.model)
    device = _device(package.device if device is None else device)

    model_file = folder / written.model
    if _changed(package.model, saved["model"]):
        model = spec.resolve(saved["model"][spec.TARGET])(**vars(package.model))
    elif initial_weights:
        model = LIONmodel.instantiate(model_file, initial_weights=True)
    else:
        model = LIONmodel.load(model_file, supress_warnings=True)[0]

    data = saved["dataset_params"]
    experiment = _made(data["experiment"], package.experiment)
    same_experiment = not _changed(package.experiment, data["experiment"])

    def made(which, record):
        dataset = experiment.get_dataset(which.split)
        if same_experiment and which.split == record["split"]:
            recorded = (record.get("subset") or {}).get("indices")
            if which.subset == recorded:
                _check_samples(dataset, LIONParameter(**record), which.subset)
        return dataset if which.subset is None else Subset(dataset, which.subset)

    training = saved["training"]
    training_data = made(package.training_data, data)
    validation_data = None
    if package.validation_data is not None:
        validation_data = made(package.validation_data, training["validation"]["data"])

    optimizer = spec.resolve(training["optimizer"][spec.TARGET])(
        model.parameters(), **vars(package.optimizer)
    )
    solver = spec.resolve(training["solver"][spec.TARGET])(
        model=model, optimizer=optimizer, device=device, **vars(package.solver)
    )
    t = package.training
    solver.set_training(
        DataLoader(
            training_data,
            batch_size=t.batch_size,
            shuffle=t.shuffle,
            drop_last=t.drop_last,
        )
    )
    if package.validation is not None:
        v = package.validation
        reconstructor = None
        if v.reconstructor is not None:
            reconstructor = _made(
                training["validation"]["reconstructor"], v.reconstructor
            )
        solver.set_validation(
            DataLoader(validation_data, batch_size=v.batch_size),
            v.every,
            v.function,
            reconstructor=reconstructor,
        )

    evaluation = None
    if package.evaluation is not None:
        evaluation = Evaluation(**vars(package.evaluation), device=device)

    built = LIONParameter()
    built.model = model
    built.experiment = experiment
    built.training_data = training_data
    built.validation_data = validation_data
    built.solver = solver
    built.evaluation = evaluation
    built.device = device
    return built


def load_and_build(folder, *, device=None) -> LIONParameter:
    """``build(load(folder))``: the package's components, as published."""
    return build(load(folder), device=device)


def train_again(package, save_folder, name="model", *, device=None):
    """Train the model of ``package`` (a ``load``-ed package, changed or not,
    or its folder) again, from its initial weights (new ones if its arguments
    were changed), for its epochs; saved as ``save_folder/name``, as a solver
    saves (``name`` and, if validated, its best validation). Returns the
    trained model. ``device``: as in ``build``."""
    if not isinstance(package, LIONParameter):
        package = load(package)
    built = build(package, device=device, initial_weights=True)
    save_folder = pathlib.Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    built.solver.set_saving(save_folder, name)
    built.solver.train(package.training.epochs)
    built.solver.save_final_results()
    return built.model


def _warn_changed_files(folder, written):
    changed = []
    for name, digest in written["sha256"].items():
        path = folder / name
        if not path.is_file():
            changed.append(f"{name} (missing)")
        elif file_sha256(path) != digest:
            changed.append(name)
    if changed:
        warnings.warn(
            f"{folder}: changed since it was published: {', '.join(changed)}",
            stacklevel=3,
        )


def _arguments(written) -> LIONParameter:
    """A spec's arguments, made (a LIONParameter of them), without its call."""
    _, kwargs = spec.build_arguments(written)
    arguments = LIONParameter()
    for name, value in kwargs.items():
        setattr(arguments, name, value)
    return arguments


def _which(record, training) -> LIONParameter:
    """Which data a record describes: its ``split`` and ``subset`` indices."""
    which = LIONParameter()
    which.split = record["split"]
    which.subset = (record.get("subset") or {}).get("indices")
    if record.get("experiment") != training.get("experiment"):
        warnings.warn(
            "the validation data were made by another experiment than the "
            "training data: they are made with the training's",
            stacklevel=3,
        )
    return which


def _changed(arguments, written) -> bool:
    """Whether ``arguments`` (``_arguments(written)``, maybe changed) differ
    from the recorded ones."""
    recorded = {k: v for k, v in written.items() if k != spec.TARGET}
    return {k: spec.encode(v) for k, v in vars(arguments).items()} != recorded


def _made(written, arguments):
    """The object ``written`` names, made with ``arguments``."""
    return spec.resolve(written[spec.TARGET])(**vars(arguments))


def _device(device):
    import torch

    device = torch.device(device or "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("trained on a GPU, but there is none here: on the CPU")
        return torch.device("cpu")
    return device


# ---------------------------------------------------------------------------
# Checks


def _test_record(evaluation, model_file):
    """The test record ``evaluation`` gives (see ``publish``), as a dict."""
    if evaluation is None:
        return None
    from LION.evaluation.evaluation import Evaluation, read_test_record

    if isinstance(evaluation, Evaluation):
        record = evaluation.run(model_file)
    else:
        record = read_test_record(evaluation)
    return record.serialize()


def _code_problems(saved, test) -> list:
    """LION code that isn't committed and pushed, now and when the model was
    saved and tested."""
    problems = []
    for when, record in (
        ("now", environment.environment_record()),
        ("when the model was saved", saved.get("environment")),
        ("when it was tested", None if test is None else test.get("environment")),
    ):
        if record is None:
            if when == "when the model was saved":
                problems.append(
                    "the model file doesn't record LION's code (saved before "
                    "LION recorded it): save it again with this LION"
                )
            continue
        problems += [f"{p} ({when})" for p in environment.publish_problems(record)]
    return problems


def _not_made_again(saved, test) -> list:
    """What the records name outside LION, or could only name."""
    parts = {
        "the model": saved.get("model"),
        "the data": saved.get("dataset_params"),
        "the training": saved.get("training"),
    }
    if test is not None:
        parts["the test"] = {k: test.get(k) for k in ("method", "evaluation", "data")}
        parts["the test's metrics"] = test.get("metrics")
    problems = []
    for name, part in parts.items():
        outside, unsaved = set(), []
        _walk(part, outside, unsaved)
        if outside:
            problems.append(
                f"{name} names code outside LION ({', '.join(sorted(outside))}): "
                "only LION's classes can be published"
            )
        if unsaved:
            problems.append(
                f"{name} could only be named, not saved ({'; '.join(unsaved)}), "
                "so it can't be made again: use a LION class, or "
                "functools.partial of a LION function"
            )
    return problems


def _walk(value, outside, unsaved):
    if isinstance(value, list):
        for v in value:
            _walk(v, outside, unsaved)
    elif isinstance(value, dict):
        if "unsaved" in value and "reason" in value:
            unsaved.append(str(value["unsaved"]))
            return
        target = value.get(spec.TARGET)
        if isinstance(target, str) and not spec.is_allowed(target):
            outside.add(target)
        for v in value.values():
            _walk(v, outside, unsaved)


def _initial_weights_file(model_file, training):
    """The file of the weights training started from, if the solver saved them."""
    how = training.get("initial_weights") or {}
    if how.get("from") != "file":
        return None
    path = model_file.parent / how["file"]
    if not path.is_file():
        raise PublishError(
            f"the training started from weights saved in {path}, which is missing"
        )
    return path


def _test_problems(test, weights, data, validation) -> list:
    from LION.evaluation.comparison import shared_samples

    problems = []
    tested = (test.get("method") or {}).get("model")
    if not isinstance(tested, dict) or "weights_sha256" not in tested:
        problems.append("the test record isn't of a model")
    elif tested["weights_sha256"] != weights:
        problems.append("the test record is of other weights than the model file's")
    for what, record in (("trained", data), ("validated", validation)):
        if record is None:
            continue
        shared, kind, _ = shared_samples(test["data"], record)
        if shared:
            problems.append(
                f"the model was {what} on {len(shared)} of the test data's {kind} "
                f"({', '.join(shared[:3])}{', ...' if len(shared) > 3 else ''})"
            )
    return problems


def _not_repeatable(saved, test) -> list:
    """What the package would miss to train the model again as it was: the
    initial weights, the seed, the experiment that made the data, or data
    any machine can find (in ``LION_DATA_PATH``)."""
    training, data = saved["training"], saved["dataset_params"]
    validation = (training.get("validation") or {}).get("data")
    problems = []
    if (training.get("initial_weights") or {}).get("from") is None:
        problems.append(
            "the weights training started from weren't saved and can't be made "
            "again from the model's random state: train it with a save folder "
            "(set_saving), so the solver saves them"
        )
    if training.get("seed") is None:
        problems.append("training wasn't seeded (seed=None): train it with a seed")
    for what, record in (("training", data), ("validation", validation)):
        if record is not None and "experiment" not in record:
            problems.append(
                f"the {what} data weren't made by an Experiment: make them with "
                "experiment.get_dataset(split)"
            )
    records = [data, validation]
    if test is not None:
        records += [test.get("data"), test.get("evaluation")]
    paths = []
    for record in records:
        _absolute_paths(record, paths)
    if paths:
        problems.append(
            "the data are read from a folder outside LION_DATA_PATH "
            f"({', '.join(sorted(set(paths)))}), which another machine can't find: "
            "keep the data in LION_DATA_PATH"
        )
    return problems


def _notes(saved, test) -> list:
    """What a reader should know, though the package is complete."""
    from LION.evaluation.comparison import problem_differences

    notes = []
    now = environment.environment_record().serialize()
    trained = saved.get("environment") or {}
    changed = [
        f"{name} {trained.get('packages', {}).get(name)} -> {version}"
        for name, version in now["packages"].items()
        if trained.get("packages", {}).get(name) != version
    ]
    if trained.get("python") != now["python"]:
        changed.insert(0, f"Python {trained.get('python')} -> {now['python']}")
    if changed:
        notes.append(
            "environment.txt is where it was published, which differs from where "
            f"it was trained: {', '.join(changed)}"
        )
    if test is not None:
        differences = problem_differences(test["data"], saved["dataset_params"])
        if differences:
            notes.append(
                "it was tested on another problem than it was trained on: "
                + "; ".join(differences)
            )
    return notes


def _absolute_paths(value, found):
    if isinstance(value, list):
        for v in value:
            _absolute_paths(v, found)
    elif isinstance(value, dict):
        if value.get(spec.TARGET) == "pathlib.Path":
            found.append(str(value[spec.ARGS][0]))
            return
        for v in value.values():
            _absolute_paths(v, found)


# ---------------------------------------------------------------------------
# What a package holds


def citations(saved, test=None) -> list:
    """Where the model and its components come from: each one's ``cite()``
    (BibTeX), once each, as ``{"of": ..., "class": ..., "bibtex": ...}``."""
    training, data = saved.get("training") or {}, saved.get("dataset_params") or {}
    components = [
        ("model", saved.get("model")),
        ("solver", training.get("solver")),
        ("dataset", data.get("dataset")),
        ("experiment", data.get("experiment")),
        ("validation", (training.get("validation") or {}).get("reconstructor")),
    ]
    if test is not None:
        components.append(("test", (test.get("method") or {}).get("reconstructor")))
    found, texts = [], set()
    for role, written in components:
        target = written.get(spec.TARGET) if isinstance(written, dict) else None
        if target is None:
            continue
        try:
            cls = spec.resolve(target)
        except spec.SpecError:
            continue
        text = cls.cite("bib") if hasattr(cls, "cite") else None
        if text and text not in texts:
            texts.add(text)
            found.append({"of": role, "class": target, "bibtex": text})
    return found


def environment_list() -> str:
    """The packages installed where LION runs: conda's (``name=version=build``,
    as ``conda list --export`` writes them) when it is a conda environment,
    and Python's (``name==version``). Without paths."""
    lines = [
        "# The environment the package was published in",
        f"# Python {sys.version.split()[0]}, {environment.environment_record().platform}",
    ]
    conda = pathlib.Path(sys.prefix) / "conda-meta"
    if conda.is_dir():
        lines.append("# conda packages (conda create --file can read these lines)")
        packages = []
        for path in conda.glob("*.json"):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if {"name", "version", "build"} <= set(meta):
                packages.append(f"{meta['name']}={meta['version']}={meta['build']}")
        lines += sorted(packages, key=str.lower)
    lines.append("# Python packages")
    python = {
        f"{d.metadata['Name']}=={d.version}"
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    lines += sorted(python, key=str.lower)
    return "\n".join(lines) + "\n"


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _lion(record) -> dict:
    lion = (record or {}).get("lion") or {}
    return {"version": lion.get("version"), "commit": lion.get("commit")}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=1), encoding="utf-8")


def _sentence(problems) -> str:
    return "can't publish:\n- " + "\n- ".join(problems)


# ---------------------------------------------------------------------------
# The model card


def model_card(saved, package, model, test=None) -> str:
    """The README of a package: what it is, from its records."""
    training, data = saved["training"], saved["dataset_params"]
    files = package["files"]
    model_class = _name(saved["model"])
    experiment = _built(data.get("experiment"))
    dataset = _dataset_name(data, experiment)
    title = f"{model_class} trained on {dataset}"
    if experiment is not None:
        title += f": {experiment.param.name}"

    lines = [
        f"# {title}",
        "",
        f"Published on {package['published']} with LION "
        f"{package['published_with']['version']}"
        + _commit(package["published_with"])
        + ". Generated from the package's records (`package.json` lists them); "
        "edit it freely, it isn't checksummed.",
        "",
        "## The model",
        "",
        f"- `{saved['model'][spec.TARGET]}`, "
        f"{sum(p.numel() for p in model.parameters()):,} parameters.",
        f"- The weights: {_weights(training)}.",
    ]
    normalisation = _normalisation(saved["model"])
    if normalisation:
        lines.append(f"- Normalisation: {normalisation}.")

    lines += ["", "## The problem", ""]
    if experiment is not None:
        lines.append(
            f"- Experiment: `{data['experiment'][spec.TARGET]}`, "
            f'"{experiment.param.name}".'
        )
    lines.append(f"- Dataset: {dataset} (`{_target(data.get('dataset'))}`).")
    geometry = _geometry(experiment)
    if geometry:
        lines.append(f"- Geometry: {geometry}.")
    noise = _noise(data)
    if noise:
        lines.append(f"- Noise: {noise}.")

    lines += ["", "## Training", ""]
    lines += _training(training, data)
    trained_in = saved.get("environment") or {}
    lines.append(f"- Trained with {_environment(trained_in)}.")

    if test is not None:
        from LION.evaluation.evaluation import table

        lines += [
            "",
            "## Test",
            "",
            f"On the {test['data'].get('split', 'test')} data "
            f"({_samples(test['data'])}), "
            + (
                f"metrics over the range {test['data_range']}"
                if test.get("data_range")
                else "metrics over each sample's own range"
            )
            + " (`test.json`):",
            "",
            table([LIONParameter(**test)]).strip(),
        ]

    if package["notes"]:
        lines += ["", "## Notes", ""] + [
            f"- {note[0].upper()}{note[1:]}." for note in package["notes"]
        ]

    lines += [
        "",
        "## Using it",
        "",
        "```python",
        "from LION.models.LIONmodel import LIONmodel",
        "",
        'model, _, _ = LIONmodel.load("model")',
        "```",
    ]
    lines += ["", "## Training it again", ""]
    lines += _retraining(saved, files)

    if package["citations"]:
        lines += [
            "",
            "## Citations",
            "",
            "Where the model and its components come from:",
            "",
            "```bibtex",
            "\n\n".join(c["bibtex"].strip() for c in package["citations"]),
            "```",
        ]
    return "\n".join(lines) + "\n"


def _retraining(saved, files) -> list:
    """The code that trains the model again from the package (run in it)."""
    training = saved["training"]
    validation = training.get("validation")
    device = ((saved.get("environment") or {}).get("device") or {}).get("type", "cpu")
    lines = [
        "In the package's folder, from its files alone:",
        "",
        "```python",
        "from LION.package import train_again",
        "",
        'model = train_again(".", "retrained")',
        "```",
        "",
        "To change something first (the model, the experiment, the solver, "
        "the optimizer, the training):",
        "",
        "```python",
        "from LION.package import load, train_again",
        "",
        'package = load(".")',
        "package.training.epochs = 2 * package.training.epochs  # for example",
        'model = train_again(package, "retrained")',
        "```",
        "",
        "### Step by step",
        "",
        "```python",
        "import pathlib",
        "",
        "import torch",
        "from torch.utils.data import DataLoader, Subset",
        "",
        "from LION.data_loaders.data_record import read_record",
        "from LION.experiments.ct_experiments import Experiment",
        "from LION.models.LIONmodel import LIONmodel",
        "from LION.optimizers.LIONsolver import LIONsolver",
        "from LION.optimizers.training_record import read_training_record",
    ]
    if validation is not None and validation.get("reconstructor") is not None:
        lines.append("from LION.reconstructors import LIONReconstructor")
    if validation is not None:
        lines.append("from LION.utils import spec")
    lines += [
        "",
        'record = read_training_record("solver.json")',
        "",
        "",
        "def data(file):",
        '    """The data a record describes, made by its experiment."""',
        "    record = read_record(file)",
        "    made = Experiment.instantiate(file).get_dataset(record.split)",
        '    subset = getattr(record, "subset", None)',
        "    return made if subset is None else Subset(made, subset.indices)",
        "",
        "",
        'model = LIONmodel.instantiate("model", initial_weights=True)',
        "solver = LIONsolver.instantiate(",
        f'    "solver.json", model, device=torch.device("{device}")  # where it was trained',
        ")",
        "t = record.train",
        "solver.set_training(",
        "    DataLoader(",
        f'        data("{files["training_data"]}"),',
        "        batch_size=t.batch_size,",
        "        shuffle=t.shuffle,",
        "        drop_last=t.drop_last,",
        "    )",
        ")",
    ]
    if validation is not None:
        lines += [
            "v = record.validation",
            "solver.set_validation(",
            f'    DataLoader(data("{files["validation_data"]}"), batch_size=v.batch_size),',
            "    v.every,",
            "    spec.build(v.function),",
        ]
        if validation.get("reconstructor") is not None:
            lines.append(
                "    reconstructor=LIONReconstructor.instantiate(v.reconstructor),"
            )
        lines.append(")")
    lines += [
        'pathlib.Path("retrained").mkdir(exist_ok=True)',
        'solver.set_saving("retrained", "model")',
        "solver.train(record.epochs.done)",
        "solver.save_final_results()",
        "```",
    ]
    return lines


def _training(training, data) -> list:
    solver = training["solver"]
    lines = [
        f"- Solver: `{solver[spec.TARGET]}`"
        + (
            f", loss `{_name(solver['loss_fn'])}`"
            if isinstance(solver.get("loss_fn"), dict)
            else ""
        )
        + f"; optimizer `{_name(training['optimizer'])}`"
        + (
            f" (learning rate {training['optimizer']['lr']})"
            if isinstance(training.get("optimizer"), dict)
            and "lr" in training["optimizer"]
            else ""
        )
        + ".",
    ]
    feed = training.get("train") or {}
    lines.append(
        f"- Data: the {data.get('split', '?')} data, {_samples(data)}; "
        f"batch size {feed.get('batch_size')}"
        + (", shuffled" if feed.get("shuffle") else "")
        + "."
    )
    validation = training.get("validation")
    if validation is not None:
        lines.append(
            f"- Validation: {_samples(validation.get('data') or {})}, every "
            f"{_count(validation.get('every'), 'epoch')}, with "
            f"`{_name(validation.get('function'))}`"
            + (
                f", reconstructed by `{_name(validation['reconstructor'])}`"
                if validation.get("reconstructor")
                else ""
            )
            + "."
        )
    epochs = training.get("epochs") or {}
    history = training.get("history") or {}
    last = []
    if history.get("train_loss"):
        last.append(f"training loss {history['train_loss'][-1]:.4g}")
    if history.get("validation") and history["validation"][-1] is not None:
        last.append(f"validation {history['validation'][-1]:.4g}")
    lines.append(
        f"- {_count(epochs.get('done'), 'epoch')} of {epochs.get('asked')}"
        + (f"; at the end, {' and '.join(last)}" if last else "")
        + "."
    )
    how = (training.get("initial_weights") or {}).get("from")
    lines.append(
        f"- Seed {training.get('seed')}; initial weights "
        + {
            "random_state": "made again from the random state saved with the model",
            "file": f"saved in `{(training.get('initial_weights') or {}).get('file')}`",
        }[how]
        + "."
    )
    return lines


def _samples(record) -> str:
    samples = record.get("samples") or {}
    text = _count(samples.get("count"), "sample")
    if samples.get("groups"):
        text += f" from {_count(len(samples['groups']), 'group')}"
    return text


def _count(n, word) -> str:
    """``"1 epoch"``, ``"2 epochs"``; ``"? epochs"`` if unknown."""
    if n is None:
        return f"? {word}s"
    return f"{n} {word}" + ("" if n == 1 else "s")


def _weights(training) -> str:
    weights = training.get("weights") or {}
    kind = weights.get("kind")
    if kind == "best_validation":
        return (
            f"the best validation ({weights.get('value'):.4g}), after epoch "
            f"{weights.get('epoch')}"
        )
    return f"{kind or 'saved'}, after epoch {weights.get('epoch')}"


def _environment(record) -> str:
    packages = record.get("packages") or {}
    torch_record = record.get("torch") or {}
    device = record.get("device") or {}
    text = f"LION {_lion(record)['version']}{_commit(_lion(record))}"
    text += f", Python {record.get('python')}, torch {packages.get('torch')}"
    if device.get("type") == "cuda":
        text += (
            f", on {', '.join(device.get('gpus') or ['a GPU'])} (CUDA "
            f"{torch_record.get('cuda')}); deterministic: "
            + (
                "yes"
                if torch_record.get("deterministic_algorithms")
                or torch_record.get("cudnn_deterministic")
                else "no, so runs on a GPU differ slightly"
            )
        )
    else:
        text += f", on the {device.get('type', '?')}"
    return text


def _commit(lion) -> str:
    return f" (commit {lion['commit'][:10]})" if lion.get("commit") else ""


def _name(written) -> str:
    target = _target(written)
    return target.rsplit(".", 1)[-1] if target else "?"


def _target(written):
    return written.get(spec.TARGET) if isinstance(written, dict) else None


def _built(written):
    """The object a spec describes, or None if it can't be made here."""
    if not isinstance(written, dict) or spec.TARGET not in written:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return spec.build(written)
    except Exception:  # only for the card: it says less, rather than fail
        return None


def _dataset_name(data, experiment) -> str:
    if experiment is not None and getattr(experiment, "dataset", None):
        return str(experiment.dataset)
    return _name(data.get("dataset"))


def _geometry(experiment) -> str:
    geometry = getattr(experiment, "geometry", None)
    if geometry is None:
        return ""
    parts = [f"{getattr(geometry, 'mode', '?')} beam"]
    angles = getattr(geometry, "angles", None)
    if angles is not None:
        parts.append(f"{len(angles)} angles")
    for name, label in (("detector_shape", "detector"), ("image_shape", "image")):
        value = getattr(geometry, name, None)
        if value is not None:
            parts.append(f"{label} {'x'.join(str(int(v)) for v in value)}")
    return ", ".join(parts)


def _noise(data) -> str:
    noise = data.get("sinogram_transform")
    if not isinstance(noise, dict):
        return ""
    settings = {
        k: v
        for k, v in noise.items()
        if k not in spec._KEYS and not isinstance(v, dict)
    }
    text = ", ".join(f"{k} {v}" for k, v in settings.items())
    seed = data.get("sinogram_transform_seed")
    return text + (f" (seed {seed})" if seed is not None else "")


def _normalisation(model_spec) -> str:
    parameters = model_spec.get("model_parameters") or {}
    value = parameters.get("normalisation")
    if not isinstance(value, dict):
        return ""
    return ", ".join(
        f"{k} {v}" for k, v in value.items() if k not in spec._KEYS and v is not None
    )
