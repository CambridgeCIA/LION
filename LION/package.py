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
``Experiment``, ``LIONDataset``, ``LIONsolver``, ``Evaluation``); the model
card shows how to train the model again from them.

``publish`` refuses, with an error and before writing anything:

- LION code that isn't committed and pushed (``environment.publish_problems``):
  the code running now, the code that saved the model and the code that tested it;
- a model file without a training record or a data record (one not saved by a
  LION solver from a LION dataset);
- records naming code outside LION (other than torch's layers, losses and
  optimizers): a package is made again with LION alone, without ``allow=``;
- records that could only name something (e.g. a lambda loss), which can't be
  made again;
- a test record of other weights than the model's, or test data the model was
  trained or validated on;
- a folder that isn't empty.

What can be made again but not exactly (e.g. initial weights that weren't
saved) is a note, in ``package.json`` and the model card, and a warning.
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
    notes = _notes(saved, test, initial)
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
    if "experiment" in data and not spec.is_unsaved(data["experiment"]):
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


def _notes(saved, test, initial) -> list:
    """What can be made again, but not exactly, or isn't the usual case."""
    from LION.evaluation.comparison import problem_differences

    training, data = saved["training"], saved["dataset_params"]
    notes = []
    how = (training.get("initial_weights") or {}).get("from")
    if how is None:
        notes.append(
            "the weights training started from can't be made again, so training "
            "can be repeated, but not exactly"
        )
    if training.get("seed") is None:
        notes.append("training wasn't seeded, so it can't be repeated exactly")
    epochs = training.get("epochs") or {}
    if epochs.get("done") is not None and epochs.get("done") != epochs.get("asked"):
        notes.append(
            f"training stopped after {epochs.get('done')} of {epochs.get('asked')} epochs"
        )
    if "experiment" not in data:
        notes.append("the data weren't made by an Experiment")
    paths = []
    _absolute_paths(data, paths)
    if paths:
        notes.append(
            "the data are read from a folder outside LION_DATA_PATH "
            f"({', '.join(sorted(set(paths)))}): elsewhere, give the folder they are in"
        )
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
        differences = problem_differences(test["data"], data)
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
        "In the package's folder, from its files alone"
        + (
            ""
            if (training.get("initial_weights") or {}).get("from")
            else " (but for the initial weights: see the notes)"
        )
        + ":",
        "",
        "```python",
        "import pathlib",
        "",
        "import torch",
        "from torch.utils.data import DataLoader, Subset",
        "",
        "from LION.data_loaders.data_record import read_record",
        "from LION.experiments.ct_experiments import Experiment"
        if "experiment" in files
        else "from LION.data_loaders.LIONDataset import LIONDataset",
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
        "    record = read_record(file)",
    ]
    if "experiment" in files:
        lines += [
            '    made = Experiment.instantiate("experiment.json").get_dataset(record.split)',
            '    subset = getattr(record, "subset", None)',
            "    return made if subset is None else Subset(made, subset.indices)",
        ]
    else:
        lines.append("    return LIONDataset.instantiate(file)")
    initial = bool((training.get("initial_weights") or {}).get("from"))
    lines += [
        "",
        "",
        f'model = LIONmodel.instantiate("model", initial_weights={initial})',
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
        "solver.train(record.epochs.asked)",
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
        }.get(how, "not saved")
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
