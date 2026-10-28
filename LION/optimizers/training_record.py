# =============================================================================
# This file is part of LION library
# License : GPL-3
# =============================================================================
"""The record of how a model was trained.

A solver writes it, as the ``training`` of every model file it saves (final,
best validation, checkpoints), next to the record of the training data
(``dataset_params``, ``LION.data_loaders.data_record``):

- ``solver``: the solver's spec, without the model (the file's ``model``), the
  optimizer (below) and where it ran (``device``, ``verbose``, ``save_folder``);
  its ``solver_params`` include values it computed (e.g. E2I's ``noise_std``);
- ``optimizer``: the optimizer's class and hyperparameters;
- ``seed``: the training seed (``None``: unseeded), and
  ``initial_weights_sha256``, a hash of the weights training started from;
- ``train``: how the training data were fed (``batch_size``, ``shuffle``,
  ``drop_last``); the data themselves are the file's ``dataset_params``;
- ``validation``: its data record, how they were fed, ``every`` (epochs),
  the validation ``function``, and the ``reconstructor`` it reconstructs with
  (its spec: operator and parameters), if one was given;
- ``epochs``: ``asked`` (``train(n)``) and ``done``;
- ``weights``: which weights the file holds: ``kind`` ("final",
  "best_validation", "checkpoint"), after which ``epoch`` (counted from 1), and
  for the best validation its ``value``;
- ``history``: the training loss and validation value of each epoch done
  (``null`` when not validated).

What can't be written as a spec (a lambda loss) is named, with a warning.
A record is a ``LIONParameter`` (``record.epochs.done``) whose specs
(``record.solver``, ``record.optimizer``) stay dicts; it is saved as plain JSON
and read back as a ``LIONParameter``, like the data record.
"""

import hashlib
import math

from torch.utils.data import RandomSampler, SequentialSampler

from LION.utils import spec
from LION.utils.parameter import LIONParameter

RECORD_KEY = "lion_training_record"
RECORD_VERSION = 1


def is_training_record(value) -> bool:
    """A training record, as a ``LIONParameter`` or as the dict read from JSON."""
    if isinstance(value, LIONParameter):
        return hasattr(value, RECORD_KEY)
    return isinstance(value, dict) and RECORD_KEY in value


def read_training_record(source) -> LIONParameter:
    """The training record in ``source``: a record, a model's metadata holding
    one (as its ``training``), or the path of a ``.json`` with either."""
    saved = spec.read(source)
    if is_training_record(saved):
        return LIONParameter(**saved)
    if is_training_record(saved.get("training")):
        return LIONParameter(**saved["training"])
    raise ValueError(
        "no training record found: give a record, or the file of a model a "
        "LION solver saved"
    )


def weights_sha256(model) -> str:
    """A hash of a model's weights (names and values, in order)."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def loader_settings(loader) -> dict:
    """How a ``DataLoader`` feeds its data: ``batch_size``, ``shuffle``,
    ``drop_last``. A sampler other than torch's sequential and random ones is
    named in ``sampler``."""
    settings = {"batch_size": loader.batch_size, "drop_last": loader.drop_last}
    if isinstance(loader.sampler, RandomSampler):
        settings["shuffle"] = True
    elif isinstance(loader.sampler, SequentialSampler):
        settings["shuffle"] = False
    else:
        settings["shuffle"] = None
        settings["sampler"] = repr(loader.sampler)
    return settings


def history(values, done: int) -> list:
    """The first ``done`` values as JSON numbers (``None`` for NaN)."""
    if values is None:
        return []
    return [None if math.isnan(float(v)) else float(v) for v in list(values)[:done]]
