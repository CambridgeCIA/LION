# =============================================================================
# This file is part of LION library
# License : GPL-3
# =============================================================================
"""The base class of reconstructors: methods that reconstruct with a trained
model (e.g. a denoiser as a prior, a learned regulariser) inside an algorithm.

A reconstructor is made with everything that decides its result, then run:

    pnp = PnP(operator, model, parameters)   # parameters: PnP.default_parameters()
    images = pnp.reconstruct(measurements)

- ``operator``: operator parameters (e.g. a CT ``Geometry``; the operator is made from them) or any LION
  ``Operator``; kept as given, and as ``self.op`` to compute with;
- ``model``: the trained network (or any function of an image). Optional: a
  reconstructor can be made without one (``instantiate``) and given it later
  (``reconstructor.model = model``); running it without one is an error;
- ``parameters``: a ``LIONParameter`` with the algorithm's settings, from the
  class's ``default_parameters()``. ``reconstruct`` takes only the data, a
  starting image ``x0`` and ``verbose`` (how much progress is shown), so a
  reconstructor with its parameters is one method, recorded as such
  (``LION.evaluation``).

Its spec (``LION.utils.spec``) holds the operator and the parameters, not the
model: a model's spec would make it with new weights. Records of an evaluation
keep the model's identity (spec and weights hash) next to it, and
``instantiate(source, model)`` checks the model given against it.

``reconstructor.save(file)`` writes its own record, ``file.json``: its spec,
the file its model was loaded from (``LIONmodel.load``) and the model's
weights hash, and the environment. The weights stay in the model's file.
``LIONReconstructor.load(file)`` makes it again with that model;
``instantiate(file)`` makes it without one (give it another model).

A new reconstructor derives from this class, defines ``default_parameters()``
and ``reconstruct_sample(y, x0, verbose)``, and reads its settings from
``self.parameters``. ``reconstruct_sample`` shows a progress bar over its
iterations when ``verbose >= 1``.
"""

from __future__ import annotations

import json
import pathlib
import warnings
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from LION.operators import Operator
from LION.utils import spec
from LION.utils.parameter import (
    DEFAULT_RECONSTRUCTION,
    LIONParameter,
    OperatorParameters,
    resolve_reconstruction,
)

RECORD_KEY = "lion_reconstructor_record"
RECORD_VERSION = 1


class LIONReconstructor(ABC):
    """Base class of reconstructors (see the module docstring)."""

    def __init__(
        self,
        operator: OperatorParameters | Operator,
        model=None,
        parameters: LIONParameter | None = None,
    ):
        if isinstance(operator, Operator):
            self.op = operator
        elif isinstance(operator, OperatorParameters):
            self.op = operator.make_operator()
        else:
            raise ValueError(
                "operator must be a LION Operator or operator parameters "
                "(OperatorParameters, e.g. a CT Geometry), not "
                f"{type(operator).__name__}"
            )
        self.operator = operator
        self.model = model
        self.parameters = (
            self.default_parameters() if parameters is None else parameters
        )
        self._resolve_initial_reconstruction()

    @staticmethod
    def default_parameters() -> LIONParameter:
        """The algorithm's settings (none here; reconstructors define theirs)."""
        return LIONParameter()

    def _resolve_initial_reconstruction(self):
        """Replace a ``"default"`` initial reconstruction (for reconstructors
        whose parameters list one) by the operator's default reconstruction
        function, so the record says which was used. An operator without one
        leaves it "default": ``initial_guess`` then says what to do."""
        value = getattr(self.parameters, "initial_reconstruction", None)
        if value != DEFAULT_RECONSTRUCTION:
            return
        try:
            self.parameters.initial_reconstruction = resolve_reconstruction(
                value, self.op, "initial_reconstruction"
            )
        except NotImplementedError:
            pass

    def initial_guess(self, y: torch.Tensor) -> torch.Tensor:
        """Where the algorithm starts for measurements ``y``: the parameters'
        ``initial_reconstruction(y, op)``. Its default, ``"default"``, is the
        operator's default reconstruction (FDK for CT)."""
        self._resolve_initial_reconstruction()
        value = getattr(self.parameters, "initial_reconstruction", None)
        if value is None or value == DEFAULT_RECONSTRUCTION:
            raise ValueError(
                f"{type(self.op).__name__} has no default reconstruction: set the "
                "parameters' initial_reconstruction, a function f(y, op), or give "
                "x0 (the starting image)"
            )
        return value(y, self.op)

    # -- running -------------------------------------------------------------

    def reconstruct(
        self, y: torch.Tensor, x0: torch.Tensor | None = None, verbose: int = 1
    ) -> torch.Tensor:
        """Reconstruct measurements ``y``: one sample (the operator's range
        shape) or a batch of them, each on its own. ``x0``: the starting
        image(s), where the algorithm takes one.

        ``verbose``: 0, nothing shown; 1 (default), one progress bar: over the
        samples of a batch, or over the iterations of one sample; 2, also the
        iterations of each sample of a batch."""
        if self.model is None:
            raise ValueError(
                f"{type(self).__name__} has no model: set it first "
                f"(reconstructor.model = model, or instantiate(source, model))"
            )
        # a batch has one more dimension than the operator's measurements
        # (B, C, angles, detectors for CT; B, M for a matrix operator)
        if y.dim() == len(self.op.range_shape) + 1:
            images = torch.zeros(
                y.size(0), *self.op.domain_shape, dtype=y.dtype, device=y.device
            )
            samples = tqdm(
                y, desc=type(self).__name__, unit="sample", disable=verbose < 1
            )
            for i, sample in enumerate(samples):
                start = None if x0 is None else x0[i]
                images[i] = self.reconstruct_sample(sample, start, verbose - 1)
            return images
        return self.reconstruct_sample(y, x0, verbose)

    @abstractmethod
    def reconstruct_sample(
        self, y: torch.Tensor, x0: torch.Tensor | None, verbose: int
    ) -> torch.Tensor:
        """Reconstruct one measurement, with the settings in
        ``self.parameters``."""

    # -- records ---------------------------------------------------------------

    def spec_kwargs(self) -> dict:
        """The spec's arguments: the operator and the parameters, not the model
        (a model's spec would make it with new weights; see the module
        docstring)."""
        kwargs = spec._constructor_arguments(self)
        kwargs.pop("model", None)
        return kwargs

    def save(self, fname):
        """Write this reconstructor's record to ``fname`` (``.json``): its
        spec (operator and parameters), its model by the file it was loaded
        from and its weights hash (a model made in the script: only the hash;
        a function: its spec), and the environment. Made again by ``load``
        (with its model) or ``instantiate`` (without)."""
        from LION.utils import environment

        model = self.model
        device = None
        if isinstance(model, torch.nn.Module):
            first = next(model.parameters(), None)
            device = None if first is None else first.device
        written = {
            RECORD_KEY: RECORD_VERSION,
            "reconstructor": spec.to_spec(self),
            "model": _model_reference(model),
            "environment": environment.environment_record(device).serialize(),
        }
        path = pathlib.Path(fname).with_suffix(".json")
        path.write_text(json.dumps(written, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, fname, *, allow=()):
        """The reconstructor ``fname`` (a file ``save`` wrote) describes, with
        its model loaded from the recorded file (a warning if its weights
        changed since). A model that wasn't loaded from a file can't be:
        then it has no model (a warning; give it one). A model file given by
        a relative path is relative to ``fname``'s folder. ``allow``: as in
        ``LION.utils.spec.build``. The reconstructor keeps its file as
        ``source_file``."""
        fname = pathlib.Path(fname)
        saved = spec.read(fname)
        if RECORD_KEY not in saved:
            raise ValueError(f"{fname} isn't a reconstructor's file (save())")
        model = _load_model(saved.get("model"), fname, allow)
        reconstructor = cls.instantiate(saved, model, allow=allow)
        reconstructor.source_file = fname.with_suffix("")
        return reconstructor

    @classmethod
    def instantiate(cls, source, model=None, *, allow=(), **arguments):
        """The reconstructor ``source`` describes: its file (``save``), its
        spec, or a test record (``LION.evaluation``) of it, or that record's
        file. ``model``: the trained model to use (optional: set it later with
        ``.model = ...``); with a file or a test record, a warning if its
        weights aren't the recorded ones.
        ``arguments`` replace recorded ones. Called on ``LIONReconstructor``
        it makes the class the spec names; on a reconstructor class, that
        class."""
        saved, recorded_model = _recorded(source)
        written = {
            k: v for k, v in saved.items() if k == spec.TARGET or k not in arguments
        }
        written.pop("model", None)
        path, kwargs = spec.build_arguments(written, allow)
        target = spec.target_class(path, cls, LIONReconstructor, allow)
        if model is not None and recorded_model is not None:
            _check_model(model, recorded_model)
        return target(**kwargs, **arguments, model=model)

    @classmethod
    def parameters_from_file(cls, source, *, allow=()):
        """What ``default_parameters()`` gives, as ``source`` (as in
        ``instantiate``) recorded it: the algorithm's settings, to change and
        make the reconstructor with (``PnP(operator, model, parameters)``)."""
        saved, _ = _recorded(source)
        return spec.parameters(saved, "parameters", cls, LIONReconstructor, allow)

    @staticmethod
    def cite(cite_format="bib"):
        """Where the reconstructor comes from: its paper's citation, BibTeX ("bib") or
        "MLA" (``LION.utils.citation``); None if it has none. Reconstructors
        from a paper define it."""
        return None


def _recorded(source):
    """The reconstructor's spec in ``source`` (its file, a spec, a test
    record or its file), and the model the record names (or None)."""
    saved = spec.read(source)
    recorded_model = None
    if RECORD_KEY in saved:
        saved, recorded_model = saved["reconstructor"], saved.get("model")
    elif "lion_test_record" in saved:
        method = saved["method"]
        if method.get("kind") != "reconstructor":
            raise ValueError(f"the test record is of a {method.get('kind')}")
        saved, recorded_model = method["reconstructor"], method.get("model")
    if spec.is_unsaved(saved):
        raise ValueError(
            f"the record only names the reconstructor ({saved['unsaved']}): "
            "it can't be made again"
        )
    return saved, recorded_model


def _model_reference(model):
    """A reconstructor's model in its file: a network by the file it was
    loaded from (if it was) and its weights hash; a function by its spec."""
    if model is None:
        return None
    if isinstance(model, torch.nn.Module):
        from LION.optimizers.training_record import weights_sha256

        reference = {"weights_sha256": weights_sha256(model)}
        if getattr(model, "source_file", None) is not None:
            reference["file"] = spec.encode(pathlib.Path(model.source_file))
        return reference
    return {"function": spec.encode_or_name(model, "the model")}


def _load_model(reference, fname, allow):
    """The model a reconstructor's file refers to (see ``_model_reference``)."""
    if reference is None:
        return None
    if "function" in reference:
        if spec.is_unsaved(reference["function"]):
            warnings.warn(
                f"{fname}: its model, {reference['function']['unsaved']}, could "
                "only be named: give it (reconstructor.model = ...)",
                stacklevel=3,
            )
            return None
        return spec.build(reference["function"], allow)
    if "file" not in reference:
        warnings.warn(
            f"{fname}: its model wasn't loaded from a file, so it can't be "
            "loaded: give it (reconstructor.model = ...)",
            stacklevel=3,
        )
        return None
    from LION.models.LIONmodel import LIONmodel

    path = spec.build(reference["file"], allow)
    if not path.is_absolute():
        path = fname.parent / path
    return LIONmodel.load(path, allow=allow, supress_warnings=True)[0]


def _check_model(model, recorded):
    """Warn if ``model`` isn't the one a record names (by its weights)."""
    from LION.optimizers.training_record import weights_sha256

    expected = recorded.get("weights_sha256") if isinstance(recorded, dict) else None
    if expected is None or not isinstance(model, torch.nn.Module):
        return
    if weights_sha256(model) != expected:
        warnings.warn(
            "the model given has other weights than the recorded one: the "
            "reconstructions won't be the recorded ones",
            stacklevel=3,
        )
