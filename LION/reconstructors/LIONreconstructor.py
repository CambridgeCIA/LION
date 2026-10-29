# =============================================================================
# This file is part of LION library
# License : GPL-3
# =============================================================================
"""The base class of reconstructors: methods that reconstruct with a trained
model (e.g. a denoiser as a prior, a learned regulariser) inside an algorithm.

A reconstructor is made with everything that decides its result, then run:

    pnp = PnP(operator, model, parameters)   # parameters: PnP.default_parameters()
    images = pnp.reconstruct(measurements)

- ``operator``: a CT ``Geometry`` (the CT operator is made from it) or any LION
  ``Operator``; kept as given, and as ``self.op`` to compute with;
- ``model``: the trained network (or any function of an image). Optional: a
  reconstructor can be made without one (``instantiate``) and given it later
  (``reconstructor.model = model``); running it without one is an error;
- ``parameters``: a ``LIONParameter`` with the algorithm's settings, from the
  class's ``default_parameters()``. ``reconstruct`` takes only the data, a
  starting image ``x0`` and ``prog_bar``, so a reconstructor with its
  parameters is one method, recorded as such (``LION.evaluation``).

Its spec (``LION.utils.spec``) holds the operator and the parameters, not the
model: a model's spec would make it with new weights. Records of an evaluation
keep the model's identity (spec and weights hash) next to it, and
``instantiate(source, model)`` checks the model given against it.

A new reconstructor derives from this class, defines ``default_parameters()``
and ``reconstruct_sample(y, x0, prog_bar)``, and reads its settings from
``self.parameters``.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import torch

from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from LION.operators import Operator
from LION.utils import spec
from LION.utils.parameter import LIONParameter


class LIONReconstructor(ABC):
    """Base class of reconstructors (see the module docstring)."""

    def __init__(
        self,
        operator: Geometry | Operator,
        model=None,
        parameters: LIONParameter | None = None,
    ):
        if isinstance(operator, Operator):
            self.op = operator
        elif isinstance(operator, Geometry):
            self.op = make_operator(operator)
        else:
            raise ValueError(
                "operator must be a LION Operator or a CT Geometry, not "
                f"{type(operator).__name__}"
            )
        self.operator = operator
        self.model = model
        self.parameters = (
            self.default_parameters() if parameters is None else parameters
        )

    @staticmethod
    def default_parameters() -> LIONParameter:
        """The algorithm's settings (none here; reconstructors define theirs)."""
        return LIONParameter()

    # -- running -------------------------------------------------------------

    def reconstruct(
        self, y: torch.Tensor, x0: torch.Tensor | None = None, prog_bar: bool = False
    ) -> torch.Tensor:
        """Reconstruct measurements ``y``: one sample (the operator's range
        shape) or a batch of them, each on its own. ``x0``: the starting
        image(s), where the algorithm takes one."""
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
            for i, sample in enumerate(y):
                start = None if x0 is None else x0[i]
                images[i] = self.reconstruct_sample(sample, start, prog_bar)
            return images
        return self.reconstruct_sample(y, x0, prog_bar)

    @abstractmethod
    def reconstruct_sample(
        self, y: torch.Tensor, x0: torch.Tensor | None, prog_bar: bool
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

    @classmethod
    def instantiate(cls, source, model=None, *, allow=(), **arguments):
        """The reconstructor ``source`` describes: its spec, or a test record
        (``LION.evaluation``) of it, or that record's file. ``model``: the
        trained model to use (optional: set it later with ``.model = ...``);
        with a test record, a warning if its weights aren't the recorded ones.
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
    """The reconstructor's spec in ``source`` (a spec, a test record or its
    file), and the model the test record names (or None)."""
    saved = spec.read(source)
    recorded_model = None
    if "lion_test_record" in saved:
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
