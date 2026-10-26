# =============================================================================
# This file is part of LION library
# License : GPL-3
# =============================================================================
"""Specs: a LION component written as the call that builds it, in JSON.

LION writes them, as metadata, when it saves a component; they are not meant to
be written by hand. A spec is plain JSON in which a dict with ``"_target_"``
means "call this":

    {"_target_": "LION.models.CNNs.UNets.Unet.UNet",
     "model_parameters": {"_target_": "LION.models.LIONmodel.LIONModelParameter",
                          "inChan": 1, ...}}

``_target_`` is the import path of a class, ``_args_`` its positional arguments
and every other key a keyword argument. ``"_call_": false`` refers to a class or
function without calling it. A dict without ``_target_`` (in older files) is a
``LIONParameter``. The keys follow Hydra's ``instantiate`` convention.

- ``to_spec(obj)`` writes the spec of an object. An object's arguments are its
  attributes with the names of its constructor's arguments (LION's components
  keep each argument under its own name; torch losses and metrics do too). A
  class can define ``spec_kwargs()`` to give them itself. A torch optimizer is
  written with its hyperparameters; its parameters are given when building.
- ``build(spec)`` makes the object again. A spec names code to run, so only
  targets in allowed modules are imported, and only classes are called (plus
  ``numpy.array``); functions are only referred to. Add your own modules with
  ``allow=``.
"""

import enum
import importlib
import inspect
import pathlib
import warnings

import numpy as np
import torch

from LION.utils.parameter import LIONParameter

TARGET = "_target_"
ARGS = "_args_"
CALL = "_call_"
_KEYS = (TARGET, ARGS, CALL)

FORMAT_VERSION = 1

#: Module prefixes whose classes ``build`` may import and call.
ALLOWED_MODULES = ("LION.", "torch.nn.", "torch.optim.")
#: Exact targets allowed outside those modules.
ALLOWED_TARGETS = (
    "builtins.tuple",
    "builtins.list",
    "builtins.dict",
    "builtins.set",
    "builtins.frozenset",
    "builtins.slice",
    "builtins.complex",
    "numpy.array",
    "pathlib.Path",
)
# Constructor arguments that describe where an object runs, not what it is.
RUNTIME_ARGUMENTS = ("device", "verbose")


class SpecError(ValueError):
    """An object that can't be written as a spec, or a spec that can't be built."""


def import_path(x) -> str:
    """The import path of a class or function."""
    return f"{x.__module__}.{x.__qualname__}"


# ---------------------------------------------------------------------------
# Writing


def encode(value):
    """A value as JSON: plain JSON types as they are, everything else as the
    spec of the call that makes it."""
    if isinstance(value, enum.Enum):  # before int: some enums are ints
        return {TARGET: import_path(type(value)), ARGS: [encode(value.value)]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.ndim == 0:  # np.array(None) etc.
                return encode(value.item())
            raise SpecError(f"a numpy array of objects can't be saved: {value!r}")
        return {
            TARGET: "numpy.array",
            ARGS: [value.tolist()],
            "dtype": str(value.dtype),
        }
    if isinstance(value, torch.Tensor):
        raise SpecError(
            "a tensor can't be saved as a parameter (tensors belong in the weights); "
            f"got shape {tuple(value.shape)}"
        )
    if isinstance(value, pathlib.PurePath):
        return {TARGET: "pathlib.Path", ARGS: [str(value)]}
    if isinstance(value, (tuple, set, frozenset)):
        return {
            TARGET: f"builtins.{type(value).__name__}",
            ARGS: [[encode(v) for v in value]],
        }
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) and k not in _KEYS for k in value):
            return {TARGET: "builtins.dict", **{k: encode(v) for k, v in value.items()}}
        pairs = [[encode(k), encode(v)] for k, v in value.items()]
        return {TARGET: "builtins.dict", ARGS: [pairs]}
    if isinstance(value, LIONParameter):
        values = value.spec_kwargs() if hasattr(value, "spec_kwargs") else vars(value)
        return {
            TARGET: import_path(type(value)),
            **{k: encode(v) for k, v in values.items()},
        }
    if inspect.isclass(value) or inspect.isroutine(value):
        if value.__name__ == "<lambda>" or "<locals>" in value.__qualname__:
            raise SpecError(
                f"{value!r} can't be saved: a lambda or a function defined inside "
                "another one has no import path"
            )
        return {TARGET: import_path(value), CALL: False}
    return to_spec(value)


def to_spec(obj) -> dict:
    """The spec of an object: its class and the arguments that build it."""
    if isinstance(obj, LIONParameter):
        return encode(obj)
    if hasattr(obj, "spec_kwargs"):
        kwargs = obj.spec_kwargs()
    elif isinstance(obj, torch.optim.Optimizer):
        kwargs = dict(obj.defaults)
    else:
        kwargs = _constructor_arguments(obj)
    return {TARGET: import_path(type(obj)), **{k: encode(v) for k, v in kwargs.items()}}


def _constructor_arguments(obj) -> dict:
    """The constructor's arguments, from the attributes with the same names.

    An argument with a default that the object doesn't keep is left out, so the
    default is used when building.
    """
    cls = type(obj)
    kwargs = {}
    parameters = list(inspect.signature(cls.__init__).parameters.values())[1:]
    for p in parameters:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or p.name in RUNTIME_ARGUMENTS:
            continue
        if hasattr(obj, p.name):
            kwargs[p.name] = getattr(obj, p.name)
        elif p.default is p.empty:
            raise SpecError(
                f"{cls.__name__} can't be saved: it doesn't keep its argument "
                f"'{p.name}' as an attribute of the same name (or define "
                f"{cls.__name__}.spec_kwargs())"
            )
    return kwargs


# ---------------------------------------------------------------------------
# Building


def _is_allowed(path: str, allow) -> bool:
    return path in ALLOWED_TARGETS or path.startswith(
        tuple(ALLOWED_MODULES) + tuple(allow)
    )


def resolve(path: str, allow=()):
    """The class or function at an import path, if it is allowed."""
    if not isinstance(path, str) or not _is_allowed(path, allow):
        raise SpecError(
            f"'{path}' is not in the modules a spec may use ({', '.join(ALLOWED_MODULES)} "
            "and a few builtins). To build your own classes, pass their module, e.g. "
            "allow=['my_package.']"
        )
    parts = path.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        try:
            for part in parts[i:]:
                obj = getattr(obj, part)
        except AttributeError:
            break
        return obj
    raise SpecError(f"'{path}' can't be imported")


def build(spec, allow=(), **kwargs):
    """Make the object a spec describes.

    ``allow``: more module prefixes (e.g. ``["my_package."]``) whose classes may
    be built. ``kwargs`` are added to the top-level call, for arguments that
    are live objects, e.g. ``build(optimizer_spec, params=model.parameters())``.
    """
    if isinstance(spec, list):
        return [build(v, allow) for v in spec]
    if not isinstance(spec, dict):
        return spec
    if TARGET not in spec:
        return _parameter(LIONParameter, {k: build(v, allow) for k, v in spec.items()})

    path = spec[TARGET]
    target = resolve(path, allow)
    if spec.get(CALL, True) is False:
        return target
    if not inspect.isclass(target) and path not in ALLOWED_TARGETS:
        raise SpecError(
            f"'{path}' is a function: a spec only calls classes (functions can be "
            'referred to, with "_call_": false)'
        )
    if target is dict and ARGS in spec:
        return {build(k, allow): build(v, allow) for k, v in spec[ARGS][0]}
    args = [build(a, allow) for a in spec.get(ARGS, [])]
    named = {k: build(v, allow) for k, v in spec.items() if k not in _KEYS}
    named.update(kwargs)
    if inspect.isclass(target) and issubclass(target, LIONParameter) and not args:
        return _parameter(target, named)
    return target(*args, **named)


def _parameter(cls, values: dict):
    """A LIONParameter (or subclass) with these attributes. Set one by one:
    ``LIONParameter(**values)`` would turn dict values into LIONParameters."""
    try:
        parameter = cls()
    except TypeError:  # a subclass that needs arguments
        return cls(**values)
    for name, value in values.items():
        setattr(parameter, name, value)
    return parameter


def build_arguments(spec: dict, allow=()):
    """A spec's target path and its built keyword arguments, without calling it
    (to call another class, e.g. the one ``Model.load()`` is called on)."""
    if not isinstance(spec, dict) or TARGET not in spec:
        raise SpecError("not a spec: no '_target_'")
    if ARGS in spec:
        raise SpecError("a spec with positional arguments can't be split into keywords")
    named = {k: build(v, allow) for k, v in spec.items() if k not in _KEYS}
    return spec[TARGET], named


def unknown_names(
    parameters: LIONParameter, defaults: LIONParameter, ignore=()
) -> list:
    """Names in ``parameters`` that ``defaults`` doesn't have (likely typos)."""
    known = set(vars(defaults)) | set(ignore)
    return sorted(set(vars(parameters)) - known)


def warn_unknown_names(owner: str, parameters, defaults, ignore=()):
    """Warn about names in ``parameters`` that ``defaults`` doesn't have."""
    names = unknown_names(parameters, defaults, ignore)
    if names:
        warnings.warn(
            f"{owner}: unknown parameter(s) {', '.join(names)}, not in its "
            "default_parameters(); they are not used (a typo?)",
            stacklevel=3,
        )
