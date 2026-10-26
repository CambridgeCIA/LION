# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


import json
import warnings

import numpy as np

from LION.utils.parameter import LIONParameter

# Each triple (shape, size, spacing) is fixed by any two of its values:
# size = shape * spacing.
IMAGE = ("image_shape", "image_size", "voxel_size")
DETECTOR = ("detector_shape", "detector_size", "pixel_size")
_TRIPLES = (IMAGE, DETECTOR)
_TRIPLE_NAMES = set(IMAGE + DETECTOR)
# Stored as numpy arrays, however they are given.
_ARRAY_NAMES = _TRIPLE_NAMES | {"image_pos", "dso", "dsd", "angles"}
MODES = ("fan", "parallel")
_RTOL = 1e-6


def _triple_values(geometry, triple):
    """The values of a triple that were set (not computed), as a dict."""
    d = vars(geometry)
    return {name: d[name] for name in triple if name in d}


def _computed(given: dict, triple, name):
    """``name`` computed from the two other values of its triple, or None."""
    shape_name, size_name, spacing_name = triple
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            if name == spacing_name and shape_name in given and size_name in given:
                return given[size_name] / given[shape_name]
            if name == size_name and shape_name in given and spacing_name in given:
                return given[shape_name] * given[spacing_name]
            if name == shape_name and size_name in given and spacing_name in given:
                shape = given[size_name] / given[spacing_name]
                # Whole pixels, or the float that shows it isn't (see check_geometry)
                if np.all(np.isfinite(shape)) and np.allclose(
                    shape, np.rint(shape), rtol=_RTOL, atol=0
                ):
                    return np.rint(shape).astype(int)
                return shape
    except (
        TypeError,
        ValueError,
    ):  # values that don't fit together (check_geometry says)
        return None
    return None


def _agree(a, b) -> bool:
    try:
        return bool(np.allclose(a, b, rtol=_RTOL, atol=1e-12))
    except (TypeError, ValueError):  # values that can't be compared
        return False


class Geometry(LIONParameter):
    """
    Class holding a CT geometry.

    The image is given by any two of ``image_shape`` (pixels), ``image_size``
    (mm) and ``voxel_size`` (mm), the detector by any two of
    ``detector_shape``, ``detector_size`` and ``pixel_size``: size = shape *
    spacing. The geometry keeps the values that were set, and computes the
    third one when it is read, so it is never out of date when the others
    change, e.g. ``geo = Geometry.default_parameters(); geo.image_shape = ...``.

    ``check_geometry()`` checks that a geometry is complete and consistent. It
    runs when an operator is built from the geometry and when the geometry is
    saved. ``__init__`` only checks what is given at once (three values of a
    triple that disagree), because a geometry can be filled in afterwards.
    """

    def __init__(self, **kwargs):
        # __init__ also runs again on an existing geometry (default_geo(), load()):
        # forget values set before.
        for name in _TRIPLE_NAMES:
            self.__dict__.pop(name, None)

        self.mode = kwargs.get("mode", None)

        for name in _TRIPLE_NAMES:
            if kwargs.get(name, None) is not None:
                setattr(self, name, kwargs[name])
        self.image_pos = kwargs.get("image_pos", [0, 0, 0])

        self.dso = kwargs.get("dso", None)
        self.dsd = kwargs.get("dsd", None)

        self.angles = kwargs.get("angles", None)

        for triple in _TRIPLES:
            given = _triple_values(self, triple)
            if len(given) == 3 and not _agree(
                given[triple[1]], given[triple[0]] * given[triple[2]]
            ):
                raise ValueError(_disagree_message(triple, given))

    def __setattr__(self, name, value):
        if name in _TRIPLE_NAMES and value is None:
            # Unset: computed from the two others from now on
            self.__dict__.pop(name, None)
            return
        if name in _ARRAY_NAMES:
            value = np.array(value)
        super().__setattr__(name, value)

    def __getattr__(self, name):
        # Only called for attributes that are not set: a triple's third value is
        # computed from the two others (None if they are not both set).
        if name in _TRIPLE_NAMES:
            for triple in _TRIPLES:
                if name in triple:
                    return _computed(_triple_values(self, triple), triple, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setstate__(self, state):
        # Unpickling (also copy.deepcopy). Geometries pickled before October 2026
        # (e.g. in a model's .pt file) store voxel_size and pixel_size, which can
        # be out of date: where they disagree, they are computed again.
        state = dict(state)
        for triple in _TRIPLES:
            if all(name in state for name in triple) and not _agree(
                np.asarray(state[triple[1]]),
                np.asarray(state[triple[0]]) * np.asarray(state[triple[2]]),
            ):
                state = _saved_triples(state)
                break
        self.__dict__.update(state)

    def serialize(self):
        """The geometry as a dict, after ``check_geometry()``: only the values
        that were set, not the computed ones."""
        check_geometry(self)
        return super().serialize()

    def load(self, fname):
        """Load a geometry saved as JSON (see ``init_from_parameter`` for files
        saved before October 2026)."""
        with open(fname, encoding="utf-8") as f:
            d = json.load(f)
        self.__init__(**_saved_triples(d))

    # PLEASE SOMEONE FIND A SMARTER WAY TO DO THIS
    @classmethod
    def init_from_parameter(cls, parameter: LIONParameter):
        """The Geometry of a geometry saved as JSON and loaded as a
        LIONParameter.

        Files saved before October 2026 also store ``voxel_size`` and
        ``pixel_size``, which LION then always computed from the sizes and
        shapes, once, so they can be out of date. Where a file has all three
        values of a triple, the spacing is computed again from the size and
        shape (with a warning if the stored one disagreed).
        """
        names = ("mode", "image_pos", "dso", "dsd", "angles") + IMAGE + DETECTOR
        kwargs = {
            name: getattr(parameter, name) for name in names if hasattr(parameter, name)
        }
        return cls(**_saved_triples(kwargs))

    @staticmethod
    def default_parameters():
        return Geometry(
            image_shape=[1, 512, 512],
            image_size=[300.0 / 512.0, 300, 300],
            detector_shape=[1, 900],
            detector_size=[1, 900],
            dso=575,
            dsd=1050,
            mode="fan",
            angles=np.linspace(0, 2 * np.pi, 360, endpoint=False),
        )

    @staticmethod
    def sparse_view_parameters():
        return Geometry(
            image_shape=[1, 512, 512],
            image_size=[300 / 512, 300, 300],
            detector_shape=[1, 900],
            detector_size=[1, 900],
            dso=575,
            dsd=1050,
            mode="fan",
            angles=np.linspace(0, 2 * np.pi, 50, endpoint=False),
        )

    @staticmethod
    def sparse_angle_parameters():
        return Geometry(
            image_shape=[1, 512, 512],
            image_size=[300 / 512, 300, 300],
            detector_shape=[1, 900],
            detector_size=[1, 900],
            dso=575,
            dsd=1050,
            mode="fan",
            angles=np.linspace(0, 2 * np.pi / 6, 60, endpoint=False),
        )

    @staticmethod
    def parallel_default_parameters(image_shape=None):
        if image_shape is None:
            image_shape = [1, 512, 512]
        return Geometry(
            image_shape=image_shape,
            image_size=image_shape,
            detector_shape=image_shape[0:2],
            detector_size=image_shape[0:2],
            dso=image_shape[1] * 2,
            dsd=image_shape[1] * 4,
            mode="parallel",
            angles=np.linspace(0, 2 * np.pi, 360, endpoint=False),
        )

    @staticmethod
    def parallel_sparse_view_parameters(image_shape=None):
        if image_shape is None:
            image_shape = [1, 512, 512]
        return Geometry(
            image_shape=image_shape,
            image_size=image_shape,
            detector_shape=image_shape[0:2],
            detector_size=image_shape[0:2],
            dso=image_shape[1] * 2,
            dsd=image_shape[1] * 4,
            mode="parallel",
            angles=np.linspace(0, 2 * np.pi, 50, endpoint=False),
        )

    @staticmethod
    def parallel_sparse_angle_parameters(image_shape=None):
        if image_shape is None:
            image_shape = [1, 512, 512]
        return Geometry(
            image_shape=image_shape,
            image_size=image_shape,
            detector_shape=image_shape[0:2],
            detector_size=image_shape[0:2],
            dso=image_shape[1] * 2,
            dsd=image_shape[1] * 4,
            mode="parallel",
            angles=np.linspace(0, 2 * np.pi / 6, 60, endpoint=False),
        )

    def default_geo(self):
        self.__init__(
            image_shape=[1, 512, 512],
            image_size=[300 / 512, 300, 300],
            detector_shape=[1, 900],
            detector_size=[1, 900],
            dso=575,
            dsd=1050,
            mode="fan",
            angles=np.linspace(0, 2 * np.pi, 360, endpoint=False),
        )

    def __str__(self):
        string = []
        string.append("CT Geometry")
        string.append("CT type/mode = " + self.mode)
        string.append("-----")
        string.append(
            "Distance from source to detector (DSD) = " + str(self.dsd) + " mm"
        )
        string.append("Distance from source to origin (DSO)= " + str(self.dso) + " mm")
        string.append("-----")
        string.append("Detector")
        string.append("Detector shape = " + str(self.detector_shape))
        string.append("Detector size = " + str(self.detector_size) + " mm")
        string.append("-----")
        string.append("Image")
        string.append("Image shape = " + str(self.image_shape))
        string.append("Image size = " + str(self.image_size) + " mm")
        string.append("Number of angles = " + str(self.angles.shape))
        return "\n".join(string)


def _disagree_message(triple, given) -> str:
    shape_name, size_name, spacing_name = triple
    return (
        f"{size_name} {np.asarray(given[size_name]).tolist()} is not "
        f"{shape_name} {np.asarray(given[shape_name]).tolist()} * {spacing_name} "
        f"{np.asarray(given[spacing_name]).tolist()}. Give two of them; the third "
        f"is computed (set one to None to have it computed)."
    )


def _saved_triples(d: dict) -> dict:
    """Saved geometry values, with the spacing left out of any triple whose three
    values are all there (it is computed again from size and shape).

    Files saved before October 2026 store the spacings as LION computed them
    once, so they can be out of date (e.g. the 2DeteCT geometry's). Files saved
    since store only the values that were set, and check them first, so their
    three values (if all set) agree and nothing is lost.
    """
    d = dict(d)
    for triple in _TRIPLES:
        shape_name, size_name, spacing_name = triple
        if all(d.get(name, None) is not None for name in triple):
            shape, size = np.array(d[shape_name]), np.array(d[size_name])
            spacing = d.pop(spacing_name)
            if not _agree(size, shape * np.array(spacing)):
                warnings.warn(
                    f"The saved {spacing_name} {np.asarray(spacing).tolist()} does "
                    f"not match {size_name} / {shape_name}; it is computed again as "
                    f"{(size / shape).tolist()} (files saved before October 2026 "
                    f"could store an out-of-date {spacing_name})."
                )
    return d


def _is_unset(value) -> bool:
    return value is None or (
        isinstance(value, np.ndarray)
        and value.dtype == object
        and value.size == 1
        and value.item() is None
    )


def _finite_numbers(value, length=None) -> bool:
    """True if ``value`` is numbers (``length`` of them, if given), all finite."""
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    if length is not None and (a.ndim != 1 or a.size != length):
        return False
    return bool(np.all(np.isfinite(a)))


def check_geometry(geometry: Geometry) -> None:
    """Check that a CT geometry is complete and consistent, and raise a
    ValueError listing every problem if not.

    Run when an operator is built from the geometry (``CTProjectionOp``) and
    when the geometry is saved. Checks:

    - ``mode`` is "fan" or "parallel";
    - image and detector: at least two of shape, size and spacing are set, and
      if all three are, they agree (size = shape * spacing);
    - shapes are positive whole numbers (3 for the image, 2 for the detector),
      sizes and spacings positive;
    - ``image_pos`` is 3 finite numbers;
    - ``angles`` is a 1-D array of finite numbers;
    - fan beam: ``dso`` and ``dsd`` are positive, with ``dsd >= dso``.
    """
    if not isinstance(geometry, Geometry):
        raise ValueError(
            f"Expected a LION.CTtools.ct_geometry.Geometry, got {type(geometry).__name__}"
        )
    problems = []

    if geometry.mode not in MODES:
        problems.append(f"mode is {geometry.mode!r}; it must be one of {MODES}")

    for triple, length in ((IMAGE, 3), (DETECTOR, 2)):
        shape_name, size_name, spacing_name = triple
        given = _triple_values(geometry, triple)
        if len(given) < 2:
            problems.append(
                f"two of {', '.join(triple)} are needed; "
                f"set: {', '.join(given) if given else 'none'}"
            )
            continue
        if len(given) == 3 and not _agree(
            given[size_name], given[shape_name] * given[spacing_name]
        ):
            problems.append(_disagree_message(triple, given))
            continue
        shape = getattr(geometry, shape_name)
        if not _finite_numbers(shape, length):
            problems.append(f"{shape_name} must be {length} numbers, it is {shape}")
        elif not np.allclose(shape, np.rint(shape), rtol=_RTOL, atol=0):
            problems.append(
                f"{shape_name} must be whole numbers of pixels, it is "
                f"{np.asarray(shape).tolist()} ({size_name} / {spacing_name})"
            )
        elif np.any(np.asarray(shape) <= 0):
            problems.append(f"{shape_name} must be positive, it is {shape}")
        for name in (size_name, spacing_name):
            value = getattr(geometry, name)
            if not _finite_numbers(value, length) or np.any(np.asarray(value) <= 0):
                problems.append(
                    f"{name} must be {length} positive numbers, it is {value}"
                )

    if not _finite_numbers(geometry.image_pos, 3):
        problems.append(f"image_pos must be 3 numbers, it is {geometry.image_pos}")

    angles = geometry.angles
    if _is_unset(angles) or not _finite_numbers(angles) or np.ndim(angles) != 1:
        problems.append(f"angles must be a 1-D array of numbers, it is {angles}")
    elif np.size(angles) == 0:
        problems.append("angles is empty")

    if geometry.mode == "fan":
        ok = True
        for name in ("dso", "dsd"):
            value = getattr(geometry, name)
            if _is_unset(value) or not _finite_numbers(value) or np.ndim(value) != 0:
                problems.append(f"{name} must be set for a fan beam, it is {value}")
                ok = False
            elif float(value) <= 0:
                problems.append(f"{name} must be positive, it is {value}")
                ok = False
        if ok and float(geometry.dsd) < float(geometry.dso):
            problems.append(
                f"dsd ({float(geometry.dsd)}) must be at least dso "
                f"({float(geometry.dso)}): the detector is beyond the centre"
            )

    if problems:
        raise ValueError("Invalid CT geometry:\n- " + "\n- ".join(problems))
