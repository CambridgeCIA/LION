## This file contains a general class to store LIONParameter files.
## You should inherit from this class for any LIONParameter setting.
import json
from pathlib import Path

from LION.utils.utils import JSONParamEncoder


class LIONParameter:
    """General class to store parameters (e.g. for experiments).

    Allows nested `LIONParameter`'s.

    Should allow runtime assignments of parameters.
    """

    def __init__(self, **kwargs):
        """
        Initialize LIONParameter from dictionary

        Parameters
        ----------
        **kwargs : dict
            Contains parameters to initialize the `LIONParameter`.
            If a value is a `dict`, it will be turned into a `LIONParameter`,
            unless it is a spec (a dict with ``"_target_"``, see
            ``LION.utils.spec``): a spec describes an object and stays a dict.
        """
        for item in kwargs:
            # TODO (low importance): Is there an `is_dict_like` that can include more types of key-based objects? If not, should we make one?
            if isinstance(kwargs[item], dict) and "_target_" not in kwargs[item]:
                # If we get a dict, turn it into a LIONParameter (we can have nested LIONParameter's)
                setattr(self, item, LIONParameter(**(kwargs[item])))
            else:
                setattr(self, item, kwargs[item])

    def is_filled(self):
        """
        Check if the values are not None
        """
        # TODO: This only check truthiness, not presence. E.g. if a parameter is False or 0, it will also be considered not filled.
        if not all(vars(self).values()):
            raise ValueError("Not all parameters set")

    def serialize(self):
        """
        This only exists to allow LIONParameter() inside LIONParameter()
        otherwise, its equivalent to vars(self)
        """
        # Build a new dict: vars(self) is the object's own __dict__, and
        # modifying it would change the parameter object itself.
        d = {}
        for k, v in vars(self).items():
            if isinstance(v, LIONParameter):
                d[k] = v.serialize()
            elif isinstance(v, Path):
                d[k] = str(v)
            else:
                d[k] = v
        return d

    def deserialize(self):
        """
        Produces a dict of LIONParameter()
        """
        # TODO: Should we raise `NotImplementedError` instead of letting it pass?

    def save(self, fname):
        """
        Save LIONParameter into a JSON file
        """
        if isinstance(fname, str):
            fname = Path(fname)
        if fname.suffix != ".json":
            fname = fname.with_suffix(".json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(
                self.serialize(), f, ensure_ascii=False, indent=4, cls=JSONParamEncoder
            )

    def load(self, fname):
        """
        Load LIONParameter from JSON file, and initialize instance
        """
        if isinstance(fname, str):
            fname = Path(fname)
        with open(fname, encoding="utf-8") as f:
            d = with_current_names(json.load(f))
        self.__init__(**d)

    def __str__(self):
        """
        Overload the str for printing.
        """
        d = vars(self)
        string = []
        for x in d:
            string.append(str(x) + " : " + str(d[x]))
        return "\n".join(string)

    def unpack(self):
        """
        If the LIONParameter is made of other Parameters, unpack those.
        """
        # Get all attributes
        attributes = vars(self).copy()
        # pop all attributes that are NOT a LIONParameter()
        for key in list(attributes.keys()):
            if not isinstance(attributes[key], LIONParameter):
                del attributes[key]
            else:
                delattr(self, key)
        return attributes


class OperatorParameters(LIONParameter):
    """The parameters of an operator, which make it again: one class per kind
    of operator. For CT, it's the ``Geometry`` (``LION.CTtools.ct_geometry``):
    wherever LION asks for ``operator_params``, a CT user gives their geometry.

    Components that need an operator take its parameters (``operator_params``)
    and call ``make_operator()``, so they work with any kind of operator."""

    def make_operator(self):
        """The operator these parameters describe."""
        raise NotImplementedError(
            f"{type(self).__name__} doesn't say how to make its operator: "
            "define make_operator()"
        )

    # Splitting into sub-operators, for methods that use parts of the
    # measurements (Noise2Inverse, Sparse2Inverse). A kind of operator that
    # can be split defines all three; the others raise.

    @property
    def split_count(self) -> int:
        """How many measurement units the operator can be split along (for CT,
        the angles)."""
        raise NotImplementedError(self._cannot_split())

    def split(self, indices) -> "OperatorParameters":
        """New parameters for the measurement units ``indices`` only (for CT,
        those angles). These parameters are not changed."""
        raise NotImplementedError(self._cannot_split())

    def split_measurements(self, y, indices):
        """The measurement units ``indices`` of ``y`` (batched or not): the
        measurements of the operator ``split(indices)`` makes."""
        raise NotImplementedError(self._cannot_split())

    def _cannot_split(self) -> str:
        return (
            f"{type(self).__name__} can't be split into sub-operators: "
            "define split_count, split and split_measurements"
        )


#: Names files saved before November 2026 used for the operator's parameters.
OLD_OPERATOR_NAMES = ("geometry", "geometry_params", "geometry_parameters")


def with_current_names(value):
    """A record read from a file, with the old names of the operator's
    parameters (``OLD_OPERATOR_NAMES``, at any depth) as ``operator_params``."""
    if isinstance(value, list):
        return [with_current_names(v) for v in value]
    if not isinstance(value, dict):
        return value
    renamed = {}
    for key, item in value.items():
        if key in OLD_OPERATOR_NAMES and "operator_params" not in value:
            key = "operator_params"
        renamed[key] = with_current_names(item)
    return renamed
