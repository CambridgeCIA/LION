## This file contains a general class to store LIONParameter files.
## You should inherit from this class for any LIONParameter setting.
import json
from dataclasses import dataclass
from pathlib import Path

from LION.utils.utils import JSONParamEncoder


@dataclass
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
            If a value is a `dict`, it will be turned into a `LIONParameter`.
        """
        for item in kwargs:
            # TODO (low importance): Is there an `is_dict_like` that can include more types of key-based objects? If not, should we make one?
            if isinstance(kwargs[item], dict):
                # If we get a dict, turn it into a LIONParameter (we can have nested LIONParameter's)
                setattr(self, item, LIONParameter(**(kwargs[item])))
            else:
                setattr(self, item, kwargs[item])

    def is_filled(self):
        """
        Check if the values are not None
        """
        # TODO: This only check truthiness. E.g. if a parameter is False or 0, it will also be considered not filled.
        if not all(vars(self).values()):
            raise ValueError("Not all parameters set")

    def serialize(self):
        """
        This only exists to allow LIONParameter() inside LIONParameter()
        otherwise, its equivalent to vars(self)
        """
        d = vars(self)
        for k, v in d.copy().items():
            if isinstance(v, LIONParameter):
                d.pop(k)
                d[k] = v.serialize()  # love some recursivity
            if isinstance(v, Path):
                d[k] = str(v)
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
            fname.joinpath(".json")
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
            d = json.load(f)
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
