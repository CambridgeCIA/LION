## This file contains a general class to store LIONParameter files.
## You should inherit from this class for any LIONParameter setting.
import json

from LION.utils.utils import NumpyEncoder
from pathlib import Path


class LIONParameter:
    def __init__(self, **kwargs):
        """
        Initialize LIONParameter from dictionary
        """
        for item in kwargs:
            if isinstance(kwargs[item], dict):
                setattr(self, item, LIONParameter(**(kwargs[item])))
            else:
                setattr(self, item, kwargs[item])

    def is_filled(self):
        """
        Check if the values are not None
        """
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
        pass

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
                self.serialize(), f, ensure_ascii=False, indent=4, cls=NumpyEncoder
            )

    def load(self, fname):
        """
        Load LIONParameter from JSON file, and initialize instance
        """
        if isinstance(fname, str):
            fname = Path(fname)
        with open(fname, "r", encoding="utf-8") as f:
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
