## This file contains a general class to store parameter files.
## You should inherit from this class for any parameter setting.
import json

from AItomotools.utils.utils import NumpyEncoder
from pathlib import Path


class Parameter:
    def __init__(self, **kwargs):
        """
        Initialize parameter from dictionary
        """
        for item in kwargs:
            if isinstance(kwargs[item], dict):
                setattr(self, item, Parameter(**(kwargs[item])))
            else:
                setattr(self, item, kwargs[item])

    def is_filled(self):
        """
        Check if the values are not None
        """
        if all(vars(self).values()):
            raise ValueError("Not all parameters set")

    def serialize(self):
        """
        This only exists to allow Parameter() inside Parameter()
        otherwise, its equivalent to vars(self)
        """
        d = vars(self)
        for k, v in d.copy().items():
            if isinstance(v, Parameter):
                d.pop(k)
                d[k] = v.serialize()  # love some recursivity
        return d

    def save(self, fname):
        """
        Save parameter into a JSON file
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
        Load parameter from JSON file, and initialize instance
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
