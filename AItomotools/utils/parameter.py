## This file contains a general class to store parameter files. 
## You should inherit from this class for any parameter setting. 
import json
from AItomotools.utils.utils import NumpyEncoder
from pathlib import Path

class Parameter():

    def __init__(self,**kwargs):
        for item in kwargs:
            setattr(self, item, kwargs[item])

    def is_filled(self):
        if all(vars(self).values()):
            raise ValueError("Not all parameters set")

    def save(self,fname):
        if isinstance(fname, str):
            fname=Path(fname)
        if fname.suffix != ".json" :
            fname.joinpath('.json')
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(vars(self), f, ensure_ascii=False, indent=4,cls=NumpyEncoder)
    

    def load(self,fname):
        if isinstance(fname, str):
            fname=Path(fname)
        with open(fname, 'r', encoding='utf-8') as f:
            d=json.load(f)
        self.__init__(**d)

    def __str__(self):
        d=vars(self)
        string=[]
        for x in d:
            string.append(str(x)+' : '+str(d[x]))
        return "\n".join(string)

