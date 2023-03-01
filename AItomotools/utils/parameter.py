## This file contains a general class to store parameter files. 
## You should inherit from this class for any parameter setting. 
import json
from AItomotools.utils.utils import NumpyEncoder


class Parameter():


    def is_filled(self):
        if all(vars(self).values()):
            raise ValueError("Not all parameters set")

    def save(self,fname):
        if fname.suffix != ".json" :
            fname.joinpath('.json')
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(vars(self), f, ensure_ascii=False, indent=4,cls=NumpyEncoder)
    

    def load(self,fname):
        with open(fname, 'r', encoding='utf-8') as f:
            d=json.load(f)
        self.__init__(**d)

    def __str__(self):
        d=vars(self)
        for x in d:
            print (x)
            for y in d[x]:
                print (y,':',d[x][y])
