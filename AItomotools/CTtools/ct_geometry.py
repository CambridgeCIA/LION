import numpy as np
import torch
import json
from pathlib import Path
from AItomotools.utils.utils import NumpyEncoder

class Geometry():
    """
    Class holding a CT geometry
    """
    def __init__(self, **kwargs):
        if kwargs:
            self.initialize_geo_values(**kwargs)

        
    def initialize_geo_values(self,**kwargs):
        self.mode=kwargs.get('mode',None)

        self.image_size=np.array(kwargs.get('image_size',None))
        self.image_shape=np.array(kwargs.get('image_shape',None))
        if self.image_shape.all()  and self.image_size.all():
            self.voxel_size=self.image_size/self.image_shape

        print(kwargs)
        self.detector_shape=np.array(kwargs.get('detector_shape',None))
        self.detector_size=np.array(kwargs.get('detector_size',None))
        if self.detector_size.all() and self.detector_shape.all():
            self.pixel_size=self.detector_size/self.detector_shape

        self.dso=np.array(kwargs.get('dso',None))
        self.dsd=np.array(kwargs.get('dsd',None))

    def default_geo(self):
        self.initialize_geo_values(
            image_shape=[1,512,512],
            image_size=[5,300,300],
            detector_shape=[1,900],
            detector_size=[1,900],
            dso=575,
            dsd=1050,
            mode="fan")

    def check_geo(self):

        if all(vars(self).values()):
            raise ValueError("Not all geometry parameters set")

    def save(self,fname):
        if fname.suffix is not ".json" :
            fname.joinpath('.json')
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(vars(self), f, ensure_ascii=False, indent=4,cls=NumpyEncoder)

    def load(self,fname):
        with open(fname, 'w', encoding='utf-8') as f:
            self.initialize_geo_values(json.load(f))
