# =============================================================================
# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


import pathlib

import numpy as np
import json

from AItomotools.utils.parameter import Parameter


class Geometry(Parameter):
    """
    Class holding a CT geometry
    """

    def __init__(self, **kwargs):
        self.mode = kwargs.get("mode", None)

        self.image_size = np.array(kwargs.get("image_size", None))
        self.image_shape = np.array(kwargs.get("image_shape", None))
        if self.image_shape.all() and self.image_size.all():
            self.voxel_size = self.image_size / self.image_shape

        self.detector_shape = np.array(kwargs.get("detector_shape", None))
        self.detector_size = np.array(kwargs.get("detector_size", None))
        if self.detector_size.all() and self.detector_shape.all():
            self.pixel_size = self.detector_size / self.detector_shape

        self.dso = np.array(kwargs.get("dso", None))
        self.dsd = np.array(kwargs.get("dsd", None))

        self.angles = np.array(kwargs.get("angles", None))

    @staticmethod
    def default_parameters():
        return Geometry(
            image_shape=[1, 512, 512],
            image_size=[300 / 512, 300, 300],
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
