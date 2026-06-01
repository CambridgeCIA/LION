# =============================================================================
# This file is part of LION library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


import pathlib

import numpy as np
import json

from LION.utils.parameter import LIONParameter


class Geometry(LIONParameter):
    """
    Class holding a CT geometry
    """

    def __init__(self, **kwargs):
        self.mode = kwargs.get("mode", None)

        self.native_image_shape = np.array(kwargs.get("native_image_shape", None))
        self.native_image_size = np.array(kwargs.get("native_image_size", None))
        self.image_scaling = kwargs.get("image_scaling", 1.0)

        self.image_shape = np.array(
            kwargs.get(
                "image_shape",
                np.array(
                    [
                        self.native_image_shape[0],
                        self.native_image_shape[1] * self.image_scaling,
                        self.native_image_shape[2] * self.image_scaling,
                    ]
                ),
            ),
            dtype=int,
        )
        self.image_size = np.array(
            kwargs.get(
                "image_size",
                [
                    self.native_image_size[0] / self.image_scaling,
                    self.native_image_size[1],
                    self.native_image_size[2],
                ],
            )
        )
        if not self.native_image_shape.all() and not self.native_image_size.all():
            self.native_image_shape = self.image_shape
            self.native_image_size = self.image_size

        if self.image_shape.all() and self.image_size.all():
            self.voxel_size = self.image_size / self.image_shape
        if "image_pos" in kwargs:
            self.image_pos = np.array(kwargs.get("image_pos", None))
        else:
            self.image_pos = np.array([0, 0, 0])

        self.detector_shape = np.array(kwargs.get("detector_shape", None))
        self.detector_size = np.array(kwargs.get("detector_size", None))
        if self.detector_size.all() and self.detector_shape.all():
            self.pixel_size = self.detector_size / self.detector_shape

        self.dso = np.array(kwargs.get("dso", None))
        self.dsd = np.array(kwargs.get("dsd", None))

        self.angles = np.array(kwargs.get("angles", None))

    # PLEASE SOMEONE FIND A SMARTER WAY TO DO THIS
    @classmethod
    def init_from_parameter(cls, parameter: LIONParameter):
        if hasattr(parameter, "native_image_shape"):
            native_image_shape = parameter.native_image_shape
        else:
            native_image_shape = parameter.image_shape
        if hasattr(parameter, "native_image_size"):
            native_image_size = parameter.native_image_size
        else:
            native_image_size = parameter.image_size
        if hasattr(parameter, "image_scaling"):
            image_scaling = parameter.image_scaling
        else:
            image_scaling = 1.0
        return cls(
            image_shape=parameter.image_shape,
            image_size=parameter.image_size,
            native_image_shape=native_image_shape,
            native_image_size=native_image_size,
            image_scaling=image_scaling,
            detector_shape=parameter.detector_shape,
            detector_size=parameter.detector_size,
            dso=parameter.dso,
            dsd=parameter.dsd,
            mode=parameter.mode,
            angles=parameter.angles,
        )

    @staticmethod
    def default_parameters(image_scaling=1.0):
        native_image_shape = [1, 512, 512]
        native_image_size = [300.0 / 512.0, 300, 300]
        return Geometry(
            native_image_shape=native_image_shape,
            native_image_size=native_image_size,
            image_scaling=image_scaling,
            image_shape=np.array(
                [
                    native_image_shape[0],
                    native_image_shape[1] * image_scaling,
                    native_image_shape[2] * image_scaling,
                ],
                dtype=int,
            ),
            image_size=[
                native_image_size[0] / image_scaling,
                native_image_size[1],
                native_image_size[2],
            ],
            detector_shape=[1, 900],
            detector_size=[1, 900],
            dso=575,
            dsd=1050,
            mode="fan",
            angles=np.linspace(0, 2 * np.pi, 360, endpoint=False),
        )

    @staticmethod
    def sparse_view_parameters(image_scaling=1.0):
        native_image_shape = [1, 512, 512]
        native_image_size = [300.0 / 512.0, 300, 300]
        return Geometry(
            native_image_shape=native_image_shape,
            native_image_size=native_image_size,
            image_scaling=image_scaling,
            image_shape=np.array(
                [
                    native_image_shape[0],
                    native_image_shape[1] * image_scaling,
                    native_image_shape[2] * image_scaling,
                ],
                dtype=int,
            ),
            image_size=[
                native_image_size[0] / image_scaling,
                native_image_size[1],
                native_image_size[2],
            ],
            detector_shape=[1, 900],
            detector_size=[1, 900],
            dso=575,
            dsd=1050,
            mode="fan",
            angles=np.linspace(0, 2 * np.pi, 50, endpoint=False),
        )

    @staticmethod
    def sparse_angle_parameters(image_scaling=1.0):
        native_image_shape = [1, 512, 512]
        native_image_size = [300.0 / 512.0, 300, 300]
        return Geometry(
            native_image_shape=native_image_shape,
            native_image_size=native_image_size,
            image_scaling=image_scaling,
            image_shape=np.array(
                [
                    native_image_shape[0],
                    native_image_shape[1] * image_scaling,
                    native_image_shape[2] * image_scaling,
                ],
                dtype=int,
            ),
            image_size=[
                native_image_size[0] / image_scaling,
                native_image_size[1],
                native_image_size[2],
            ],
            detector_shape=[1, 900],
            detector_size=[1, 900],
            dso=575,
            dsd=1050,
            mode="fan",
            angles=np.linspace(0, 2 * np.pi / 6, 60, endpoint=False),
        )

    @staticmethod
    def parallel_default_parameters(image_shape=None, image_scaling=1.0):
        if image_shape is None:
            native_image_shape = [1, 512, 512]
            image_shape = np.array(
                [
                    native_image_shape[0],
                    native_image_shape[1] * image_scaling,
                    native_image_shape[2] * image_scaling,
                ],
                dtype=int,
            )
        else:
            native_image_shape = image_shape
            if image_scaling != 1.0:
                raise Exception("If image_shape is provided, image_scaling must be 1.0")
        return Geometry(
            native_image_shape=native_image_shape,
            native_image_size=native_image_shape,
            image_scaling=image_scaling,
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
    def parallel_sparse_view_parameters(image_shape=None, image_scaling=1.0):
        if image_shape is None:
            native_image_shape = [1, 512, 512]
            image_shape = np.array(
                [
                    native_image_shape[0],
                    native_image_shape[1] * image_scaling,
                    native_image_shape[2] * image_scaling,
                ],
                dtype=int,
            )
        else:
            native_image_shape = image_shape
            if image_scaling != 1.0:
                raise Exception("If image_shape is provided, image_scaling must be 1.0")
        return Geometry(
            native_image_shape=native_image_shape,
            native_image_size=native_image_shape,
            image_scaling=image_scaling,
            image_shape=image_shape,
            image_size=image_shape,
            detector_shape=image_shape[0:2],
            detector_size=image_shape[0:2],
            dso=image_shape[1] * 2,
            dsd=image_shape[1] * 4,
            mode="parallel",
            angles=np.linspace(0, 2 * np.pi, 50, endpoint=False),
        )

    staticmethod

    def parallel_sparse_angle_parameters(image_shape=None, image_scaling=1.0):
        if image_shape is None:
            native_image_shape = [1, 512, 512]
            image_shape = np.array(
                [
                    native_image_shape[0],
                    native_image_shape[1] * image_scaling,
                    native_image_shape[2] * image_scaling,
                ],
                dtype=int,
            )
        else:
            native_image_shape = image_shape
            if image_scaling != 1.0:
                raise Exception("If image_shape is provided, image_scaling must be 1.0")
        return Geometry(
            native_image_shape=native_image_shape,
            native_image_size=native_image_shape,
            image_scaling=image_scaling,
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
        _native_image_shape = ([1, 512, 512],)
        _native_image_size = ([300.0 / 512.0, 300, 300],)
        _image_scaling = (1.0,)
        self.__init__(
            native_image_shape=[1, 512, 512],
            native_image_size=[300.0 / 512.0, 300, 300],
            image_scaling=_image_scaling,
            image_shape=np.array(
                [
                    _native_image_shape[0],
                    _native_image_shape[1] * _image_scaling,
                    _native_image_shape[2] * _image_scaling,
                ],
                dtype=int,
            ),
            image_size=[
                _native_image_size[0] / _image_scaling,
                _native_image_size[1],
                _native_image_size[2],
            ],
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
        string.append("Image scaling = " + str(self.image_scaling))
        return "\n".join(string)
