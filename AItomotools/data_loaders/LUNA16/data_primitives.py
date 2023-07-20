# OS/path imports
import os
from genericpath import exists
import pathlib

# utils
from tqdm import tqdm
import tifffile as tif
import csv
import pydicom as dicom
from natsort import natsorted, ns

# scientific imports
import numpy as np
import tomosipo as ts
import scipy.io
from scipy import ndimage

import matplotlib.pyplot as plt
from matplotlib import testing

import skimage.transform
from skimage import filters
from skimage import measure
import skimage.io as skio


# Class that holds CT image info
class CTimage:
    """
    Container for the images on the CT images from different datasets

     ...

    Attributes
    ----------
    folder : str
        folder where the data is located

    file_name : str
        file_name of this image

    image : np.array
        pixel image information

    offset: np.array (3,1)
        offset of image center

    spacing: np.array (3,1)
        size of each voxel in mm

    dimsize : np.array(3,1)
        same as image.shape, but of the original data.

    shape: tuple
        same as dimsize but in np.shape compatible format

    Methods
    -------

    """

    def __init__(
        self, folder: pathlib.Path, file_path: pathlib.Path, load_data=False
    ) -> None:
        """
        Initializes an image, given a folder and a file_path.
        Optionally loads the data from disk too, if `load_data` is set to True
        """
        if not file_path.suffix == ".mhd" and not file_path.suffix == ".dcm":
            raise ValueError("File should be .mhd")
        self.folder = folder
        self.file_path = file_path
        self.data = None
        self.unit = None
        self.shape = None
        self.size = None
        self.dimsize = None
        self.load_metadata()
        if load_data:
            self.load_data()

    def load_data(self):
        """
        Loads the image information from disk
        """
        if self.file_path.suffix == ".mhd":
            self.data = skio.imread(
                self.folder.joinpath(self.file_path), plugin="simpleitk"
            )
        elif self.file_path.suffix == ".dcm":
            # get file_name
            fname = self.file_path.stem
            if fname == "*":
                # find all dcm
                file_names = []
                for file in os.listdir(self.folder):
                    if file.endswith(".dcm"):
                        file_names.append(file)
                file_names = natsorted(file_names, key=lambda y: y.lower())

                # read first for allocation purposes
                first = dicom.dcmread(self.folder.joinpath(file_names[0]))
                self.data = np.zeros((len(file_names), *first.pixel_array.shape))
                self.data[0] = first.pixel_array
                for i in range(1, len(file_names)):
                    dcm = dicom.dcmread(self.folder.joinpath(file_names[i]))
                    self.data[i] = dcm.pixel_array
                self.shape = (len(file_names), *first.pixel_array.shape)
        else:
            raise ValueError(
                f"file_path extension not supported: {self.file_path.stem}"
            )

    def unload_data(self):
        """
        Deletes image information (but not metadata)
        """
        self.data = None

    def load_metadata(self):
        """
        Loads metadata of the image.
        """
        if self.file_path.suffix == ".mhd":
            for line in open(self.folder.joinpath(self.file_path)):
                if line.startswith("Offset = "):
                    self.offset = np.array(
                        [float(i) for i in line.rsplit(sep=" = ")[1].rsplit()]
                    )[::-1]
                if line.startswith("ElementSpacing = "):
                    self.spacing = np.array(
                        [float(i) for i in line.rsplit(sep=" = ")[1].rsplit()]
                    )[::-1]
                if line.startswith("DimSize = "):
                    self.dimsize = np.array(
                        [float(i) for i in line.rsplit(sep=" = ")[1].rsplit()]
                    )[::-1]
                    self.shape = tuple(self.dimsize.astype(int))

    def coords2index(self, coords):
        """
        gives indices of real world coordinates for image
        """
        if self.offset is None or self.spacing is None:
            raise RuntimeError("Metadata not read")
        return (coords - self.offset) / self.spacing

    def __getitem__(self, key):
        """
        So you can slice data directly.
        """
        return self.data[key]

    def resample(self, resolution):
        """
        Resamples the data (modifying it) and the mask
        """
        if isinstance(resolution, list) or isinstance(resolution, np.ndarray):
            resolution = tuple(resolution)
        if len(resolution) != len(self.dimsize):
            raise ValueError(
                "Required resolution length not the same number of dimensions as data"
            )

        self.spacing = self.spacing * np.array(self.data.shape) / np.array(resolution)
        self.data = skimage.transform.resize(self.data, resolution)
        self.dimsize = np.array(self.data.shape)
        self.shape = self.data.shape

    def crop_z(self, slices):
        """
        Crops the desired slices in Z
        """

        if not isinstance(slices, list):
            if int(slices) == slices:
                slices = [slices]
            else:
                raise ValueError(
                    "Input slices need to be an index (or list of slices) of the images to keep"
                )
        self.data = self.data[slices]
        self.shape = self.data.shape
        self.size = self.spacing * len(slices)
        self.dimsize = np.array((self.shape))
        # TODO change offset
