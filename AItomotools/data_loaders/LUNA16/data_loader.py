# OS/path imports
import os
from multiprocessing.sharedctypes import Value  # what the heck is this
import fnmatch


# utils
import tifffile as tif
from tqdm import tqdm
import h5py
import csv

# math/science imports
from cmath import nan
import skimage.io as io
from skimage.util import random_noise
from matplotlib import pyplot as plt
import numpy as np
import scipy.io

# my imports
from AItomotools.data_loaders.data_primitives import CTimage
import AItomotools.CTtools.ct_utils as ct


class CT_data_loader:
    def __init__(self) -> None:
        pass

    def load_data(self, indices):
        """
        Loads the actual data from disk
        """
        if not self.images:
            self.load_metadata()
        if not isinstance(indices, list):
            if int(indices) == indices:
                indices = [indices]
            else:
                raise ValueError(
                    "Input indices need to be an index (or list of indices) of the images to load"
                )
        if any(i > len(self.images) for i in indices):
            raise ValueError("Index exceeds Image range")
        # load data
        for i in indices:
            self.images[i].load_data()
            # conver to the right unit. They are in HUs in memory
            if self.unit == "normal":
                self.images[i].unit = "normal"
                self.images[i].data = ct.from_HU_to_normal(self.images[i].data)
            if self.unit == "mu":
                self.images[i].unit = "mu"
                self.images[i].data = ct.from_HU_to_mu(self.images[i].data)

    def unload_data(self, indices=None):
        """
        Unloads data from disk. By default, all data, otherwise, give the indices desired to unload
        """
        if indices is None:
            indices = list(range(len(self.images)))
        if not isinstance(indices, list):
            if int(indices) == indices:
                indices = [indices]
            else:
                raise ValueError(
                    "Input indices need to be an index (or list of indices) of the images to load"
                )
        for i in indices:
            self.images[i].unload_data()

    def from_HU_to_normal(self):
        """
        Converts any loaded values that are in HUs to [0-1] range
        """
        if (
            self.unit == "HU"
        ):  # No need to check individual images, they are loaded in the units of the loader.
            self.unit = "normal"
            for i in self.images:
                i.unit = "normal"
                if i.data is not None:
                    i.data = ct.from_HU_to_normal(i.data)
        else:
            raise AssertionError("Data must be in HUs for HU to normal conversion")

    def from_HU_to_mu(self):
        """
        Converts any loaded values that are in HUs to mu (linear attenuation coefficient)
        """
        if (
            self.unit == "HU"
        ):  # No need to check individual images, they are loaded in the units of the loader.
            self.unit = "mu"
            for i in self.images:
                i.unit = "mu"
                if i.data is not None:
                    i.data = ct.from_HU_to_mu(i.data)
        else:
            raise AssertionError("Data must be in HUs for HU to mu conversion")


# class LIDC_IDRI(CT_data_loader):
#     """
#     Class that loads LIDC_IDRI dataset and metadata.
#     """
#     def __init__(self,folder, verbose=True,load_data=False) -> None:
#         self.images=[]
#         self.subfolder=None
#         self.verbose=verbose
#         self.unit="HU" # Default in HUs
#         self.folder=folder
#         if load_data:
#             self.load_metadata()

#     def load_metadata(self,folder=None,mode="CT"):
#         """
#         Loads metadata of scans in the dataset and filters out undesired
#         """

#         if mode != "CT":
#             raise AttributeError("Only CT images supported, DX is still not supported")

#         if folder is None:
#             folder=self.folder

#         # Find nodule metadata
#         meta_file=os.path.join(folder,'metadata.csv')
#         with open(meta_file, newline='') as csvfile:
#             csvreader = csv.reader(csvfile, delimiter=',')
#             next(csvreader) # remove header
#             for row in csvreader:
#                 if row[10]==mode:
#                     full_folder=os.path.join(folder,row[15][2:].replace("\\","/"))
#                     self.images.append(CTimage(full_folder,"*.dcm"))
