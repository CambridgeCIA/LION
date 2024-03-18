# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Emilien Valat
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import warnings
from LION.utils.paths import DETECT_PROCESSED_DATASET_PATH
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ctgeo
import LION.CTtools.ct_utils as ct
from LION.CTtools.ct_utils import make_operator
from ts_algorithms import fdk, nag_ls


class deteCT(Dataset):
    def __init__(
        self,
        mode,
        geometry_params: ct.Geometry = None,
        parameters: LIONParameter = None,
    ):
        if parameters is None:
            parameters = self.default_parameters()

        # Get geometry default, or validate if given geometry is valid for 2DeteCT, as its raw sinogram data, we can't allow any geometry
        if geometry_params is None:
            self.geo = self.get_default_geometry()
            self.angle_index = list(range(len(self.geo.angles)))
        else:
            self.__is_valid_geo(geometry_params)
            self.geo = geometry_params
            if not np.array_equal(self.geo.angles, self.get_default_geometry().angles):
                # if we are here we already know that the angles are at least valid.
                self.angle_index = []
                full_angles = self.get_default_geometry().angles
                for angle in self.geo.angles:
                    index = next(
                        i
                        for i, _ in enumerate(full_angles)
                        if np.isclose(_, angle, 10e-4)
                    )
                    self.angle_index.append(index)
            else:
                self.angle_index = list(range(len(self.geo.angles)))

        if parameters.sinogram_mode != parameters.reconstruction_mode:
            warnings.warn(
                "Sinogram mode and reconstruction mode don't match, so reconstruction is not from the sinogram you are getting... \n This should be an error, but I trust that you won what you are doing"
            )
        if parameters.query == "" and (
            not parameters.flat_field_correction or not parameters.dark_field_correction
        ):
            warnings.warn(
                "You are not using any detector query, but you are not using flat field or dark field correction. This is not recommended, as different detectors perform differently"
            )
        ### Defining the path to data
        self.path_to_dataset = parameters.path_to_dataset
        """
        The path_to_dataset attribute (pathlib.Path) points towards the folder
        where the data is stored
        """
        ### Defining the path to scan data
        self.path_to_scan_data = self.path_to_dataset.joinpath("scan_settings.json")
        ### Defining the path to the data record
        self.path_to_data_record = self.path_to_dataset.joinpath(
            f"default_data_records.csv"
        )

        ### Defining the data record
        self.data_record = pd.read_csv(self.path_to_data_record)
        """
        The data_record (pd.Dataframe) maps a slice index/identifier to
            - the sample index of the sample it belongs to
            - the number of slices expected
            - the number of slices actually sampled
            - the first slice of the sample to which a given slice belongs
            - the last slice of the sample to which a given slice belongs
            - the mix 
            - the detector it was sampled with
        """
        # Defining the sinogram mode
        self.sinogram_mode = parameters.sinogram_mode
        """
        The sinogram_mode (str) argument is a keyword defining what sinogram mode of the dataset to use:
                         |  mode1   |   mode2  |  mode3
            Tube Voltage |   90kV   |   90kV   |  60kV
            Tube power   |    3W    |    90W   |  60W
            Filter       | Thoraeus | Thoraeus | No Filter
        """
        # Defining the reconstruction mode
        self.reconstruction_mode = parameters.reconstruction_mode
        """
        The reconstruction_mode (str) argument is a keyword defining what image mode of the dataset to use:
                         |  mode1   |   mode2  |  mode3
            Tube Voltage |   90kV   |   90kV   |  60kV
            Tube power   |    3W    |    90W   |  60W
            Filter       | Thoraeus | Thoraeus | No Filter
        """
        # Defining the task
        self.task = parameters.task
        """
        The task (str) argument is a keyword defining what is the dataset used for:
            - task == 'reconstruction' -> the dataset returns the sinogram and the reconstruction
            - task == 'segmentation' -> the dataset returns the reconstruction and the segmentation
            - task == 'joint' -> the dataset returns the sinogram, the reconstruction and the segmentation
        """

        assert self.task in [
            "reconstruction",
            "segmentation",
            "joint",
        ], f'Wrong task argument, must be in ["reconstruction", "segmentation", "joint"]'
        assert mode in [
            "train",
            "validation",
            "test",
        ], f'Wrong mode argument, must be in ["train", "validation", "test"]'
        # Defining the training mode
        self.mode = mode
        """
        The train (bool) argument defines if the dataset is used for training or testing
        """
        self.training_proportion = parameters.training_proportion
        self.validation_proportion = parameters.validation_proportion

        # Defining the train proportion
        """
        The training_proportion (float) argument defines the proportion of the training dataset used for training
        """
        self.transforms = None  # parameters.transforms

        ### We query the dataset subset
        self.slice_dataframe: pd.DataFrame
        """
        The slice_dataframe (pd.Dataframe) is the subset of interest of the dataset.
        The self.data_record argument becomes the slice_dataframe once we have queried it
        Example of query: 'detector==1'
        If the no query argument is passed, data_record_subset == data_record
        """
        if parameters.query:
            self.slice_dataframe = self.data_record.query(parameters.query)
        else:
            self.slice_dataframe = self.data_record

        ### We split the dataset between training and testing
        self.compute_sample_dataframe()
        self.sample_dataframe: pd.DataFrame
        """
        The sample_dataframe (pd.Dataframe) is a dataframe linking sample index to slices.
        It is used to partition the dataset on a sample basis, rather than on a slice basis,
        avoiding 'data leakage' between training, validation and testing
        """
        if self.mode == "train":
            self.sample_dataframe = self.sample_dataframe.head(
                int(len(self.sample_dataframe) * self.training_proportion)
            )
        elif self.mode == "validation":
            self.sample_dataframe = self.sample_dataframe.iloc[
                int(len(self.sample_dataframe) * self.training_proportion) : int(
                    len(self.sample_dataframe)
                    * (self.training_proportion + self.validation_proportion)
                )
            ]
        else:
            self.sample_dataframe = self.sample_dataframe.tail(
                int(
                    len(self.sample_dataframe)
                    * (1 - self.training_proportion - self.validation_proportion)
                )
            )

        self.slice_dataframe = self.slice_dataframe[
            self.slice_dataframe["sample_index"].isin(
                self.sample_dataframe["sample_index"].unique()
            )
        ]

        self.flat_field_correction = parameters.flat_field_correction
        self.dark_field_correction = parameters.dark_field_correction
        self.log_transform = parameters.log_transform
        self.do_recon = parameters.do_recon
        self.recon_algo = parameters.recon_algo

    @staticmethod
    def default_parameters():
        param = LIONParameter()
        param.path_to_dataset = DETECT_PROCESSED_DATASET_PATH
        param.sinogram_mode = "mode2"
        param.reconstruction_mode = "mode2"
        param.task = "reconstruction"
        param.training_proportion = 0.8
        param.validation_proportion = 0.1
        param.test_proportion = 0.1

        param.do_recon = False
        param.recon_algo = "nag_ls"
        param.flat_field_correction = True
        param.dark_field_correction = True
        param.log_transform = True
        param.query = ""

        param.geo = deteCT.get_default_geometry()
        return param

    @staticmethod
    def get_default_geometry():
        geo = ctgeo.Geometry.default_parameters()
        # From Max Kiss code
        SOD = 431.019989
        SDD = 529.000488
        detPix = 0.0748
        detSubSamp = 2
        detPixSz = detSubSamp * detPix
        nPix = 956
        det_width = detPixSz * nPix
        FOV_width = det_width * SOD / SDD
        nVox = 1024
        voxSz = FOV_width / nVox
        scaleFactor = 1.0 / voxSz
        SDD = SDD * scaleFactor
        SOD = SOD * scaleFactor
        detPixSz = detPixSz * scaleFactor

        geo.dsd = SDD
        geo.dso = SOD
        geo.detector_shape = [1, 956]
        geo.detector_size = [detPixSz, detPixSz * 956]
        geo.image_shape = [1, 1024, 1024]
        geo.image_size = [1, 1024, 1024]
        geo.image_pos = [0, -1, -1]
        geo.angles = -np.linspace(0, 2 * np.pi, 3600, endpoint=False) + np.pi
        return geo

    def __is_valid_geo(self, geo):
        if not isinstance(geo, ctgeo.Geometry):
            raise ValueError("geo must be a ctgeo.Geometry object")
        default_geo = deteCT.get_default_geometry()

        if geo.dsd != default_geo.dsd or geo.dso != default_geo.dso:
            raise ValueError(
                f"dsd and dso must be the same as the default geometry: DSO = {default_geo.dso}, DSD = {default_geo.dsd}"
            )
        if geo.detector_shape[0] != default_geo.detector_shape[0]:
            raise ValueError(f"detector_shape[0] must be 1")
        if geo.detector_shape[1] != default_geo.detector_shape[1]:
            warnings.warn(
                f"Raw data detector_shape[1] must be 956, it is {geo.detector_shape[1]}. Interpolation will be used, but note that you are not using exactly the raw data"
            )
        if geo.detector_size != default_geo.detector_size:
            raise ValueError(f"detector_size must be {default_geo.detector_size}")
        if geo.image_shape != default_geo.image_shape:
            warnings.warn(
                f"image_shape must be {default_geo.image_shape}, it is {geo.image_shape}. Interpolation will be used, but note that you are not using exactly the raw data"
            )
        if geo.image_size[1:] != default_geo.image_size[1:]:
            raise ValueError(f"image_size must be {default_geo.image_size}")
        if geo.image_pos != default_geo.image_pos:
            raise ValueError(f"image_pos must be {default_geo.image_pos}")

        full_angles = self.get_default_geometry().angles
        if not np.array_equal(geo.angles, full_angles):
            # I think this is slow, maybe there is abetter way

            for angle in geo.angles:
                index = next(
                    i for i, _ in enumerate(full_angles) if np.isclose(_, angle, 10e-4)
                )
                if index is None:
                    raise ValueError(
                        f"Given angles must be part of existing ones. Check the array deteCT.get_default_geometry().angles for the existing angles. The given angle {angle} is not part of the existing angles."
                    )
            return True

    @staticmethod
    def get_default_operator():
        geo = deteCT.get_default_geometry()
        return make_operator(geo)

    def get_operator(self):
        return make_operator(self.geo)

    def compute_sample_dataframe(self):
        unique_identifiers = self.slice_dataframe["sample_index"].unique()
        record = {"sample_index": [], "first_slice": [], "last_slice": []}
        for identifier in unique_identifiers:
            record["sample_index"].append(identifier)
            subset = self.slice_dataframe[
                self.slice_dataframe["sample_index"] == identifier
            ]
            record["first_slice"].append(subset["first_slice"].iloc[0])
            record["last_slice"].append(subset["last_slice"].iloc[0])
        self.sample_dataframe = pd.DataFrame.from_dict(record)

    def __len__(self):
        return (
            self.sample_dataframe["last_slice"].iloc[-1]
            - self.sample_dataframe["first_slice"].iloc[0]
            + 1
        )

    def __getitem__(self, index):
        slice_row = self.slice_dataframe.iloc[index]
        path_to_sinogram = self.path_to_dataset.joinpath(
            f"{slice_row['slice_identifier']}/{self.sinogram_mode}"
        )
        path_to_reconstruction = self.path_to_dataset.joinpath(
            f"{slice_row['slice_identifier']}/{self.reconstruction_mode}"
        )
        path_to_segmentation = self.path_to_dataset.joinpath(
            f"{slice_row['slice_identifier']}/mode2"
        )

        if self.task == ["segmentation", "joint"]:
            segmentation = torch.from_numpy(
                np.load(path_to_segmentation.joinpath("segmentation.npy"))
            ).unsqueeze(0)

        if self.task in ["reconstruction", "joint"]:
            sinogram = torch.from_numpy(
                np.load(path_to_sinogram.joinpath("sinogram.npy"))
            ).unsqueeze(0)
            if self.flat_field_correction:
                flat = torch.from_numpy(
                    np.load(path_to_sinogram.joinpath("flat.npy"))
                ).unsqueeze(0)
            else:
                flat = 1
            if self.dark_field_correction:
                dark = torch.from_numpy(
                    np.load(path_to_sinogram.joinpath("dark.npy"))
                ).unsqueeze(0)
            else:
                dark = 0
            sinogram = (sinogram - dark) / (flat - dark)
            if self.log_transform:
                sinogram = -torch.log(sinogram)

            sinogram = torch.flip(sinogram, [2])
            sinogram = sinogram[:, self.angle_index, :]

            # Interpolate if geometry is not default
            if self.geo.detector_shape != self.get_default_geometry().detector_shape:
                sinogram = torch.nn.functional.interpolate(
                    sinogram.unsqueeze(0),
                    size=(sinogram.shape[1], self.geo.detector_shape[1]),
                    mode="bilinear",
                )
                sinogram = torch.squeeze(sinogram, 0)
        if self.do_recon:
            op = deteCT.get_operator()
            if self.recon_algo == "nag_ls":
                reconstruction = nag_ls(op, sinogram, 100, min_constraint=0)
            elif self.recon_algo == "fdk":
                reconstruction = fdk(op, sinogram)
        else:
            reconstruction = torch.from_numpy(
                np.load(path_to_reconstruction.joinpath("reconstruction.npy"))
            ).unsqueeze(0)
            # Interpolate if geometry is not default
            if self.geo.image_shape != self.get_default_geometry().image_shape:
                reconstruction = torch.nn.functional.interpolate(
                    reconstruction.unsqueeze(0),
                    size=(self.geo.image_shape[1], self.geo.image_shape[2]),
                    mode="bilinear",
                )
                reconstruction = torch.squeeze(reconstruction, 0)

        if self.task == "reconstruction":
            return sinogram, reconstruction
        if self.task == "segmentation":
            return reconstruction, segmentation
        if self.task == "joint":
            return sinogram, reconstruction, segmentation
