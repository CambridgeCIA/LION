# =============================================================================
# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Emilien Valat
# =============================================================================
# =============================================================================
# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Emilien Valat
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset


def format_slice_index(index: int) -> str:
    str_index = str(index)
    while len(str_index) < 5:
        str_index = "0" + str_index
    return f"slice{str_index}"


def format_mode_index(index: int) -> str:
    return f"mode{index}"


class ScanDataset(Dataset):
    def __init__(
        self,
        query: str,
        path_to_dataset: Path,
        mode: str,
        pipeline: str,
        train: bool,
        training_proportion: float,
        transforms=None,
        default=True,
    ):
        ### Defining the path to data
        self.path_to_dataset = path_to_dataset
        """
        The path_to_dataset attribute (pathlib.Path) points towards the folder
        where the data is stored
        """
        ### Defining the path to scan data
        self.path_to_scan_data = self.path_to_dataset.joinpath("scan_settings.json")
        ### Defining the path to the data record
        if default:
            self.path_to_data_record = self.path_to_dataset.joinpath(
                f"default_data_records.csv"
            )
        else:
            self.path_to_data_record = self.path_to_dataset.joinpath(
                f"complete_data_records.csv"
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
            - the mix (if indicated)
            - the detector it was sampled with
        """
        # Defining the mode
        self.mode = mode
        """
        The mode (str) argument is a keyword defining what mode of the dataset to use:
                         |  mode1   |   mode2  |  mode3
            Tube Voltage |   90kV   |   90kV   |  60kV
            Tube power   |    3W    |    90W   |  60W
            Filter       | Thoraeus | Thoraeus | No Filter
        """
        # Defining the pipeline
        assert pipeline in [
            "reconstruction",
            "segmentation",
            "joint",
        ], f'Wrong pipeline argument, must be in ["reconstruction", "segmentation", "joint"]'
        if pipeline in ["segmentation", "joint"]:
            assert (
                self.mode == "mode2"
            ), f"Inconsistent arguments. The mode is set on {self.mode} which does not contain the segmentation that {pipeline} pipeline requires"
        self.pipeline = pipeline
        """
        The pipeline (str) argument is a keyword defining what is the dataset used for:
            - pipeline == 'reconstruction' -> the dataset returns the sinogram and the reconstruction
            - pipeline == 'segmentation' -> the dataset returns the reconstruction and the segmentation
            - pipeline == 'joint' -> the dataset returns the sinogram, the reconstruction and the segmentation
        """
        # Defining the training mode
        self.train = train
        """
        The train (bool) argument defines if the dataset is used for training or testing
        """
        self.training_proportion = training_proportion
        # Defining the train proportion
        """
        The training_proportion (float) argument defines the proportion of the training dataset used for training
        """
        self.transforms = transforms

        ### We query the dataset subset
        self.slice_dataframe: pd.DataFrame
        """
        The slice_dataframe (pd.Dataframe) is the subset of interest of the dataset.
        The self.data_record argument becomes the slice_dataframe once we have queried it
        Example of query: 'detector==1'
        If the no query argument is passed, data_record_subset == data_record
        """
        if query:
            self.slice_dataframe = self.data_record.query(query)
        else:
            self.slice_dataframe = self.data_record

        ### We split the dataset between training and testing
        self.compute_sample_dataframe()
        self.sample_dataframe: pd.DataFrame
        """
        The sample_dataframe (pd.Dataframe) is a dataframe linking sample index to slices.
        It is used to partition the dataset on a sample basis, rather than on a slice basis,
        avoiding 'data leakage' between training and testing
        """
        if self.train:
            self.sample_dataframe = self.sample_dataframe.head(
                int(len(self.sample_dataframe) * self.training_proportion)
            )
        else:
            self.sample_dataframe = self.sample_dataframe.tail(
                int(len(self.sample_dataframe) * (1 - self.training_proportion))
            )

        self.slice_dataframe = self.slice_dataframe[
            self.slice_dataframe["sample_index"].isin(
                self.sample_dataframe["sample_index"].unique()
            )
        ]

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
        path_to_slice = self.path_to_dataset.joinpath(
            f"{slice_row['slice_identifier']}/{self.mode}"
        )
        reconstruction = torch.from_numpy(
            np.load(path_to_slice.joinpath("reconstruction.npy"))
        ).unsqueeze(0)
        if self.pipeline == "segmentation":
            segmentation = torch.from_numpy(
                np.load(path_to_slice.joinpath("reconstruction.npy"))
            ).unsqueeze(0)
            tensor_dict = {
                "reconstruction": reconstruction,
                "segmentation": segmentation,
            }
        if self.pipeline in "reconstruction":
            sinogram = torch.from_numpy(
                np.load(path_to_slice.joinpath("sinogram.npy"))
            ).unsqueeze(0)
            tensor_dict = {"reconstruction": reconstruction, "sinogram": sinogram}
        elif self.pipeline == "joint":
            segmentation = torch.from_numpy(
                np.load(path_to_slice.joinpath("reconstruction.npy"))
            ).unsqueeze(0)
            sinogram = torch.from_numpy(
                np.load(path_to_slice.joinpath("sinogram.npy"))
            ).unsqueeze(0)
            tensor_dict = {
                "reconstruction": reconstruction,
                "segmentation": segmentation,
                "sinogram": sinogram,
            }
        else:
            raise ValueError("Wrong pipeline argument")

        if self.transforms:
            tensor_dict = self.transforms(tensor_dict)
        return tensor_dict
