# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: Emilien Valat
# =============================================================================


import torch
from torch.utils.data import Dataset

from AItomotools.utils.paths import LUNA_PROCESSED_DATASET_PATH
from AItomotools.CTtools.ct_geometry import Geometry
from AItomotools.utils.parameter import Parameter
import AItomotools.CTtools.ct_utils as ct


def parse_index(index: int) -> str:
    index_str = str(index)

    while len(index_str) != 6:
        index_str = "0" + index_str
    return index_str


class Luna16Dataset(Dataset):
    def __init__(
        self,
        device: torch.device,
        mode: str,
        geo: Geometry() = None,
        sinogram_transform=None,
        image_transform=None,
    ) -> None:
        # Input parsing
        assert mode in [
            "testing",
            "training",
            "validation",
        ], f'Mode argument {mode} not in ["testing", "training", "validation"]'

        self.device = device
        self.mode = mode
        if geo is not None:
            self.operator = ct.make_operator(geo)
        self.geometry = geo
        self.sinogram_transform = sinogram_transform
        self.image_transform = image_transform

        self.samples_path = LUNA_PROCESSED_DATASET_PATH.joinpath(mode)
        self.images_list = list(self.samples_path.glob("image_*"))
        self.pre_processing_params = Parameter().load(
            LUNA_PROCESSED_DATASET_PATH.joinpath("parameter.json")
        )

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):

        image: torch.Tensor = torch.load(self.images_list[index]).to(self.device)
        sinogram = self.compute_clean_sinogram(index, image.float())
        assert image.size() == torch.Size(
            self.geometry.image_shape
        ), f"Queried image size {image.size()} != expected size from geometry {self.geometry.image_shape}"
        if self.sinogram_transform is not None:
            sinogram = self.sinogram_transform(sinogram)
        if self.image_transform is not None:
            image = self.image_transform(image)

        return sinogram.float().to(self.device), image.float().to(self.device)

    def compute_clean_sinogram(self, index, image=None):

        if self.operator is None:
            raise AttributeError("CT oeprator not know. Have you give a ct geometry?")
        if image is None:
            image: torch.Tensor = torch.load(self.images_list[index]).to(self.device)
        sinogram = self.operator(image)
        return sinogram

    def set_sinogram_transform(self, sinogram_transform):
        self.sinogram_transform = sinogram_transform
