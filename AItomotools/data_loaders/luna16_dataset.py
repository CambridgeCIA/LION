import torch
from torch.utils.data import Dataset

from AItomotools.utils.paths import LUNA_PROCESSED_DATASET_PATH
from AItomotools.CTtools.ct_geometry import Geometry


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
        self.sinogram_transform = sinogram_transform
        self.image_transform = image_transform

        self.samples_path = LUNA_PROCESSED_DATASET_PATH.joinpath(mode)
        self.images_list = list(self.samples_path.glob("image_*"))

    def __len__(self):
        return len(self.sinograms_list)

    def __getitem__(self, index):

        image: torch.Tensor = torch.load(self.images_list[index])
        assert image.size() == torch.Size(
            self.geometry.image_shape
        ), f"Queried image size {image.size()} != expected size from geometry {self.geometry.image_shape}"
        if self.sinogram_transform is not None:
            sinogram = self.sinogram_transform(sinogram)
        if self.image_transform is not None:
            image = self.image_transform(image)

        return sinogram.float().to(self.device), image.float().to(self.device)

    def get_clean_sinogram(self, index):
        sinogram: torch.Tensor = torch.load(self.sinograms_list[index]).unsqueeze(0)
        return sinogram.to(self.device)

    def set_sinogram_transform(self, sinogram_transform):
        self.sinogram_transform = sinogram_transform
