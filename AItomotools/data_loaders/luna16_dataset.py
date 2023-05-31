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
        self.sinograms_list = list(self.samples_path.glob("sino_*"))
        self.images_list = list(self.samples_path.glob("image_*"))
        assert len(self.sinograms_list) == len(
            self.images_list
        ), f"Wrong number of files: Sinograms {len(self.sinograms_list)} != Images {len(self.images_list)}"

        self.geometry_file_path = LUNA_PROCESSED_DATASET_PATH.joinpath("geometry.json")
        if self.geometry_file_path.is_file():
            self.geometry = Geometry()
            self.geometry.load(self.geometry_file_path)
        else:
            raise FileNotFoundError(
                f"The geometry file geometry.json not found at {self.geometry_file_path}"
            )

    def __len__(self):
        return len(self.sinograms_list)

    def __getitem__(self, index):
        sinogram: torch.Tensor = torch.load(self.sinograms_list[index])
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
