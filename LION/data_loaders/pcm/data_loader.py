"""Load PCM data."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Literal, TypeVar

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch.utils.data import Dataset

from LION.data_loaders.pcm.data_primitives import PCMDataCategory, PCMDataParameter

PCMDatasetName = Literal[
    "synthetic_png", "example_images_256", "example_images_512", "example_measurements"
]
GrayscaleImage2D = Float[torch.Tensor, "height width"]
Measurement1D = Float[torch.Tensor, "num_measurements"]

ItemType = TypeVar("ItemType")


class PCMSyntheticPNGDataset(Dataset, ABC, Generic[ItemType]):
    """Dataset for PCM synthetic PNG data loader."""

    def __init__(self, data_param: PCMDataParameter) -> None:
        """Initialize the dataset."""
        assert data_param.samples is not None
        assert data_param.samples.ids is not None
        self.sample_ids = data_param.samples.ids
        assert data_param.categories is not None
        self.categories = data_param.categories
        self.data_dir = data_param.root

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        num_categories = len(self.categories)
        num_ids_per_category = len(self.sample_ids)
        return num_categories * num_ids_per_category

    def category_dir(self, category: PCMDataCategory) -> Path:
        """Get the directory for a given category.

        The category directory is assumed to be a subdirectory of the root data directory, named after the category.
        """
        return self.data_dir / category

    def filename(self, sample_id: int) -> str:
        """Get the filename for a given sample ID.

        The filename is in the format "imageXXXX.png" where XXXX is the zero-padded sample ID.
        """
        return f"image{sample_id:04d}.png"

    def sample_png_path(self, category: PCMDataCategory, sample_id: int) -> Path:
        """Get the path to the PNG file for a given category and sample ID."""
        return self.category_dir(category) / self.filename(sample_id)

    @abstractmethod
    def make_item(self, sample_png_path: Path) -> ItemType:
        """Create a dataset item from the PNG file path."""

    def __getitem__(self, index: int) -> ItemType:
        """Get the sample at the given index."""
        num_ids_per_category = len(self.sample_ids)
        category = self.categories[index // num_ids_per_category]
        sample_id = self.sample_ids[index % num_ids_per_category]
        sample_png_path = self.sample_png_path(category, sample_id)
        return self.make_item(sample_png_path)


class PCMSyntheticPNGTorchDataset(PCMSyntheticPNGDataset[GrayscaleImage2D]):
    """Dataset for PCM synthetic PNG grayscale images loader."""

    def make_item(self, sample_png_path: Path) -> GrayscaleImage2D:
        """Create a torch tensor."""
        sample_np = np.array(Image.open(sample_png_path).convert("L")) / 255.0
        return torch.from_numpy(sample_np).float()
