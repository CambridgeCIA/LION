from typing import Union
import pathlib
from statistics import mean

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.paths import WALNUT_DATASET_PATH


def parse_index(index: int) -> str:
    index_str = str(index)

    while len(index_str) != 6:
        index_str = "0" + index_str
    return index_str


def load_tif(tif_path: pathlib.Path) -> np.ndarray:
    return imageio.v2.imread(tif_path)


def standard_transform(np_array: np.ndarray) -> np.ndarray:
    return np.transpose(np.flipud(np_array))


def load_and_transform(tif_path: pathlib.Path) -> np.ndarray:
    return standard_transform(load_tif(tif_path))


def negative_log(linearised_measurement: np.ndarray) -> np.ndarray:
    return np.negative(np.log(linearised_measurement))


def to_tensor(np_array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np_array).unsqueeze(0)


class WalnutDataset(Dataset):
    def __init__(
        self,
        dimension: int,
        device: torch.cuda.device,
        walnut_index: int,
        tube_index: int,
        middle_slices_only=True,
        transform=None,
    ) -> None:
        assert 1 <= walnut_index <= 42
        assert 1 <= tube_index <= 3
        assert 2 <= dimension <= 3

        if tube_index != 2 and dimension == 2:
            raise ValueError(
                f"Must have tube_index == 2 for dimension == 2, currently is {tube_index}"
            )

        self.dimension = dimension
        self.device = device
        self.middle_slices_only = middle_slices_only

        walnut_path = WALNUT_DATASET_PATH.joinpath(f"Walnut{walnut_index}")
        projections_path = walnut_path.joinpath("Projections")
        reconstructions_path = walnut_path.joinpath("Reconstructions")
        self.tube_path = projections_path.joinpath(f"tubeV{tube_index}")
        self.corrected_geometry_path = self.tube_path.joinpath(
            f"scan_geom_corrected.geom"
        )

        self.transform = transform
        dark_field_path = self.tube_path.joinpath(f"di000000.tif")
        flat_field_1_path = self.tube_path.joinpath(f"io000000.tif")
        flat_field_2_path = self.tube_path.joinpath(f"io000001.tif")

        self.flat_field = 0.5 * (
            load_and_transform(flat_field_1_path)
            + load_and_transform(flat_field_2_path)
        )
        self.dark_field = load_and_transform(dark_field_path)

        if self.dimension == 2:
            self.get_distances()
            self.reconstruction = load_tif(
                reconstructions_path.joinpath(f"full_AGD_50_000250.tiff")
            )
        else:
            raise NotImplementedError(f"Not implemented for dimension != 2")

    def get_distances(self):
        assert self.dimension == 2
        source_origin_list = []
        origin_det_list = []
        for angle in np.loadtxt(self.corrected_geometry_path):
            source_origin_list.append(
                (angle[0] ** 2 + angle[1] ** 2 + angle[2] ** 2) ** 0.5
            )
            origin_det_list.append(
                (angle[3] ** 2 + angle[4] ** 2 + angle[5] ** 2) ** 0.5
            )
        self.source_origin = mean(source_origin_list)
        self.origin_det = mean(origin_det_list)

    def linearise(self, measurement: np.ndarray) -> np.ndarray:
        return (measurement - self.dark_field) / (self.flat_field - self.dark_field)

    def measurement_transform(self, measurement_path: pathlib.Path):
        return to_tensor(
            negative_log(self.linearise(load_and_transform(measurement_path)))
        )

    def pre_process_middle_slices(self):
        assert self.dimension == 2
        for i in tqdm(range(self.__len__())):
            parsed_index = parse_index(i)
            load_path = self.tube_path.joinpath(f"scan_{parsed_index}.tif")
            save_path = self.tube_path.joinpath(f"middle_slice_{parsed_index}.pt")
            if not save_path.is_file():
                middle_slice = self.measurement_transform(load_path)[:, 486, :]
                torch.save(middle_slice, save_path)

    def make_sinogram(self) -> torch.Tensor:
        if self.dimension == 2:
            sinogram = torch.zeros((1, self.__len__(), 768))
            for i in tqdm(range(self.__len__())):
                parsed_index = parse_index(i)
                if self.tube_path.joinpath(f"middle_slice_{parsed_index}.pt").is_file():
                    sinogram[:, :, i] = torch.load(
                        self.tube_path.joinpath(f"middle_slice_{parsed_index}.pt")
                    )
                else:
                    sinogram[:, :, i] = self.measurement_transform(
                        self.tube_path.joinpath(f"scan_{parsed_index}.tif")
                    )[0, 486, :]
            return sinogram
        else:
            sinogram = torch.zeros((1, self.__len__(), 972, 768))
            for i in tqdm(range(self.__len__())):
                parsed_index = parse_index(i)
                sinogram[:, :, i] = self.measurement_transform(
                    self.tube_path.joinpath(f"scan_{parsed_index}.tif")
                )[0]
            return sinogram

    def __len__(self):
        return 1201

    def __getitem__(self, index) -> Union[torch.Tensor, float]:
        parsed_index = parse_index(index)

        if self.middle_slices_only:
            return torch.load(
                self.tube_path.joinpath(f"middle_slice_{parsed_index}.pt")
            ).to(self.device)
        else:
            measurement = self.measurement_transform(
                self.tube_path.joinpath(f"scan_{parsed_index}.tif")
            )
            if self.transform is not None:
                measurement = self.transform(measurement)

            return measurement.to(self.device)
