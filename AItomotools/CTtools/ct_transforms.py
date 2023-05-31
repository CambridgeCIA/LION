from typing import List, Dict

import torch
import torch.nn as nn
from torchvision.transforms import Compose

"""
class TransfromTemplate(object):
    def __init__(self):
        pass

    def __call__(self, sample:torch.Tensor) -> torch.Tensor:
        return sample
"""


class FieldsCorrect(object):
    def __init__(
        self, sinogram_size: torch.Size, flat_field=None, dark_field=None
    ) -> None:
        if dark_field is None:
            self.dark_field = torch.zeros(sinogram_size)
        else:
            self.dark_field = dark_field
        if flat_field is None:
            self.flat_field = torch.ones(sinogram_size)
        else:
            self.flat_field = flat_field

    def __call__(self, input_dict: Dict) -> Dict:
        input_dict.update(
            {
                "sinogram": (input_dict["sinogram"] - self.dark_field)
                / (self.flat_field - self.dark_field)
            }
        )
        return input_dict


class FieldsUncorrect(object):
    def __init__(
        self, sinogram_size: torch.Size, flat_field=None, dark_field=None
    ) -> None:
        if dark_field is None:
            self.dark_field = torch.zeros(sinogram_size)
        else:
            self.dark_field = dark_field
        if flat_field is None:
            self.flat_field = torch.ones(sinogram_size)
        else:
            self.flat_field = flat_field

    def __call__(self, input_dict: Dict) -> Dict:
        input_dict.update(
            {
                "sinogram": input_dict["sinogram"] * (self.flat_field - self.dark_field)
                + self.dark_field
            }
        )
        return input_dict


class Exponentiate(object):
    def __init__(self, IO: int):
        self.IO = IO

    def __call__(self, input_dict: Dict) -> Dict:
        input_dict.update(
            {
                "sinogram": self.IO
                * torch.exp(-input_dict["sinogram"] / input_dict["max_value"])
            }
        )
        return input_dict


class Logarithm(object):
    def __init__(self, IO: int):
        self.IO = IO

    def __call__(self, input_dict: Dict) -> torch.Tensor:
        sinogram = (
            -torch.log(input_dict["sinogram"] / self.IO) * input_dict["max_value"]
        )
        sinogram[sinogram < 0] = 0
        return sinogram


class PoissonNoise(object):
    def __init__(self):
        pass

    def __call__(self, input_dict: Dict) -> Dict:
        input_dict.update({"sinogram": torch.poisson(input_dict["sinogram"])})
        return input_dict


class DetectorCrossTalk(object):
    def __init__(self, cross_talk: float):
        kernel = torch.tensor(
            [[0.0, 0.0, 0.0], [cross_talk, 1, cross_talk], [0.0, 0.0, 0.0]]
        ).view(1, 1, 3, 3).repeat(1, 1, 1, 1) / (1 + 2 * cross_talk)

        self.conv = nn.Conv2d(1, 1, 3, bias=False, padding="same")
        with torch.no_grad():
            self.conv.weight = nn.Parameter(kernel)

    def __call__(self, input_dict: Dict) -> Dict:
        input_dict.update({"sinogram": self.conv(input_dict["sinogram"])})
        return input_dict


class ElectronicNoise(object):
    def __init__(self, sigma: int):
        self.sigma = sigma

    def __call__(self, input_dict: Dict) -> Dict:
        noisy_tensor = (
            input_dict["sinogram"]
            + self.sigma * torch.zeros(size=input_dict["sinogram"].size()).normal_()
        )
        noisy_tensor[noisy_tensor <= 0] = 1e-6
        input_dict.update({"sinogram": noisy_tensor})
        return input_dict


class CompleteSinogramTransform(object):
    def __init__(
        self,
        I0: int,
        sinogram_size: torch.Size,
        sigma: int,
        cross_talk: float,
        flat_field=None,
        dark_field=None,
    ):
        self.sinogram_transform = Compose(
            [
                Exponentiate(I0),
                FieldsUncorrect(sinogram_size, flat_field, dark_field),
                PoissonNoise(),
                DetectorCrossTalk(cross_talk),
                ElectronicNoise(sigma),
                FieldsCorrect(sinogram_size, flat_field, dark_field),
                Logarithm(I0),
            ]
        )

    def __call__(self, sinogram: torch.Tensor):
        assert torch.is_tensor(sinogram) == True, f"Torch Tensor expected"
        max_value = torch.max(sinogram)
        sample_dict = {"sinogram": sinogram, "max_value": max_value}
        return self.sinogram_transform(sample_dict)
