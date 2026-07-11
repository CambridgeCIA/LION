from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import torch
import warnings
from LION.models.LIONmodel import ModelInputType


class SURE(nn.Module):
    def __init__(self, noise_std: float, epsilon: Optional[float] = None) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.noise_std = noise_std
        self.mse = nn.MSELoss()
        warnings.warn(
            "SURE expects Gaussian noise, which is not the case in noisy recosntruction of CT, so this may not work as expected"
        )
        raise NotImplementedError(
            "This is not working as expected, it is not implemented for CT reconstruction. See issue #144 to develop this"
        )

    def forward(self, noisy, model):

        if model.get_input_type() != ModelInputType.IMAGE:
            raise NotImplementedError(
                "Generalized SURE loss is not implemented yet, this only works for denoising networks"
            )

        epsilon = (
            self.epsilon if self.epsilon is not None else self.default_epsilon(noisy)
        )

        b = torch.normal(0.0, 1.0, noisy.shape).to(noisy.device)
        model_y = model(noisy)
        mc_div = torch.mean(
            b * (model(noisy + epsilon * b) - model_y) / epsilon, dim=(1, 2, 3)
        )

        return self.mse(noisy, model_y) - torch.sum(
            self.noise_std**2 + 2 * (self.noise_std**2) * mc_div, dim=0
        )  # the sum over the batch is already being applied by mse

    @staticmethod
    def default_epsilon(y):
        return torch.max(y) / 1000

    @staticmethod
    def cite(cite_format="MLA"):

        if cite_format == "MLA":
            print("Metzler, Christopher A., et al.")
            print('"Unsupervised learning with Steins unbiased risk estimator."')
            print("\x1B[3m  arXiv preprint arXiv:1805.10531 \x1B[0m")
            print("2018")

        elif cite_format == "bib":
            string = """
            @article{metzler2018unsupervised,
            title={Unsupervised learning with Stein's unbiased risk estimator},
            author={Metzler, Christopher A and Mousavi, Ali and Heckel, Reinhard and Baraniuk, Richard G},
            journal={arXiv preprint arXiv:1805.10531},
            year={2018}
            }"""
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )
