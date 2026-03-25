# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : T. Trung (Troy) Vu
# =============================================================================
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, cast

from LION.utils.parameter import LIONParameter


@dataclass
class PCMReconParameter(LIONParameter):
    """Common parameters for PCM reconstruction experiments.

    Parameters
    ----------
    name : str
        Name of the experiment.
    physics : LIONParameter
        Parameters for PCM forward operator.
    noise_params : LIONParameter
        Settings for sinogram noise simulation.
    data_loader_params : LIONParameter
        Parameters to build the data loader.
    """

    name: str
    physics: LIONParameter
    noise_params: LIONParameter
    data_loader_params: LIONParameter


class PCMPhysics:
    @classmethod
    def default_undersampling_parameters(cls) -> LIONParameter:
        return LIONParameter(
            randomising_scheme="uniform",
            sampling_ratios=(0.2,),
            coarse_j_values=None,
            coarse_j_offset_from_j_order=2,
            reverse_test_cases=True,
        )


PCMDataset = Literal["example_images_256", "example_images_512", "example_measurements"]


@dataclass
class PCMExperiment(ABC):
    """PCM reconstruction experiment base class.

    Defines the common interface for PCM reconstruction experiments.
    """

    def __init__(
        self,
        experiment_params: LIONParameter | None = None,
        dataset: PCMDataset = "example_images_256",
        datafolder: str | None = None,
    ):
        """Base class for PCM reconstruction experiments.

        Parameters
        ----------
        experiment_params : LIONParameter, optional
            Custom experiment parameters. If None, default parameters will be used. Default is None.
            NOT RECOMMENDED. The purpose of the experiment class is to have reliable and repeatable experiments.
            If you want to change the parameters, please create a new class with the new parameters.
        dataset : PCMDataset, optional
            The name of the dataset to be used for the experiment. See `__get_dataset` for supported datasets.
        datafolder : str or Path, optional
            The folder where the dataset is stored. If None, the default folder will be used. Default is None.
        """
        super().__init__()  # Initialize parent classes.
        if experiment_params is None:
            self.param: PCMReconParameter = cast(
                PCMReconParameter, self.default_parameters(dataset=dataset)
            )
        else:
            self.param = cast(PCMReconParameter, experiment_params)
        if datafolder is not None:
            self.param.data_loader_params.folder = datafolder
        self.dataset = dataset

    @staticmethod
    @abstractmethod  # crash if not defined in derived class
    def default_parameters(dataset: PCMDataset) -> LIONParameter:
        """Default experiment parameters."""

    def __get_dataset(self, mode: str):
        raise NotImplementedError()

    def get_training_dataset(self):
        return self.__get_dataset("train")

    def get_validation_dataset(self):
        return self.__get_dataset("validation")

    def get_testing_dataset(self):
        return self.__get_dataset("test")

    def __str__(self):
        return f"Experiment parameters: \n {self.param} \n Dataset: \n {self.dataset} \n Physics parameters: \n {self.physics}"

    @staticmethod
    def get_dataset_parameters(dataset: PCMDataset, physics=None) -> LIONParameter:
        if dataset == "example_images_256":
            dataset_param = LIONParameter(
                data_name="example_images_256",
                data_type="image",
                j_order=8,
                inverse_sign=False,
                tests_scale_ground_truth=False,
                is_out_of_distribution=True,
                r_high=2e-5,
                r_low=-2e-6,
                scale_eps=1e-12,
            )
        elif dataset == "example_images_512":
            dataset_param = LIONParameter(
                data_name="example_images_512",
                data_type="image",
                j_order=9,
                inverse_sign=False,
                tests_scale_ground_truth=False,
                is_out_of_distribution=True,
                r_high=2e-5,
                r_low=-2e-6,
                scale_eps=1e-12,
            )
        elif dataset == "example_measurements":
            dataset_param = LIONParameter(
                data_name="example_measurements",
                data_type="original_measurement_data",
                j_order=9,
                inverse_sign=True,
                tests_scale_ground_truth=False,
                is_out_of_distribution=True,
                r_high=1e-6,
                r_low=-1e-6,
                scale_eps=1e-12,
            )
        else:
            raise NotImplementedError(
                f"Dataset {dataset} not implemented. Supported datasets: 'example_images_256', 'example_images_512', 'example_measurements'."
            )
        return dataset_param


@dataclass
class PCMUndersamplingRecon(PCMExperiment):
    """PCM reconstruction experiment with 256x256 images.

    Fixed values (also see `PCMRecon.default_parameters()`):
        - ...
    """

    def __init__(self, datafolder=None):
        super().__init__(None, "example_images_256", datafolder)

    @staticmethod
    def default_parameters(dataset: PCMDataset) -> LIONParameter:
        param = LIONParameter()
        param.name = "PCM undersampling reconstruction experiment"
        param.physics = PCMPhysics.default_undersampling_parameters()
        param.noise_params = LIONParameter()
        param.data_loader_params = PCMExperiment.get_dataset_parameters(
            dataset, physics=param.physics
        )
        return param
