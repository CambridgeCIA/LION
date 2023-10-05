# =============================================================================
# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Emilien Valat
# Modifications: Michelle Limbach, Ander Biguri
# =============================================================================


from typing import List, Dict
import pathlib
import random
import math

import torch
import numpy as np
import json
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


from AItomotools.utils.paths import LIDC_IDRI_PROCESSED_DATASET_PATH
import AItomotools.CTtools.ct_utils as ct
from AItomotools.utils.parameter import Parameter


def format_index(index: int) -> str:
    str_index = str(index)
    while len(str_index) < 4:
        str_index = "0" + str_index
    assert len(str_index) == 4
    return str_index


def load_json(file_path: pathlib.Path):
    if not file_path.is_file():
        raise FileNotFoundError(f"No file found at {file_path}")
    return json.load(open(file_path))


def choose_random_annotation(
    nodule_annotations_list: List,
) -> str:
    return random.choice(nodule_annotations_list)


def create_consensus_annotation(
    path_to_patient_folder: pathlib.Path,
    slice_index: int,
    nodule_index: str,
    nodule_annotations_list: List,
    clevel: float,
) -> torch.int16:
    masks = []
    if isinstance(path_to_patient_folder, str):
        path_to_patient_folder = pathlib.Path(path_to_patient_folder)
    for annotation in nodule_annotations_list:
        path_to_mask = path_to_patient_folder.joinpath(
            f"mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation}.npy"
        )
        current_annotation_mask = np.load(path_to_mask)
        masks.append(current_annotation_mask)

    nodule_mask = torch.from_numpy(np.mean(masks, axis=0) >= clevel)
    return nodule_mask


class LIDC_IDRI(Dataset):
    def __init__(
        self,
        mode,
        geometry_parameters: ct.Geometry = None,
        parameters: Parameter = None,
    ):
        """
        Initializes LIDC-IDRI dataset.

        Parameters:
            - device (torch.device): Selects the device to use for the data loader.
            - task (str): Defines pipeline on how to use data. Distinguish between "joint", "end_to_end", "segmentation", "reconstruction" and "diagnostic".
                          Dataset will return, for each task:
                          "segmentation"    -> (image, segmentation_label)
                          "reconstruction"  -> (sinogram, image_label)
                          "diagnostic"      -> (segmented_nodule, diagnostic_label)
                          "joint"           -> ?????
                          "end_to_end"      -> ?????
            - training_proportion (float): Defines training % of total data.
            - mode (str): Defines "training" or "testing" mode.
            - Task (str): Defines what task is the Dataset being used for. "segmentation" (default) returns (gt_image,segmentation) pairs while "reconstruction" returns (sinogram, gt_image) pairs
            - annotation (str): Defines what annotation mode to use. Distinguish between "random" and "consensus". Default "consensus"
            - max_num_slices_per_patient (int): Defines the maximum number of slices to take per patient. Default is -1, which takes all slices we have of each patient and pcg_slices_nodule gets ignored.
            - pcg_slices_nodule (float): Defines percentage of slices with nodule in dataset. 0 meaning "no nodules at all" and 1 meaning "just take slices that contain annotated nodules". Only used if max_num_slices_per_patient != -1. Default is 0.5.
            - clevel (float): Defines consensus level if annotation=consensus. Value between 0-1. Default is 0.5.
            - geo: Geometry() type, if sinograms are requied (e.g. fo "reconstruction")

        """

        # Input parsing
        assert mode in [
            "testing",
            "training",
            "validation",
        ], f'Mode argument {mode} not in ["testing", "training", "validation"]'

        if parameters is None:
            parameters = LIDC_IDRI.default_parameters()
        self.params = parameters

        task = self.params.task
        assert task in [
            "joint",
            "end_to_end",
            "segmentation",
            "reconstruction",
            "diagnostic",
        ], f'task argument {task} not in ["joint", "end_to_end", "segmentation", "reconstruction", "diagnostic"]'

        if task not in ["segmentation", "reconstruction"]:
            raise NotImplementedError(f"task {task} not implemented yet")

        if (
            task in ["reconstruction"]
            and geometry_parameters is None
            and self.params.geo is None
        ):
            raise ValueError("geo input required for recosntruction modes")

        # Aux variable setting
        self.sinogram_transform = None
        self.image_transform = None
        self.device = self.params.device

        if task in ["reconstruction"]:
            self.image_transform = ct.from_HU_to_mu

        if geometry_parameters is not None:
            self.params.geo = geometry_parameters
            self.operator = ct.make_operator(geometry_parameters)
        elif self.params.geo is not None:
            self.operator = ct.make_operator(self.params.geo)
        # Start of Patient pre-processing

        self.path_to_processed_dataset = pathlib.Path(self.params.folder)
        self.patients_masks_dictionary = load_json(
            self.path_to_processed_dataset.joinpath("patients_masks.json")
        )

        # Add patients without masks, for now hardcoded, find a solution in preprocessing
        self.patients_masks_dictionary["LIDC-IDRI-0238"] = {}
        self.patients_masks_dictionary["LIDC-IDRI-0585"] = {}

        self.patients_diagnosis_dictionary = load_json(
            self.path_to_processed_dataset.joinpath("patient_id_to_diagnosis.json")
        )
        self.total_patients = (
            len(list(self.path_to_processed_dataset.glob("LIDC-IDRI-*"))) + 1
        )
        self.num_slices_per_patient = self.params.max_num_slices_per_patient
        self.pcg_slices_nodule = self.params.pcg_slices_nodule
        self.annotation = self.params.annotation
        self.clevel = (
            self.params.clevel
        )  # consensus level, only used if annotation == consensus
        self.patient_index_to_n_slices_dict: Dict = {
            f"LIDC-IDRI-{format_index(index)}": len(
                list(
                    self.path_to_processed_dataset.joinpath(
                        f"LIDC-IDRI-{format_index(index)}"
                    ).glob("slice_*.npy")
                )
            )
            for index in range(1, self.total_patients)
        }

        # Dict with all slices of each patient
        self.patient_index_to_slices_index_dict: Dict = {
            f"LIDC-IDRI-{format_index(index)}": list(
                np.arange(
                    0,
                    self.patient_index_to_n_slices_dict[
                        f"LIDC-IDRI-{format_index(index)}"
                    ],
                    1,
                )
            )
            for index in range(1, self.total_patients)
        }

        # Dict with all nodule slices of each patient
        # Converts the keys from self.patients_masks_dictionary to integer
        self.patient_index_to_nodule_slices_index_dict: Dict = {
            f"LIDC-IDRI-{format_index(index)}": [
                int(item)
                for item in list(
                    self.patients_masks_dictionary[
                        f"LIDC-IDRI-{format_index(index)}"
                    ].keys()
                )
            ]
            for index in range(1, self.total_patients)
        }

        # Dict with all non-nodule slices of each patient
        # Computes as difference of all slices dict and dict with nodules
        self.patient_index_to_non_nodule_slices_index_dict: Dict = {
            f"LIDC-IDRI-{format_index(index)}": list(
                set(
                    self.patient_index_to_slices_index_dict[
                        f"LIDC-IDRI-{format_index(index)}"
                    ]
                )
                - set(
                    self.patient_index_to_nodule_slices_index_dict[
                        f"LIDC-IDRI-{format_index(index)}"
                    ]
                )
            )
            for index in range(1, self.total_patients)
        }

        # Corrupted data handling
        # Delete all slices that contain a nodule that has more than 4 annotations
        self.removed_slices: Dict = {}
        for (
            patient_id,
            nodule_slices_list,
        ) in self.patient_index_to_nodule_slices_index_dict.items():
            self.removed_slices[patient_id] = []
            for slice_index in nodule_slices_list:
                all_nodules_dict: Dict = self.patients_masks_dictionary[patient_id][
                    f"{slice_index}"
                ]
                for _, nodule_annotations_list in all_nodules_dict.items():
                    if len(nodule_annotations_list) > 4:
                        self.removed_slices[patient_id].append(slice_index)
                        break

        self.patient_index_to_nodule_slices_index_dict: Dict = {
            f"LIDC-IDRI-{format_index(index)}": list(
                set(
                    self.patient_index_to_nodule_slices_index_dict[
                        f"LIDC-IDRI-{format_index(index)}"
                    ]
                )
                - set(self.removed_slices[f"LIDC-IDRI-{format_index(index)}"])
            )
            for index in range(1, self.total_patients)
        }

        ##% Divide dataset in training/validation/testing
        self.training_proportion = self.params.training_proportion
        self.validation_proportion = self.params.validation_proportion
        self.params.mode = mode
        # Commpute number if images for each
        self.n_patients_training = math.floor(
            self.training_proportion * (self.total_patients)
        )
        self.n_patients_validation = math.floor(
            self.validation_proportion * (self.total_patients)
        )
        self.n_patients_testing = (
            self.total_patients - self.n_patients_training - self.n_patients_validation
        )

        assert self.total_patients == (
            self.n_patients_training
            + self.n_patients_testing
            + self.n_patients_validation
        ), print(
            f"Total patients: {self.total_patients}, \n training patients {self.n_patients_training}, \n validation patients {self.n_patients_validation}, \n testing patients {self.n_patients_testing}"
        )

        # Get patient IDs for each
        self.patient_ids = list(self.patient_index_to_n_slices_dict.keys())
        self.training_patients_list = self.patient_ids[: self.n_patients_training]
        self.validation_patients_list = self.patient_ids[
            self.n_patients_training : self.n_patients_training
            + self.n_patients_validation
            - 1
        ]
        self.testing_patients_list = self.patient_ids[
            self.n_patients_training + self.n_patients_validation - 1 :
        ]

        assert len(self.patient_ids) == len(self.training_patients_list) + len(
            self.testing_patients_list
        ) + len(self.validation_patients_list), print(
            f"Len patients ids: {len(self.patient_ids)}, \n len training patients {len(self.training_patients_list)},\n len validation patients {len(self.validation_patients_list)}, \n len testing patients {len(self.testing_patients_list)}"
        )

        print("Preparing patient list, this may take time....")
        if self.params.mode == "training":
            patient_list_to_load = self.training_patients_list
        elif self.params.mode == "validation":
            patient_list_to_load = self.validation_patients_list
        elif self.params.mode == "testing":
            patient_list_to_load = self.testing_patients_list
        else:
            raise NotImplementedError(
                f"mode {self.params.mode} not implemented, try training, validation or testing"
            )

        self.slices_to_load = self.get_slices_to_load(
            patient_list_to_load,
            self.patient_index_to_non_nodule_slices_index_dict,
            self.patient_index_to_nodule_slices_index_dict,
            self.num_slices_per_patient,
            self.pcg_slices_nodule,
        )
        self.slice_index_to_patient_id_list = self.get_slice_index_to_patient_id_list(
            self.slices_to_load
        )
        self.patient_id_to_first_index_dict = self.get_patient_id_to_first_index_dict(
            self.slices_to_load
        )

        print(f"Patient lists ready for {self.params.mode} dataset")

    @staticmethod
    def default_parameters(geo=None, task="reconstruction"):
        param = Parameter()
        param.name = "LIDC-IDRI Data Loader"
        param.training_proportion = 0.8
        param.validation_proportion = 0.1
        param.testing_proportion = (
            1 - param.training_proportion - param.validation_proportion
        )  # not used, but for metadata
        param.max_num_slices_per_patient = -1  # i.e. all
        param.pcg_slices_nodule = 0.5
        param.task = task
        param.folder = LIDC_IDRI_PROCESSED_DATASET_PATH
        if task == "reconstruction" and geo is None:
            raise ValueError(
                "For reconstruction task geometry needs to be input to default_parameters(geo=geometry_param)"
            )

        # segmentation specific
        param.clevel = 0.5
        param.annotation = "consensus"
        param.device = torch.cuda.current_device()
        param.geo = geo
        return param

    def get_slices_to_load(
        self,
        patient_list: List,
        non_nodule_slices_dict: Dict,
        nodule_slices_dict: Dict,
        num_slices_per_patient: int,
        pcg_slices_nodule: float,
    ):
        """
        Returns a dictionary that contains patient_id's as keys and list of slices to load as values for each patient.

        Parameters:
            - patient_list (List): List that contains patient_id of all patients.
            - non_nodule_slices_dict (Dict): Dict that contains all slices without nodule of each patient_id.
            - nodule_slices_dict (Dict): Dict that contains all slices with nodule of each patient_id.
            - num_slices_per_patient (int): Defines maximum number of slices we want per patient. If num_slices_per_patient=-1 take all slices we have of each patient.
            - pcg_slices_nodule (float): Defines amount of slices that should contain a nodule. Value between 0-1. Only used if num_slices_per_patient != -1.
        Returns:
            - patient_id_to_slices_to_load_dict which contains patient_id as key and list of slices to load as values
        """

        patient_id_to_slices_to_load_dict = (
            {}
        )  # Empty dict which should contain patient id as key and slice ids as array of values

        if (
            num_slices_per_patient == -1
        ):  # Default: Take all slices we have of each patient
            for patient_id in patient_list:  # Loop over every patient

                # Get non-nodule slices and nodule slices of each patient and afterwards sort the list in increasing order
                patient_id_to_slices_to_load_dict[patient_id] = (
                    non_nodule_slices_dict[patient_id] + nodule_slices_dict[patient_id]
                )
                patient_id_to_slices_to_load_dict[patient_id].sort()

        else:  # Take maximum of n slices per patient with p% containing a nodule
            for patient_id in patient_list:  # Loop over every patient
                number_of_slices = min(
                    num_slices_per_patient,
                    min(
                        len(non_nodule_slices_dict[patient_id]),
                        len(nodule_slices_dict[patient_id]),
                    ),
                )

                # Get amount of slices we want without nodule
                number_of_slices_without_nodule = int(
                    np.ceil(number_of_slices * (1 - pcg_slices_nodule))
                )
                # Get amount of slices we want with nodule
                number_of_slices_with_nodule = (
                    number_of_slices - number_of_slices_without_nodule
                )

                # Get linspace of non-nodule and nodule slices of each patient and afterwards sort the list in increasing order
                patient_id_to_slices_to_load_dict[patient_id] = list(
                    np.array(non_nodule_slices_dict[patient_id])[
                        np.linspace(
                            0,
                            len(non_nodule_slices_dict[patient_id]),
                            number_of_slices_without_nodule,
                            dtype=int,
                            endpoint=False,
                        )
                    ]
                ) + list(
                    np.array(nodule_slices_dict[patient_id])[
                        np.linspace(
                            0,
                            len(nodule_slices_dict[patient_id]),
                            number_of_slices_with_nodule,
                            dtype=int,
                            endpoint=False,
                        )
                    ]
                )
                patient_id_to_slices_to_load_dict[patient_id].sort()

        return patient_id_to_slices_to_load_dict

    def get_patient_id_to_first_index_dict(self, patient_with_slices_to_load: Dict):
        """
        Returns a dictionary that contains patient_id's as keys and start index of each patient in self.slice_index_to_patient_id_list as value.

        Parameters:
            - patient_with_slices_to_load (Dict): Dict that defines which slices to load per patient.
        Returns:
            - patient_id_to_first_index_dict (Dict): Defines start index of each patient in self.slice_index_to_patient_id_list. Needed for mapping of global index to slice index.
        """
        patient_id_to_first_index_dict = {}
        global_index = 0
        for patient_id in patient_with_slices_to_load:
            path_to_folder = self.path_to_processed_dataset.joinpath(patient_id)
            patient_id_to_first_index_dict[patient_id] = global_index

            if len(patient_with_slices_to_load[patient_id]) < len(
                list(path_to_folder.glob("slice_*.npy"))
            ):
                global_index += len(patient_with_slices_to_load[patient_id])
            else:
                global_index += len(list(path_to_folder.glob("slice_*.npy")))
        return patient_id_to_first_index_dict

    def get_slice_index_to_patient_id_list(self, patient_with_slices_to_load: Dict):
        """
        Returns a list that contains "number of slices" times each patient id.

        Parameters:
            - patient_with_slices_to_load (Dict): Dict that defines which slices to load per patient.
        Returns:
            - slice_index_to_patient_id_list (List): Contains number of slices times each patient id. Needed for mapping of global index to slice index.
        """
        slice_index_to_patient_id_list = []
        for patient_id in patient_with_slices_to_load:
            path_to_folder = self.path_to_processed_dataset.joinpath(patient_id)

            if len(patient_with_slices_to_load[patient_id]) < len(
                list(path_to_folder.glob("slice_*.npy"))
            ):
                n_slices = len(patient_with_slices_to_load[patient_id])
            else:
                n_slices = len(list(path_to_folder.glob("slice_*.npy")))

            for slice_index in range(n_slices):
                slice_index_to_patient_id_list.append(patient_id)
        return slice_index_to_patient_id_list

    def get_reconstruction_tensor(self, file_path: pathlib.Path) -> torch.Tensor:
        tensor = torch.from_numpy(np.load(file_path)).unsqueeze(0).to(self.device)
        return tensor

    def set_sinogram_transform(self, sinogram_transform):
        self.sinogram_transform = sinogram_transform

    def set_image_transform(self, image_transform):
        self.image_transform = image_transform

    def compute_clean_sinogram(self, image=None) -> torch.Tensor:

        if self.operator is None:
            raise AttributeError("CT operator not know. Have you given a ct geometry?")
        sinogram = self.operator(image)
        return sinogram

    def get_mask_tensor(self, patient_id: str, slice_index: int) -> torch.Tensor:
        ## First, assess if the slice has a nodule
        try:
            mask = torch.zeros((512, 512), dtype=torch.bool)
            all_nodules_dict: Dict = self.patients_masks_dictionary[patient_id][
                f"{slice_index}"
            ]
            for nodule_index, nodule_annotations_list in all_nodules_dict.items():

                if self.annotation == "random":
                    ## If a nodule was not segmented by all the clinicians, the other annotations should not always be seen
                    while len(nodule_annotations_list) < 4:
                        nodule_annotations_list.append("")

                    annotation = choose_random_annotation(nodule_annotations_list)
                    if annotation == "":
                        # Hopefully, that exists the try to return an empty mask
                        nodule_mask = torch.zeros((512, 512), dtype=torch.bool)
                    else:
                        path_to_mask = self.path_to_processed_dataset.joinpath(
                            f"{patient_id}/mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation}.npy"
                        )
                        # print(path_to_mask)
                        nodule_mask = torch.from_numpy(np.load(path_to_mask))

                elif self.annotation == "consensus":
                    # Create consensus annotation out of all annotations of this nodule
                    path_to_patient_folder = self.path_to_processed_dataset.joinpath(
                        f"{patient_id}/"
                    )
                    nodule_mask = create_consensus_annotation(
                        path_to_patient_folder,
                        slice_index,
                        nodule_index,
                        nodule_annotations_list,
                        self.clevel,
                    )

                else:
                    raise NotImplementedError(
                        f"annotation {self.annotation} not implemented, try random or consensus"
                    )

                mask = mask.bitwise_or(nodule_mask)

        except KeyError:
            mask = torch.zeros((512, 512), dtype=torch.bool)
        # byte inversion
        background = ~mask
        return torch.stack((background, mask))

    def __len__(self):
        return len(self.slice_index_to_patient_id_list)

    def get_specific_slice(self, patient_index, slice_index):
        ## Assumes slice and mask exist
        file_path = self.path_to_processed_dataset.joinpath(
            f"{patient_index}/slice_{slice_index}.npy"
        )
        return self.get_reconstruction_tensor(file_path), self.get_mask_tensor(
            patient_index, slice_index
        )

    def __getitem__(self, index):
        patient_id = self.slice_index_to_patient_id_list[index]
        first_slice_index = self.patient_id_to_first_index_dict[patient_id]
        dict_slice_index = index - first_slice_index
        slice_index_to_load = self.slices_to_load[patient_id][dict_slice_index]
        # print(f'Index, {index}, Patient Id : {patient_id}, first_slice_index : {first_slice_index}, slice_index : {dict_slice_index} ', slice_to_load : {slice_index_to_load})
        file_path = self.path_to_processed_dataset.joinpath(
            f"{patient_id}/slice_{slice_index_to_load}.npy"
        )

        if self.params.task in [
            "joint",
            "end_to_end",
            "segmentation",
            "reconstruction",
        ]:
            reconstruction_tensor = self.get_reconstruction_tensor(file_path)
            if self.image_transform is not None:
                reconstruction_tensor = self.image_transform(reconstruction_tensor)

        if self.params.task in ["joint", "end_to_end", "segmentation"]:
            mask_tensor = self.get_mask_tensor(patient_id, slice_index_to_load)
            return reconstruction_tensor, mask_tensor

        elif self.params.task == "reconstruction":
            sinogram = self.compute_clean_sinogram(reconstruction_tensor.float())

            if self.sinogram_transform is not None:
                sinogram = self.sinogram_transform(sinogram)
            return sinogram, reconstruction_tensor

        elif self.params.task == "diagnostic":
            return self.patients_diagnosis_dictionary[patient_id]
        else:
            raise NotImplementedError
