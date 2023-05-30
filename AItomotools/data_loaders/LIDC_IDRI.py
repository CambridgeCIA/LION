from typing import List, Tuple, Dict
import pathlib
import random
import math

import torch
import numpy as np
import json
from torch.utils.data import Dataset

from backends.odl import ODLBackend

def format_index(index:int) -> str:
    str_index = str(index)
    while len(str_index) < 4:
        str_index = '0' + str_index
    assert len(str_index) == 4
    return str_index

def load_json(file_path:pathlib.Path):
    if not file_path.is_file():
        raise FileNotFoundError (f'No file found at {file_path}')
    return json.load(open(file_path))

class LIDC_IDRI(Dataset):
    def __init__(
            self,
            pipeline:str,
            training_proportion:float,
            mode:str,
            ):

        self.path_to_processed_dataset = pathlib.Path('/local/scratch/public/AItomotools/processed/LIDC-IDRI')
        self.patients_masks_dictionary = load_json(self.path_to_processed_dataset.joinpath('patients_masks.json'))
        self.patients_diagnosis_dictionary = load_json(self.path_to_processed_dataset.joinpath('patients_id_to_diagnosis.json'))
        self.total_patients = 1012

        self.pipeline = pipeline
        self.patient_index_to_n_slices_dict :Dict = {
            f'LIDC-IDRI-{format_index(index)}' : len(list(self.path_to_processed_dataset.joinpath(f'LIDC-IDRI-{format_index(index)}').glob('slice_*.npy'))) for index in range(1,self.total_patients)
        }
        self.training_proportion = training_proportion
        self.mode = mode

        self.n_patients_training = math.floor(self.training_proportion*self.total_patients)
        self.n_patients_testing  = math.ceil((1-self.training_proportion)*self.total_patients)
        assert self.total_patients == (self.n_patients_training + self.n_patients_testing), print(
            f'Total patients: {self.total_patients}, \n training patients {self.n_patients_training}, \n testing patients {self.n_patients_testing}'
            )

        self.patient_ids = list(self.patient_index_to_n_slices_dict.keys())
        self.training_patients_list = self.patient_ids[:self.n_patients_training]
        self.testing_patients_list = self.patient_ids[self.n_patients_training:]
        assert len(self.patient_ids) == len(self.training_patients_list) + len(self.testing_patients_list), print(
            f'Len patients ids: {len(self.patient_ids)}, \n len training patients {len(self.training_patients_list)}, \n len testing patients {len(self.testing_patients_list)}'
            )

        print('Preparing patient list, this may take time....')
        if self.mode == 'training':
            self.slice_index_to_patient_id_list = self.get_slice_index_to_patient_id_list(self.training_patients_list)
            self.patient_id_to_first_index_dict = self.get_patient_id_to_first_index_dict(self.training_patients_list)

        elif self.mode == 'testing':
            self.slice_index_to_patient_id_list = self.get_slice_index_to_patient_id_list(self.testing_patients_list)
            self.patient_id_to_first_index_dict = self.get_patient_id_to_first_index_dict(self.testing_patients_list)

        else:
            raise NotImplementedError(f'mode {self.mode} not implemented, try training or testing')

        print(f'Patient lists ready for {self.mode} dataset')

    def get_patient_id_to_first_index_dict(self, patient_list:List):
        patient_id_to_first_index_dict = {}
        global_index = 0
        for patient_id in patient_list:
            path_to_folder = self.path_to_processed_dataset.joinpath(patient_id)
            patient_id_to_first_index_dict[patient_id] = global_index
            global_index += len(list(path_to_folder.glob('slice_*.npy')))
        return patient_id_to_first_index_dict

    def get_slice_index_to_patient_id_list(self, patient_list:List):
        slice_index_to_patient_id_list = []
        for patient_id in patient_list:
            path_to_folder = self.path_to_processed_dataset.joinpath(patient_id)
            n_slices = len(list(path_to_folder.glob('slice_*.npy')))
            for slice_index in range(n_slices):
                slice_index_to_patient_id_list.append(patient_id)
        return slice_index_to_patient_id_list

    def get_reconstruction_tensor(self, file_path:pathlib.Path) -> torch.Tensor:
        tensor = torch.from_numpy(np.load(file_path)).unsqueeze(0)
        return tensor

    def get_sinogram_tensor(self, file_path:pathlib.Path, backend:ODLBackend) -> torch.Tensor:
        #### EXPENSIVE ####
        return backend.get_sinogram(self.get_reconstruction_tensor(file_path))

    def get_mask_tensor(self, patient_id:str, slice_index:int) -> torch.Tensor:
        ## First, assess if the slice has a nodule
        try:
            mask = torch.zeros((512,512))
            for nodule_index, nodule_annotations_list in enumerate(self.patients_masks_dictionary[patient_id][slice_index]):
                ## If a nodule was not segmented by all the clinicians, the other annotations should not always be seen
                while len(nodule_annotations_list) < 4:
                    nodule_annotations_list.append('')

                annotation = random.choice(nodule_annotations_list)
                if annotation == '':
                    # Hopefully, that exists the try to return an empty mask
                    nodule_mask = torch.zeros((512,512))
                else:
                    path_to_mask = self.path_to_processed_dataset.joinpath(f'mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation}.npy')
                    nodule_mask = torch.from_numpy(np.load(path_to_mask))

                mask.bitwise_and(nodule_mask)

        except KeyError:
            mask = torch.zeros((512,512))
        # byte inversion
        background = ~mask
        return torch.stack((background, mask))

    def __len__(self):
        return len(self.slice_index_to_patient_id_list)

    def __getitem__(self, index):
        patient_id = self.slice_index_to_patient_id_list[index]
        first_slice_index = self.patient_id_to_first_index_dict[patient_id]
        slice_index = index - first_slice_index
        #print(f'Index, {index}, Patient Id : {patient_id}, first_slice_index : {first_slice_index}, slice_index : {slice_index} ')
        file_path = self.path_to_processed_dataset.joinpath(f'{patient_id}/slice_{slice_index}.npy')

        ### WE NEVER RETURN THE SINOGRAM TO AVOID COMPUTING IT PER SAMPLE ###
        if self.pipeline == "joint" or self.pipeline == "end_to_end" or self.pipeline == "segmentation":
            reconstruction_tensor = self.get_reconstruction_tensor(file_path)
            mask_tensor = self.get_mask_tensor(patient_id, slice_index)
            return reconstruction_tensor, mask_tensor

        elif self.pipeline == "reconstruction":
            reconstruction_tensor = self.get_reconstruction_tensor(file_path)
            return reconstruction_tensor

        elif self.pipeline == "diagnostic":
            return self.patients_diagnosis_dictionary[patient_id]

        else:
            raise NotImplementedError

