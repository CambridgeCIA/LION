from typing import List, Tuple, Dict
import pathlib
import random
import math

import torch
import numpy as np
import json
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pickle

from AItomotools.backends.odl import ODLBackend

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

def choose_random_annotation(
    nodule_annotations_list: List,
) -> str:
    return random.choice(nodule_annotations_list)

def create_consensus_annotation(
    path_to_patient_folder:str, slice_index:str, nodule_index:str, nodule_annotations_list: List, clevel: float
) -> torch.int16:
    masks = []
    for annotation in nodule_annotations_list:
        path_to_mask = path_to_patient_folder.joinpath(f'mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation}.npy')              
        current_annotation_mask = np.load(path_to_mask)
        masks.append(current_annotation_mask)
    
    nodule_mask = torch.from_numpy(np.mean(masks, axis=0) >= clevel)
    return nodule_mask

class LIDC_IDRI(Dataset):
    def __init__(
            self,
            pipeline:str,
            training_proportion:float,
            mode:str,
            annotation: str,
            clevel=0.5,
            ):

        self.path_to_processed_dataset = pathlib.Path('/local/scratch/public/AItomotools/processed/LIDC-IDRI')
        self.patients_masks_dictionary = load_json(self.path_to_processed_dataset.joinpath('patients_masks.json'))

        # Add patients without masks, for now hardcoded, find a solution in preprocessing
        self.patients_masks_dictionary['LIDC-IDRI-0238'] = {}
        self.patients_masks_dictionary['LIDC-IDRI-0585'] = {}

        self.patients_diagnosis_dictionary = load_json(self.path_to_processed_dataset.joinpath('patient_id_to_diagnosis.json'))
        self.total_patients = len(list(self.path_to_processed_dataset.glob('LIDC-IDRI-*')))
        self.num_slice_per_patient = 40
        self.pcg_slices_nodule = 0.5
        self.pipeline = pipeline
        self.annotation = annotation
        self.clevel = clevel # consensus level, only used if annotation == consensus
        self.patient_index_to_n_slices_dict :Dict = {
            f'LIDC-IDRI-{format_index(index)}' : len(list(self.path_to_processed_dataset.joinpath(f'LIDC-IDRI-{format_index(index)}').glob('slice_*.npy'))) for index in range(1,self.total_patients+1)
        }

        # Dict with all slices of each patient
        self.patient_index_to_slices_index_dict :Dict = {
            f'LIDC-IDRI-{format_index(index)}' : list(np.arange(0, self.patient_index_to_n_slices_dict[f'LIDC-IDRI-{format_index(index)}'], 1)) for index in range(1,self.total_patients+1)
        }

        # Dict with all nodule slices of each patient
        # Converts the keys from self.patients_masks_dictionary to integer
        self.patient_index_to_nodule_slices_index_dict :Dict = {
            f'LIDC-IDRI-{format_index(index)}' : [int(item) for item in list(self.patients_masks_dictionary[f'LIDC-IDRI-{format_index(index)}'].keys())] for index in range(1,self.total_patients+1)  
        } 

        # Dict with all non-nodule slices of each patient
        # Computes as difference of all slices dict and dict with nodules
        self.patient_index_to_non_nodule_slices_index_dict :Dict = {
            f'LIDC-IDRI-{format_index(index)}' : list(set(self.patient_index_to_slices_index_dict[f'LIDC-IDRI-{format_index(index)}']) - set(self.patient_index_to_nodule_slices_index_dict[f'LIDC-IDRI-{format_index(index)}'])) for index in range(1,self.total_patients+1)
        }
        
        # Corrupted data handling
        # Delete all slices that contain a nodule that has more than 4 annotations
        self.slices_to_remove :Dict = {}
        for patient_id, nodule_slices_list in self.patient_index_to_nodule_slices_index_dict.items():
            self.slices_to_remove[patient_id] = []
            for slice_index in nodule_slices_list:
                all_nodules_dict:Dict = self.patients_masks_dictionary[patient_id][f'{slice_index}']
                for _, nodule_annotations_list in all_nodules_dict.items():
                    if len(nodule_annotations_list) > 4:
                        self.slices_to_remove[patient_id].append(slice_index)
                        break
        
        self.patient_index_to_nodule_slices_index_dict :Dict = {
            f'LIDC-IDRI-{format_index(index)}' : list(set(self.patient_index_to_nodule_slices_index_dict[f'LIDC-IDRI-{format_index(index)}']) - set(self.slices_to_remove[f'LIDC-IDRI-{format_index(index)}'])) for index in range(1,self.total_patients+1)
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
            self.slices_to_load = self.get_slices_to_load_new(self.training_patients_list, self.patient_index_to_non_nodule_slices_index_dict, self.patient_index_to_nodule_slices_index_dict, self.num_slice_per_patient, self.pcg_slices_nodule)
            self.slice_index_to_patient_id_list = self.get_slice_index_to_patient_id_list_new(self.slices_to_load)
            self.patient_id_to_first_index_dict = self.get_patient_id_to_first_index_dict_new(self.slices_to_load)


        elif self.mode == 'testing':
            self.slices_to_load = self.get_slices_to_load_new(self.testing_patients_list, self.patient_index_to_non_nodule_slices_index_dict, self.patient_index_to_nodule_slices_index_dict, self.num_slice_per_patient, self.pcg_slices_nodule)
            self.slice_index_to_patient_id_list = self.get_slice_index_to_patient_id_list_new(self.slices_to_load)
            self.patient_id_to_first_index_dict = self.get_patient_id_to_first_index_dict_new(self.slices_to_load)

        else:
            raise NotImplementedError(f'mode {self.mode} not implemented, try training or testing')

        
        print(f'Patient lists ready for {self.mode} dataset')

    def get_slices_to_load(self, patient_list:List, patient_index_to_slices_index_dict:Dict, num_slice_per_patient:int):
        patient_id_to_slices_to_load_dict = {} # Empty dict which should contain patient id as key and slice ids as array of values
        for patient_id in patient_list: # Loop over every patient
            number_of_slices = len(patient_index_to_slices_index_dict[patient_id]) # Gives the total number of slices of this patient
            if number_of_slices < num_slice_per_patient: # If the total number of slices of this patient is smaller than the desired size then just use all slices this patient has
                patient_id_to_slices_to_load_dict[patient_id] = patient_index_to_slices_index_dict[patient_id]
            else: patient_id_to_slices_to_load_dict[patient_id] = np.linspace(0, number_of_slices, num_slice_per_patient, dtype=int, endpoint=False).tolist() # Take a linspace from the given slices and exclude the last because this slice does not exist
        return patient_id_to_slices_to_load_dict
    

    def get_slices_to_load_new(self, patient_list:List, non_nodule_slices_dict:Dict, nodule_slices_dict:Dict, num_slice_per_patient:int, pcg_slices_nodule:float):
        patient_id_to_slices_to_load_dict = {} # Empty dict which should contain patient id as key and slice ids as array of values
        for patient_id in patient_list: # Loop over every patient
            number_of_slices = min(num_slice_per_patient, min(len(non_nodule_slices_dict[patient_id]), len(nodule_slices_dict[patient_id])))

            # Get amount of slices we want with nodule
            number_of_slices_with_nodule = int(number_of_slices * pcg_slices_nodule)
            # Get amount of slices we want without nodule
            number_of_slices_without_nodule = int(number_of_slices * (1 - pcg_slices_nodule))

            patient_id_to_slices_to_load_dict[patient_id] = list(np.array(non_nodule_slices_dict[patient_id])[np.linspace(0, len(non_nodule_slices_dict[patient_id]), number_of_slices_without_nodule, dtype=int, endpoint=False)]) + list(np.array(nodule_slices_dict[patient_id])[np.linspace(0, len(nodule_slices_dict[patient_id]), number_of_slices_with_nodule, dtype=int, endpoint=False)])
            patient_id_to_slices_to_load_dict[patient_id].sort()
        
        return patient_id_to_slices_to_load_dict


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
    
    def get_patient_id_to_first_index_dict_new(self, patient_with_slices_to_load:Dict):
        patient_id_to_first_index_dict = {}
        global_index = 0
        for patient_id in patient_with_slices_to_load:
            path_to_folder = self.path_to_processed_dataset.joinpath(patient_id)
            patient_id_to_first_index_dict[patient_id] = global_index
            
            if len(patient_with_slices_to_load[patient_id]) < len(list(path_to_folder.glob('slice_*.npy'))):
                global_index += len(patient_with_slices_to_load[patient_id])
            else: global_index += len(list(path_to_folder.glob('slice_*.npy')))
        return patient_id_to_first_index_dict
    
    def get_slice_index_to_patient_id_list_new(self, patient_with_slices_to_load:Dict):
        slice_index_to_patient_id_list = []
        for patient_id in patient_with_slices_to_load:
            path_to_folder = self.path_to_processed_dataset.joinpath(patient_id)

            if len(patient_with_slices_to_load[patient_id]) < len(list(path_to_folder.glob('slice_*.npy'))):
                n_slices = len(patient_with_slices_to_load[patient_id])
            else: n_slices = len(list(path_to_folder.glob('slice_*.npy')))

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
            mask = torch.zeros((512,512), dtype=torch.bool)
            all_nodules_dict:Dict = self.patients_masks_dictionary[patient_id][f'{slice_index}']
            for nodule_index, nodule_annotations_list in all_nodules_dict.items():

                # if len(nodule_annotations_list) > 4:
                #     # If there are more than 4 annotations for a nodule consider this slice as corrupted and skip
                #     return None

                if self.annotation == 'random':
                    ## If a nodule was not segmented by all the clinicians, the other annotations should not always be seen
                    while len(nodule_annotations_list) < 4:
                        nodule_annotations_list.append('')

                    annotation = choose_random_annotation(nodule_annotations_list)
                    if annotation == '':
                        # Hopefully, that exists the try to return an empty mask
                        nodule_mask = torch.zeros((512,512), dtype=torch.bool)
                    else:
                        path_to_mask = self.path_to_processed_dataset.joinpath(f'{patient_id}/mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation}.npy')
                        print(path_to_mask)
                        nodule_mask = torch.from_numpy(np.load(path_to_mask))
                
                elif self.annotation == 'consensus':
                    # Create consensus annotation out of all annotations of this nodule
                    path_to_patient_folder = self.path_to_processed_dataset.joinpath(f'{patient_id}/')
                    nodule_mask = create_consensus_annotation(path_to_patient_folder, slice_index, nodule_index, nodule_annotations_list, self.clevel)

                else:
                    raise NotImplementedError(
                        f"annotation {self.annotation} not implemented, try random or consensus"
                    )

                mask = mask.bitwise_or(nodule_mask)

        except KeyError:
            mask = torch.zeros((512,512), dtype=torch.bool)
        # byte inversion
        background = ~mask
        return torch.stack((background, mask))

    def __len__(self):
        return len(self.slice_index_to_patient_id_list)

    def get_specific_slice(self, patient_index, slice_index):
        ## Assumes slice and mask exist
        file_path = self.path_to_processed_dataset.joinpath(f'{patient_index}/slice_{slice_index}.npy')
        return self.get_reconstruction_tensor(file_path), self.get_mask_tensor(patient_index, slice_index)

    def __getitem__(self, index):
        patient_id = self.slice_index_to_patient_id_list[index]
        print(patient_id)
        first_slice_index = self.patient_id_to_first_index_dict[patient_id]
        dict_slice_index = index - first_slice_index
        slice_index_to_load = self.slices_to_load[patient_id][dict_slice_index]
        #print(f'Index, {index}, Patient Id : {patient_id}, first_slice_index : {first_slice_index}, slice_index : {slice_index} ')
        file_path = self.path_to_processed_dataset.joinpath(f'{patient_id}/slice_{slice_index_to_load}.npy')

        ### WE NEVER RETURN THE SINOGRAM TO AVOID COMPUTING IT PER SAMPLE ###
        if self.pipeline == "joint" or self.pipeline == "end_to_end" or self.pipeline == "segmentation":
            reconstruction_tensor = self.get_reconstruction_tensor(file_path)
            mask_tensor = self.get_mask_tensor(patient_id, slice_index_to_load)
            return reconstruction_tensor, mask_tensor

        elif self.pipeline == "reconstruction":
            reconstruction_tensor = self.get_reconstruction_tensor(file_path)
            return reconstruction_tensor

        elif self.pipeline == "diagnostic":
            return self.patients_diagnosis_dictionary[patient_id]

        else:
            raise NotImplementedError
