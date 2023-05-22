from typing import List, Tuple, Dict
import pathlib
import random
import math

import torch
import numpy as np
import pylidc as pl
import json
from torch.utils.data import Dataset
import pydicom as dicom

from backends.odl import ODLBackend

def format_index(index:int) -> str:
    str_index = str(index)
    while len(str_index) < 4:
        str_index = '0' + str_index
    assert len(str_index) == 4
    return str_index

def get_n_slices(scan:pl.Scan) -> int:
    n_slices = len(list(pathlib.Path(scan.get_path_to_dicom_files()).glob('*')))
    print(f'Scan with PID {scan.patient_id} has {n_slices} slices')
    return n_slices

def get_slice(scan:pl.Scan, slice_index:int) -> np.ndarray:
    image = scan.load_all_dicom_images()[slice_index]
    path_to_dicom_files = scan.get_path_to_dicom_files()
    image = dicom.read_file(pathlib.Path(path_to_dicom_files).joinpath(f'{slice_index}.dcm'))
    input()
    return (image.pixel_array * image.RescaleSlope + image.RescaleIntercept).astype(np.int16)

def choose_random_annotation(list_of_nodule_annotations:List[pl.Annotation]) -> pl.Annotation:
    return random.choice(list_of_nodule_annotations)

def is_nodule_on_slice(annotation:pl.Annotation, slice_index:int, verbose=False) -> Tuple[bool, int]:
    zmin  = annotation.bbox()[-1].start
    zmax = annotation.bbox()[-1].stop
    if zmin <= slice_index <= zmax:
        if verbose :
            print(f'The annotated nodule {annotation.id} is present on slice {slice_index}')
        return True, zmin

    if verbose :
        print(f'The annotated nodule {annotation.id} is absent on slice {slice_index}')
    return False, zmin

def get_nodule_boundaries(annotation:pl.Annotation) -> Tuple[int, int, int, int]:
    return annotation.bbox()[0].start, annotation.bbox()[0].stop, annotation.bbox()[1].start, annotation.bbox()[1].stop

def rescaled_z_index(z, z_min, z_max) -> int:
    return int(z + (z_max-z_min)/(2*z_min+z_max))

def get_slice_mask(list_of_annotated_nodules:List[pl.Annotation], slice_index:int) -> np.ndarray:
    background_array = np.ones((512,512), dtype=np.int16)
    nodule_array = np.zeros((512,512), dtype=np.int16)
    for annotated_nodule in list_of_annotated_nodules:

        annotation = choose_random_annotation(annotated_nodule)
        tumor_on_slice, zmin = is_nodule_on_slice(annotation, slice_index)

        if tumor_on_slice:

            ### mask = annotation.boolean_mask()[:,:,slice_index - zmin]
            ### WARNING indexing shenanigans,
            ### is there a mismatch between the number of indices when a volume is considered as a volume AND when a single slice is queried?
            ### EV gets errors once every other sample
            ### Temporary solution below
            try:
                mask = annotation.boolean_mask()[:,:,slice_index - zmin]
            except IndexError:
                mask = annotation.boolean_mask()[:,:,slice_index - zmin - 1] ## yuck

            xmin, xmax, ymin, ymax = get_nodule_boundaries(annotation)

            nodule_array[xmin:xmax, ymin:ymax] = np.bitwise_or(nodule_array[xmin:xmax, ymin:ymax], mask)
            background_array[xmin:xmax, ymin:ymax] = np.bitwise_and(background_array[xmin:xmax, ymin:ymax], np.invert(mask))

    return np.dstack((background_array, nodule_array)).transpose((2,1,0)).transpose((0,2,1))

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

    def get_mask_tensor(self, scan:pl.Scan, slice_index:int) -> torch.Tensor:
        return torch.from_numpy(get_slice_mask(scan.cluster_annotations(), slice_index)) #type:ignore

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
            scan:pl.Scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()  #type:ignore
            reconstruction_tensor = self.get_reconstruction_tensor(file_path)
            mask_tensor = self.get_mask_tensor(scan, slice_index)
            return reconstruction_tensor, mask_tensor

        elif self.pipeline == "reconstruction":
            reconstruction_tensor = self.get_reconstruction_tensor(file_path)
            return reconstruction_tensor

        else:
            raise NotImplementedError

