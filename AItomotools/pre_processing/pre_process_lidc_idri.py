import pathlib
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pylidc as pl
import pandas as pd
import json

def format_index(index:int) -> str:
    str_index = str(index)
    while len(str_index) < 4:
        str_index = '0' + str_index
    assert len(str_index) == 4
    return str_index

def get_nodule_boundaries(annotation:pl.Annotation) -> Tuple[int, int, int, int, int, int]:
    return annotation.bbox()[0].start, annotation.bbox()[0].stop, annotation.bbox()[1].start, annotation.bbox()[1].stop, annotation.bbox()[2].start, annotation.bbox()[2].stop

def process_patient_mask(scan, path_to_processed_volume_folder:pathlib.Path) -> Dict:
    patient_nodules_dict = {}

    ## Fetching n_slices for the volume
    n_slices = len(list(path_to_processed_volume_folder.glob('*')))
    for slice_index in range(n_slices):
        patient_nodules_dict[slice_index] = {}
    list_of_annotated_nodules:List[List[pl.Annotation]] = scan.cluster_annotations() # type:ignore
    for nodule_index, annotated_nodule in enumerate(list_of_annotated_nodules):
        for annotation in annotated_nodule:
            xmin, xmax, ymin,ymax, zmin, zmax = get_nodule_boundaries(annotation)
            delta_z = zmax-zmin
            for slice_index in range(zmin, zmin+delta_z):
                patient_nodules_dict[slice_index][nodule_index] = []

        for annotation in annotated_nodule:
            xmin, xmax, ymin,ymax, zmin, zmax = get_nodule_boundaries(annotation)
            delta_z = zmax-zmin

            mask = annotation.boolean_mask() # type:ignore
            assert delta_z == mask.shape[-1]

            for slice_index in range(zmin, zmin+delta_z):
                patient_nodules_dict[slice_index][nodule_index].append(annotation.id)
                nodule_array = np.zeros((512,512), dtype=np.int16)
                nodule_array[xmin:xmax, ymin:ymax] = np.bitwise_or(nodule_array[xmin:xmax, ymin:ymax], mask[:,:,slice_index - zmin])
                path_to_mask_slice = path_to_processed_volume_folder.joinpath(f'mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation.id}.npy')
                np.save(path_to_mask_slice, nodule_array)

    return {k: v for k, v in patient_nodules_dict.items() if v}

def process_volume(scan:pl.Scan, path_to_processed_volume_folder:pathlib.Path):
    volume = scan.to_volume()
    n_slices = volume.shape[-1]
    for slice_index in range(n_slices):
        path_to_slice = pathlib.Path(path_to_processed_volume_folder.joinpath(f'slice_{slice_index}.npy'))
        assert np.shape(volume[:,:,slice_index]) == (512,512), print(np.shape(volume[:,:,slice_index]))
        np.save(path_to_slice,volume[:,:,slice_index])

def process_patient_statistics(scan:pl.Scan):
    list_of_annotated_nodules = scan.cluster_annotations()
    n_slices_patient = 0
    for annotated_nodule in list_of_annotated_nodules:
        for annotation in annotated_nodule:
            n_slices_patient += annotation.bbox()[-1].stop -annotation.bbox()[-1].start  #type:ignore
    return len(list_of_annotated_nodules), int(n_slices_patient)

def pre_process_dataset(path_to_processed_dataset:pathlib.Path):
    path_to_processed_dataset.mkdir(exist_ok=True, parents=True)

    patient_id_to_n_segmented_slices = {}
    patient_id_to_n_nodules = {}
    patients_masks = {}

    for patient_index in range(1,1012):
        formatted_index = format_index(patient_index)
        formatted_index = f'LIDC-IDRI-{formatted_index}'
        path_to_processed_volume_folder = path_to_processed_dataset.joinpath(formatted_index)
        path_to_processed_volume_folder.mkdir(exist_ok=True, parents=True)
        print(f'Processing patient with PID {formatted_index}')
        scan:pl.Scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == formatted_index).first() #type:ignore
        if type(scan) == type(None):
            print(f'Current patient {formatted_index} has no scan')
        else:
            if len(list(path_to_processed_volume_folder.glob('*'))) == 0:

                ### Pre-process the volume
                process_volume(scan, path_to_processed_volume_folder)
                ### Compute volume statistics
                n_annotated_nodules, n_segmented_slices = process_patient_statistics(scan)

                patient_id_to_n_nodules[formatted_index] = n_annotated_nodules
                patient_id_to_n_segmented_slices[formatted_index] = n_segmented_slices

                ### Pre-process the masks
                patients_masks[formatted_index] = process_patient_mask(scan, path_to_processed_volume_folder)

            else:
                print(f'\t Patient already sampled, passing...')

        with open(path_to_processed_dataset.joinpath('patient_id_to_n_segmented_slices.json'), 'w') as out_f:
            json.dump(patient_id_to_n_segmented_slices, out_f)
        with open(path_to_processed_dataset.joinpath('patient_id_to_n_nodules.json'), 'w') as out_f:
            json.dump(patient_id_to_n_nodules, out_f)
        with open(path_to_processed_dataset.joinpath('patients_masks.json'), 'w') as out_f:
            json.dump(patients_masks, out_f, indent=4)

def compute_diagnosis_file(diagnosis_file_path:pathlib.Path, diagnosis_dict_save_path:pathlib.Path):
    if not diagnosis_file_path.is_file():
        raise FileNotFoundError(f'No file found at {diagnosis_file_path}')

    if diagnosis_dict_save_path.is_file():
        raise FileExistsError(f'There is already a file at {diagnosis_dict_save_path}, passing')
    diagnosis_df = pd.read_excel(diagnosis_file_path)

    diagnosis_dict = {}

    patients_column_name  = diagnosis_df.columns[0]
    diagnosis_column_name = diagnosis_df.columns[1]

    for patient_index in range(1,1012):
        formatted_index = format_index(patient_index)
        formatted_index = f'LIDC-IDRI-{formatted_index}'
        is_index_in_df = (diagnosis_df[patients_column_name] == formatted_index).any()
        if is_index_in_df:
            diagnosis = diagnosis_df.loc[diagnosis_df[patients_column_name] == formatted_index, diagnosis_column_name].iloc[0]
        else:
            diagnosis = 0

        diagnosis_dict[formatted_index] = int(diagnosis)

    with open(diagnosis_dict_save_path, 'w') as out_file:
        json.dump(diagnosis_dict, out_file)

if __name__ == '__main__':
    path_to_raw_dataset = pathlib.Path('/local/scratch/public/AItomotools/raw/LIDC-IDRI')
    path_to_diagnosis_file = path_to_raw_dataset.joinpath('tcia-diagnosis-data-2012-04-20.xls')
    path_to_processed_dataset = pathlib.Path('/local/scratch/public/AItomotools/processed/LIDC-IDRI')
    print('LIDC-IDRI dataset pre-processing functions')
    # pre_process_dataset(path_to_processed_dataset)
    compute_diagnosis_file(path_to_diagnosis_file, path_to_processed_dataset.joinpath('patient_id_to_diagnosis.json'))

