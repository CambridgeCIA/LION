# =============================================================================
# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Emilien Valat
# Modifications:-
# =============================================================================

import pathlib
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pylidc as pl
import pandas as pd
import json

import importlib.util
import sys
import os


spec = importlib.util.spec_from_file_location(
    "aipaths", pathlib.Path(__file__).resolve().parents[2] / "utils/paths.py"
)
foo = importlib.util.module_from_spec(spec)
sys.modules["aipaths"] = foo
spec.loader.exec_module(foo)

from aipaths import LIDC_IDRI_PROCESSED_DATASET_PATH, LIDC_IDRI_PATH


def format_index(index: int) -> str:
    str_index = str(index)
    while len(str_index) < 4:
        str_index = "0" + str_index
    assert len(str_index) == 4
    return str_index


def get_nodule_boundaries(
    annotation: pl.Annotation,
) -> Tuple[int, int, int, int, int, int]:
    return (
        annotation.bbox()[0].start,
        annotation.bbox()[0].stop,
        annotation.bbox()[1].start,
        annotation.bbox()[1].stop,
        annotation.bbox()[2].start,
        annotation.bbox()[2].stop,
    )


def process_patient_mask(scan, path_to_processed_volume_folder: pathlib.Path) -> Dict:
    patient_nodules_dict = {}

    ## Fetching n_slices for the volume
    n_slices = len(list(path_to_processed_volume_folder.glob("*")))
    for slice_index in range(n_slices):
        patient_nodules_dict[slice_index] = {}
    list_of_annotated_nodules: List[
        List[pl.Annotation]
    ] = scan.cluster_annotations()  # type:ignore
    for nodule_index, annotated_nodule in enumerate(list_of_annotated_nodules):
        for annotation in annotated_nodule:
            xmin, xmax, ymin, ymax, zmin, zmax = get_nodule_boundaries(annotation)
            delta_z = zmax - zmin
            for slice_index in range(zmin, zmin + delta_z):
                patient_nodules_dict[slice_index][nodule_index] = []

        for annotation in annotated_nodule:
            xmin, xmax, ymin, ymax, zmin, zmax = get_nodule_boundaries(annotation)
            delta_z = zmax - zmin

            mask = annotation.boolean_mask()  # type:ignore
            assert delta_z == mask.shape[-1]

            for slice_index in range(zmin, zmin + delta_z):
                patient_nodules_dict[slice_index][nodule_index].append(annotation.id)
                nodule_array = np.zeros((512, 512), dtype=np.int16)
                nodule_array[xmin:xmax, ymin:ymax] = np.bitwise_or(
                    nodule_array[xmin:xmax, ymin:ymax], mask[:, :, slice_index - zmin]
                )
                path_to_mask_slice = path_to_processed_volume_folder.joinpath(
                    f"mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation.id}.npy"
                )
                np.save(path_to_mask_slice, nodule_array)

    return {k: v for k, v in patient_nodules_dict.items() if v}


def process_volume(scan: pl.Scan, path_to_processed_volume_folder: pathlib.Path):
    volume = scan.to_volume()
    n_slices = volume.shape[-1]
    for slice_index in range(n_slices):
        path_to_slice = pathlib.Path(
            path_to_processed_volume_folder.joinpath(f"slice_{slice_index}.npy")
        )
        assert np.shape(volume[:, :, slice_index]) == (512, 512), print(
            np.shape(volume[:, :, slice_index])
        )
        np.save(path_to_slice, volume[:, :, slice_index])


def compute_slice_thickness(path_to_processed_dataset: pathlib.Path):
    file_name = "patient_id_to_slice_thickness.json"
    patient_dict_to_slice_thickness = {}

    for patient_index in range(1, 1012):
        formatted_index = format_index(patient_index)
        formatted_index = f"LIDC-IDRI-{formatted_index}"
        scan: pl.Scan = (
            pl.query(pl.Scan).filter(pl.Scan.patient_id == formatted_index).first()
        )  # type:ignore
        if type(scan) == type(None):
            print(f"Current patient {formatted_index} has no scan")
        else:
            patient_dict_to_slice_thickness[patient_index] = scan.slice_thickness

    with open(path_to_processed_dataset.joinpath(file_name), "w") as out_file:
        print(f"Writing {file_name}...")
        json.dump(patient_dict_to_slice_thickness, out_file)


def pre_process_dataset(path_to_processed_dataset: pathlib.Path):
    path_to_processed_dataset.mkdir(exist_ok=True, parents=True)

    patients_masks = {}

    for patient_index in range(1, 1012):
        formatted_index = format_index(patient_index)
        formatted_index = f"LIDC-IDRI-{formatted_index}"
        path_to_processed_volume_folder = path_to_processed_dataset.joinpath(
            formatted_index
        )
        path_to_processed_volume_folder.mkdir(exist_ok=True, parents=True)
        print(f"Processing patient with PID {formatted_index}")
        scan: pl.Scan = (
            pl.query(pl.Scan).filter(pl.Scan.patient_id == formatted_index).first()
        )  # type:ignore
        if type(scan) == type(None):
            print(f"Current patient {formatted_index} has no scan")
        else:
            if len(list(path_to_processed_volume_folder.glob("*"))) == 0:

                ### Pre-process the volume
                process_volume(scan, path_to_processed_volume_folder)
                ### Pre-process the masks
                patients_masks[formatted_index] = process_patient_mask(
                    scan, path_to_processed_volume_folder
                )

            else:
                print(f"\t Patient already sampled, passing...")

    path_to_patients_masks_dict = path_to_processed_dataset.joinpath(
        "patients_masks.json"
    )
    if path_to_patients_masks_dict.is_file():
        print(f"Patients masks file already written, passing...")
    else:
        with open(path_to_patients_masks_dict, "w") as out_f:
            json.dump(patients_masks, out_f, indent=4)


def compute_diagnosis_file(
    diagnosis_file_path: pathlib.Path, diagnosis_dict_save_path: pathlib.Path
):
    if not diagnosis_file_path.is_file():
        raise FileNotFoundError(f"No file found at {diagnosis_file_path}")

    if diagnosis_dict_save_path.is_file():
        print(f"There is already a file at {diagnosis_dict_save_path}, passing")
    else:
        diagnosis_df = pd.read_excel(diagnosis_file_path)

        diagnosis_dict = {}

        patients_column_name = diagnosis_df.columns[0]
        diagnosis_column_name = diagnosis_df.columns[1]

        for patient_index in range(1, 1012):
            formatted_index = format_index(patient_index)
            formatted_index = f"LIDC-IDRI-{formatted_index}"
            is_index_in_df = (
                diagnosis_df[patients_column_name] == formatted_index
            ).any()
            if is_index_in_df:
                diagnosis = diagnosis_df.loc[
                    diagnosis_df[patients_column_name] == formatted_index,
                    diagnosis_column_name,
                ].iloc[0]
            else:
                diagnosis = 0

            diagnosis_dict[formatted_index] = int(diagnosis)

        with open(diagnosis_dict_save_path, "w") as out_file:
            json.dump(diagnosis_dict, out_file)


def compute_patient_with_nodule_subtetly(
    path_to_processed_dataset: pathlib.Path, nodule_subtelty: int, operator: str
) -> None:
    accepter_operator_argument = [
        "inferior",
        "superior",
        "inferior_or_equal",
        "superior_or_equal",
    ]
    assert nodule_subtelty in [1, 2, 3, 4, 5], print(
        f"Wrong nodule_subtelty argument, cf https://pylidc.github.io/annotation.html"
    )
    assert operator in accepter_operator_argument, print(
        f"Wrong operator_argument, must be in {accepter_operator_argument}"
    )

    file_name = f"patients_with_nodule_subtlety_{operator}_to_{nodule_subtelty}.json"

    if operator == "superior":
        annotations: List[pl.Annotation] = (
            pl.query(pl.Annotation)
            .filter(nodule_subtelty < pl.Annotation.subtlety)
            .all()
        )
    elif operator == "superior_or_equal":
        annotations: List[pl.Annotation] = (
            pl.query(pl.Annotation)
            .filter(nodule_subtelty <= pl.Annotation.subtlety)
            .all()
        )
    elif operator == "inferior":
        annotations: List[pl.Annotation] = (
            pl.query(pl.Annotation)
            .filter(pl.Annotation.subtlety < nodule_subtelty)
            .all()
        )
    elif operator == "inferior_or_equal":
        annotations: List[pl.Annotation] = (
            pl.query(pl.Annotation)
            .filter(pl.Annotation.subtlety <= nodule_subtelty)
            .all()
        )
    else:
        raise ValueError(
            f"Wrong operator_argument, must be in {accepter_operator_argument}"
        )

    patient_list = []

    for annotation in annotations:
        if annotation.scan.patient_id not in patient_list:
            patient_list.append(annotation.scan.patient_id)

    with open(path_to_processed_dataset.joinpath(file_name), "w") as out_file:
        print(f"Writing {file_name}...")
        json.dump(patient_list, out_file)


if __name__ == "__main__":

    path_to_raw_dataset = pathlib.Path(LIDC_IDRI_PATH)
    path_to_diagnosis_file = path_to_raw_dataset.joinpath(
        "tcia-diagnosis-data-2012-04-20.xls"
    )
    path_to_processed_dataset = pathlib.Path(LIDC_IDRI_PROCESSED_DATASET_PATH)
    print("LIDC-IDRI dataset pre-processing functions")
    pre_process_dataset(path_to_processed_dataset)
    compute_diagnosis_file(
        path_to_diagnosis_file,
        path_to_processed_dataset.joinpath("patient_id_to_diagnosis.json"),
    )
    compute_patient_with_nodule_subtetly(
        path_to_processed_dataset, nodule_subtelty=4, operator="superior"
    )
    compute_slice_thickness(path_to_processed_dataset)
