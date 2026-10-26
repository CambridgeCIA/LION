# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Emilien Valat
# Modifications:-
# =============================================================================

import argparse
import pathlib
from pathlib import Path
from typing import Tuple, List, Dict
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pylidc as pl
import pandas as pd
import json

import importlib.util
import sys


spec = importlib.util.spec_from_file_location(
    "aipaths", pathlib.Path(__file__).resolve().parents[2] / "utils/paths.py"
)
foo = importlib.util.module_from_spec(spec)
sys.modules["aipaths"] = foo
spec.loader.exec_module(foo)

# from LION.utils.paths import LIDC_IDRI_PROCESSED_DATASET_PATH, LIDC_IDRI_PATH
# from ...utils.paths import LIDC_IDRI_PROCESSED_DATASET_PATH, LIDC_IDRI_PATH
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.paths import LIDC_IDRI_PROCESSED_DATASET_PATH, LIDC_IDRI_PATH

# resampling.py is next to this script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from resampling import (
    PIXEL_SPACING_MM,
    IMAGE_SHAPE,
    resample_ct_slice,
    resample_mask,
)

# Slices are resampled to PIXEL_SPACING_MM (500 mm over 512 pixels, the largest
# spacing in LIDC-IDRI, so nothing is cropped). Each patient folder records its
# spacing in this file; the dataset folder gets PRE_PROCESSING_FILE once every
# patient is resampled (the data loader reads it). Datasets processed before
# October 2026 have neither: their slices are at each scan's own spacing.
PATIENT_SPACING_FILE = "pixel_spacing.json"
PRE_PROCESSING_FILE = "pre_processing.json"


def configure_pylidc(path_to_raw_dataset: pathlib.Path) -> None:
    config_path = pathlib.Path.home().joinpath(".pylidcrc")
    dicom_path = path_to_raw_dataset.joinpath("LIDC-IDRI").resolve()
    with open(config_path, "w") as config_file:
        config_file.write(f"[dicom]\npath = {dicom_path}\n")
    print(f"Set pylidc DICOM path to {dicom_path} in {config_path}")


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
    n_slices = len(list(path_to_processed_volume_folder.glob("slice_*.npy")))
    for slice_index in range(n_slices):
        patient_nodules_dict[slice_index] = {}
    list_of_annotated_nodules: List[
        List[pl.Annotation]
    ] = scan.cluster_annotations()  # type:ignore
    print(scan.annotations)

    print(f"\t Found {len(list_of_annotated_nodules)} nodules")
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
                # resampled as the slices
                np.save(
                    path_to_mask_slice,
                    resample_mask(nodule_array, scan.pixel_spacing),
                )

    return {k: v for k, v in patient_nodules_dict.items() if v}


def process_volume(scan: pl.Scan, path_to_processed_volume_folder: pathlib.Path):
    volume = scan.to_volume()
    n_slices = volume.shape[-1]
    for slice_index in range(n_slices):
        path_to_slice = pathlib.Path(
            path_to_processed_volume_folder.joinpath(f"slice_{slice_index}.npy")
        )
        if np.shape(volume[:, :, slice_index]) != (512, 512):
            raise ValueError(
                f"Unexpected slice shape {np.shape(volume[:, :, slice_index])}"
            )
        np.save(
            path_to_slice,
            resample_ct_slice(volume[:, :, slice_index], scan.pixel_spacing),
        )
    write_patient_spacing(path_to_processed_volume_folder, scan.pixel_spacing)


def write_patient_spacing(path_to_processed_volume_folder, original_spacing):
    write_json(
        path_to_processed_volume_folder.joinpath(PATIENT_SPACING_FILE),
        {
            "original_pixel_spacing_mm": float(original_spacing),
            "pixel_spacing_mm": PIXEL_SPACING_MM,
        },
    )


def resample_processed_patient(
    path_to_processed_volume_folder: pathlib.Path, original_spacing: float
) -> None:
    """Resample a patient folder processed before October 2026 (slices and
    masks at the scan's own spacing) to PIXEL_SPACING_MM.

    The resampled folder is written next to it and then swapped in, so an
    interrupted run leaves either the old or the new folder, never a mix.
    """
    folder = path_to_processed_volume_folder
    tmp = folder.with_name(folder.name + ".resampling.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    for path in folder.iterdir():
        try:
            if path.name.startswith("slice_"):
                image = resample_ct_slice(np.load(path), original_spacing)
                np.save(tmp / path.name, image)
            elif path.name.startswith("mask_"):
                np.save(tmp / path.name, resample_mask(np.load(path), original_spacing))
            else:
                shutil.copy2(path, tmp / path.name)
        except Exception as error:
            raise RuntimeError(f"{path}: {error}") from error
    write_patient_spacing(tmp, original_spacing)
    old = folder.with_name(folder.name + ".old")
    folder.rename(old)
    tmp.rename(folder)
    shutil.rmtree(old)


def is_resampled(path_to_processed_volume_folder: pathlib.Path) -> bool:
    return path_to_processed_volume_folder.joinpath(PATIENT_SPACING_FILE).is_file()


def resample_processed_dataset(path_to_processed_dataset: pathlib.Path) -> None:
    """Resample every patient of a dataset processed before October 2026, in
    place. Needs pylidc (for the pixel spacings, in its own database) but not
    the DICOM files. Patients already resampled are skipped, so this can be
    run again after an interruption."""
    failed = []
    for folder in sorted(path_to_processed_dataset.glob("LIDC-IDRI-????")):
        if not folder.is_dir() or is_resampled(folder):
            continue
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == folder.name).first()
        if scan is None:
            print(f"Patient {folder.name} has no scan in pylidc, skipping")
            continue
        print(f"Resampling {folder.name} from {scan.pixel_spacing} mm")
        try:
            resample_processed_patient(folder, scan.pixel_spacing)
        except Exception as error:  # e.g. a truncated .npy file
            # the patient's folder is left as it was; the others go on
            failed.append((folder.name, str(error)))
            print(f"\t Failed, patient left as it was: {error}")
            tmp = folder.with_name(folder.name + ".resampling.tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
    if failed:
        print(f"{len(failed)} patients could not be resampled:")
        for patient, error in failed:
            print(f"  {patient}: {error}")
    write_pre_processing_file(path_to_processed_dataset)


def write_pre_processing_file(path_to_processed_dataset: pathlib.Path) -> None:
    """Record the dataset's pixel spacing, with the patients that could not
    be resampled (e.g. a damaged file): the data loader leaves them out."""
    folders = [
        f for f in path_to_processed_dataset.glob("LIDC-IDRI-????") if f.is_dir()
    ]
    missing = sorted(f.name for f in folders if not is_resampled(f))
    if missing:
        print(
            f"Not resampled: {missing}. The data loader will leave them out; "
            "repair them (see the errors above) and run this again to add them."
        )
    write_json(
        path_to_processed_dataset.joinpath(PRE_PROCESSING_FILE),
        {
            "pixel_spacing_mm": PIXEL_SPACING_MM,
            "image_shape": list(IMAGE_SHAPE),
            "image_size_mm": [PIXEL_SPACING_MM * n for n in IMAGE_SHAPE],
            "interpolation": "linear, Gaussian anti-aliasing (resampling.py)",
            "not_resampled": missing,
        },
    )


def load_json(path: pathlib.Path, default):
    if path.is_file():
        with open(path, "r") as in_file:
            return json.load(in_file)
    return default


def write_json(path: pathlib.Path, data) -> None:
    with open(path, "w") as out_file:
        json.dump(data, out_file, indent=4)


def remove_error_for_patient(error_list: List, patient_id: str) -> List:
    return [error for error in error_list if error[0] != patient_id]


def compute_pixel_spacing(path_to_processed_dataset: pathlib.Path):
    """The pixel spacing of each scan before resampling, in mm (pylidc)."""
    file_name = "patient_id_to_pixel_spacing.json"
    patient_dict_to_pixel_spacing = {}
    for patient_index in range(1, 1012):
        formatted_index = f"LIDC-IDRI-{format_index(patient_index)}"
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == formatted_index).first()
        if scan is not None:
            patient_dict_to_pixel_spacing[patient_index] = scan.pixel_spacing
    with open(path_to_processed_dataset.joinpath(file_name), "w") as out_file:
        print(f"Writing {file_name}...")
        json.dump(patient_dict_to_pixel_spacing, out_file)


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

    path_to_patients_masks_dict = path_to_processed_dataset.joinpath(
        "patients_masks.json"
    )
    path_to_error_list = path_to_processed_dataset.joinpath("error_list.json")
    patients_masks = load_json(path_to_patients_masks_dict, {})
    error_list = load_json(path_to_error_list, [])

    for patient_index in range(1, 1012):
        # for patient_index in [197]:
        formatted_index = format_index(patient_index)
        formatted_index = f"LIDC-IDRI-{formatted_index}"
        path_to_processed_volume_folder = path_to_processed_dataset.joinpath(
            formatted_index
        )
        tmp_processed_volume_folder = path_to_processed_dataset.joinpath(
            f"{formatted_index}.tmp"
        )

        if path_to_processed_volume_folder.is_dir():
            if not is_resampled(path_to_processed_volume_folder):
                # processed before October 2026, at the scan's own spacing
                scan = (
                    pl.query(pl.Scan)
                    .filter(pl.Scan.patient_id == formatted_index)
                    .first()
                )
                print(f"\t Patient {formatted_index}: resampling the processed slices")
                resample_processed_patient(
                    path_to_processed_volume_folder, scan.pixel_spacing
                )
            else:
                print(f"\t Patient {formatted_index} already sampled, passing...")
            continue

        try:
            print(f"Processing patient with PID {formatted_index}")
            scan: pl.Scan = (
                pl.query(pl.Scan).filter(pl.Scan.patient_id == formatted_index).first()
            )  # type:ignore
            if type(scan) == type(None):
                print(f"Current patient {formatted_index} has no scan")
                patients_masks[formatted_index] = {}
                error_list = remove_error_for_patient(error_list, formatted_index)
                write_json(path_to_patients_masks_dict, patients_masks)
                write_json(path_to_error_list, error_list)
            else:
                if tmp_processed_volume_folder.exists():
                    shutil.rmtree(tmp_processed_volume_folder)
                tmp_processed_volume_folder.mkdir(exist_ok=True, parents=True)

                ### Pre-process the volume
                process_volume(scan, tmp_processed_volume_folder)
                ### Pre-process the masks
                patients_masks[formatted_index] = process_patient_mask(
                    scan, tmp_processed_volume_folder
                )
                error_list = remove_error_for_patient(error_list, formatted_index)
                write_json(path_to_patients_masks_dict, patients_masks)
                write_json(path_to_error_list, error_list)
                tmp_processed_volume_folder.rename(path_to_processed_volume_folder)
        except Exception as e:
            print(
                f"An error occurred while processing patient {formatted_index}, skipping..."
            )
            if tmp_processed_volume_folder.exists():
                shutil.rmtree(tmp_processed_volume_folder)
            error_list = remove_error_for_patient(error_list, formatted_index)
            error_list.append([formatted_index, str(e)])
            write_json(path_to_error_list, error_list)
            write_json(path_to_patients_masks_dict, patients_masks)

    # exit()


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
    parser = argparse.ArgumentParser(description="Pre-process LIDC-IDRI for LION")
    parser.add_argument(
        "--resample-only",
        action="store_true",
        help="Only resample a dataset processed before October 2026 (slices at "
        "each scan's own spacing) to LION's pixel spacing, in place. Needs pylidc "
        "but not the raw DICOM files.",
    )
    parser.add_argument(
        "--folder",
        default=None,
        help="the processed LIDC-IDRI folder (the one with LIDC-IDRI-0001, ...). "
        "Default: LION_DATA_PATH/processed/LIDC-IDRI",
    )
    args = parser.parse_args()

    path_to_raw_dataset = pathlib.Path(LIDC_IDRI_PATH)
    path_to_processed_dataset = pathlib.Path(
        args.folder if args.folder else LIDC_IDRI_PROCESSED_DATASET_PATH
    )
    if args.resample_only:
        resample_processed_dataset(path_to_processed_dataset)
        compute_pixel_spacing(path_to_processed_dataset)
        sys.exit(0)

    configure_pylidc(path_to_raw_dataset)
    path_to_diagnosis_file = path_to_raw_dataset.joinpath(
        "tcia-diagnosis-data-2012-04-20.xls"
    )
    print("LIDC-IDRI dataset pre-processing functions")
    pre_process_dataset(path_to_processed_dataset)
    write_pre_processing_file(path_to_processed_dataset)
    compute_diagnosis_file(
        path_to_diagnosis_file,
        path_to_processed_dataset.joinpath("patient_id_to_diagnosis.json"),
    )
    compute_patient_with_nodule_subtetly(
        path_to_processed_dataset, nodule_subtelty=4, operator="superior"
    )
    compute_slice_thickness(path_to_processed_dataset)
    compute_pixel_spacing(path_to_processed_dataset)
