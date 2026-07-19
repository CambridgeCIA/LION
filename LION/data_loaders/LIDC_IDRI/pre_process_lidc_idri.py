"""Download recovery and preprocessing utilities for LIDC-IDRI.

The script converts patient DICOM volumes and pylidc annotations into the
per-slice NumPy layout consumed by :class:`LION.data_loaders.LIDC_IDRI`.  It can
resume interrupted downloads, verify expected patient/file counts, and remove
raw patients only after their processed outputs are complete.
"""

# =============================================================================
# This file is part of LION library
# License : BSD-3
#
# Author  : Emilien Valat
# Modifications:-
# =============================================================================

import argparse
import importlib.util
import json
import pathlib
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pylidc as pl
import pandas as pd


LIDC_IDRI_PATIENT_COUNT = 1012
EXPECTED_PROCESSED_FILE_COUNT = 282776
EXPECTED_PROCESSED_PATIENT_DIR_COUNT = 1010


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


def configure_pylidc(path_to_raw_dataset: pathlib.Path) -> None:
    """Write pylidc's user configuration for the selected raw DICOM root."""
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


def iter_lidc_patient_indices():
    """Return the canonical one-based LIDC-IDRI patient index range."""
    return range(1, LIDC_IDRI_PATIENT_COUNT + 1)


def format_patient_id(index: int) -> str:
    return f"LIDC-IDRI-{format_index(index)}"


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
    """Rasterise all pylidc nodule annotations for one scan."""
    patient_nodules_dict = {}

    ## Fetching n_slices for the volume
    n_slices = len(list(path_to_processed_volume_folder.glob("*")))
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
                np.save(path_to_mask_slice, nodule_array)

    return {k: v for k, v in patient_nodules_dict.items() if v}


def process_volume(scan: pl.Scan, path_to_processed_volume_folder: pathlib.Path):
    """Save one DICOM scan as indexed axial HU slices."""
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


def load_json(path: pathlib.Path, default):
    if path.is_file():
        with open(path, "r") as in_file:
            return json.load(in_file)
    return default


def write_json(path: pathlib.Path, data) -> None:
    with open(path, "w") as out_file:
        json.dump(data, out_file, indent=4)


def get_raw_dicom_root(path_to_raw_dataset: pathlib.Path) -> pathlib.Path:
    return path_to_raw_dataset.joinpath("LIDC-IDRI").resolve()


def get_raw_patient_dir(
    path_to_raw_dataset: pathlib.Path, patient_id: str
) -> Optional[pathlib.Path]:
    patient_dir = get_raw_dicom_root(path_to_raw_dataset).joinpath(patient_id)
    if patient_dir.is_dir():
        return patient_dir
    return None


def processed_patient_has_slices(
    path_to_processed_volume_folder: pathlib.Path,
) -> bool:
    return path_to_processed_volume_folder.is_dir() and any(
        path_to_processed_volume_folder.glob("slice_*.npy")
    )


def remove_raw_patient_dir(
    path_to_raw_dataset: pathlib.Path,
    patient_id: str,
    dry_run: bool = False,
) -> None:
    dicom_root = get_raw_dicom_root(path_to_raw_dataset)
    patient_dir = get_raw_patient_dir(path_to_raw_dataset, patient_id)

    if patient_dir is None:
        print(f"\t Raw files for {patient_id} were not found, passing...")
        return

    resolved_patient_dir = patient_dir.resolve()
    if patient_dir.is_symlink():
        raise ValueError(
            f"Refusing to delete symlinked patient directory {patient_dir}"
        )
    if patient_dir.name != patient_id:
        raise ValueError(
            f"Refusing to delete {patient_dir}: expected directory named {patient_id}"
        )
    if resolved_patient_dir.parent != dicom_root:
        raise ValueError(
            f"Refusing to delete {patient_dir}: it is not directly under {dicom_root}"
        )

    if dry_run:
        print(f"\t Would remove raw files for {patient_id}: {patient_dir}")
        return

    print(f"\t Removing raw files for {patient_id}: {patient_dir}")
    shutil.rmtree(patient_dir)


def flush_raw_delete_block(
    path_to_raw_dataset: pathlib.Path,
    patients_to_delete: List[str],
    dry_run: bool = False,
) -> None:
    if not patients_to_delete:
        return

    print(f"\t Cleaning raw files for {len(patients_to_delete)} processed patients")
    for patient_id in patients_to_delete:
        remove_raw_patient_dir(path_to_raw_dataset, patient_id, dry_run=dry_run)
    patients_to_delete.clear()


def remove_error_for_patient(error_list: List, patient_id: str) -> List:
    return [error for error in error_list if error[0] != patient_id]


def count_regular_files(path: pathlib.Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for file_path in path.rglob("*") if file_path.is_file())


def processed_patient_ids(path_to_processed_dataset: pathlib.Path) -> List[str]:
    if not path_to_processed_dataset.is_dir():
        return []
    return sorted(
        path.name
        for path in path_to_processed_dataset.glob("LIDC-IDRI-*")
        if path.is_dir()
    )


def format_patient_list(patient_ids: List[str], limit: int = 20) -> str:
    if len(patient_ids) <= limit:
        return ", ".join(patient_ids)
    shown_patients = ", ".join(patient_ids[:limit])
    return f"{shown_patients}, ... ({len(patient_ids) - limit} more)"


def patient_has_scan(patient_id: str) -> bool:
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    return scan is not None


def print_download_recovery_hint() -> None:
    print(
        "\t If this is due to missing raw DICOM files, rerun the downloader "
        "from LION/data_loaders/LIDC_IDRI:"
    )
    print("\t     NBIA_RESUME_CHOICE=M ./download_LIDC_IDRI.sh")
    print(
        "\t If NBIA still leaves missing or corrupt series, redownload all "
        "series when prompted:"
    )
    print("\t     NBIA_RESUME_CHOICE=A ./download_LIDC_IDRI.sh")
    print(
        "\t Then rerun preprocessing with the dedicated environment, e.g. "
        "`conda activate lidc_idri && python pre_process_lidc_idri.py`."
    )


def validate_processed_dataset_completeness(
    path_to_processed_dataset: pathlib.Path,
) -> None:
    processed_file_count = count_regular_files(path_to_processed_dataset)
    processed_patients = processed_patient_ids(path_to_processed_dataset)
    processed_patient_set = set(processed_patients)
    expected_patient_ids = [
        format_patient_id(index) for index in iter_lidc_patient_indices()
    ]
    missing_patient_ids = [
        patient_id
        for patient_id in expected_patient_ids
        if patient_id not in processed_patient_set
    ]
    missing_patients_with_scans = [
        patient_id for patient_id in missing_patient_ids if patient_has_scan(patient_id)
    ]
    error_list = load_json(path_to_processed_dataset.joinpath("error_list.json"), [])

    print("LIDC-IDRI processed dataset completeness check")
    print(
        f"\t Regular files: {processed_file_count} "
        f"(expected at least {EXPECTED_PROCESSED_FILE_COUNT})"
    )
    print(
        f"\t Processed patient directories: {len(processed_patients)} "
        f"(expected at least {EXPECTED_PROCESSED_PATIENT_DIR_COUNT})"
    )

    is_complete = (
        processed_file_count >= EXPECTED_PROCESSED_FILE_COUNT
        and len(processed_patients) >= EXPECTED_PROCESSED_PATIENT_DIR_COUNT
        and not missing_patients_with_scans
        and not error_list
    )
    if is_complete:
        print("\t Complete: all expected processed files are present.")
        return

    print("\t WARNING: processed LIDC-IDRI output appears incomplete.")
    if missing_patient_ids:
        print(
            f"\t Missing patient directories: "
            f"{format_patient_list(missing_patient_ids)}"
        )
    if missing_patients_with_scans:
        print(
            "\t Missing patient directories with pylidc scan records: "
            f"{format_patient_list(missing_patients_with_scans)}"
        )
    if error_list:
        print("\t Current processing errors:")
        for patient_id, message in error_list:
            print(f"\t     {patient_id}: {message}")
    print_download_recovery_hint()


def compute_slice_thickness(path_to_processed_dataset: pathlib.Path):
    file_name = "patient_id_to_slice_thickness.json"
    patient_dict_to_slice_thickness = {}

    for patient_index in iter_lidc_patient_indices():
        formatted_index = format_patient_id(patient_index)
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


def pre_process_dataset(
    path_to_processed_dataset: pathlib.Path,
    path_to_raw_dataset: Optional[pathlib.Path] = None,
    delete_raw_after_processing: bool = False,
    raw_delete_block_size: int = 1,
    dry_run_raw_delete: bool = False,
):
    if raw_delete_block_size < 1:
        raise ValueError("raw_delete_block_size must be at least 1")
    if delete_raw_after_processing and path_to_raw_dataset is None:
        raise ValueError("path_to_raw_dataset must be set when deleting raw files")

    path_to_processed_dataset.mkdir(exist_ok=True, parents=True)

    path_to_patients_masks_dict = path_to_processed_dataset.joinpath(
        "patients_masks.json"
    )
    path_to_error_list = path_to_processed_dataset.joinpath("error_list.json")
    patients_masks = load_json(path_to_patients_masks_dict, {})
    error_list = load_json(path_to_error_list, [])
    raw_delete_block: List[str] = []

    def queue_raw_delete(patient_id: str) -> None:
        if not delete_raw_after_processing:
            return
        if path_to_raw_dataset is None:
            raise ValueError("path_to_raw_dataset must be set when deleting raw files")

        raw_delete_block.append(patient_id)
        if len(raw_delete_block) >= raw_delete_block_size:
            flush_raw_delete_block(
                path_to_raw_dataset,
                raw_delete_block,
                dry_run=dry_run_raw_delete,
            )

    for patient_index in iter_lidc_patient_indices():
        # for patient_index in [197]:
        formatted_index = format_patient_id(patient_index)
        path_to_processed_volume_folder = path_to_processed_dataset.joinpath(
            formatted_index
        )
        tmp_processed_volume_folder = path_to_processed_dataset.joinpath(
            f"{formatted_index}.tmp"
        )

        if path_to_processed_volume_folder.is_dir():
            print(f"\t Patient {formatted_index} already sampled, passing...")
            if (
                delete_raw_after_processing
                and formatted_index in patients_masks
                and processed_patient_has_slices(path_to_processed_volume_folder)
            ):
                queue_raw_delete(formatted_index)
            continue

        patient_ready_for_raw_delete = False
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
                patient_ready_for_raw_delete = True
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

        if (
            patient_ready_for_raw_delete
            and delete_raw_after_processing
            and processed_patient_has_slices(path_to_processed_volume_folder)
        ):
            queue_raw_delete(formatted_index)

    if delete_raw_after_processing:
        if path_to_raw_dataset is None:
            raise ValueError("path_to_raw_dataset must be set when deleting raw files")
        flush_raw_delete_block(
            path_to_raw_dataset,
            raw_delete_block,
            dry_run=dry_run_raw_delete,
        )

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

        for patient_index in iter_lidc_patient_indices():
            formatted_index = format_patient_id(patient_index)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-process the LIDC-IDRI raw DICOM dataset."
    )
    parser.add_argument(
        "--delete-raw-after-processing",
        action="store_true",
        help=(
            "Remove each raw per-patient DICOM directory after that patient has "
            "been successfully written to the processed dataset."
        ),
    )
    parser.add_argument(
        "--raw-delete-block-size",
        type=int,
        default=1,
        help=(
            "Number of successfully processed patients to collect before removing "
            "their raw DICOM directories. Default: 1."
        ),
    )
    parser.add_argument(
        "--dry-run-raw-delete",
        action="store_true",
        help=(
            "Print the raw patient directories that would be removed, without "
            "deleting them. This also enables raw cleanup scanning."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    path_to_raw_dataset = pathlib.Path(LIDC_IDRI_PATH)
    configure_pylidc(path_to_raw_dataset)
    path_to_diagnosis_file = path_to_raw_dataset.joinpath(
        "tcia-diagnosis-data-2012-04-20.xls"
    )
    path_to_processed_dataset = pathlib.Path(LIDC_IDRI_PROCESSED_DATASET_PATH)
    print("LIDC-IDRI dataset pre-processing functions")
    pre_process_dataset(
        path_to_processed_dataset,
        path_to_raw_dataset=path_to_raw_dataset,
        delete_raw_after_processing=(
            args.delete_raw_after_processing or args.dry_run_raw_delete
        ),
        raw_delete_block_size=args.raw_delete_block_size,
        dry_run_raw_delete=args.dry_run_raw_delete,
    )
    compute_diagnosis_file(
        path_to_diagnosis_file,
        path_to_processed_dataset.joinpath("patient_id_to_diagnosis.json"),
    )
    compute_patient_with_nodule_subtetly(
        path_to_processed_dataset, nodule_subtelty=4, operator="superior"
    )
    compute_slice_thickness(path_to_processed_dataset)
    validate_processed_dataset_completeness(path_to_processed_dataset)
