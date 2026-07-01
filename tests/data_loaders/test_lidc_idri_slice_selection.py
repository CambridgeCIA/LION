import json

import torch

from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
from LION.utils.parameter import LIONParameter


def test_lidc_minus_one_uses_all_slices_without_nodule_balancing():
    patient_list = ["LIDC-IDRI-0001"]
    all_slices = {"LIDC-IDRI-0001": [0, 1, 2, 3, 4, 5, 6]}
    non_nodule = {"LIDC-IDRI-0001": [0, 2, 4, 6]}
    nodule = {"LIDC-IDRI-0001": [1]}

    selected = LIDC_IDRI.get_slices_to_load(
        None,
        patient_list,
        non_nodule,
        nodule,
        -1,
        0.5,
        all_slices,
    )

    assert selected == all_slices


def test_lidc_subset_selection_keeps_existing_balanced_behavior():
    patient_list = ["LIDC-IDRI-0001"]
    all_slices = {"LIDC-IDRI-0001": list(range(10))}
    non_nodule = {"LIDC-IDRI-0001": [0, 2, 4, 6, 8]}
    nodule = {"LIDC-IDRI-0001": [1, 3, 5, 7, 9]}

    selected = LIDC_IDRI.get_slices_to_load(
        None,
        patient_list,
        non_nodule,
        nodule,
        4,
        0.5,
        all_slices,
    )

    assert len(selected["LIDC-IDRI-0001"]) == 4
    selected_nodule_slices = set(selected["LIDC-IDRI-0001"]).intersection(
        nodule["LIDC-IDRI-0001"]
    )
    assert len(selected_nodule_slices) == 2


def test_lidc_initialization_uses_processed_patient_folders(tmp_path):
    for patient_id in ("LIDC-IDRI-0001", "LIDC-IDRI-0003"):
        patient_folder = tmp_path / patient_id
        patient_folder.mkdir()
        for slice_index in range(2):
            (patient_folder / f"slice_{slice_index}.npy").touch()

    (tmp_path / "patients_masks.json").write_text(
        json.dumps({"LIDC-IDRI-0001": {"1": {"0": ["0"]}}})
    )
    (tmp_path / "patient_id_to_diagnosis.json").write_text(json.dumps({}))

    params = LIONParameter(
        task="image_prior",
        folder=tmp_path,
        training_proportion=1.0,
        validation_proportion=0.0,
        max_num_slices_per_patient=-1,
        pcg_slices_nodule=0.5,
        annotation="consensus",
        clevel=0.5,
        device=torch.device("cpu"),
        geometry=None,
    )

    dataset = LIDC_IDRI("train", parameters=params)

    assert dataset.patient_ids == ["LIDC-IDRI-0001", "LIDC-IDRI-0003"]
    assert "LIDC-IDRI-0002" not in dataset.patient_index_to_n_slices_dict
    assert dataset.patients_masks_dictionary["LIDC-IDRI-0003"] == {}
    assert dataset.slices_to_load == {
        "LIDC-IDRI-0001": [0, 1],
        "LIDC-IDRI-0003": [0, 1],
    }
