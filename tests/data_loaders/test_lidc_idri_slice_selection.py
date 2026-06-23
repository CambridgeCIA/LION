from LION.data_loaders.LIDC_IDRI import LIDC_IDRI


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
