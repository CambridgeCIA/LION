"""LIDC-IDRI resampling to one pixel spacing (pre-processing), its record in
the processed folder, and the matching geometry of the data loader and the
experiments. No LIDC-IDRI data needed: fake slices and a stub pylidc."""

import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import types

# the datasets' default folders are in LION_DATA_PATH
os.environ.setdefault("LION_DATA_PATH", tempfile.mkdtemp())

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import LION.data_loaders.LIDC_IDRI as lidc  # noqa: E402
import LION.experiments.ct_experiments as ct_experiments  # noqa: E402

FOLDER = pathlib.Path(lidc.__file__).parent / "LIDC_IDRI"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resampling = load_file("lidc_resampling", FOLDER / "resampling.py")


def disk(radius_mm, spacing, centre_mm=(0.0, 0.0), value=1000.0, background=-1000.0):
    """A 512x512 slice (HU) of a disk, with pixels of ``spacing`` mm."""
    c = (np.arange(512) - 255.5) * spacing
    y, x = np.meshgrid(c, c, indexing="ij")
    inside = (y - centre_mm[0]) ** 2 + (x - centre_mm[1]) ** 2 < radius_mm**2
    return np.where(inside, value, background).astype(np.int16)


@pytest.mark.parametrize("spacing", [0.4609375, 0.586, 0.703125, 0.9])
def test_objects_keep_their_size_and_position(spacing):
    """A disk of 60 mm radius, 40 mm off centre, has the same size and position
    in mm after resampling; outside the scan it is air."""
    out = resampling.resample_ct_slice(disk(60, spacing, (40.0, -20.0)), spacing)
    new = resampling.PIXEL_SPACING_MM
    inside = out > 0
    assert inside.sum() * new**2 == pytest.approx(np.pi * 60**2, rel=0.02)
    rows, cols = np.nonzero(inside)
    assert (rows.mean() - 255.5) * new == pytest.approx(40.0, abs=0.5)
    assert (cols.mean() - 255.5) * new == pytest.approx(-20.0, abs=0.5)
    # the scan's field of view is 512 * spacing; beyond it, air
    edge = int(256 - 256 * spacing / new) - 2
    assert np.all(out[:, :edge] == -1000)


def test_masks_stay_binary_with_their_area():
    mask = (disk(10, 0.6) > 0).astype(np.int16)
    out = resampling.resample_mask(mask, 0.6)
    assert set(np.unique(out)) <= {0, 1} and out.dtype == np.int16
    assert out.sum() * resampling.PIXEL_SPACING_MM**2 == pytest.approx(
        mask.sum() * 0.6**2, rel=0.1
    )


# --------------------------------------------------------------------------
# The pre-processing script, with a stub pylidc (spacings from its database)


SPACINGS = {"LIDC-IDRI-0001": 0.703125, "LIDC-IDRI-0002": 0.5}


@pytest.fixture
def preprocessing(monkeypatch):
    pylidc = types.ModuleType("pylidc")

    class Scan:
        patient_id = types.SimpleNamespace(__eq__=None)

    class Query:
        def __init__(self):
            self.patient = None

        def filter(self, condition):
            self.patient = condition
            return self

        def first(self):
            return types.SimpleNamespace(pixel_spacing=SPACINGS[self.patient])

    class PatientId:
        def __eq__(self, other):
            return other

    Scan.patient_id = PatientId()
    pylidc.Scan, pylidc.Annotation = Scan, object
    pylidc.query = lambda cls: Query()
    monkeypatch.setitem(sys.modules, "pylidc", pylidc)
    return load_file("pre_process_lidc_idri", FOLDER / "pre_process_lidc_idri.py")


def fake_processed_patient(folder, spacing):
    folder.mkdir(parents=True)
    for k in range(3):
        np.save(folder / f"slice_{k}.npy", disk(50 + 10 * k, spacing))
    np.save(
        folder / "mask_1_nodule_0_annotation_7.npy",
        (disk(15, spacing) > 0).astype(np.int16),
    )


def test_old_dataset_is_resampled_in_place(preprocessing, tmp_path):
    for patient, spacing in SPACINGS.items():
        fake_processed_patient(tmp_path / patient, spacing)
    (tmp_path / "patients_masks.json").write_text("{}")
    with pytest.warns(UserWarning, match="before October 2026"):
        assert lidc.LIDC_IDRI.read_pixel_spacing(tmp_path) is None

    preprocessing.resample_processed_dataset(tmp_path)

    for patient, spacing in SPACINGS.items():
        folder = tmp_path / patient
        record = json.loads((folder / "pixel_spacing.json").read_text())
        assert record == {
            "original_pixel_spacing_mm": spacing,
            "pixel_spacing_mm": lidc.PIXEL_SPACING_MM,
        }
        for k in range(3):
            np.testing.assert_array_equal(
                np.load(folder / f"slice_{k}.npy"),
                resampling.resample_ct_slice(disk(50 + 10 * k, spacing), spacing),
            )
        mask = np.load(folder / "mask_1_nodule_0_annotation_7.npy")
        assert set(np.unique(mask)) == {0, 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "LIDC-IDRI-0001",
        "LIDC-IDRI-0002",
        "patients_masks.json",
        "pre_processing.json",
    ]  # no temporary folders left
    assert lidc.LIDC_IDRI.read_pixel_spacing(tmp_path) == lidc.PIXEL_SPACING_MM

    # running it again changes nothing (resampled patients are skipped)
    before = np.load(tmp_path / "LIDC-IDRI-0001" / "slice_0.npy")
    preprocessing.resample_processed_dataset(tmp_path)
    np.testing.assert_array_equal(
        np.load(tmp_path / "LIDC-IDRI-0001" / "slice_0.npy"), before
    )


# --------------------------------------------------------------------------
# Geometry


def test_default_geometry_covers_the_field_of_view():
    geometry = lidc.LIDC_IDRI.default_geometry()
    assert list(geometry.image_shape) == [1, 512, 512]
    assert geometry.image_size[-1] / geometry.image_shape[-1] == lidc.PIXEL_SPACING_MM
    # radius of the circle the fan beam sees: dso * sin(half fan angle)
    half_fan = np.arctan(geometry.detector_size[-1] / 2 / geometry.dsd)
    assert geometry.dso * np.sin(half_fan) >= 250  # the 500 mm field of view
    assert len(geometry.angles) == 360


def test_loader_warns_if_the_geometry_does_not_match():
    dataset = lidc.LIDC_IDRI.__new__(lidc.LIDC_IDRI)  # no data needed
    dataset.pixel_spacing_mm = lidc.PIXEL_SPACING_MM
    dataset.parameters = types.SimpleNamespace(
        geometry=lidc.LIDC_IDRI.default_geometry()
    )
    dataset.check_geometry()  # no warning
    dataset.parameters.geometry = ct_experiments.ctgeo.Geometry.default_parameters()
    with pytest.warns(UserWarning, match="pixel size"):
        dataset.check_geometry()


@pytest.mark.parametrize(
    "cls, preset",
    [
        (ct_experiments.ExtremeLowDoseCTRecon, "default_parameters"),
        (ct_experiments.LimitedAngleCTRecon, "sparse_angle_parameters"),
        (ct_experiments.SparseAngleCTRecon, "sparse_view_parameters"),
    ],
)
def test_lidc_experiments_use_the_lidc_geometry_with_the_same_angles(cls, preset):
    geometry = cls.default_parameters().geometry
    old = getattr(ct_experiments.ctgeo.Geometry, preset)()
    np.testing.assert_allclose(geometry.angles, old.angles)
    assert geometry.image_size[-1] == 500 and geometry.detector_shape[-1] == 1024


def test_new_patients_are_resampled(preprocessing, tmp_path):
    """process_volume resamples the scan's slices and records the spacing."""
    volume = np.stack([disk(80, 0.6), disk(90, 0.6)], axis=-1)
    scan = types.SimpleNamespace(to_volume=lambda: volume, pixel_spacing=0.6)
    preprocessing.process_volume(scan, tmp_path)
    for k in range(2):
        np.testing.assert_array_equal(
            np.load(tmp_path / f"slice_{k}.npy"),
            resampling.resample_ct_slice(volume[:, :, k], 0.6),
        )
    assert preprocessing.is_resampled(tmp_path)


# --------------------------------------------------------------------------
# Damaged files: left out with a warning


def test_incomplete_npy_files_are_found(tmp_path):
    good, bad = tmp_path / "good.npy", tmp_path / "bad.npy"
    np.save(good, np.zeros((512, 512), dtype=np.int16))
    np.save(bad, np.zeros((512, 512), dtype=np.int16))
    bad.write_bytes(bad.read_bytes()[:131072])  # as a cut-off copy
    assert lidc.npy_problem(good) is None
    assert "incomplete" in lidc.npy_problem(bad)
    assert "unreadable" in lidc.npy_problem(tmp_path / "missing.npy")


@pytest.mark.parametrize("task", ["reconstruction", "segmentation"])
def test_loader_leaves_out_damaged_slices(tmp_path, task):
    for patient in ("LIDC-IDRI-0001", "LIDC-IDRI-0002", "LIDC-IDRI-0003"):
        (tmp_path / patient).mkdir()
        for k in range(3):
            np.save(
                tmp_path / patient / f"slice_{k}.npy", np.zeros((512, 512), np.int16)
            )
    np.save(
        tmp_path / "LIDC-IDRI-0001" / "mask_2_nodule_0_annotation_5.npy",
        np.zeros((512, 512), np.int16),
    )
    damaged = tmp_path / "LIDC-IDRI-0001" / "slice_1.npy"
    damaged.write_bytes(damaged.read_bytes()[:1000])
    mask = tmp_path / "LIDC-IDRI-0001" / "mask_2_nodule_0_annotation_5.npy"
    mask.write_bytes(mask.read_bytes()[:1000])

    dataset = lidc.LIDC_IDRI.__new__(lidc.LIDC_IDRI)  # no full dataset needed
    dataset.parameters = types.SimpleNamespace(task=task)
    dataset.path_to_processed_dataset = tmp_path
    dataset.not_resampled = ["LIDC-IDRI-0003"]
    selected = {
        p: [0, 1, 2] for p in ("LIDC-IDRI-0001", "LIDC-IDRI-0002", "LIDC-IDRI-0003")
    }
    with pytest.warns(UserWarning, match="left out"):
        kept = dataset.drop_damaged_slices(selected)
    # slice 2's mask only matters when masks are loaded
    assert kept["LIDC-IDRI-0001"] == ([0] if task == "segmentation" else [0, 2])
    assert kept["LIDC-IDRI-0002"] == [0, 1, 2]
    assert kept["LIDC-IDRI-0003"] == []


def test_dataset_record_lists_patients_not_resampled(preprocessing, tmp_path):
    fake_processed_patient(tmp_path / "LIDC-IDRI-0001", 0.7)
    fake_processed_patient(tmp_path / "LIDC-IDRI-0002", 0.5)
    preprocessing.resample_processed_patient(tmp_path / "LIDC-IDRI-0002", 0.5)
    preprocessing.write_pre_processing_file(tmp_path)
    assert lidc.LIDC_IDRI.read_not_resampled(tmp_path) == ["LIDC-IDRI-0001"]
    assert lidc.LIDC_IDRI.read_pixel_spacing(tmp_path) == lidc.PIXEL_SPACING_MM


def test_conversion_goes_on_after_a_damaged_patient(preprocessing, tmp_path):
    for patient, spacing in SPACINGS.items():
        fake_processed_patient(tmp_path / patient, spacing)
    damaged = tmp_path / "LIDC-IDRI-0001" / "slice_1.npy"
    original = damaged.read_bytes()[:131072]
    damaged.write_bytes(original)
    preprocessing.resample_processed_dataset(tmp_path)
    assert not preprocessing.is_resampled(tmp_path / "LIDC-IDRI-0001")
    assert damaged.read_bytes() == original  # left as it was
    assert preprocessing.is_resampled(tmp_path / "LIDC-IDRI-0002")
    assert not list(tmp_path.glob("*.tmp"))
    assert lidc.LIDC_IDRI.read_not_resampled(tmp_path) == ["LIDC-IDRI-0001"]


# --------------------------------------------------------------------------
# LIDC-IDRI: the patient list


def fake_lidc(folder, numbers, extra=()):
    """Patient folders LIDC-IDRI-NNNN (two slices each), and other entries."""
    import json

    for n in numbers:
        patient = folder / f"LIDC-IDRI-{n:04d}"
        patient.mkdir()
        for k in range(2):
            np.save(patient / f"slice_{k}.npy", np.zeros((512, 512), np.int16))
    for name in extra:
        (folder / name).mkdir()
    ids = [f"LIDC-IDRI-{n:04d}" for n in range(1, 13)]
    (folder / "patients_masks.json").write_text(json.dumps({i: {} for i in ids}))
    (folder / "patient_id_to_diagnosis.json").write_text(
        json.dumps({i: 0 for i in ids})
    )
    (folder / "pre_processing.json").write_text(
        json.dumps({"pixel_spacing_mm": lidc.PIXEL_SPACING_MM, "not_resampled": []})
    )


def test_lidc_patient_list(tmp_path):
    """The patients are the LIDC-IDRI-NNNN folders; the legacy list (before
    October 2026) is 1..N for N entries, which skipped the last patients when
    some number had no folder, and counted leftover folders."""
    fake_lidc(tmp_path, [1, 2, 4, 5, 6], extra=["LIDC-IDRI-0002.tmp"])
    patients, total = lidc.LIDC_IDRI.patient_list(tmp_path)
    assert patients == [f"LIDC-IDRI-{n:04d}" for n in (1, 2, 4, 5, 6)]
    assert total == 5
    patients, total = lidc.LIDC_IDRI.patient_list(tmp_path, legacy=True)
    assert patients == [f"LIDC-IDRI-{n:04d}" for n in range(1, 7)]  # 0003: none
    assert total == 7


@pytest.mark.parametrize("saved_before", [False, True])
def test_lidc_loader_uses_the_legacy_list_for_old_parameters(tmp_path, saved_before):
    fake_lidc(tmp_path, range(1, 11), extra=["LIDC-IDRI-0010.old"])
    params = lidc.LIDC_IDRI.default_parameters(task="segmentation")
    params.folder = tmp_path
    assert params.legacy_patient_list is False
    if saved_before:  # parameter files from before October 2026
        del params.legacy_patient_list
    data = lidc.LIDC_IDRI("test", parameters=params)
    if saved_before:  # 11 entries: patients 1-11, proportions of 12
        assert data.patient_ids[-1] == "LIDC-IDRI-0011"
        assert data.testing_patients_list == ["LIDC-IDRI-0011"]
    else:  # the 10 patient folders
        assert data.patient_ids == [f"LIDC-IDRI-{n:04d}" for n in range(1, 11)]
        assert data.testing_patients_list == ["LIDC-IDRI-0010"]
