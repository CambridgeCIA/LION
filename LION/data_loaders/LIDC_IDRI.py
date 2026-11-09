# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Emilien Valat
# Modifications: Michelle Limbach, Ander Biguri
# =============================================================================


from typing import List, Dict
import pathlib
import math
import re
import warnings

import torch
import numpy as np
import json


from LION.utils import paths
import LION.operators.ct.ct_utils as ct
from LION.operators.ct.ct_geometry import Geometry
from LION.utils.parameter import LIONParameter
from LION.utils.device import default_device
from LION.data_loaders.LIONDataset import LIONDataset
from LION.utils.rng import seeded
from LION.utils.citation import citation

# The pre-processing (LIDC_IDRI/pre_process_lidc_idri.py, resampling.py)
# resamples every slice to this pixel spacing on 512x512: 500 mm, the largest
# field of view in LIDC-IDRI, so nothing is cropped.
PIXEL_SPACING_MM = 500.0 / 512.0
# written by the pre-processing once every patient is resampled
PRE_PROCESSING_FILE = "pre_processing.json"


def npy_problem(path: pathlib.Path):
    """None if the .npy file exists and holds all the data its header says,
    else what is wrong. Reads the header only."""
    try:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(f)
            else:
                shape, _, dtype = np.lib.format.read_array_header_2_0(f)
            expected = f.tell() + int(np.prod(shape)) * dtype.itemsize
    except Exception as error:  # missing, or not a .npy file
        return f"unreadable: {error}"
    size = path.stat().st_size
    if size != expected:
        return f"incomplete, {size} of {expected} bytes"
    return None


def format_index(index: int) -> str:
    """A patient number as LIDC-IDRI writes it, 4 digits: 7 -> "0007"."""
    str_index = f"{index:04d}"
    if len(str_index) != 4:
        raise ValueError(f"index must have at most 4 digits, not {index}")
    return str_index


def load_json(file_path: pathlib.Path):
    """The contents of the JSON file ``file_path`` (FileNotFoundError if there
    is none)."""
    if not file_path.is_file():
        raise FileNotFoundError(f"No file found at {file_path}")
    with open(file_path) as file:
        return json.load(file)


def choose_random_annotation(
    nodule_annotations_list: List,
) -> str:
    """One of the annotations, drawn with torch's random numbers (so
    ``rng.seeded`` makes the choice reproducible)."""
    return nodule_annotations_list[int(torch.randint(len(nodule_annotations_list), ()))]


def create_consensus_annotation(
    path_to_patient_folder: pathlib.Path,
    slice_index: int,
    nodule_index: str,
    nodule_annotations_list: List,
    clevel: float,
) -> torch.int16:
    """The consensus mask of one nodule on one slice: the pixels that at least
    a fraction ``clevel`` of its annotations (``nodule_annotations_list``)
    mark."""
    masks = []
    if isinstance(path_to_patient_folder, str):
        path_to_patient_folder = pathlib.Path(path_to_patient_folder)
    for annotation in nodule_annotations_list:
        path_to_mask = path_to_patient_folder.joinpath(
            f"mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation}.npy"
        )
        current_annotation_mask = np.load(path_to_mask)
        masks.append(current_annotation_mask)

    nodule_mask = torch.from_numpy(np.mean(masks, axis=0) >= clevel)
    return nodule_mask


def _evenly_spaced(slices: List, n: int) -> List:
    """n of the given slices, evenly spaced (the first one included)."""
    indices = np.linspace(0, len(slices), n, dtype=int, endpoint=False)
    return list(np.array(slices)[indices])


def select_slices(
    non_nodule_slices: List,
    nodule_slices: List,
    max_slices: int,
    pcg_slices_nodule: float,
    legacy: bool = False,
) -> List:
    """Slices to load for one patient, sorted.

    Takes up to ``max_slices`` slices (all of them if -1), a fraction
    ``pcg_slices_nodule`` of them with a nodule. If the patient has too few
    slices of one kind, more of the other kind are taken; patients without
    nodule slices contribute non-nodule slices.

    ``legacy=True`` reproduces LION before October 2026: at most
    min(max_slices, #non-nodule, #nodule) slices, so patients without nodule
    slices contributed none, and -1 meant at most 1000 of each.
    """
    n_non_available, n_nodule_available = len(non_nodule_slices), len(nodule_slices)
    if legacy:
        if max_slices == -1:
            max_slices = 1000
        n = min(max_slices, n_non_available, n_nodule_available)
        n_non = int(np.ceil(n * (1 - pcg_slices_nodule)))
        n_nodule = n - n_non
    elif max_slices == -1:
        n_non, n_nodule = n_non_available, n_nodule_available
    else:
        n = min(max_slices, n_non_available + n_nodule_available)
        # same split as the legacy rule when both kinds are available...
        n_non = min(int(np.ceil(n * (1 - pcg_slices_nodule))), n_non_available)
        n_nodule = min(n - n_non, n_nodule_available)
        # ...and the rest from the other kind when one runs out
        n_non = min(n - n_nodule, n_non_available)
    selected = _evenly_spaced(non_nodule_slices, n_non) + _evenly_spaced(
        nodule_slices, n_nodule
    )
    return sorted(selected)


class LIDC_IDRI(LIONDataset):
    """The LIDC-IDRI dataset: lung CT scans of 1010 patients with nodules
    annotated by up to four radiologists, pre-processed into 512x512 slices
    (``LIDC_IDRI/pre_process_lidc_idri.py``). Patients are split into training,
    validation and test (``mode``); each patient's slices stay in one split.

    What a sample is depends on ``parameters.task``:

    - "reconstruction": (sinogram, image): the image in mass density, and its
      sinogram with the operator of ``operator_params`` (the geometry), with
      ``sinogram_transform`` (e.g. an experiment's noise) applied;
    - "segmentation", "joint", "end_to_end": (image, mask): the mask is
      (background, nodule) channels, from the annotations (``annotation``);
    - "diagnostic": the patient's diagnosis.

    ``parameters`` (``default_parameters()``):

    - ``training_proportion``, ``validation_proportion``: of the patients (0.8,
      0.1; ``testing_proportion`` is the rest, recorded only);
    - ``max_num_slices_per_patient``: slices taken per patient (5; -1: all);
    - ``pcg_slices_nodule``: the fraction of them with a nodule (0.5; more of
      the other kind when a patient has too few);
    - ``legacy_slice_selection``, ``legacy_patient_list``: reproduce the
      selection of LION before October 2026 (False);
    - ``task``, ``folder`` (the pre-processed dataset), ``operator_params``;
    - ``annotation``: "consensus" (the pixels at least ``clevel`` of the
      annotations mark; 0.5) or "random" (one annotation per nodule, chosen
      with ``annotation_seed``; 0).

    Samples are on LION's device (``default_device()``). Cite it with
    ``cite()``."""

    def __init__(
        self,
        mode,
        operator_params: Geometry = None,
        parameters: LIONParameter = None,
    ):
        """The ``mode`` split ("train", "validation" or "test") with
        ``parameters`` (``default_parameters(operator_params)`` if None);
        ``operator_params``: the CT geometry, for sinograms.

        Pixel spacing: the pre-processing resamples every slice to
        ``PIXEL_SPACING_MM`` (500 mm over 512x512 pixels, the largest field of
        view in the dataset, so no scan is cropped; smaller ones are surrounded
        by air), so all images have the same physical size.
        ``LIDC_IDRI.default_geometry()`` is the matching CT geometry. Datasets
        processed before October 2026 have each scan's own spacing (0.46 to
        0.98 mm) and no ``pre_processing.json``: the loader warns, and ``python
        LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py --resample-only``
        resamples them in place (no DICOM files needed).

        Damaged data: the download and pre-processing can leave incomplete
        files. The loader leaves out, with a warning, selected slices whose
        files are incomplete or unreadable, and the patients
        ``pre_processing.json`` lists as not resampled.

        Images: for reconstruction, the CT images (Hounsfield units) are turned
        into mass density in g/cm^3 (``ct_utils.from_HU_to_density``: air
        0.0012, water 1.0137, 500 HU 1.52). With LION's geometries in mm, the
        sinograms are density line integrals in g/cm^3 * mm, which is what the
        noise model of ``ct_utils.sinogram_add_noise`` expects: the experiments
        (``ct_experiments``) add the noise."""

        # Input parsing
        if mode not in ["train", "validation", "test"]:
            raise ValueError(
                f'Wrong mode argument {mode!r}, must be in ["train", "validation", "test"]'
            )

        if parameters is None:
            parameters = LIDC_IDRI.default_parameters(operator_params=operator_params)
        self.parameters = parameters
        self.mode = mode
        self.operator_params = operator_params

        task = self.parameters.task
        tasks = ["joint", "end_to_end", "segmentation", "reconstruction", "diagnostic"]
        if task not in tasks:
            raise ValueError(f"task argument {task!r} not in {tasks}")

        if task not in ["segmentation", "reconstruction", "end_to_end"]:
            raise NotImplementedError(f"task {task} not implemented yet")

        if (
            task in ["reconstruction"]
            and operator_params is None
            and self.parameters.operator_params is None
        ):
            raise ValueError("geometry input required for recosntruction modes")

        # Aux variable setting
        self.sinogram_transform = None
        self.sinogram_transform_seed = None
        self.image_transform = None
        # the device selected in LION (a device in parameter files is not used)
        self.device = default_device()

        if task in ["reconstruction"]:
            self.image_transform = (
                ct.from_HU_to_density
            )  # g/cm^3 (was from_HU_to_mu, same values)
        if task in ["segmentation"]:
            self.image_transform = ct.from_HU_to_normal

        # parameters loaded from JSON have the geometry as a plain LIONParameter
        if self.parameters.operator_params is not None and not isinstance(
            self.parameters.operator_params, Geometry
        ):
            self.parameters.operator_params = Geometry.init_from_parameter(
                self.parameters.operator_params
            )
        if operator_params is not None:
            self.parameters.operator_params = operator_params
            self.operator = operator_params.make_operator()
        elif self.parameters.operator_params is not None:
            self.operator = self.parameters.operator_params.make_operator()
        # Start of Patient pre-processing

        self.path_to_processed_dataset = pathlib.Path(self.parameters.folder)
        self.pixel_spacing_mm = self.read_pixel_spacing(self.path_to_processed_dataset)
        # patients the pre-processing could not resample (e.g. a damaged file)
        self.not_resampled = self.read_not_resampled(self.path_to_processed_dataset)
        self.check_geometry()
        self.patients_masks_dictionary = load_json(
            self.path_to_processed_dataset.joinpath("patients_masks.json")
        )

        # Add patients without masks, for now hardcoded, find a solution in preprocessing
        self.patients_masks_dictionary["LIDC-IDRI-0238"] = {}
        self.patients_masks_dictionary["LIDC-IDRI-0585"] = {}

        self.patients_diagnosis_dictionary = load_json(
            self.path_to_processed_dataset.joinpath("patient_id_to_diagnosis.json")
        )
        self.patient_ids_all, self.total_patients = self.patient_list(
            self.path_to_processed_dataset,
            legacy=self.parameters.legacy_patient_list,
        )
        self.num_slices_per_patient = self.parameters.max_num_slices_per_patient
        self.pcg_slices_nodule = self.parameters.pcg_slices_nodule
        self.annotation = self.parameters.annotation
        self.clevel = (
            self.parameters.clevel
        )  # consensus level, only used if annotation == consensus
        self.patient_index_to_n_slices_dict: Dict = {
            patient_id: len(
                list(
                    self.path_to_processed_dataset.joinpath(patient_id).glob(
                        "slice_*.npy"
                    )
                )
            )
            for patient_id in self.patient_ids_all
        }

        # Dict with all slices of each patient
        self.patient_index_to_slices_index_dict: Dict = {
            patient_id: list(
                np.arange(
                    0,
                    self.patient_index_to_n_slices_dict[patient_id],
                    1,
                )
            )
            for patient_id in self.patient_ids_all
        }

        # Dict with all nodule slices of each patient
        # Converts the keys from self.patients_masks_dictionary to integer
        self.patient_index_to_nodule_slices_index_dict: Dict = {
            patient_id: [
                int(item)
                for item in list(
                    self.patients_masks_dictionary.get(patient_id, {}).keys()
                )
            ]
            for patient_id in self.patient_ids_all
        }

        # Dict with all non-nodule slices of each patient
        # Computes as difference of all slices dict and dict with nodules
        self.patient_index_to_non_nodule_slices_index_dict: Dict = {
            patient_id: list(
                set(self.patient_index_to_slices_index_dict[patient_id])
                - set(self.patient_index_to_nodule_slices_index_dict[patient_id])
            )
            for patient_id in self.patient_ids_all
        }

        # Corrupted data handling
        # Delete all slices that contain a nodule that has more than 4 annotations
        self.removed_slices: Dict = {}
        for (
            patient_id,
            nodule_slices_list,
        ) in self.patient_index_to_nodule_slices_index_dict.items():
            self.removed_slices[patient_id] = []
            for slice_index in nodule_slices_list:
                all_nodules_dict: Dict = self.patients_masks_dictionary[patient_id][
                    f"{slice_index}"
                ]
                for _, nodule_annotations_list in all_nodules_dict.items():
                    if len(nodule_annotations_list) > 4:
                        self.removed_slices[patient_id].append(slice_index)
                        break

        self.patient_index_to_nodule_slices_index_dict: Dict = {
            patient_id: list(
                set(self.patient_index_to_nodule_slices_index_dict[patient_id])
                - set(self.removed_slices[patient_id])
            )
            for patient_id in self.patient_ids_all
        }

        ##% Divide dataset in training/validation/testing
        self.training_proportion = self.parameters.training_proportion
        self.validation_proportion = self.parameters.validation_proportion
        # Commpute number if images for each
        self.n_patients_training = math.floor(
            self.training_proportion * (self.total_patients)
        )
        self.n_patients_validation = math.floor(
            self.validation_proportion * (self.total_patients)
        )
        self.n_patients_testing = (
            self.total_patients - self.n_patients_training - self.n_patients_validation
        )

        # Get patient IDs for each
        self.patient_ids = list(self.patient_index_to_n_slices_dict.keys())
        self.training_patients_list = self.patient_ids[: self.n_patients_training]
        self.validation_patients_list = self.patient_ids[
            self.n_patients_training : self.n_patients_training
            + self.n_patients_validation
        ]
        self.testing_patients_list = self.patient_ids[
            self.n_patients_training + self.n_patients_validation :
        ]
        if self.mode == "train":
            patient_list_to_load = self.training_patients_list
        elif self.mode == "validation":
            patient_list_to_load = self.validation_patients_list
        elif self.mode == "test":
            patient_list_to_load = self.testing_patients_list
        else:
            raise NotImplementedError(
                f"mode {self.mode} not implemented, try training, validation or testing"
            )

        self.slices_to_load = self.get_slices_to_load(
            patient_list_to_load,
            self.patient_index_to_non_nodule_slices_index_dict,
            self.patient_index_to_nodule_slices_index_dict,
            self.num_slices_per_patient,
            self.pcg_slices_nodule,
        )
        self.slices_to_load = self.drop_damaged_slices(self.slices_to_load)
        self.slice_index_to_patient_id_list = self.get_slice_index_to_patient_id_list(
            self.slices_to_load
        )
        self.patient_id_to_first_index_dict = self.get_patient_id_to_first_index_dict(
            self.slices_to_load
        )

    @staticmethod
    def patient_list(path_to_processed_dataset, legacy=False):
        """The patients of the dataset, in order, and the number the
        train/validation/test proportions are taken of.

        The patients are the folders named LIDC-IDRI-NNNN, so patients without a
        scan (0238 and 0585) or that failed pre-processing are not in the list,
        and other folders (e.g. a leftover ``LIDC-IDRI-0002.tmp``) are ignored.

        ``legacy=True`` is LION's list before October 2026 (parameter files
        without ``legacy_patient_list`` use it): patients 1 to N, N the number
        of entries named LIDC-IDRI-*, with proportions of N + 1. As 0238 and 0585
        have no scan, it leaves out the last two patients (1011 and 1012), and any
        other entry (a missing patient, a leftover folder) moves the splits.
        """
        folder = pathlib.Path(path_to_processed_dataset)
        if legacy:
            n = len(list(folder.glob("LIDC-IDRI-*")))
            return [f"LIDC-IDRI-{format_index(i)}" for i in range(1, n + 1)], n + 1
        patients = sorted(
            p.name
            for p in folder.glob("LIDC-IDRI-*")
            if p.is_dir() and re.fullmatch(r"LIDC-IDRI-\d{4}", p.name)
        )
        return patients, len(patients)

    @staticmethod
    def read_not_resampled(path_to_processed_dataset):
        """Patients the pre-processing could not resample (pre_processing.json)."""
        path = pathlib.Path(path_to_processed_dataset).joinpath(PRE_PROCESSING_FILE)
        if not path.is_file():
            return []
        return load_json(path).get("not_resampled", [])

    def drop_damaged_slices(self, slices_to_load: Dict) -> Dict:
        """Leave out, with a warning, the slices that cannot be loaded: those of
        patients the pre-processing could not resample, and incomplete or
        unreadable files (and, for tasks with masks, their mask files).

        A damaged file can come from an interrupted download, pre-processing or
        copy. Only the selected slices are checked (their headers and sizes),
        so this is quick unless every slice is selected.
        """
        needs_masks = self.parameters.task in ["joint", "end_to_end", "segmentation"]
        kept, dropped = {}, []
        for patient_id, slice_indices in slices_to_load.items():
            if patient_id in self.not_resampled:
                dropped += [f"{patient_id} (not resampled)"]
                kept[patient_id] = []
                continue
            folder = self.path_to_processed_dataset.joinpath(patient_id)
            masks = list(folder.glob("mask_*.npy")) if needs_masks else []
            kept[patient_id] = []
            for slice_index in slice_indices:
                files = [folder.joinpath(f"slice_{slice_index}.npy")]
                files += [m for m in masks if m.name.split("_")[1] == str(slice_index)]
                problems = [
                    f"{f.name}: {npy_problem(f)}" for f in files if npy_problem(f)
                ]
                if problems:
                    dropped += [f"{patient_id}/{p}" for p in problems]
                else:
                    kept[patient_id].append(slice_index)
        if dropped:
            warnings.warn(
                f"LIDC-IDRI: left out {len(dropped)} damaged or unusable files "
                f"(re-run the pre-processing for these patients): {dropped}"
            )
        return kept

    @staticmethod
    def read_pixel_spacing(path_to_processed_dataset):
        """The pixel spacing (mm) of the processed slices, or None for a
        dataset processed before October 2026 (each scan's own spacing)."""
        path = pathlib.Path(path_to_processed_dataset).joinpath(PRE_PROCESSING_FILE)
        if not path.is_file():
            warnings.warn(
                f"{path_to_processed_dataset} was pre-processed before October 2026: "
                "its slices have each scan's own pixel spacing (0.46 to 0.98 mm), "
                "not one for all. Run `python LION/data_loaders/LIDC_IDRI/"
                "pre_process_lidc_idri.py --resample-only --folder "
                f"{path_to_processed_dataset}` to resample them."
            )
            return None
        return load_json(path)["pixel_spacing_mm"]

    def check_geometry(self):
        """Warn if the geometry's pixel size is not the data's pixel spacing:
        then sizes, and the physical noise model's path lengths, are wrong."""
        operator_params = self.parameters.operator_params
        if self.pixel_spacing_mm is None or operator_params is None:
            return
        pixel_size = operator_params.image_size[-1] / operator_params.image_shape[-1]
        if not math.isclose(pixel_size, self.pixel_spacing_mm, rel_tol=1e-6):
            warnings.warn(
                f"The geometry's pixel size ({pixel_size:.4g} mm) is not the pixel "
                f"spacing of the data ({self.pixel_spacing_mm:.4g} mm); "
                "LIDC_IDRI.default_geometry() matches the data."
            )

    @staticmethod
    def default_geometry(angles=None):
        """The fan-beam geometry of the pre-processed slices: 512x512 pixels of
        ``PIXEL_SPACING_MM`` (500 mm), source to centre 575 mm, source to
        detector 1050 mm, as LION's default geometry, with a 1024 x 1 mm
        detector: it covers a 504 mm circle, like the ~50 degree fan of a
        clinical scanner. ``angles``: in radians (None: 360 over 360 degrees).
        """
        if angles is None:
            angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        return Geometry(
            image_shape=[1, 512, 512],
            image_size=[PIXEL_SPACING_MM, 500.0, 500.0],
            detector_shape=[1, 1024],
            detector_size=[1, 1024],
            dso=575,
            dsd=1050,
            mode="fan",
            angles=np.asarray(angles),
        )

    @staticmethod
    def default_parameters(operator_params=None, task="reconstruction"):
        """The parameters (see the class), for ``task`` ("reconstruction" needs
        ``operator_params``, the geometry)."""
        param = LIONParameter()
        param.name = "LIDC-IDRI Data Loader"
        param.training_proportion = 0.8
        param.validation_proportion = 0.1
        param.testing_proportion = (
            1 - param.training_proportion - param.validation_proportion
        )  # not used, but for metadata
        param.max_num_slices_per_patient = 5
        param.pcg_slices_nodule = 0.5
        # True reproduces the slice selection of LION before October 2026
        # (patients without nodule slices were left out); see select_slices
        param.legacy_slice_selection = False
        # True reproduces the patient list of LION before October 2026, which
        # left out the last two patients; see LIDC_IDRI.patient_list
        param.legacy_patient_list = False
        param.task = task
        param.folder = paths.LIDC_IDRI_PROCESSED_DATASET_PATH
        if task == "reconstruction" and operator_params is None:
            raise ValueError(
                "For reconstruction task geometry needs to be input to default_parameters(geometry=geometry_param)"
            )

        # segmentation specific
        param.clevel = 0.5
        param.annotation = "consensus"
        # annotation="random": the seed of each slice's choice (see rng.seeded)
        param.annotation_seed = 0
        param.operator_params = operator_params
        return param

    def get_slices_to_load(
        self,
        patient_list: List,
        non_nodule_slices_dict: Dict,
        nodule_slices_dict: Dict,
        num_slices_per_patient: int,
        pcg_slices_nodule: float,
    ):
        """The slices to load of each patient of ``patient_list``: ``{patient:
        [slice, ...]}``, ``num_slices_per_patient`` of them (-1: all), a
        fraction ``pcg_slices_nodule`` with a nodule (``select_slices``), from
        each patient's slices without (``non_nodule_slices_dict``) and with
        (``nodule_slices_dict``) a nodule."""

        patient_id_to_slices_to_load_dict = (
            {}
        )  # Empty dict which should contain patient id as key and slice ids as array of values

        legacy = self.parameters.legacy_slice_selection
        for patient_id in patient_list:  # Loop over every patient
            patient_id_to_slices_to_load_dict[patient_id] = select_slices(
                non_nodule_slices_dict[patient_id],
                nodule_slices_dict[patient_id],
                num_slices_per_patient,
                pcg_slices_nodule,
                legacy=legacy,
            )

        return patient_id_to_slices_to_load_dict

    def get_patient_id_to_first_index_dict(self, patient_with_slices_to_load: Dict):
        """The index of each patient's first sample: ``{patient: index}``, from
        the slices to load (``patient_with_slices_to_load``)."""
        patient_id_to_first_index_dict = {}
        global_index = 0
        for patient_id in patient_with_slices_to_load:
            path_to_folder = self.path_to_processed_dataset.joinpath(patient_id)
            patient_id_to_first_index_dict[patient_id] = global_index

            if len(patient_with_slices_to_load[patient_id]) < len(
                list(path_to_folder.glob("slice_*.npy"))
            ):
                global_index += len(patient_with_slices_to_load[patient_id])
            else:
                global_index += len(list(path_to_folder.glob("slice_*.npy")))
        return patient_id_to_first_index_dict

    def get_slice_index_to_patient_id_list(self, patient_with_slices_to_load: Dict):
        """The patient of each sample, in order: each patient once per slice it
        has to load."""
        slice_index_to_patient_id_list = []
        for patient_id in patient_with_slices_to_load:
            path_to_folder = self.path_to_processed_dataset.joinpath(patient_id)

            if len(patient_with_slices_to_load[patient_id]) < len(
                list(path_to_folder.glob("slice_*.npy"))
            ):
                n_slices = len(patient_with_slices_to_load[patient_id])
            else:
                n_slices = len(list(path_to_folder.glob("slice_*.npy")))

            for slice_index in range(n_slices):
                slice_index_to_patient_id_list.append(patient_id)
        return slice_index_to_patient_id_list

    def get_reconstruction_tensor(self, file_path: pathlib.Path) -> torch.Tensor:
        """The slice saved at ``file_path``, (1, 512, 512), on the dataset's
        device."""
        tensor = torch.from_numpy(np.load(file_path)).unsqueeze(0).to(self.device)
        return tensor

    def set_sinogram_transform(self, sinogram_transform, seed=None):
        """A function applied to each clean sinogram, e.g. the noise of an
        experiment.

        ``seed``: an integer makes its random numbers depend only on the seed
        and the slice (patient and slice file): each slice gets the same
        noisy sinogram in every epoch, run, order and DataLoader worker, as a
        scan measures it once. None: fresh random numbers each time.
        """
        self.sinogram_transform = sinogram_transform
        self.sinogram_transform_seed = seed

    def set_image_transform(self, image_transform):
        """A function applied to each image, after loading it."""
        self.image_transform = image_transform

    def compute_clean_sinogram(self, image=None) -> torch.Tensor:

        """The sinogram of ``image`` with the dataset's operator (from
        ``operator_params``), without noise."""
        if self.operator is None:
            raise AttributeError("CT operator not know. Have you given a ct geometry?")
        sinogram = self.operator(image)
        return sinogram

    def get_mask_tensor(self, patient_id: str, slice_index: int) -> torch.Tensor:
        ## First, assess if the slice has a nodule
        """The nodule mask of one slice (``slice_index`` of ``patient_id``):
        (2, 512, 512), background and nodule channels, from the annotations as
        ``annotation`` says; empty if the slice has no nodule."""
        try:
            mask = torch.zeros((512, 512), dtype=torch.bool)
            all_nodules_dict: Dict = self.patients_masks_dictionary[patient_id][
                f"{slice_index}"
            ]
            for nodule_index, nodule_annotations_list in all_nodules_dict.items():

                if self.annotation == "random":
                    ## If a nodule was not segmented by all the clinicians, the other annotations should not always be seen
                    while len(nodule_annotations_list) < 4:
                        nodule_annotations_list.append("")

                    # annotation_seed: each nodule of each slice gets one
                    # annotation, the same every time it is read; None or
                    # (or None) a new one each time
                    with seeded(
                        self.parameters.annotation_seed,
                        "LIDC-IDRI-annotation",
                        patient_id,
                        slice_index,
                        nodule_index,
                    ):
                        annotation = choose_random_annotation(nodule_annotations_list)
                    if annotation == "":
                        # Hopefully, that exists the try to return an empty mask
                        nodule_mask = torch.zeros((512, 512), dtype=torch.bool)
                    else:
                        path_to_mask = self.path_to_processed_dataset.joinpath(
                            f"{patient_id}/mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation}.npy"
                        )
                        nodule_mask = torch.from_numpy(np.load(path_to_mask))

                elif self.annotation == "consensus":
                    # Create consensus annotation out of all annotations of this nodule
                    path_to_patient_folder = self.path_to_processed_dataset.joinpath(
                        f"{patient_id}/"
                    )
                    nodule_mask = create_consensus_annotation(
                        path_to_patient_folder,
                        slice_index,
                        nodule_index,
                        nodule_annotations_list,
                        self.clevel,
                    )

                else:
                    raise NotImplementedError(
                        f"annotation {self.annotation} not implemented, try random or consensus"
                    )

                mask = mask.bitwise_or(nodule_mask)

        except KeyError:
            mask = torch.zeros((512, 512), dtype=torch.bool)
        # byte inversion
        background = ~mask
        return torch.stack((background, mask))

    def __len__(self):
        return len(self.slice_index_to_patient_id_list)

    def _sample(self, index):
        """The patient and slice file index of sample ``index``."""
        patient_id = self.slice_index_to_patient_id_list[index]
        position = index - self.patient_id_to_first_index_dict[patient_id]
        return patient_id, self.slices_to_load[patient_id][position]

    def sample_ids(self):
        """What each sample is, in order: ``"LIDC-IDRI-NNNN/slice_K"``."""
        return [
            "{}/slice_{}".format(*self._sample(index)) for index in range(len(self))
        ]

    def sample_groups(self):
        """The group of each sample, in order: its patient. Slices of one
        patient are near-copies, so a patient's slices are kept in one split,
        and evaluations check that no test patient was trained on."""
        return [self._sample(index)[0] for index in range(len(self))]

    def sample_files(self):
        """The files the samples are read from, in order: each slice, and for
        tasks with masks the slice's mask files."""
        files = []
        for index in range(len(self)):
            patient_id, slice_index = self._sample(index)
            folder = self.path_to_processed_dataset.joinpath(patient_id)
            files.append(folder.joinpath(f"slice_{slice_index}.npy"))
            if self.parameters.task in ["joint", "end_to_end", "segmentation"]:
                files += sorted(folder.glob(f"mask_{slice_index}_*.npy"))
        return files

    def get_specific_slice(self, patient_index, slice_index):
        ## Assumes slice and mask exist
        """The image and mask of one slice (``slice_index`` of
        ``patient_index``), whether or not it is in the dataset's selection."""
        file_path = self.path_to_processed_dataset.joinpath(
            f"{patient_index}/slice_{slice_index}.npy"
        )
        return self.get_reconstruction_tensor(file_path), self.get_mask_tensor(
            patient_index, slice_index
        )

    def __getitem__(self, index):
        if index >= len(self.slice_index_to_patient_id_list):
            raise IndexError(
                f"Index {index} out of range, max index is {len(self.slice_index_to_patient_id_list)-1}"
            )
        patient_id = self.slice_index_to_patient_id_list[index]
        first_slice_index = self.patient_id_to_first_index_dict[patient_id]
        dict_slice_index = index - first_slice_index
        slice_index_to_load = self.slices_to_load[patient_id][dict_slice_index]
        file_path = self.path_to_processed_dataset.joinpath(
            f"{patient_id}/slice_{slice_index_to_load}.npy"
        )

        if self.parameters.task in [
            "joint",
            "end_to_end",
            "segmentation",
            "reconstruction",
        ]:
            reconstruction_tensor = self.get_reconstruction_tensor(file_path)
            if self.image_transform is not None:
                reconstruction_tensor = self.image_transform(reconstruction_tensor)

        if self.parameters.task in ["joint", "end_to_end", "segmentation"]:
            mask_tensor = self.get_mask_tensor(patient_id, slice_index_to_load)
            return reconstruction_tensor, mask_tensor

        elif self.parameters.task == "reconstruction":
            sinogram = self.compute_clean_sinogram(reconstruction_tensor.float())

            if self.sinogram_transform is not None:
                with seeded(
                    self.sinogram_transform_seed,
                    "LIDC-IDRI",
                    patient_id,
                    slice_index_to_load,
                    device=sinogram.device,
                ):
                    sinogram = self.sinogram_transform(sinogram)
            return sinogram, reconstruction_tensor

        elif self.parameters.task == "diagnostic":
            return self.patients_diagnosis_dictionary[patient_id]
        else:
            raise NotImplementedError

    @staticmethod
    def cite(cite_format="bib"):
        """The LIDC-IDRI dataset: the paper and the data (TCIA), BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{armato2011lidc,
            title={The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans},
            author={Armato III, Samuel G and McLennan, Geoffrey and Bidaut, Luc and others},
            journal={Medical Physics},
            volume={38},
            number={2},
            pages={915--931},
            year={2011}
            }
            @misc{lidc2015data,
            title={Data From LIDC-IDRI},
            howpublished={The Cancer Imaging Archive},
            doi={10.7937/K9/TCIA.2015.LO9QL9SX}
            }
            """,
            mla="""
            Armato III, Samuel G., et al. "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans." Medical Physics 38.2 (2011): 915-931.
            Data From LIDC-IDRI. The Cancer Imaging Archive. doi:10.7937/K9/TCIA.2015.LO9QL9SX
            """,
        )
