# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Emilien Valat
# Modifications: Michelle Limbach, Ander Biguri
# =============================================================================


from typing import List, Dict
import pathlib
import random
import math
import warnings

import torch
import numpy as np
import json
from torch.utils.data import Dataset


from LION.utils.paths import LIDC_IDRI_PROCESSED_DATASET_PATH
import LION.CTtools.ct_utils as ct
from LION.utils.parameter import LIONParameter
from LION.utils.device import default_device


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
    str_index = str(index)
    while len(str_index) < 4:
        str_index = "0" + str_index
    assert len(str_index) == 4
    return str_index


def load_json(file_path: pathlib.Path):
    if not file_path.is_file():
        raise FileNotFoundError(f"No file found at {file_path}")
    with open(file_path) as file:
        return json.load(file)


def choose_random_annotation(
    nodule_annotations_list: List,
) -> str:
    return random.choice(nodule_annotations_list)


def create_consensus_annotation(
    path_to_patient_folder: pathlib.Path,
    slice_index: int,
    nodule_index: str,
    nodule_annotations_list: List,
    clevel: float,
) -> torch.int16:
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


class LIDC_IDRI(Dataset):
    def __init__(
        self,
        mode,
        geometry_parameters: ct.Geometry = None,
        parameters: LIONParameter = None,
    ):
        """
        Initializes LIDC-IDRI dataset.

        Pixel spacing: the pre-processing resamples every slice to
        ``PIXEL_SPACING_MM`` (500 mm over 512x512 pixels, the largest field of
        view in the dataset, so no scan is cropped; smaller ones are surrounded
        by air), so all images have the same physical size.
        ``LIDC_IDRI.default_geometry()`` is the matching CT geometry. Datasets
        processed before October 2026 have each scan's own spacing (0.46 to
        0.98 mm) and no ``pre_processing.json``: the loader warns, and
        ``python LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py
        --resample-only`` resamples them in place (no DICOM files needed).

        Damaged data: the LIDC-IDRI download and pre-processing can leave
        incomplete files. The loader leaves out, with a warning, selected slices
        whose files are incomplete or unreadable, and the patients
        ``pre_processing.json`` lists as not resampled.

        Images: for reconstruction, the CT images (Hounsfield units) are turned
        into mass density in g/cm^3 (``ct_utils.from_HU_to_density``: air 0.0012,
        water 1.0137, 500 HU 1.52). Before October 2026 the same values came from
        ``from_HU_to_mu``, which called them attenuation. With LION's geometries
        in mm, the sinograms are then density line integrals in g/cm^3 * mm,
        which is what the physical noise model of
        ``ct_utils.sinogram_add_noise`` expects: the experiments
        (``ct_experiments``) add the noise.

        Parameters:
            - device (torch.device): Selects the device to use for the data loader.
            - task (str): Defines pipeline on how to use data. Distinguish between "joint", "end_to_end", "segmentation", "reconstruction" and "diagnostic".
                          Dataset will return, for each task:
                          "segmentation"    -> (image, segmentation_label)
                          "reconstruction"  -> (sinogram, image_label)
                          "diagnostic"      -> (segmented_nodule, diagnostic_label)
                          "joint"           -> ?????
                          "end_to_end"      -> ?????
            - training_proportion (float): Defines training % of total data.
            - mode (str): Defines "train", "validation" or "test" mode.
            - Task (str): Defines what task is the Dataset being used for. "segmentation" (default) returns (gt_image,segmentation) pairs while "reconstruction" returns (sinogram, gt_image) pairs
            - annotation (str): Defines what annotation mode to use. Distinguish between "random" and "consensus". Default "consensus"
            - max_num_slices_per_patient (int): Defines the maximum number of slices to take per patient. -1 takes all slices we have of each patient and pcg_slices_nodule gets ignored. Default is 5.
            - pcg_slices_nodule (float): Defines percentage of slices with nodule in dataset. 0 meaning "no nodules at all" and 1 meaning "just take slices that contain annotated nodules". Only used if max_num_slices_per_patient != -1. If a patient has too few slices of one kind, more of the other kind are taken. Default is 0.5.
            - legacy_slice_selection (bool): True reproduces the selection of LION before October 2026, which left out patients without nodule slices. Default False (missing in old parameter files, which then use the old selection).
            - clevel (float): Defines consensus level if annotation=consensus. Value between 0-1. Default is 0.5.
            - geometry: Geometry() type, if sinograms are requied (e.g. fo "reconstruction")

        """

        # Input parsing
        assert mode in [
            "train",
            "validation",
            "test",
        ], f'Wrong mode argument, must be in ["train", "validation", "test"]'

        if parameters is None:
            parameters = LIDC_IDRI.default_parameters(geometry=geometry_parameters)
        self.params = parameters

        task = self.params.task
        assert task in [
            "joint",
            "end_to_end",
            "segmentation",
            "reconstruction",
            "diagnostic",
        ], f'task argument {task} not in ["joint", "end_to_end", "segmentation", "reconstruction", "diagnostic"]'

        if task not in ["segmentation", "reconstruction", "end_to_end"]:
            raise NotImplementedError(f"task {task} not implemented yet")

        if (
            task in ["reconstruction"]
            and geometry_parameters is None
            and self.params.geometry is None
        ):
            raise ValueError("geometry input required for recosntruction modes")

        # Aux variable setting
        self.sinogram_transform = None
        self.image_transform = None
        self.device = self.params.device

        if task in ["reconstruction"]:
            self.image_transform = (
                ct.from_HU_to_density
            )  # g/cm^3 (was from_HU_to_mu, same values)
        if task in ["segmentation"]:
            self.image_transform = ct.from_HU_to_normal

        # parameters loaded from JSON have the geometry as a plain LIONParameter
        if self.params.geometry is not None and not isinstance(
            self.params.geometry, ct.Geometry
        ):
            self.params.geometry = ct.Geometry.init_from_parameter(self.params.geometry)
        if geometry_parameters is not None:
            self.params.geometry = geometry_parameters
            self.operator = ct.make_operator(geometry_parameters)
        elif self.params.geometry is not None:
            self.operator = ct.make_operator(self.params.geometry)
        # Start of Patient pre-processing

        self.path_to_processed_dataset = pathlib.Path(self.params.folder)
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
        self.total_patients = (
            len(list(self.path_to_processed_dataset.glob("LIDC-IDRI-*"))) + 1
        )
        self.num_slices_per_patient = self.params.max_num_slices_per_patient
        self.pcg_slices_nodule = self.params.pcg_slices_nodule
        self.annotation = self.params.annotation
        self.clevel = (
            self.params.clevel
        )  # consensus level, only used if annotation == consensus
        self.patient_index_to_n_slices_dict: Dict = {
            f"LIDC-IDRI-{format_index(index)}": len(
                list(
                    self.path_to_processed_dataset.joinpath(
                        f"LIDC-IDRI-{format_index(index)}"
                    ).glob("slice_*.npy")
                )
            )
            for index in range(1, self.total_patients)
        }

        # Dict with all slices of each patient
        self.patient_index_to_slices_index_dict: Dict = {
            f"LIDC-IDRI-{format_index(index)}": list(
                np.arange(
                    0,
                    self.patient_index_to_n_slices_dict[
                        f"LIDC-IDRI-{format_index(index)}"
                    ],
                    1,
                )
            )
            for index in range(1, self.total_patients)
        }

        # Dict with all nodule slices of each patient
        # Converts the keys from self.patients_masks_dictionary to integer
        self.patient_index_to_nodule_slices_index_dict: Dict = {
            f"LIDC-IDRI-{format_index(index)}": [
                int(item)
                for item in list(
                    self.patients_masks_dictionary[
                        f"LIDC-IDRI-{format_index(index)}"
                    ].keys()
                )
            ]
            for index in range(1, self.total_patients)
        }

        # Dict with all non-nodule slices of each patient
        # Computes as difference of all slices dict and dict with nodules
        self.patient_index_to_non_nodule_slices_index_dict: Dict = {
            f"LIDC-IDRI-{format_index(index)}": list(
                set(
                    self.patient_index_to_slices_index_dict[
                        f"LIDC-IDRI-{format_index(index)}"
                    ]
                )
                - set(
                    self.patient_index_to_nodule_slices_index_dict[
                        f"LIDC-IDRI-{format_index(index)}"
                    ]
                )
            )
            for index in range(1, self.total_patients)
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
            f"LIDC-IDRI-{format_index(index)}": list(
                set(
                    self.patient_index_to_nodule_slices_index_dict[
                        f"LIDC-IDRI-{format_index(index)}"
                    ]
                )
                - set(self.removed_slices[f"LIDC-IDRI-{format_index(index)}"])
            )
            for index in range(1, self.total_patients)
        }

        ##% Divide dataset in training/validation/testing
        self.training_proportion = self.params.training_proportion
        self.validation_proportion = self.params.validation_proportion
        self.params.mode = mode
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

        assert self.total_patients == (
            self.n_patients_training
            + self.n_patients_testing
            + self.n_patients_validation
        ), print(
            f"Total patients: {self.total_patients}, \n training patients {self.n_patients_training}, \n validation patients {self.n_patients_validation}, \n testing patients {self.n_patients_testing}"
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
        assert len(self.patient_ids) == len(self.training_patients_list) + len(
            self.testing_patients_list
        ) + len(self.validation_patients_list), print(
            f"Len patients ids: {len(self.patient_ids)}, \n len training patients {len(self.training_patients_list)},\n len validation patients {len(self.validation_patients_list)}, \n len testing patients {len(self.testing_patients_list)}"
        )

        print("Preparing patient list, this may take time....")
        if self.params.mode == "train":
            patient_list_to_load = self.training_patients_list
        elif self.params.mode == "validation":
            patient_list_to_load = self.validation_patients_list
        elif self.params.mode == "test":
            patient_list_to_load = self.testing_patients_list
        else:
            raise NotImplementedError(
                f"mode {self.params.mode} not implemented, try training, validation or testing"
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

        print(f"Patient lists ready for {self.params.mode} dataset")

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
        needs_masks = self.params.task in ["joint", "end_to_end", "segmentation"]
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
        geometry = self.params.geometry
        if self.pixel_spacing_mm is None or geometry is None:
            return
        pixel_size = geometry.image_size[-1] / geometry.image_shape[-1]
        if not math.isclose(pixel_size, self.pixel_spacing_mm, rel_tol=1e-6):
            warnings.warn(
                f"The geometry's pixel size ({pixel_size:.4g} mm) is not the pixel "
                f"spacing of the data ({self.pixel_spacing_mm:.4g} mm); "
                "LIDC_IDRI.default_geometry() matches the data."
            )

    @staticmethod
    def default_geometry(angles=None):
        """The fan-beam geometry for the pre-processed slices.

        512x512 pixels of ``PIXEL_SPACING_MM`` (500 mm). Source to centre 575
        mm, source to detector 1050 mm, as LION's default geometry, with a
        1024 x 1 mm detector: it covers a 504 mm circle, like the ~50 degree fan
        of a clinical scanner (LION's 900 mm detector covers 453 mm).

        Args:
            angles: projection angles in radians. Default: 360 over 360 degrees.
        """
        if angles is None:
            angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        return ct.Geometry(
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
    def default_parameters(geometry=None, task="reconstruction"):
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
        param.task = task
        param.folder = LIDC_IDRI_PROCESSED_DATASET_PATH
        if task == "reconstruction" and geometry is None:
            raise ValueError(
                "For reconstruction task geometry needs to be input to default_parameters(geometry=geometry_param)"
            )

        # segmentation specific
        param.clevel = 0.5
        param.annotation = "consensus"
        # a string (e.g. "cuda:2"), so the parameters stay JSON-serialisable
        param.device = str(default_device())
        param.geometry = geometry
        return param

    def get_slices_to_load(
        self,
        patient_list: List,
        non_nodule_slices_dict: Dict,
        nodule_slices_dict: Dict,
        num_slices_per_patient: int,
        pcg_slices_nodule: float,
    ):
        """
        Returns a dictionary that contains patient_id's as keys and list of slices to load as values for each patient.

        Parameters:
            - patient_list (List): List that contains patient_id of all patients.
            - non_nodule_slices_dict (Dict): Dict that contains all slices without nodule of each patient_id.
            - nodule_slices_dict (Dict): Dict that contains all slices with nodule of each patient_id.
            - num_slices_per_patient (int): Defines maximum number of slices we want per patient. If num_slices_per_patient=-1 take all slices we have of each patient.
            - pcg_slices_nodule (float): Defines amount of slices that should contain a nodule. Value between 0-1.
        Returns:
            - patient_id_to_slices_to_load_dict which contains patient_id as key and list of slices to load as values
        """

        patient_id_to_slices_to_load_dict = (
            {}
        )  # Empty dict which should contain patient id as key and slice ids as array of values

        # parameter files from before legacy_slice_selection existed used the old rule
        legacy = getattr(self.params, "legacy_slice_selection", True)
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
        """
        Returns a dictionary that contains patient_id's as keys and start index of each patient in self.slice_index_to_patient_id_list as value.

        Parameters:
            - patient_with_slices_to_load (Dict): Dict that defines which slices to load per patient.
        Returns:
            - patient_id_to_first_index_dict (Dict): Defines start index of each patient in self.slice_index_to_patient_id_list. Needed for mapping of global index to slice index.
        """
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
        """
        Returns a list that contains "number of slices" times each patient id.

        Parameters:
            - patient_with_slices_to_load (Dict): Dict that defines which slices to load per patient.
        Returns:
            - slice_index_to_patient_id_list (List): Contains number of slices times each patient id. Needed for mapping of global index to slice index.
        """
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
        tensor = torch.from_numpy(np.load(file_path)).unsqueeze(0).to(self.device)
        return tensor

    def set_sinogram_transform(self, sinogram_transform):
        self.sinogram_transform = sinogram_transform

    def set_image_transform(self, image_transform):
        self.image_transform = image_transform

    def compute_clean_sinogram(self, image=None) -> torch.Tensor:

        if self.operator is None:
            raise AttributeError("CT operator not know. Have you given a ct geometry?")
        sinogram = self.operator(image)
        return sinogram

    def get_mask_tensor(self, patient_id: str, slice_index: int) -> torch.Tensor:
        ## First, assess if the slice has a nodule
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

                    annotation = choose_random_annotation(nodule_annotations_list)
                    if annotation == "":
                        # Hopefully, that exists the try to return an empty mask
                        nodule_mask = torch.zeros((512, 512), dtype=torch.bool)
                    else:
                        path_to_mask = self.path_to_processed_dataset.joinpath(
                            f"{patient_id}/mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation}.npy"
                        )
                        # print(path_to_mask)
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

    def get_specific_slice(self, patient_index, slice_index):
        ## Assumes slice and mask exist
        file_path = self.path_to_processed_dataset.joinpath(
            f"{patient_index}/slice_{slice_index}.npy"
        )
        return self.get_reconstruction_tensor(file_path), self.get_mask_tensor(
            patient_index, slice_index
        )

    def __getitem__(self, index):
        assert index < len(self.slice_index_to_patient_id_list), print(
            f"Index {index} out of range, max index is {len(self.slice_index_to_patient_id_list)-1}"
        )
        patient_id = self.slice_index_to_patient_id_list[index]
        first_slice_index = self.patient_id_to_first_index_dict[patient_id]
        dict_slice_index = index - first_slice_index
        slice_index_to_load = self.slices_to_load[patient_id][dict_slice_index]
        # print(f'Index, {index}, Patient Id : {patient_id}, first_slice_index : {first_slice_index}, slice_index : {dict_slice_index} ', slice_to_load : {slice_index_to_load})
        file_path = self.path_to_processed_dataset.joinpath(
            f"{patient_id}/slice_{slice_index_to_load}.npy"
        )

        if self.params.task in [
            "joint",
            "end_to_end",
            "segmentation",
            "reconstruction",
        ]:
            reconstruction_tensor = self.get_reconstruction_tensor(file_path)
            if self.image_transform is not None:
                reconstruction_tensor = self.image_transform(reconstruction_tensor)

        if self.params.task in ["joint", "end_to_end", "segmentation"]:
            mask_tensor = self.get_mask_tensor(patient_id, slice_index_to_load)
            return reconstruction_tensor, mask_tensor

        elif self.params.task == "reconstruction":
            sinogram = self.compute_clean_sinogram(reconstruction_tensor.float())

            if self.sinogram_transform is not None:
                sinogram = self.sinogram_transform(sinogram)
            return sinogram, reconstruction_tensor

        elif self.params.task == "diagnostic":
            return self.patients_diagnosis_dictionary[patient_id]
        else:
            raise NotImplementedError
