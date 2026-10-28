# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Emilien Valat, Ander Biguri
# =============================================================================

from pathlib import Path

import pandas as pd
import torch
import numpy as np
import warnings
from LION.utils.paths import DETECT_PROCESSED_DATASET_PATH
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ctgeo
import LION.CTtools.ct_utils as ct
from LION.CTtools.ct_utils import make_operator
from LION.utils.tomosipo_import import import_tomosipo
from LION.utils.rng import seeded
from LION.data_loaders.LIONDataset import LIONDataset
from LION.utils.citation import citation

# Low-dose (mode1) noise of 2DeteCT's two detectors, as the I0 of
# deteCT.add_sinogram_noise that reproduces it when added to mode2: measured
# by Ander on 100 slices (25 Oct 2026; physical noise model of the sinogram
# variance, fitted per slice). Detector 2 gets about a third of the photons
# of detector 1 (its mode2 too: ~4,400 vs ~14,600 effective photons).
MODE1_I0_PER_DETECTOR = {1: 290, 2: 100}


class deteCT(LIONDataset):
    """
    The 2DeteCT dataset: 5,000 slices of 109 scanned samples (fruit, nuts and
    other objects), each measured in three modes:

                     |  mode1   |   mode2  |  mode3
        Tube Voltage |   90kV   |   90kV   |  60kV
        Tube power   |    3W    |    90W   |  60W
        Filter       | Thoraeus | Thoraeus | No Filter
        Meaning      | Low Dose | High Dose| Beam hardening

    Detectors: the data were acquired with two detectors, recorded in the
    ``detector`` column of the data record. Detector 1: samples 1-63 (slices
    1-2830, mixes 1-2); detector 2: samples 64-109 (slices 2831-5000, mixes
    2-3). Their noise differs: with the same tube settings, detector 2 gets
    about a third of the photons of detector 1 (``MODE1_I0_PER_DETECTOR``).
    Samples are split in order (training_proportion, validation_proportion),
    so with the default 0.8/0.1/0.1 the validation and test sets are all
    detector 2, and the training set is mostly detector 1.

    Added noise (``add_noise`` and ``noise_params``): ``add_sinogram_noise``
    adds Poisson noise and detector cross-talk to the measured (already noisy)
    sinograms, as in Kiss et al. 2024. ``noise_params.I0`` is either
    - a number: the same I0 for every slice, as LION did before October 2026
      and as Kiss et al. 2024 did (I0 = 200 to look like mode1); or
    - a dict {detector: I0}: one I0 per detector, so that the simulated noise
      follows each detector's real noise. ``deteCT.noise_parameters()`` gives
      ``MODE1_I0_PER_DETECTOR``, which looks like mode1 on each detector.
    """

    def __init__(
        self,
        mode,
        geometry_params: ct.Geometry = None,
        parameters: LIONParameter = None,
    ):
        if parameters is None:
            parameters = self.default_parameters()
        self.parameters = parameters
        self.geometry_params = geometry_params

        # Get geometry default, or validate if given geometry is valid for 2DeteCT, as its raw sinogram data, we can't allow any geometry
        if geometry_params is None:
            self.geometry = self.get_default_geometry()
            self.angle_index = list(range(len(self.geometry.angles)))
        else:
            self.angle_index = self.__is_valid_geo(geometry_params)
            self.geometry = geometry_params

        if parameters.query == "" and (
            not parameters.flat_field_correction or not parameters.dark_field_correction
        ):
            warnings.warn(
                "You are not using any detector query, but you are not using flat field or dark field correction. This is not recommended, as different detectors perform differently"
            )
        ### Defining the path to data
        self.path_to_dataset = parameters.path_to_dataset
        """
        The path_to_dataset attribute (pathlib.Path) points towards the folder
        where the data is stored
        """
        ### Defining the path to scan data
        self.path_to_scan_data = self.path_to_dataset.joinpath("scan_settings.json")
        ### Defining the path to the data record
        self.path_to_data_record = self.path_to_dataset.joinpath(
            f"default_data_records.csv"
        )

        ### Defining the data record
        self.data_record = pd.read_csv(self.path_to_data_record)
        """
        The data_record (pd.Dataframe) maps a slice index/identifier to
            - the sample index of the sample it belongs to
            - the number of slices expected
            - the number of slices actually sampled
            - the first slice of the sample to which a given slice belongs
            - the last slice of the sample to which a given slice belongs
            - the mix 
            - the detector it was sampled with
        """
        # Defining the input and target mode
        self.input_mode = parameters.input_mode
        self.target_mode = parameters.target_mode

        """
        The input_mode (str) argument is a keyword defining what input mode of the dataset to use:
                         |  mode1   |   mode2  |  mode3
            Tube Voltage |   90kV   |   90kV   |  60kV
            Tube power   |    3W    |    90W   |  60W
            Filter       | Thoraeus | Thoraeus | No Filter
            Meaning      | Low Dose | High Dose| Beam hardening
        """

        # Defining the task
        self.task = parameters.task
        """
        The task (str) argument is a keyword defining what is the dataset used for:
            - task == 'sino2sino' -> input and target are both sinograms
            - task == 'sino2recon' -> input is a sinogram and target is a reconstruction
            - task == 'recon2recon' -> input and target are both reconstructions
            - task == 'recon2seg' -> input is a reconstruction and target is a segmentation
            - task == 'sino2seg' -> input is a sinogram and target is a segmentation
            - task == 'joint' -> input is a sinogram, target is a reconstruction and segmentation
            - task == 'groundtruth' -> there is no input, target is a reconstruction
        """

        assert self.task in [
            "sino2sino",
            "sino2recon",
            "recon2recon",
            "recon2seg",
            "sino2seg",
            "joint",
            "groundtruth",
        ], f'Wrong task argument, must be in ["sino2sino", "sino2recon", "recon2recon", "recon2seg", "sino2seg", "joint"]'

        assert mode in [
            "train",
            "validation",
            "test",
        ], f'Wrong mode argument, must be in ["train", "validation", "test"]'
        # Defining the training mode
        self.mode = mode
        """
        The train (bool) argument defines if the dataset is used for training or testing
        """
        self.training_proportion = parameters.training_proportion
        self.validation_proportion = parameters.validation_proportion

        # Defining the train proportion
        """
        The training_proportion (float) argument defines the proportion of the training dataset used for training
        """
        self.transforms = None  # parameters.transforms

        ### We query the dataset subset
        self.slice_dataframe: pd.DataFrame
        """
        The slice_dataframe (pd.Dataframe) is the subset of interest of the dataset.
        The self.data_record argument becomes the slice_dataframe once we have queried it
        Example of query: 'detector==1'
        If the no query argument is passed, data_record_subset == data_record
        """
        if parameters.query:
            self.slice_dataframe = self.data_record.query(parameters.query)
        else:
            self.slice_dataframe = self.data_record

        ### We split the dataset between training and testing
        self.compute_sample_dataframe()
        self.sample_dataframe: pd.DataFrame
        """
        The sample_dataframe (pd.Dataframe) is a dataframe linking sample index to slices.
        It is used to partition the dataset on a sample basis, rather than on a slice basis,
        avoiding 'data leakage' between training, validation and testing
        """
        # Consecutive blocks of samples, with the same boundaries as before
        # (floor of n * proportion). The epsilon stops floating point error
        # (e.g. 0.8 + 0.1 = 0.9000...01 or 0.7 + 0.2 = 0.8999...) from moving a
        # boundary, and the test split takes all the remaining samples.
        n_samples = len(self.sample_dataframe)
        end_train = int(n_samples * self.training_proportion + 1e-9)
        end_validation = int(
            n_samples * (self.training_proportion + self.validation_proportion) + 1e-9
        )
        if self.mode == "train":
            self.sample_dataframe = self.sample_dataframe.iloc[:end_train]
        elif self.mode == "validation":
            self.sample_dataframe = self.sample_dataframe.iloc[end_train:end_validation]
        else:
            self.sample_dataframe = self.sample_dataframe.iloc[end_validation:]

        self.slice_dataframe = self.slice_dataframe[
            self.slice_dataframe["sample_index"].isin(
                self.sample_dataframe["sample_index"].unique()
            )
        ]

        if hasattr(parameters, "add_noise") and parameters.add_noise:
            self.add_noise = parameters.add_noise
            self.noise_params = parameters.noise_params
            if self.task not in ["sino2sino", "sino2recon", "joint", "sino2seg"]:
                warnings.warn(
                    "You have defined noise parameters, but the task is not 'sino2sino', 'sino2recon', 'joint', or 'sino2seg'. Noise will not be added"
                )
                self.add_noise = False
            if not hasattr(parameters, "noise_params") and self.add_noise:
                raise ValueError(
                    "You have to define noise parameters if you want to add noise"
                )
        else:
            self.add_noise = False

        self.flat_field_correction = parameters.flat_field_correction
        self.dark_field_correction = parameters.dark_field_correction
        self.log_transform = parameters.log_transform
        self.do_recon = parameters.do_recon
        self.recon_algo = parameters.recon_algo

    @staticmethod
    def default_parameters():
        param = LIONParameter()
        param.path_to_dataset = DETECT_PROCESSED_DATASET_PATH
        param.input_mode = "mode2"
        param.target_mode = "mode2"
        param.task = "sino2recon"
        param.training_proportion = 0.8
        param.validation_proportion = 0.1
        param.test_proportion = 0.1

        param.do_recon = False
        param.recon_algo = "nag_ls"
        param.flat_field_correction = True
        param.dark_field_correction = True
        param.log_transform = True
        param.query = ""

        param.add_noise = False

        param.geometry = deteCT.get_default_geometry()
        return param

    @staticmethod
    def get_default_geometry():
        # From Max Kiss code
        SOD = 431.019989
        SDD = 529.000488
        detPix = 0.0748
        detSubSamp = 2
        detPixSz = detSubSamp * detPix
        nPix = 956
        det_width = detPixSz * nPix
        FOV_width = det_width * SOD / SDD
        nVox = 1024
        voxSz = FOV_width / nVox
        scaleFactor = 1.0 / voxSz
        SDD = SDD * scaleFactor
        SOD = SOD * scaleFactor
        detPixSz = detPixSz * scaleFactor

        return ctgeo.Geometry(
            mode="fan",
            dsd=SDD,
            dso=SOD,
            detector_shape=[1, 956],
            detector_size=[detPixSz, detPixSz * 956],
            image_shape=[1, 1024, 1024],
            image_size=[1, 1024, 1024],
            image_pos=[0, -1, -1],
            angles=-np.linspace(0, 2 * np.pi, 3600, endpoint=False) + np.pi,
        )

    @staticmethod
    def _get_angle_indices(full_angles, angles):
        """Map requested angles to unique views in the raw deteCT sinogram."""
        angle_indices = []
        for angle in angles:
            matches = np.flatnonzero(
                np.isclose(full_angles, angle, rtol=0.0, atol=1e-8)
            )
            if matches.size != 1:
                raise ValueError(
                    "Given angles must be part of existing ones. Check the array "
                    "deteCT.get_default_geometry().angles for the existing angles. "
                    f"The given angle {angle} is not part of the existing angles."
                )
            angle_indices.append(int(matches[0]))
        return angle_indices

    def __is_valid_geo(self, geometry):
        if not isinstance(geometry, ctgeo.Geometry):
            raise ValueError("geometry must be a ctgeo.Geometry object")
        default_geo = deteCT.get_default_geometry()

        if geometry.dsd != default_geo.dsd or geometry.dso != default_geo.dso:
            raise ValueError(
                f"dsd and dso must be the same as the default geometry: DSO = {default_geo.dso}, DSD = {default_geo.dsd}"
            )
        if geometry.detector_shape[0] != default_geo.detector_shape[0]:
            raise ValueError(f"detector_shape[0] must be 1")
        if geometry.detector_shape[1] != default_geo.detector_shape[1]:
            warnings.warn(
                f"Raw data detector_shape[1] must be 956, it is {geometry.detector_shape[1]}. Interpolation will be used, but note that you are not using exactly the raw data"
            )
        if not np.array_equal(geometry.detector_size, default_geo.detector_size):
            raise ValueError(f"detector_size must be {default_geo.detector_size}")
        if not np.array_equal(geometry.image_shape, default_geo.image_shape):
            warnings.warn(
                f"image_shape must be {default_geo.image_shape}, it is {geometry.image_shape}. Interpolation will be used, but note that you are not using exactly the raw data"
            )
        if not np.array_equal(geometry.image_size[1:], default_geo.image_size[1:]):
            raise ValueError(f"image_size must be {default_geo.image_size}")
        if not np.array_equal(geometry.image_pos, default_geo.image_pos):
            raise ValueError(f"image_pos must be {default_geo.image_pos}")

        full_angles = self.get_default_geometry().angles
        if np.array_equal(geometry.angles, full_angles):
            return list(range(len(full_angles)))
        return self._get_angle_indices(full_angles, geometry.angles)

    @staticmethod
    def get_default_operator():
        geometry = deteCT.get_default_geometry()
        return make_operator(geometry)

    def get_operator(self):
        return make_operator(self.geometry)

    def sample_ids(self):
        """What each sample is, in order: its slice identifier."""
        return [str(i) for i in self.slice_dataframe["slice_identifier"]]

    def sample_groups(self):
        """The group of each sample, in order: the physical sample (mix) its
        slice is from, by which the splits are made."""
        return [f"sample_{i}" for i in self.slice_dataframe["sample_index"]]

    def sample_files(self):
        """The files the samples are read from, in order: those of each
        slice's input and target modes."""
        files = []
        modes = sorted({self.input_mode, self.target_mode})
        for identifier in self.slice_dataframe["slice_identifier"]:
            for mode in modes:
                files += sorted(
                    self.path_to_dataset.joinpath(str(identifier), mode).glob("*")
                )
        return files

    def set_sinogram_transform(self, sinogram_transform):
        self.sinogram_transform = sinogram_transform

    def compute_sample_dataframe(self):
        unique_identifiers = self.slice_dataframe["sample_index"].unique()
        record = {"sample_index": [], "first_slice": [], "last_slice": []}
        for identifier in unique_identifiers:
            record["sample_index"].append(identifier)
            subset = self.slice_dataframe[
                self.slice_dataframe["sample_index"] == identifier
            ]
            record["first_slice"].append(subset["first_slice"].iloc[0])
            record["last_slice"].append(subset["last_slice"].iloc[0])
        self.sample_dataframe = pd.DataFrame.from_dict(record)

    def __len__(self):
        # __getitem__ indexes the rows of slice_dataframe, which need not be
        # consecutive slices (e.g. with a query)
        return len(self.slice_dataframe)

    @staticmethod
    def noise_parameters(I0=None, cross_talk=0.05):
        """
        noise_params for ``add_noise``: by default the I0 of each detector
        that makes mode2 plus the added noise look like that detector's mode1
        (``MODE1_I0_PER_DETECTOR``). I0 can be a number (the same for every
        slice, LION's behaviour before October 2026) or a dict
        {detector: I0}. cross_talk: fraction of the signal of the neighbouring
        detector pixels (0.05, from XCIST).
        """
        noise = LIONParameter()
        noise.I0 = dict(MODE1_I0_PER_DETECTOR) if I0 is None else I0
        noise.cross_talk = cross_talk
        # Each slice's noise is drawn once (see __load_and_preprocess_sinogram__)
        noise.seed = 0
        return noise

    def noise_I0(self, index):
        """The I0 of the added noise for the slice at index: noise_params.I0
        if it is a number, else the I0 of the slice's detector."""
        I0 = self.noise_params.I0
        if isinstance(I0, LIONParameter):  # a dict saved and loaded as JSON
            I0 = vars(I0)
        if isinstance(I0, dict):
            detector = int(self.slice_dataframe.iloc[index]["detector"])
            # LIONParameter files may store dict keys as strings
            return I0[detector] if detector in I0 else I0[str(detector)]
        return I0

    def add_sinogram_noise(self, sinogram, I0, cross_talk=0.05):
        r"""
        Add noise to a measured sinogram.
        The reason this data loader has a bestpoke function for this is that the noise is that the
        noise simulator in LION.ct_tools.ct_utils assumes noiseles sinograms in X-ray absobtion.
        However, adding noise to a real measured sinogram is not the same as adding noise to a noiseless sinogram.

        The assumption here is that the input sinogram is measured, and contains enough photon counts as to ignore the poisson noise.
        However, effects due to the detector, electronics, etc. are still present. In particular, it assumes that there is electronic noise,
        modelled by a signal-independent Gaussian noise, and that it has detector cross-talk, modelled by a convolution.

        Under those assumptions, we can say thus that the noisy sinogram is:
        $$
        sino_{noisy} = sino_{measured} + corss_talk(P_{I0}),
        $$
        where $P_{I0}$ is the noise conponent of adding Poisson noise at $I0$ counts to the input sinogram (input is assumed at $I0=\inf$).
        This is not exactly correct, but approximately so. See Kiss et al. 2024 for more details

        Input:
        sinogram: torch.Tensor, the sinogram to add noise to. It has to be flat field corrected, and in absobrtion units (i.e. log transformed)
        I0: float, the number of counts in the measurement (lower=more noise).
            The data loader passes noise_I0(index): one I0 per detector when
            noise_params.I0 is a dict (see the class docstring).
        """
        dev = sinogram.device

        Im = I0 * torch.exp(-sinogram)
        # Add Poisson noise
        Pm = torch.poisson(Im)
        PI = Pm - Im
        # Detector cross talk

        kernel = torch.tensor(
            [[0.0, 0.0, 0.0], [cross_talk, 1, cross_talk], [0.0, 0.0, 0.0]]
        ).view(1, 1, 3, 3).repeat(1, 1, 1, 1) / (1 + 2 * cross_talk)

        conv = torch.nn.Conv2d(1, 1, 3, bias=False, padding="same")
        with torch.no_grad():
            conv.weight = torch.nn.Parameter(kernel)
        conv = conv.to(dev)

        noisy = Im + conv(PI.unsqueeze(0).to(dev))[0]
        noisy = torch.clip(noisy, min=0.0, max=I0)
        noisy[noisy <= 0] = 1e-6
        return -torch.log(noisy / I0)

    def __load_and_preprocess_sinogram__(self, index, mode):
        slice_row = self.slice_dataframe.iloc[index]
        path_to_input = self.path_to_dataset.joinpath(
            f"{slice_row['slice_identifier']}/{mode}"
        )
        sinogram = torch.from_numpy(
            np.load(path_to_input.joinpath("sinogram.npy")).astype(np.float32)
        ).unsqueeze(0)

        if self.flat_field_correction:
            flat = torch.from_numpy(
                np.load(path_to_input.joinpath("flat.npy")).astype(np.float32)
            ).unsqueeze(0)
        else:
            flat = 1
        if self.dark_field_correction:
            dark = torch.from_numpy(
                np.load(path_to_input.joinpath("dark.npy")).astype(np.float32)
            ).unsqueeze(0)
        else:
            dark = 0

        sinogram = (sinogram - dark) / (flat - dark)
        sinogram = torch.clip(sinogram, min=1e-6)
        if self.log_transform:
            sinogram = -torch.log(sinogram)

        if self.add_noise:
            # noise_params.seed: an integer gives each slice the same noise
            # every time it is read (as a scan is measured once); None or
            # missing: fresh noise each time
            with seeded(
                getattr(self.noise_params, "seed", None),
                "2DeteCT",
                slice_row["slice_identifier"],
                mode,
                device=sinogram.device,
            ):
                sinogram = self.add_sinogram_noise(
                    sinogram, self.noise_I0(index), self.noise_params.cross_talk
                )

        sinogram = torch.flip(sinogram, [2])
        sinogram = sinogram[:, self.angle_index, :]

        # Interpolate if geometry is not default
        if not np.array_equal(
            self.geometry.detector_shape, self.get_default_geometry().detector_shape
        ):
            sinogram = torch.nn.functional.interpolate(
                sinogram.unsqueeze(0),
                size=(sinogram.shape[1], self.geometry.detector_shape[1]),
                mode="bilinear",
            )
            sinogram = torch.squeeze(sinogram, 0)

        return sinogram

    def __load_and_preprocess_reconstruction__(self, index, mode):
        slice_row = self.slice_dataframe.iloc[index]
        path_to_input = self.path_to_dataset.joinpath(
            f"{slice_row['slice_identifier']}/{mode}"
        )
        reconstruction = torch.from_numpy(
            np.load(path_to_input.joinpath("reconstruction.npy")).astype(np.float32)
        ).unsqueeze(0)
        # Interpolate if geometry is not default
        if not np.array_equal(
            self.geometry.image_shape, self.get_default_geometry().image_shape
        ):
            reconstruction = torch.nn.functional.interpolate(
                reconstruction.unsqueeze(0),
                size=(self.geometry.image_shape[1], self.geometry.image_shape[2]),
                mode="bilinear",
            )
            reconstruction = torch.squeeze(reconstruction, 0)
        return reconstruction

    def __load_and_preprocess_segmentation__(self, index):
        slice_row = self.slice_dataframe.iloc[index]
        path_to_input = self.path_to_dataset.joinpath(
            f"{slice_row['slice_identifier']}/mode2"
        )
        segmentation = torch.from_numpy(
            np.load(path_to_input.joinpath("segmentation.npy"))
        ).unsqueeze(0)
        # Interpolate if geometry is not default
        if not np.array_equal(
            self.geometry.image_shape, self.get_default_geometry().image_shape
        ):
            segmentation = torch.nn.functional.interpolate(
                segmentation.unsqueeze(0),
                size=(self.geometry.image_shape[1], self.geometry.image_shape[2]),
                mode="nearest",
            )
            segmentation = torch.squeeze(segmentation, 0)
        return segmentation

    def __getitem__(self, index):
        ts_algorithms = import_tomosipo("ts_algorithms", "deteCT")
        fdk, nag_ls = ts_algorithms.fdk, ts_algorithms.nag_ls

        index = int(index)  # cast to int

        # If input is sinogram, we need to load the sinogram
        if self.task in ["sino2sino", "sino2recon", "sino2seg", "joint"]:
            input = self.__load_and_preprocess_sinogram__(index, self.input_mode)
        # if input is reconstruction, we need to load the reconstruction
        else:
            # Even if input is recon, we may want to actually do it ourselves.
            if self.do_recon:
                sinogram = self.__load_and_preprocess_sinogram__(index, self.input_mode)
                op = self.get_operator()
                if self.recon_algo == "nag_ls":
                    input = nag_ls(op, sinogram, 100, min_constraint=0)
                elif self.recon_algo == "fdk":
                    input = fdk(op, sinogram)
            else:
                # Otherwise just load the recon
                input = self.__load_and_preprocess_reconstruction__(
                    index, self.input_mode
                )

        # If target is sinogram, we need to load the sinogram
        if self.task in ["sino2sino", "recon2sino", "joint"]:
            # lets make sure we don't add noise to the target.
            noise = self.add_noise
            self.add_noise = False
            target = self.__load_and_preprocess_sinogram__(index, self.target_mode)
            self.add_noise = noise
        # if target is reconstruction, we need to load the reconstruction
        elif self.task in ["recon2recon", "sino2recon", "groundtruth"]:
            if self.do_recon:
                # lets make sure we don't add noise to the target.
                noise = self.add_noise
                self.add_noise = False
                sinogram = self.__load_and_preprocess_sinogram__(
                    index, self.target_mode
                )
                self.add_noise = noise

                op = self.get_operator()
                if self.recon_algo == "nag_ls":
                    target = nag_ls(op, sinogram, 100, min_constraint=0)
                elif self.recon_algo == "fdk":
                    target = fdk(op, sinogram)
            else:
                target = self.__load_and_preprocess_reconstruction__(
                    index, self.target_mode
                )
        elif self.task in ["recon2seg", "sino2seg"]:
            # Get paths to the dataset
            target = self.__load_and_preprocess_segmentation__(index)

        if self.task == "groundtruth":
            return target

        if self.task != "joint":
            return input, target
        else:
            return input, target, self.__load_and_preprocess_segmentation__(index)

    @staticmethod
    def cite(cite_format="bib"):
        """The 2DeteCT dataset, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{kiss2023detect,
            title={2DeteCT - A large 2D expandable, trainable, experimental Computed Tomography dataset for machine learning},
            author={Kiss, Maximilian B and Coban, Sophia B and Batenburg, K Joost and van Leeuwen, Tristan and Lucka, Felix},
            journal={Scientific Data},
            volume={10},
            pages={576},
            year={2023},
            doi={10.1038/s41597-023-02484-6}
            }
            """,
            mla="""
            Kiss, Maximilian B., Sophia B. Coban, K. Joost Batenburg, Tristan van Leeuwen, and Felix Lucka. "2DeteCT - A large 2D expandable, trainable, experimental Computed Tomography dataset for machine learning." Scientific Data 10 (2023): 576.
            """,
        )
