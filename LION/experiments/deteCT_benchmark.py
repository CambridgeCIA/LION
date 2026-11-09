# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri, Max Kiss
# =============================================================================
"""The 2DeteCT benchmark: the experiments of

    Kiss, Biguri, Shumaylov, Sherry, Batenburg, Schönlieb and Lucka,
    "Benchmarking learned algorithms for computed tomography image
    reconstruction tasks", Applied Mathematics for Modern Challenges 3 (2025)
    (``CTBenchmarkingExperiment.cite()``).

Each experiment is a reconstruction task on the 2DeteCT dataset
(``LION.data_loaders.deteCT``), with mode 2 reconstructions as targets:

- ``FullDataCTRecon``: all 3600 angles of mode 2; ``GroundTruthCT``: the mode 2
  reconstructions only (no sinograms);
- ``LimitedAngle{150,120,90,60}CTRecon``: the first 1500, 1200, 900 or 600
  angles (150 to 60 degrees);
- ``SparseAngle{720,360,180,120,90,60}CTRecon``: every 5th to every 60th
  angle;
- ``LowDoseCTRecon``: mode 1 sinograms (low dose);
- ``BeamHardeningCTRecon``: mode 3 sinograms (beam hardening).

They reproduce the paper: their parameters are not changed.
"""

from LION.data_loaders.deteCT import deteCT
from LION.experiments.ct_experiments import Experiment
from LION.utils.citation import citation
from LION.utils.parameter import LIONParameter


def benchmark_parameters(
    name: str,
    angles: slice = None,
    input_mode: str = "mode2",
    task: str = "sino2recon",
) -> LIONParameter:
    """The parameters of a 2DeteCT benchmark experiment: the dataset's default
    geometry, keeping only the projections ``angles`` (a slice; None: all of
    them; the data loader subsamples the sinograms to match), sinograms of
    ``input_mode`` and mode 2 reconstructions as targets."""
    param = LIONParameter()
    param.name = name
    param.operator_params = deteCT.get_default_geometry()
    if angles is not None:
        param.operator_params.angles = param.operator_params.angles[angles]
    param.data_loader_params = Experiment.get_dataset_parameters("2DeteCT")
    param.data_loader_params.task = task
    param.data_loader_params.input_mode = input_mode
    param.data_loader_params.target_mode = "mode2"
    return param


class CTBenchmarkingExperiment(Experiment):
    """An experiment of the 2DeteCT benchmark (see the module docstring)."""

    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        if dataset != "2DeteCT":
            raise ValueError(
                "Benchmarking experiments only supports 2DeteCT dataset, currently"
            )
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def cite(cite_format="bib"):
        """The paper of the benchmark: BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{kiss2025benchmarking,
            title={Benchmarking learned algorithms for computed tomography image reconstruction tasks},
            author={Kiss, Maximilian B. and Biguri, Ander and Shumaylov, Zakhar and Sherry, Ferdia and Batenburg, K. Joost and Sch{\\"o}nlieb, Carola-Bibiane and Lucka, Felix},
            journal={Applied Mathematics for Modern Challenges},
            volume={3},
            pages={1--43},
            year={2025},
            doi={10.3934/ammc.2025001}
            }
            """,
            mla="""
            Kiss, Maximilian B., et al. "Benchmarking learned algorithms for
            computed tomography image reconstruction tasks." Applied Mathematics
            for Modern Challenges 3 (2025): 1-43.
            """,
        )


class GroundTruthCT(CTBenchmarkingExperiment):
    def __init__(self, datafolder=None):
        super().__init__(experiment_params=None, datafolder=datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = benchmark_parameters(
            "Ground truths only, from Full Data CT reconstruction using the full angular sampling experiment from mode 2",
            task="groundtruth",
        )
        param.do_recon = False
        return param


class FullDataCTRecon(CTBenchmarkingExperiment):
    def __init__(self, datafolder=None):
        super().__init__(experiment_params=None, datafolder=datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        return benchmark_parameters(
            "Full Data CT reconstruction using the full angular sampling experiment from mode 2"
        )


class LimitedAngle150CTRecon(CTBenchmarkingExperiment):
    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        return benchmark_parameters(
            "150 degrees Limited Angle CT reconstruction experiment from mode 2",
            angles=slice(None, 1500),
        )


class LimitedAngle120CTRecon(CTBenchmarkingExperiment):
    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        return benchmark_parameters(
            "120 degrees Limited Angle CT reconstruction experiment from mode 2",
            angles=slice(None, 1200),
        )


class LimitedAngle90CTRecon(CTBenchmarkingExperiment):
    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        return benchmark_parameters(
            "90 degrees Limited Angle CT reconstruction experiment from mode 2",
            angles=slice(None, 900),
        )


class LimitedAngle60CTRecon(CTBenchmarkingExperiment):
    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        return benchmark_parameters(
            "60 degrees Limited Angle CT reconstruction experiment from mode 2",
            angles=slice(None, 600),
        )


class SparseAngle720CTRecon(CTBenchmarkingExperiment):
    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        return benchmark_parameters(
            "720 angles Sparse Angle CT reconstruction experiment from mode 2",
            angles=slice(None, None, 5),
        )


class SparseAngle360CTRecon(CTBenchmarkingExperiment):
    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        return benchmark_parameters(
            "360 angles Sparse Angle CT reconstruction experiment from mode 2",
            angles=slice(None, None, 10),
        )


class SparseAngle180CTRecon(CTBenchmarkingExperiment):
    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        return benchmark_parameters(
            "180 angles Sparse Angle CT reconstruction experiment from mode 2",
            angles=slice(None, None, 20),
        )


class SparseAngle120CTRecon(CTBenchmarkingExperiment):
    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        return benchmark_parameters(
            "120 angles Sparse Angle CT reconstruction experiment from mode 2",
            angles=slice(None, None, 30),
        )


class SparseAngle90CTRecon(CTBenchmarkingExperiment):
    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        return benchmark_parameters(
            "90 angles Sparse Angle CT reconstruction experiment from mode 2",
            angles=slice(None, None, 40),
        )


class SparseAngle60CTRecon(CTBenchmarkingExperiment):
    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        return benchmark_parameters(
            "60 angles Sparse Angle CT reconstruction experiment from mode2",
            angles=slice(None, None, 60),
        )


class LowDoseCTRecon(CTBenchmarkingExperiment):
    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        return benchmark_parameters(
            "Denoising experiment with sinograms from mode1 and reconstructions from mode2",
            input_mode="mode1",
        )


class BeamHardeningCTRecon(CTBenchmarkingExperiment):
    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        return benchmark_parameters(
            "Beam Hardening Reduction experiment with sinograms from mode3 and reconstructions from mode2",
            input_mode="mode3",
        )
