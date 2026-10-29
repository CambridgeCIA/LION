"""Parameters from a file: each component's ``parameters_from_file(source)``
gives what its ``default_parameters()`` gives, as recorded, to change and make
the component with; ``instantiate(source, **arguments)`` replaces recorded
arguments. Models, experiments, datasets, solvers and reconstructors."""

import os
import tempfile
import warnings

# the datasets' default folders are in LION_DATA_PATH
os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Subset  # noqa: E402

import LION.CTtools.ct_utils as ct_utils  # noqa: E402
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI  # noqa: E402
from LION.data_loaders.LIONDataset import LIONDataset  # noqa: E402
from LION.experiments import ct_experiments  # noqa: E402
from LION.models.CNNs.dncnn import DnCNN  # noqa: E402
from LION.models.LIONmodel import LIONmodel  # noqa: E402
from LION.optimizers.LIONsolver import LIONsolver  # noqa: E402
from LION.optimizers.Noisier2InverseSolver import Noisier2InverseSolver  # noqa: E402
from LION.optimizers.SupervisedSolver import SupervisedSolver  # noqa: E402
from LION.reconstructors import LIONReconstructor, PnP  # noqa: E402
from LION.utils import spec  # noqa: E402
from pytests.helpers import ScaleOp, small_dncnn  # noqa: E402
from pytests.test_data_record import stand_in_projection  # noqa: E402
from pytests.test_lidc_resampling import fake_lidc  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*Expected '(dataset|training)'")


def same(a, b) -> bool:
    """The same parameters (as written to a file)."""
    return spec.encode(a) == spec.encode(b)


# ---------------------------------------------------------------------------
# Models


@pytest.fixture
def model_file(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # saved without data or training
        small_dncnn().save(tmp_path / "model")
    return tmp_path / "model"


def test_a_models_parameters_from_its_file(model_file):
    parameters = DnCNN.parameters_from_file(model_file)
    assert type(parameters) is type(DnCNN.default_parameters())
    assert (parameters.blocks, parameters.int_channels) == (3, 4)
    # on the base class, the class the file names
    assert same(LIONmodel.parameters_from_file(model_file), parameters)
    # changed, then made as usual, or by instantiate
    parameters.blocks = 2
    for model in (
        DnCNN(parameters),
        DnCNN.instantiate(model_file, model_parameters=parameters),
        LIONmodel.instantiate(model_file, model_parameters=parameters),
    ):
        assert type(model) is DnCNN and model.model_parameters.blocks == 2
    # the file is unchanged
    assert DnCNN.parameters_from_file(model_file).blocks == 3


def test_a_changed_model_starts_from_new_weights(model_file):
    parameters = DnCNN.parameters_from_file(model_file)
    with pytest.raises(ValueError, match="starts from new ones"):
        DnCNN.instantiate(model_file, model_parameters=parameters, initial_weights=True)


# ---------------------------------------------------------------------------
# Experiments and datasets


@pytest.fixture
def data_record(tmp_path, monkeypatch):
    """The record of an experiment's training data (fake LIDC-IDRI, a
    stand-in projection)."""
    monkeypatch.setattr(ct_utils, "make_operator", lambda g: stand_in_projection)
    folder = tmp_path / "LIDC-IDRI"
    folder.mkdir()
    fake_lidc(folder, range(1, 11))
    for k, path in enumerate(sorted(folder.glob("LIDC-IDRI-*/slice_*.npy"))):
        np.save(path, np.full((512, 512), -500 + k, np.int16))
    experiment = ct_experiments.LowDoseCTRecon(datafolder=folder)
    return spec.read(experiment.data_record("train"))


def test_an_experiments_parameters_from_a_data_record(data_record):
    Experiment = ct_experiments.Experiment
    parameters = ct_experiments.LowDoseCTRecon.parameters_from_file(data_record)
    assert parameters.noise_params.I0 == 11000
    assert same(Experiment.parameters_from_file(data_record), parameters)
    parameters.noise_params.I0 = 5000
    for experiment in (
        ct_experiments.LowDoseCTRecon(parameters),
        Experiment.instantiate(data_record, experiment_params=parameters),
    ):
        assert type(experiment) is ct_experiments.LowDoseCTRecon
        assert experiment.param.noise_params.I0 == 5000
    assert Experiment.instantiate(data_record).param.noise_params.I0 == 11000
    # and from the experiment's own spec (a package's experiment.json)
    written = data_record["experiment"]
    assert Experiment.parameters_from_file(written).noise_params.I0 == 11000


def test_a_datasets_parameters_from_a_data_record(data_record):
    parameters = LIDC_IDRI.parameters_from_file(data_record)
    assert parameters.max_num_slices_per_patient == 5
    assert same(LIONDataset.parameters_from_file(data_record), parameters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the record's noise is left out
        recorded = LIONDataset.instantiate(data_record)
    parameters.max_num_slices_per_patient = 1
    changed = LIONDataset.instantiate(data_record, parameters=parameters)
    assert type(changed) is LIDC_IDRI and not isinstance(changed, Subset)
    assert changed.parameters.max_num_slices_per_patient == 1
    assert len(changed) < len(recorded)


# ---------------------------------------------------------------------------
# Solvers and reconstructors


def training_record(solver):
    return {"lion_training_record": 1, "solver": solver}


def test_a_solvers_parameters_from_its_training_record():
    parameters = Noisier2InverseSolver.default_parameters()
    parameters.sigma = 7
    record = training_record(
        {
            "_target_": spec.import_path(Noisier2InverseSolver),
            "solver_params": spec.encode(parameters),
        }
    )
    again = LIONsolver.parameters_from_file(record)
    assert again.sigma == 7 and again.delta == parameters.delta
    assert again.recon_fn is parameters.recon_fn
    assert Noisier2InverseSolver.parameters_from_file(record).sigma == 7
    # a solver that takes none: its defaults
    record = training_record({"_target_": spec.import_path(SupervisedSolver)})
    assert same(
        LIONsolver.parameters_from_file(record), SupervisedSolver.default_parameters()
    )


def test_a_solver_made_with_changed_parameters(monkeypatch):
    """Equivariance2Inverse, with an identity for its tomosipo operator."""
    import types

    import LION.optimizers.Equivariance2InverseSolver as e2i
    import LION.optimizers.LIONsolver as lion_solver
    from LION.CTtools.ct_geometry import Geometry

    identity = types.SimpleNamespace(
        to_autograd=lambda op, num_extra_dims: (lambda x: x)
    )
    monkeypatch.setattr(e2i, "import_tomosipo", lambda *args: identity)
    monkeypatch.setattr(lion_solver, "make_operator", lambda geometry: None)
    record = training_record(
        {
            "_target_": spec.import_path(e2i.Equivariance2InverseSolver),
            "loss_fn": spec.encode(torch.nn.MSELoss()),
            "solver_params": spec.encode(
                e2i.Equivariance2InverseSolver.default_parameters()
            ),
        }
    )
    record["optimizer"] = spec.to_spec(
        torch.optim.Adam(small_dncnn().parameters(), lr=1e-3)
    )
    parameters = LIONsolver.parameters_from_file(record)
    parameters.equivariance_strength = 0.5
    solver = LIONsolver.instantiate(
        record,
        small_dncnn(),
        solver_params=parameters,
        geometry=Geometry.default_parameters(),
        device=torch.device("cpu"),
    )
    assert type(solver) is e2i.Equivariance2InverseSolver
    assert solver.solver_params.equivariance_strength == 0.5


def test_a_reconstructors_parameters_from_its_spec():
    parameters = PnP.default_parameters("HQS")
    parameters.max_iter = 3
    written = spec.to_spec(PnP(ScaleOp(2.0, (1, 4, 4)), None, parameters))
    again = PnP.parameters_from_file(written)
    assert same(again, parameters)
    again.max_iter = 7
    assert LIONReconstructor.parameters_from_file(written).max_iter == 3
    changed = LIONReconstructor.instantiate(
        written, allow=["pytests."], parameters=again
    )
    assert type(changed) is PnP and changed.parameters.max_iter == 7


def test_another_experiments_parameters_warn():
    """The experiments differ only in their parameters: made with another's,
    it would be that experiment, recorded under the wrong class."""
    parameters = ct_experiments.LowDoseCTRecon.default_parameters()
    with pytest.warns(UserWarning, match="are LowDoseCTRecon's"):
        ct_experiments.ExtremeLowDoseCTRecon(parameters)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ct_experiments.LowDoseCTRecon(parameters)  # its own
        ct_experiments.ExtremeLowDoseCTRecon()  # its defaults
        parameters.name = "Low dose, my detector"  # another experiment
        ct_experiments.ExtremeLowDoseCTRecon(parameters)
