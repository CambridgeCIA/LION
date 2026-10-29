"""Specs (LION.utils.spec): components written as the call that builds them, and
models saved and loaded with them."""

import importlib
import inspect
import json
import os
import pathlib
import pickle
import tempfile
import warnings

# LION.utils.paths requires LION_DATA_PATH at import time
os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import numpy as np
import pytest
import torch

import LION.experiments.ct_2DeteCT_noise_experiments as noise_experiments
import LION.experiments.ct_benchmarking_experiments as benchmarking_experiments
import LION.experiments.ct_experiments as ct_experiments
import LION.models
import LION.utils.utils as lion_utils
from LION.CTtools.ct_geometry import Geometry
from LION.metrics.haarpsi import HAARPsi
from LION.metrics.psnr import PSNR
from LION.metrics.ssim import SSIM
from LION.models.CNNs.dncnn import DnCNN
from LION.models.CNNs.drunet import DRUNet
from LION.models.CNNs.MSDNet import MSDNet
from LION.models.CNNs.UNets.Unet import UNet
from LION.models.learned_regularizer.AR import AR
from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType
from LION.models.PnP.gradient_step_denoiser import GSD
from LION.models.PnP.gs_drunet import GSDRUNet
from LION.utils.parameter import LIONParameter
from LION.utils.spec import SpecError, build, encode, to_spec
from pytests.helpers import ImageMatrixOp


def same(a, b, where="") -> list:
    """Differences between two values, in type and value (empty if the same)."""
    if type(a) is not type(b):
        return [f"{where}: {type(a).__name__} -> {type(b).__name__}"]
    if isinstance(a, LIONParameter):
        if set(vars(a)) != set(vars(b)):
            return [f"{where}: names {sorted(vars(a))} -> {sorted(vars(b))}"]
        return sum(
            (same(getattr(a, k), getattr(b, k), f"{where}.{k}") for k in vars(a)), []
        )
    if isinstance(a, np.ndarray):
        ok = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
        return [] if ok else [f"{where}: array"]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{where}: length"]
        return sum(
            (same(x, y, f"{where}[{i}]") for i, (x, y) in enumerate(zip(a, b))), []
        )
    if isinstance(a, dict):
        if set(a) != set(b):
            return [f"{where}: keys"]
        return sum((same(a[k], b[k], f"{where}[{k!r}]") for k in a), [])
    return [] if a == b else [f"{where}: {a!r} -> {b!r}"]


def through_json(value):
    """``value`` written as a spec, through a JSON file's text, and built again."""
    return build(json.loads(json.dumps(encode(value))))


def parameter(**values):
    """A LIONParameter with these attributes (LIONParameter(**values) would turn
    dict values into LIONParameters, and fails on non-string keys)."""
    p = LIONParameter()
    for name, value in values.items():
        setattr(p, name, value)
    return p


def model_classes():
    """Every LIONmodel in LION/models, found by file (some folders have no
    __init__.py, which pkgutil skips). Files an import statement can't name
    (e.g. in a folder called MS-D, left in old clones) are not LION's."""
    classes = set()
    root = pathlib.Path(LION.models.__path__[0])
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).with_suffix("").parts
        if path.name == "__init__.py" or not all(p.isidentifier() for p in parts):
            continue
        name = ".".join(("LION", "models") + parts)
        module = importlib.import_module(name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, LIONmodel)
                and cls is not LIONmodel
                and cls.__module__ == module.__name__
            ):
                classes.add(cls)
    return sorted(classes, key=lambda c: c.__name__)


def experiment_parameters():
    found = {}
    for module in (ct_experiments, benchmarking_experiments, noise_experiments):
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__ or inspect.isabstract(cls):
                continue
            for dataset in ("LIDC-IDRI", "2DeteCT"):
                try:
                    found[f"{name}[{dataset}]"] = cls.default_parameters(
                        dataset=dataset
                    )
                except TypeError:
                    found[name] = cls.default_parameters()
                except (ValueError, NotImplementedError):
                    pass  # not possible with that dataset
    return found


# --------------------------------------------------------------------------
# Values


@pytest.mark.parametrize("cls", model_classes(), ids=lambda c: c.__name__)
def test_model_parameters_come_back_the_same(cls):
    """Classes (nested ones too), enums, tuples: what plain JSON loses."""
    params = cls.default_parameters()
    assert same(params, through_json(params)) == []


@pytest.mark.parametrize("name, params", experiment_parameters().items())
def test_experiment_parameters_come_back_the_same(name, params):
    """Including their Geometry, numpy arrays and Paths."""
    assert same(params, through_json(params)) == []


@pytest.mark.parametrize(
    "value",
    [
        None,
        3,
        2.5,
        "text",
        [1, (2, 3)],
        (1, 2.0, "x"),
        {1, 2},
        np.array([1, 2, 3], dtype=np.int16),
        np.linspace(0, 1, 5),
        ModelInputType.IMAGE,
        pathlib.Path("some/folder"),
        {"a": 1, "b": [2, 3]},
        {1: 290, 2: 100},  # deteCT's I0 per detector
        parameter(I0={1: 290, 2: 100}, inner=LIONParameter(depth=2)),
    ],
    ids=lambda v: type(v).__name__,
)
def test_values_come_back_the_same(value):
    assert same(value, through_json(value)) == []


def test_functions_are_referred_to_not_called():
    spec = encode(torch.nn.functional.mse_loss)
    assert spec == {"_target_": "torch.nn.functional.mse_loss", "_call_": False}
    assert build(spec) is torch.nn.functional.mse_loss


@pytest.mark.parametrize(
    "value, message",
    [
        (lambda x: x, "lambda"),
        (torch.zeros(2), "tensor"),
        (np.array([object(), object()]), "objects"),
    ],
    ids=["lambda", "tensor", "object array"],
)
def test_values_that_cannot_be_saved(value, message):
    with pytest.raises(SpecError, match=message):
        encode(value)


def test_block_without_target_is_a_lion_parameter():
    built = build({"a": 1, "inner": {"b": 2}})
    assert isinstance(built, LIONParameter) and isinstance(built.inner, LIONParameter)
    assert built.inner.b == 2


# --------------------------------------------------------------------------
# Objects: the same-name rule


@pytest.mark.parametrize(
    "obj",
    [
        torch.nn.MSELoss(),
        torch.nn.L1Loss(reduction="sum"),
        torch.nn.SmoothL1Loss(beta=0.5),
        torch.nn.HuberLoss(delta=2.0),
        PSNR(),
        SSIM(),
        HAARPsi(C=10.0),
    ],
    ids=lambda o: type(o).__name__,
)
def test_losses_and_metrics(obj):
    rebuilt = through_json(obj)
    assert type(rebuilt) is type(obj)
    public = lambda o: {k: v for k, v in vars(o).items() if not k.startswith("_")}
    assert public(rebuilt) == public(obj)


@pytest.mark.parametrize(
    "make",
    [
        lambda p: torch.optim.Adam(p, lr=1e-4, betas=(0.5, 0.9)),
        lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9),
    ],
    ids=["Adam", "SGD"],
)
def test_optimizers_get_their_parameters_when_built(make):
    net = torch.nn.Linear(2, 2)
    optimizer = make(net.parameters())
    spec = json.loads(json.dumps(to_spec(optimizer)))
    rebuilt = build(spec, params=net.parameters())
    assert type(rebuilt) is type(optimizer)
    assert rebuilt.defaults == optimizer.defaults


class KeepsItsArguments:
    def __init__(self, size, scale=1.0, device=None):
        self.size = size
        self.scale = scale
        self.device = device


class RenamesItsArgument:
    def __init__(self, size):
        self.n = size


class SaysItsArguments(RenamesItsArgument):
    def spec_kwargs(self):
        return {"size": self.n}


def test_same_name_rule():
    spec = to_spec(KeepsItsArguments((3, 4), scale=2.0, device="cpu"))
    assert spec["_target_"] == "pytests.test_spec.KeepsItsArguments"
    assert "device" not in spec  # where it runs, not what it is
    rebuilt = build(spec, allow=["pytests."])
    assert rebuilt.size == (3, 4) and rebuilt.scale == 2.0


def test_object_that_renames_its_argument_cannot_be_saved():
    with pytest.raises(SpecError, match="doesn't keep its argument 'size'"):
        to_spec(RenamesItsArgument(3))


def test_spec_kwargs_gives_the_arguments():
    rebuilt = build(to_spec(SaysItsArguments(3)), allow=["pytests."])
    assert type(rebuilt) is SaysItsArguments and rebuilt.n == 3


# --------------------------------------------------------------------------
# What a spec may run


@pytest.mark.parametrize(
    "spec",
    [
        {"_target_": "os.system", "_args_": ["echo unsafe"]},
        {"_target_": "builtins.eval", "_args_": ["1 + 1"]},
        {"_target_": "builtins.open", "_args_": ["file.txt"]},
        {"_target_": "torch.load", "_args_": ["model.pt"]},
        {"_target_": "numpy.memmap", "_args_": ["file.dat"]},
        {"_target_": "pytests.test_spec.KeepsItsArguments", "size": 1},
    ],
    ids=["os.system", "eval", "open", "torch.load", "numpy.memmap", "not allowed"],
)
def test_targets_outside_the_allowed_modules_are_refused(spec):
    with pytest.raises(SpecError, match="not in the modules"):
        build(spec)


def test_functions_are_not_called():
    with pytest.raises(SpecError, match="only calls classes"):
        build({"_target_": "torch.nn.functional.relu", "_args_": [1.0]})


# --------------------------------------------------------------------------
# Models saved and loaded


def save(model, path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # no dataset/training/geometry given
        model.save(path, **kwargs)


@pytest.fixture
def stand_in_operator(monkeypatch):
    """A CT operator that needs no tomosipo (identity on 16x16 images)."""
    op = ImageMatrixOp(torch.eye(16 * 16), (1, 16, 16))
    monkeypatch.setattr(
        LIONmodel, "_make_operator", lambda self: setattr(self, "op", op)
    )


def small_geometry():
    geometry = Geometry.default_parameters()
    geometry.image_shape = [1, 16, 16]
    return geometry


def test_saved_file(tmp_path):
    save(UNet(), tmp_path / "model")
    saved = json.loads((tmp_path / "model.json").read_text())
    assert saved["lion_format"] == 1
    assert saved["model"]["_target_"] == "LION.models.CNNs.UNets.Unet.UNet"
    assert saved["model"]["model_parameters"]["_target_"] == (
        "LION.models.LIONmodel.LIONModelParameter"
    )
    assert not {"model_name", "model_parameters", "geometry"} & set(saved)


@pytest.mark.parametrize(
    "make",
    [DnCNN, DRUNet, UNet, MSDNet, GSD, GSDRUNet],
    ids=lambda c: c.__name__,
)
def test_lionmodel_load_builds_the_saved_class(tmp_path, make):
    torch.manual_seed(0)
    model = make().eval()
    save(model, tmp_path / "model")
    loaded, options, _ = LIONmodel.load(tmp_path / "model", supress_warnings=True)
    loaded = loaded.cpu().eval()
    assert type(loaded) is make
    assert options.model_name == make.__name__
    assert same(model.model_parameters, loaded.model_parameters) == []
    x = torch.rand(1, 1, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x))


def test_model_with_geometry(tmp_path, stand_in_operator):
    model = AR(AR.default_parameters(), small_geometry())
    save(model, tmp_path / "model")
    saved = json.loads((tmp_path / "model.json").read_text())
    assert saved["model"]["geometry"]["_target_"] == "LION.CTtools.ct_geometry.Geometry"
    loaded, options, _ = LIONmodel.load(tmp_path / "model", supress_warnings=True)
    assert type(loaded) is AR
    assert same(model.geometry, loaded.geometry) == []
    assert same(model.geometry, options.geometry) == []


def test_saving_checks_the_geometry(tmp_path, stand_in_operator):
    model = AR(AR.default_parameters(), small_geometry())
    model.geometry.angles = None  # broken after the model was made
    with pytest.raises(ValueError, match="Invalid CT geometry"):
        save(model, tmp_path / "model")


def test_data_metadata_comes_back(tmp_path):
    """The geometry, dataset and training parameters given to save()."""
    dataset = parameter(name="test", I0={1: 290, 2: 100}, angles=(0, 90))
    training = LIONParameter(epochs=3, inner=LIONParameter(lr=1e-3))
    save(
        DnCNN(),
        tmp_path / "model",
        geometry=small_geometry(),
        dataset=dataset,
        training=training,
    )
    _, options, _ = DnCNN.load(tmp_path / "model", supress_warnings=True)
    assert same(options.dataset_params, dataset) == []
    assert same(options.training, training) == []
    assert same(options.geometry, small_geometry()) == []


class LocalModel(DnCNN):
    """A model class outside LION."""


def test_own_model_classes(tmp_path):
    save(LocalModel(), tmp_path / "model")
    with pytest.raises(SpecError, match="not in the modules"):
        LIONmodel.load(tmp_path / "model", supress_warnings=True)
    loaded, _, _ = LIONmodel.load(
        tmp_path / "model", supress_warnings=True, allow=["pytests."]
    )
    assert type(loaded) is LocalModel
    loaded, _, _ = LocalModel.load(tmp_path / "model", supress_warnings=True)
    assert type(loaded) is LocalModel


def test_loading_as_another_class_warns(tmp_path):
    save(DnCNN(), tmp_path / "model")
    with pytest.warns(UserWarning, match="made as a"):
        LocalModel.load(tmp_path / "model", supress_warnings=True)


# --------------------------------------------------------------------------
# instantiate(): the saved model with new weights


def dncnn_with_other_parameters():
    params = DnCNN.default_parameters()
    params.blocks = 4
    return DnCNN(params)


@pytest.mark.parametrize("given", ["path", "pt", "json", "dict", "spec"])
def test_instantiate_makes_the_saved_model_untrained(tmp_path, given):
    torch.manual_seed(0)
    model = dncnn_with_other_parameters()
    save(model, tmp_path / "model")
    saved = json.loads((tmp_path / "model.json").read_text())
    source = {
        "path": tmp_path / "model",
        "pt": str(tmp_path / "model.pt"),
        "json": tmp_path / "model.json",
        "dict": saved,
        "spec": saved["model"],
    }[given]
    made = LIONmodel.instantiate(source)
    assert type(made) is DnCNN
    assert same(model.model_parameters, made.model_parameters) == []
    weights, new = model.state_dict(), made.state_dict()
    assert weights.keys() == new.keys()
    assert not all(torch.equal(weights[k], new[k]) for k in weights)


def test_instantiate_on_a_class(tmp_path):
    save(dncnn_with_other_parameters(), tmp_path / "model")
    assert DnCNN.instantiate(tmp_path / "model").model_parameters.blocks == 4
    with pytest.warns(UserWarning, match="made as a"):
        LocalModel.instantiate(tmp_path / "model")
    with pytest.raises(SpecError, match="not a LIONmodel"):
        LIONmodel.instantiate({"_target_": "LION.utils.parameter.LIONParameter"})


def test_instantiate_unversioned_file(tmp_path):
    save_unversioned(GSD(), tmp_path / "model")
    assert type(GSD.instantiate(tmp_path / "model")) is GSD
    saved = json.loads((tmp_path / "model.json").read_text())
    with pytest.raises(ValueError, match="give its file"):
        GSD.instantiate(saved)


def test_file_from_a_newer_lion(tmp_path):
    save(DnCNN(), tmp_path / "model")
    path = tmp_path / "model.json"
    saved = json.loads(path.read_text())
    saved["lion_format"] = 99
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="format 99"):
        DnCNN.load(tmp_path / "model", supress_warnings=True)


def save_unversioned(model, path):
    """A model saved as LION did without ``lion_format``: the parameters as
    plain JSON and the class by name."""
    options = LIONParameter()
    options.commit_hash = "0" * 40
    options.model_name = type(model).__name__
    options.model_parameters = model.model_parameters
    options.save(path.with_suffix(".json"))
    torch.save({"model_state_dict": model.state_dict()}, path.with_suffix(".pt"))


def test_unversioned_file_loads_with_its_class(tmp_path):
    torch.manual_seed(0)
    model = GSD().eval()
    save_unversioned(model, tmp_path / "model")
    loaded, options, _ = GSD.load(tmp_path / "model", supress_warnings=True)
    assert options.model_name == "GSD"
    x = torch.rand(1, 1, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(loaded.cpu().eval()(x), model(x))
    with pytest.raises(ValueError, match="names its model class only by name"):
        LIONmodel.load(tmp_path / "model", supress_warnings=True)


# --------------------------------------------------------------------------
# Unknown parameter names


def test_unknown_parameter_name_warns():
    params = DnCNN.default_parameters()
    params.blokcs = 3  # a typo of "blocks"
    with pytest.warns(UserWarning, match="unknown parameter.*blokcs"):
        DnCNN(params)


def test_known_parameters_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DnCNN(DnCNN.default_parameters())
        GSD(GSD.default_parameters())


def test_saved_file_with_a_removed_parameter_loads_with_a_warning(tmp_path):
    """E.g. a parameter a model no longer has: the file still loads."""
    save(DnCNN(), tmp_path / "model")
    path = tmp_path / "model.json"
    saved = json.loads(path.read_text())
    saved["model"]["model_parameters"]["lambda_primal"] = 0.1
    path.write_text(json.dumps(saved))
    with pytest.warns(UserWarning, match="lambda_primal"):
        loaded, _, _ = DnCNN.load(tmp_path / "model", supress_warnings=True)
    assert isinstance(loaded, DnCNN)


# --------------------------------------------------------------------------
# The weights file (.pt): read with torch's safe loader


def test_pt_holds_only_safe_contents(tmp_path):
    model = DnCNN()
    optimiser = torch.optim.Adam(model.parameters())
    model(torch.rand(1, 1, 8, 8)).sum().backward()
    optimiser.step()
    save_kwargs = dict(dataset=LIONParameter(name="x"), geometry=small_geometry())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.save_checkpoint(
            tmp_path / "checkpoint",
            epoch=np.int64(3),
            loss=np.array([0.5, 0.4, 0.3]),
            optimizer=optimiser,
            training_param=LIONParameter(lr=1e-3),
            **save_kwargs,
        )
    # torch's own safe loader, with nothing extra allowed
    data = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert set(data) == {
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "loss",
        "initial_random_state",  # torch's random state when the model was made
    }
    assert data["epoch"] == 3 and data["loss"] == [0.5, 0.4, 0.3]
    assert data["initial_random_state"].dtype == torch.uint8


@pytest.mark.parametrize(
    "loss, expected",
    [
        (np.array([0.5, 0.4]), np.array([0.5, 0.4])),  # training history
        (np.float64(0.25), 0.25),  # the best validation value
        (torch.tensor([0.5, 0.4]), np.array([0.5, 0.4])),
    ],
    ids=["array", "numpy float", "tensor"],
)
def test_loss_comes_back_as_numbers(tmp_path, loss, expected):
    save(DnCNN(), tmp_path / "model", loss=loss, epoch=1)
    _, _, data = DnCNN.load(tmp_path / "model", supress_warnings=True)
    assert type(data["loss"]) is type(expected)
    np.testing.assert_allclose(data["loss"], expected)


def old_pt(path, model):
    """A .pt as LION saved it with pickled parameter objects, a geometry, a
    path and numpy values."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset_params": LIONParameter(folder=pathlib.Path("/data/LIDC-IDRI")),
            "training_params": LIONParameter(epochs=3),
            "geometry": Geometry.default_parameters(),
            "loss": np.array([0.5, 0.4], dtype=np.float32),
            "epoch": 3,
        },
        path,
    )


def test_old_pt_loads_safely(tmp_path):
    model = DnCNN()
    save(model, tmp_path / "model")
    old_pt(tmp_path / "model.pt", model)
    with pytest.raises(pickle.UnpicklingError):  # torch alone refuses it
        torch.load(tmp_path / "model.pt", weights_only=True)
    loaded, _, data = DnCNN.load(tmp_path / "model", supress_warnings=True)
    assert isinstance(data["geometry"], Geometry)
    assert data["dataset_params"].folder == pathlib.Path("/data/LIDC-IDRI")
    np.testing.assert_allclose(data["loss"], [0.5, 0.4])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key].cpu(), value.cpu())


class NotAllowed:
    """An object LION doesn't load safely."""


def test_pt_with_other_objects_needs_weights_only_false(tmp_path):
    model = DnCNN()
    save(model, tmp_path / "model")
    torch.save(
        {"model_state_dict": model.state_dict(), "extra": NotAllowed()},
        tmp_path / "model.pt",
    )
    with pytest.raises(pickle.UnpicklingError, match="weights_only=False"):
        DnCNN.load(tmp_path / "model", supress_warnings=True)
    _, _, data = DnCNN.load(
        tmp_path / "model", supress_warnings=True, weights_only=False
    )
    assert isinstance(data["extra"], NotAllowed)


def test_files_from_numpy_1_are_allowed():
    """numpy 1 named its array functions numpy.core.multiarray.*"""
    from LION.models.LIONmodel import _safe_globals

    names = {entry[1] for entry in _safe_globals() if isinstance(entry, tuple)}
    assert {
        "numpy.core.multiarray._reconstruct",
        "numpy.core.multiarray.scalar",
    } <= names
