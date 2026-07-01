import json
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

warnings.filterwarnings(
    "ignore",
    message="`torch.jit.script` is deprecated",
    category=DeprecationWarning,
)

from scripts.paper_scripts.PaDIS import PaDIS_LIDC_256 as lidc256
from scripts.paper_scripts.PaDIS import PaDIS_LIDC_512 as lidc512
from scripts.paper_scripts.PaDIS import PaDIS_LIDC_PnP_denoiser as pnp


class FakeRun:
    def __init__(self):
        self.id = "new-run-id"
        self.name = "new-run-name"
        self.summary = {}
        self.defined_metrics = []
        self.logs = []
        self.artifacts = []

    def define_metric(self, *args, **kwargs):
        self.defined_metrics.append((args, kwargs))

    def log(self, metrics, step=None, **kwargs):
        self.logs.append({"metrics": metrics, "step": step, "kwargs": kwargs})

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)


class FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []

    def add_file(self, path):
        self.files.append(path)


class FakeWandb:
    Image = str
    Artifact = FakeArtifact

    def __init__(self):
        self.run = FakeRun()
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


class FakeCheckpointSolver:
    def __init__(self, folder, pattern, current_epoch=0):
        self.checkpoint_save_folder = folder
        self.checkpoint_fname = pattern
        self.train_loss = []
        self.current_epoch = current_epoch
        self.saved_epochs = []

    def save_checkpoint(self, epoch):
        self.saved_epochs.append(epoch)
        filename = self.checkpoint_fname.replace("*", f"{epoch + 1:04d}")
        (self.checkpoint_save_folder / filename).write_text("checkpoint")


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


@pytest.mark.parametrize("module", [lidc256, lidc512])
def test_diffusion_wandb_resume_reuses_saved_run_id_without_manual_step(
    module, tmp_path, fake_wandb
):
    run_folder = tmp_path / "diffusion_run"
    run_folder.mkdir()
    (run_folder / "wandb_run.json").write_text(
        json.dumps({"id": "resume-id", "name": "old-name"})
    )
    args = module.build_arg_parser().parse_args(["--wandb-project", "PaDIS-Test"])

    run = module.init_wandb(args, run_folder, "padis-test-preset")
    log_fn = module.wandb_log_fn(run)
    log_fn({"train/seen_patches": 128, "train/loss": 0.5}, 128)

    assert fake_wandb.init_kwargs["id"] == "resume-id"
    assert fake_wandb.init_kwargs["resume"] == "allow"
    assert fake_wandb.init_kwargs["mode"] == "online"
    assert "settings" not in fake_wandb.init_kwargs
    assert (("train/seen_patches",), {}) in run.defined_metrics
    assert (
        ("train/*",),
        {"step_metric": "train/seen_patches"},
    ) in run.defined_metrics
    assert run.logs[-1]["step"] is None
    assert run.logs[-1]["metrics"]["train/seen_patches"] == 128


def test_pnp_wandb_resume_reuses_saved_run_id_without_manual_step(tmp_path, fake_wandb):
    run_folder = tmp_path / "pnp_run"
    run_folder.mkdir()
    (run_folder / "wandb_run.json").write_text(
        json.dumps({"id": "resume-pnp-id", "name": "old-name"})
    )
    args = pnp.build_arg_parser().parse_args(["--wandb-project", "PaDIS-Test"])
    solver = SimpleNamespace(
        train_loss=np.array([0.2]),
        validation_loss=np.array([0.1]),
    )

    run = pnp.init_wandb(args, run_folder, {"run_folder": str(run_folder)})
    pnp.log_epoch_to_wandb(run, solver, 0)

    assert fake_wandb.init_kwargs["id"] == "resume-pnp-id"
    assert fake_wandb.init_kwargs["resume"] == "allow"
    assert fake_wandb.init_kwargs["mode"] == "online"
    assert "settings" not in fake_wandb.init_kwargs
    assert (("epoch",), {}) in run.defined_metrics
    assert (("train_loss",), {"step_metric": "epoch"}) in run.defined_metrics
    assert run.logs[-1]["step"] is None
    assert run.logs[-1]["metrics"] == {
        "epoch": 1,
        "train_loss": 0.2,
        "validation_loss": 0.1,
    }


@pytest.mark.parametrize("module", [lidc256, lidc512, pnp])
def test_wandb_output_artifact_matches_padis_run_folder_contract(
    module, tmp_path, fake_wandb
):
    run_folder = tmp_path / "artifact_run"
    run_folder.mkdir()
    for filename in (
        "checkpoint.pt",
        "training_config.json",
        "loss.png",
        "validation_loss.png",
        "notes.txt",
    ):
        (run_folder / filename).write_text(filename)

    module.log_wandb_outputs(fake_wandb.run, run_folder, log_artifact=True)

    assert fake_wandb.run.artifacts
    artifact = fake_wandb.run.artifacts[-1]
    artifact_names = {path.split("/")[-1] for path in artifact.files}
    assert artifact.name == run_folder.name
    assert artifact.type == "padis-run"
    assert artifact_names == {
        "checkpoint.pt",
        "training_config.json",
        "loss.png",
        "validation_loss.png",
    }


@pytest.mark.parametrize("module", [lidc256, lidc512])
def test_diffusion_interruption_checkpoint_uses_next_periodic_index(module, tmp_path):
    pattern = "padis_checkpoint_*.pt"
    (tmp_path / "padis_checkpoint_0001.pt").write_text("old")
    (tmp_path / "padis_checkpoint_0002.pt").write_text("old")
    (tmp_path / "padis_checkpoint_0002.ema.pt").write_text("ema")
    solver = FakeCheckpointSolver(tmp_path, pattern)

    module.save_interruption_checkpoint(solver)

    assert solver.saved_epochs == [2]
    assert (tmp_path / "padis_checkpoint_0003.pt").is_file()


def test_pnp_interruption_checkpoint_saves_current_epoch_with_retention(tmp_path):
    pattern = "pnp_check_*.pt"
    (tmp_path / "pnp_check_0007.pt").write_text("old")
    solver = FakeCheckpointSolver(tmp_path, pattern, current_epoch=7)

    pnp.save_interruption_checkpoint(solver, max_periodic_checkpoints=1)

    assert solver.saved_epochs == [7]
    assert not (tmp_path / "pnp_check_0007.pt").exists()
    assert (tmp_path / "pnp_check_0008.pt").is_file()
