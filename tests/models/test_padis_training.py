import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from LION.CTtools.ct_geometry import Geometry
from LION.losses.PaDIS import (
    PaDISDenoisingLoss,
    build_position_grid,
    sample_patch_pair,
    zero_pad_images,
)
from LION.models.diffusion import NCSNpp
from LION.optimizers import PaDISSolver


def _tiny_padis_model(mode="padis-paper-ct-256"):
    geometry = Geometry.default_parameters(image_scaling=0.5)
    params = NCSNpp.default_parameters(mode)
    params.model_channels = 8
    params.channel_mult = [1, 1, 1, 1]
    params.num_res_blocks = 1
    return NCSNpp(params, geometry), geometry


def test_padis_paper_256_defaults():
    params = NCSNpp.default_parameters("padis-paper-ct-256")
    assert params.channel_mult == [1, 2, 2, 2]
    assert params.dropout == 0.05
    assert params.patch_sizes == [16, 32, 56]
    assert params.patch_probabilities == [0.2, 0.3, 0.5]
    assert params.pad_width == 24
    assert params.input_position_channels == 2
    assert params.centered is True
    assert params.noise_label_type == "identity"
    assert params.network_backend == "score_sde_pytorch"
    assert params.padis_embedding_compat is True
    assert params.padis_attention_compat is True
    assert params.init_scale == 1e-10


def test_padis_paper_512_defaults():
    params = NCSNpp.default_parameters("padis-paper-ct-512")
    assert params.patch_sizes == [16, 32, 64]
    assert params.pad_width == 64
    assert params.sigma_min == 0.002
    assert params.sigma_max == 40


def test_padis_paper_whole_image_defaults():
    params = NCSNpp.default_parameters("padis-paper-whole-ct-256")
    assert params.channel_mult == [1, 2, 2, 4]
    assert params.patch_sizes == [256]
    assert params.patch_probabilities == [1.0]
    assert params.pad_width == 0
    assert params.largest_patch_size == 256
    assert params.prior_mode == "whole_image"
    assert params.input_position_channels == 2


def test_padis_paper_model_parameter_counts():
    # The paper reports ~60M weights for patch-based PaDIS models. The ~110M
    # variant is the separate whole-image baseline with a wider fourth layer.
    expected_counts = {
        "padis-paper-ct-256": (0.5, 60_483_713),
        "padis-paper-ct-512": (1.0, 60_483_713),
        "padis-paper-whole-ct-256": (0.5, 114_643_585),
    }
    for mode, (image_scaling, expected_count) in expected_counts.items():
        geometry = Geometry.default_parameters(image_scaling=image_scaling)
        model = NCSNpp(NCSNpp.default_parameters(mode), geometry)
        total = sum(param.numel() for param in model.parameters())
        trainable = sum(
            param.numel() for param in model.parameters() if param.requires_grad
        )
        assert total == expected_count
        assert trainable == expected_count


def test_patch_utilities_shapes_and_ranges():
    images = torch.rand(2, 1, 256, 256)
    padded = zero_pad_images(images, 24)
    positions = build_position_grid(
        2, 304, 304, device=images.device, dtype=images.dtype
    )
    image_patch, position_patch = sample_patch_pair(padded, positions, 56)
    assert padded.shape == (2, 1, 304, 304)
    assert positions.shape == (2, 2, 304, 304)
    assert image_patch.shape == (2, 1, 56, 56)
    assert position_patch.shape == (2, 2, 56, 56)
    assert torch.all(positions >= -1)
    assert torch.all(positions <= 1)


def test_ncsnpp_patch_forward_shape():
    model, _ = _tiny_padis_model()
    x = torch.randn(2, 3, 16, 16)
    sigma = torch.ones(2)
    assert model(x, sigma).shape == (2, 1, 16, 16)


def test_padis_loss_backpropagates():
    model, _ = _tiny_padis_model()
    clean = torch.rand(2, 1, 16, 16)
    positions = build_position_grid(2, 16, 16, device=clean.device, dtype=clean.dtype)
    loss = PaDISDenoisingLoss()(model, clean, positions)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(param.grad is not None for param in model.parameters())


def test_padis_loss_uses_edm_lognormal_by_default():
    loss = PaDISDenoisingLoss()
    assert loss.sigma_distribution == "edm_lognormal"
    assert loss.P_mean == -1.2
    assert loss.P_std == 1.2
    assert loss.sigma_data == 0.5
    assert loss.reduction == "batch_mean_sum"


def test_padis_loss_uses_patch_l2_norm_not_pixel_mean():
    class ZeroModel(nn.Module):
        def forward(self, x, time_cond):
            return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)

    clean = torch.zeros(1, 1, 8, 8)
    model = ZeroModel()
    torch.manual_seed(0)
    batch_sum_loss = PaDISDenoisingLoss(reduction="batch_mean_sum")(model, clean)
    torch.manual_seed(0)
    mean_loss = PaDISDenoisingLoss(reduction="mean")(model, clean)
    assert torch.allclose(batch_sum_loss, mean_loss * clean[0].numel())


def test_padis_solver_minibatch_and_ema():
    model, geometry = _tiny_padis_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    solver_params = PaDISSolver.default_parameters("padis-paper-ct-256")
    solver_params.patch_sizes = [16]
    solver_params.patch_probabilities = [1.0]
    solver = PaDISSolver(
        model,
        optimizer,
        PaDISDenoisingLoss(),
        geometry=geometry,
        solver_params=solver_params,
        device=torch.device("cpu"),
    )
    loss = solver.mini_batch_step(torch.empty(2, 1, 1, 1), torch.rand(2, 1, 256, 256))
    loss.backward()
    optimizer.step()
    solver._update_ema(2)
    assert torch.isfinite(loss)
    assert solver.ema_state is not None


def test_padis_solver_default_checkpoint_freq_is_safe_without_checkpointing():
    model, geometry = _tiny_padis_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    solver_params = PaDISSolver.default_parameters("padis-paper-ct-256")
    solver_params.patch_sizes = [16]
    solver_params.patch_probabilities = [1.0]
    solver = PaDISSolver(
        model,
        optimizer,
        PaDISDenoisingLoss(),
        geometry=geometry,
        solver_params=solver_params,
        device=torch.device("cpu"),
    )
    assert solver.checkpoint_freq == 10**12


def test_padis_solver_trains_to_patch_budget_across_loader_restarts():
    model, geometry = _tiny_padis_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    solver_params = PaDISSolver.default_parameters("padis-paper-ct-256")
    solver_params.patch_sizes = [16]
    solver_params.patch_probabilities = [1.0]
    solver_params.patch_batch_multipliers = {16: 1}
    solver_params.lr_rampup_kimg = None
    solver = PaDISSolver(
        model,
        optimizer,
        PaDISDenoisingLoss(),
        geometry=geometry,
        solver_params=solver_params,
        device=torch.device("cpu"),
    )
    images = torch.rand(2, 1, 256, 256)
    train_loader = DataLoader(TensorDataset(images, images), batch_size=1)
    validation_loader = DataLoader(TensorDataset(images[:1], images[:1]), batch_size=1)
    solver.set_training(train_loader)
    solver.set_validation(validation_loader, validation_freq=10**12)
    logs = []
    solver.train_for_patches(
        3,
        validation_interval_patches=2,
        log_fn=lambda metrics, step: logs.append((metrics, step)),
    )
    assert solver.seen_patches == 3
    assert len(solver.train_loss) == 3
    assert len(solver.validation_loss) == 1
    assert sum("train/loss" in metrics for metrics, _ in logs) == 3
    assert sum("validation/loss" in metrics for metrics, _ in logs) == 1
    assert all(step > 0 for _, step in logs)


def test_padis_solver_uses_paper_relative_batch_multipliers():
    params = PaDISSolver.default_parameters("padis-paper-ct-256")
    assert params.patch_batch_multipliers == {16: 4, 32: 2, 56: 1}
    assert params.ema_rampup_ratio == 0.05
    assert params.lr_rampup_kimg == 10_000
    assert params.enforce_data_range is True
    params = PaDISSolver.default_parameters("padis-paper-ct-512")
    assert params.patch_batch_multipliers == {16: 4, 32: 2, 64: 1}
    params = PaDISSolver.default_parameters("padis-paper-whole-ct-256")
    assert params.prior_mode == "whole_image"
    assert params.patch_batch_multipliers == {256: 1}
    assert params.default_batch_size == 8


def test_whole_image_solver_minibatch_uses_full_image():
    model, geometry = _tiny_padis_model("padis-paper-whole-ct-256")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    solver_params = PaDISSolver.default_parameters("padis-paper-whole-ct-256")
    solver_params.lr_rampup_kimg = None
    solver = PaDISSolver(
        model,
        optimizer,
        PaDISDenoisingLoss(),
        geometry=geometry,
        solver_params=solver_params,
        device=torch.device("cpu"),
    )
    target = torch.rand(1, 1, 256, 256)
    clean, positions = solver._sample_training_patch(target, patch_size=256)
    assert clean.shape == target.shape
    assert positions.shape == (1, 2, 256, 256)
    loss = solver.mini_batch_step(torch.empty(1, 1, 1, 1), target)
    loss.backward()
    assert torch.isfinite(loss)


def test_ema_weight_swap_restores_raw_weights():
    model, geometry = _tiny_padis_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    solver_params = PaDISSolver.default_parameters("padis-paper-ct-256")
    solver = PaDISSolver(
        model,
        optimizer,
        PaDISDenoisingLoss(),
        geometry=geometry,
        solver_params=solver_params,
        device=torch.device("cpu"),
    )
    first_name, first_param = next(iter(model.named_parameters()))
    raw = first_param.detach().clone()
    solver.ema_state[first_name] = raw + 1
    raw_state = solver._apply_ema_weights()
    assert torch.allclose(first_param, raw + 1)
    solver._restore_raw_weights(raw_state)
    assert torch.allclose(first_param, raw)


def test_save_validation_writes_ema_weights_and_restores_raw_model(tmp_path):
    model, geometry = _tiny_padis_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    solver_params = PaDISSolver.default_parameters("padis-paper-ct-256")
    solver = PaDISSolver(
        model,
        optimizer,
        PaDISDenoisingLoss(),
        geometry=geometry,
        solver_params=solver_params,
        device=torch.device("cpu"),
    )
    solver.validation_fname = "padis_min_val.pt"
    solver.validation_save_folder = tmp_path
    solver.validation_fn = solver.loss_fn
    solver.validation_loss = [0.25]

    first_name, first_param = next(iter(model.named_parameters()))
    raw = first_param.detach().clone()
    solver.ema_state[first_name] = raw + 1

    solver.save_validation(0)

    assert torch.allclose(first_param, raw)
    data = torch.load(
        tmp_path / "padis_min_val.pt", map_location="cpu", weights_only=False
    )
    assert torch.allclose(data["model_state_dict"][first_name], raw + 1)
    full_data = torch.load(
        tmp_path / "padis_min_val_full.pt", map_location="cpu", weights_only=False
    )
    assert torch.allclose(full_data["model_state_dict"][first_name], raw)
    assert torch.allclose(full_data["ema_state_dict"][first_name], raw + 1)
    assert full_data["full_save_kind"] == "validation"
    assert full_data["validation_loss"] == 0.25


def test_train_for_patches_preserves_existing_min_validation_on_resume(tmp_path):
    model, geometry = _tiny_padis_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    solver_params = PaDISSolver.default_parameters("padis-paper-ct-256")
    solver_params.patch_sizes = [16]
    solver_params.patch_probabilities = [1.0]
    solver_params.patch_batch_multipliers = {16: 1}
    solver_params.lr_rampup_kimg = None
    solver = PaDISSolver(
        model,
        optimizer,
        PaDISDenoisingLoss(),
        geometry=geometry,
        solver_params=solver_params,
        device=torch.device("cpu"),
        save_folder=tmp_path,
    )
    images = torch.rand(2, 1, 256, 256)
    validation_loader = DataLoader(TensorDataset(images[:1], images[:1]), batch_size=1)
    solver.set_validation(
        validation_loader,
        validation_freq=10**12,
        validation_fname="padis_min_val.pt",
        save_folder=tmp_path,
    )
    solver.validation_loss = [0.1]
    solver.save_validation(0)

    resumed_model, resumed_geometry = _tiny_padis_model()
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
    resumed_solver = PaDISSolver(
        resumed_model,
        resumed_optimizer,
        PaDISDenoisingLoss(),
        geometry=resumed_geometry,
        solver_params=solver_params,
        device=torch.device("cpu"),
        save_folder=tmp_path,
    )
    train_loader = DataLoader(TensorDataset(images, images), batch_size=1)
    resumed_solver.set_training(train_loader)
    resumed_solver.set_validation(
        validation_loader,
        validation_freq=10**12,
        validation_fname="padis_min_val.pt",
        save_folder=tmp_path,
    )
    resumed_solver.validate = lambda: 0.2

    resumed_solver.train_for_patches(1, validation_interval_patches=1)

    assert resumed_solver.validation_loss[0] == 0.1
    assert resumed_solver.validation_loss[1] == 0.2
    data = torch.load(
        tmp_path / "padis_min_val.pt", map_location="cpu", weights_only=False
    )
    assert data["loss"] == 0.1


def test_save_final_results_writes_resumable_full_state(tmp_path):
    model, geometry = _tiny_padis_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    solver_params = PaDISSolver.default_parameters("padis-paper-ct-256")
    solver = PaDISSolver(
        model,
        optimizer,
        PaDISDenoisingLoss(),
        geometry=geometry,
        solver_params=solver_params,
        device=torch.device("cpu"),
        save_folder=tmp_path,
    )
    solver.set_saving(tmp_path, "padis_final")
    first_name, first_param = next(iter(model.named_parameters()))
    raw = first_param.detach().clone()
    solver.ema_state[first_name] = raw + 1
    solver.seen_patches = 5
    solver.train_loss = [1.0, 0.5]

    solver.save_final_results()

    lightweight = torch.load(
        tmp_path / "padis_final.pt", map_location="cpu", weights_only=False
    )
    full_data = torch.load(
        tmp_path / "padis_final_full.pt", map_location="cpu", weights_only=False
    )
    assert torch.allclose(lightweight["model_state_dict"][first_name], raw + 1)
    assert torch.allclose(full_data["model_state_dict"][first_name], raw)
    assert torch.allclose(full_data["ema_state_dict"][first_name], raw + 1)
    assert full_data["seen_patches"] == 5
    assert full_data["full_save_kind"] == "final"
    assert "optimizer_state_dict" in full_data


def test_load_checkpoint_falls_back_to_full_final_state(tmp_path):
    model, geometry = _tiny_padis_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    solver_params = PaDISSolver.default_parameters("padis-paper-ct-256")
    solver = PaDISSolver(
        model,
        optimizer,
        PaDISDenoisingLoss(),
        geometry=geometry,
        solver_params=solver_params,
        device=torch.device("cpu"),
        save_folder=tmp_path,
    )
    solver.set_saving(tmp_path, "padis_final")
    first_name, first_param = next(iter(model.named_parameters()))
    raw = first_param.detach().clone()
    solver.ema_state[first_name] = raw + 1
    solver.seen_patches = 7
    solver.save_final_results()

    resumed_model, resumed_geometry = _tiny_padis_model()
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
    resumed_solver = PaDISSolver(
        resumed_model,
        resumed_optimizer,
        PaDISDenoisingLoss(),
        geometry=resumed_geometry,
        solver_params=solver_params,
        device=torch.device("cpu"),
        save_folder=tmp_path,
    )
    resumed_solver.set_saving(tmp_path, "padis_final")
    resumed_solver.set_checkpointing(
        "missing_checkpoint_*.pt",
        checkpoint_freq=10**12,
        load_checkpoint_if_exists=True,
        save_folder=tmp_path,
    )

    resumed_solver.load_checkpoint()
    resumed_first_param = dict(resumed_model.named_parameters())[first_name]

    assert torch.allclose(resumed_first_param, raw)
    assert torch.allclose(resumed_solver.ema_state[first_name], raw + 1)
    assert resumed_solver.seen_patches == 7


def test_load_checkpoint_loads_periodic_checkpoint_without_reconstructing_model(
    tmp_path,
):
    model, geometry = _tiny_padis_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    solver_params = PaDISSolver.default_parameters("padis-paper-ct-256")
    solver = PaDISSolver(
        model,
        optimizer,
        PaDISDenoisingLoss(),
        geometry=geometry,
        solver_params=solver_params,
        device=torch.device("cpu"),
        save_folder=tmp_path,
    )
    solver.set_checkpointing(
        "padis_checkpoint_*.pt",
        checkpoint_freq=10**12,
        load_checkpoint_if_exists=True,
        save_folder=tmp_path,
    )
    first_name, first_param = next(iter(model.named_parameters()))
    raw = first_param.detach().clone()
    with torch.no_grad():
        first_param.add_(2)
    solver.ema_state[first_name] = raw + 3
    solver.seen_patches = 11
    solver.save_checkpoint(0)

    resumed_model, resumed_geometry = _tiny_padis_model()
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-4)
    resumed_solver = PaDISSolver(
        resumed_model,
        resumed_optimizer,
        PaDISDenoisingLoss(),
        geometry=resumed_geometry,
        solver_params=solver_params,
        device=torch.device("cpu"),
        save_folder=tmp_path,
    )
    resumed_solver.set_checkpointing(
        "padis_checkpoint_*.pt",
        checkpoint_freq=10**12,
        load_checkpoint_if_exists=True,
        save_folder=tmp_path,
    )

    epoch = resumed_solver.load_checkpoint()
    resumed_first_param = dict(resumed_model.named_parameters())[first_name]

    assert epoch == 1
    assert torch.allclose(resumed_first_param, raw + 2)
    assert torch.allclose(resumed_solver.ema_state[first_name], raw + 3)
    assert resumed_solver.seen_patches == 11


def test_first_ema_update_copies_model_like_padis_rampup():
    model, geometry = _tiny_padis_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    solver_params = PaDISSolver.default_parameters("padis-paper-ct-256")
    solver = PaDISSolver(
        model,
        optimizer,
        PaDISDenoisingLoss(),
        geometry=geometry,
        solver_params=solver_params,
        device=torch.device("cpu"),
    )
    first_name, first_param = next(iter(model.named_parameters()))
    with torch.no_grad():
        first_param.add_(1)
    solver._update_ema(1)
    assert torch.allclose(solver.ema_state[first_name], first_param)


def test_padis_solver_rejects_out_of_range_images():
    model, geometry = _tiny_padis_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    solver_params = PaDISSolver.default_parameters("padis-paper-ct-256")
    solver_params.patch_sizes = [16]
    solver_params.patch_probabilities = [1.0]
    solver = PaDISSolver(
        model,
        optimizer,
        PaDISDenoisingLoss(),
        geometry=geometry,
        solver_params=solver_params,
        device=torch.device("cpu"),
    )
    try:
        solver.mini_batch_step(torch.empty(1, 1, 1, 1), torch.full((1, 1, 16, 16), 2.0))
    except ValueError as exc:
        assert "[0, 1]" in str(exc)
    else:
        raise AssertionError("Expected out-of-range images to be rejected.")
