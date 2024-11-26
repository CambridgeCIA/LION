# This file is part of LION library
# License : GPL-3
#
# Author: Ferdia Sherry
# Modifications: -
# =============================================================================

from LION.CTtools.ct_utils import make_operator
from LION.experiments.ct_benchmarking_experiments import (
    FullDataCTRecon,
    LimitedAngle120CTRecon,
    LimitedAngle90CTRecon,
    LimitedAngle60CTRecon,
    SparseAngle360CTRecon,
    SparseAngle120CTRecon,
    SparseAngle60CTRecon,
    LowDoseCTRecon,
    BeamHardeningCTRecon,
)
from LION.models.LIONmodel import LIONParameter
from LION.models.PnP import GSDRUNet

import argparse
import json
import os
from skimage.metrics import structural_similarity as ssim
import torch


def psnr(x, y):
    return 10 * torch.log10((x**2).max() / ((x - y) ** 2).mean())


def my_ssim(x: torch.tensor, y: torch.tensor):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())


with open("normalisation.json", "r") as in_file:
    normalisation = json.load(in_file)
    x_min, x_max = normalisation["x_min"], normalisation["x_max"]


def get_denoiser(model):
    def denoiser(x):
        x = (x - x_min) / (x_max - x_min)
        out = model(x)
        return x_min + (x_max - x_min) * out

    return denoiser


def data_obj_grad(op, x, y):
    res = op(x[0]) - y[0]
    data_grad = op.T(res).unsqueeze(0)
    return 0.5 * (res**2).sum(), data_grad


def operator_norm(operator, N_iter=500):
    u = torch.randn(1, 1024, 1024).cuda()
    for i in range(N_iter):
        u /= u.norm()
        u = operator.T(operator(u))
    return u.norm().sqrt().item()


parser = argparse.ArgumentParser("validate_dncnn")
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--result_path", type=str, default=".")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--testing", action="store_true")
params = vars(parser.parse_args())
print(params)

torch.cuda.set_device(params["device"])
chkpt = torch.load(params["checkpoint"], map_location="cpu")
config = chkpt["config"]
model = GSDRUNet(
    LIONParameter(
        in_channels=1,
        out_channels=1,
        int_channels=config["int_channels"],
        kernel_size=(config["kernel_size"], config["kernel_size"]),
        n_blocks=config["n_blocks"],
        use_noise_level=False,
        bias_free=config["bias_free"],
        act="elu",
        enforce_positivity=False,
    )
).cuda()
model.load_state_dict(chkpt["state_dict"])
model.eval()
denoiser = get_denoiser(model)


for experiment in [
    FullDataCTRecon(),
    LimitedAngle120CTRecon(),
    LimitedAngle90CTRecon(),
    LimitedAngle60CTRecon(),
    SparseAngle360CTRecon(),
    SparseAngle120CTRecon(),
    SparseAngle60CTRecon(),
    LowDoseCTRecon(),
    BeamHardeningCTRecon(),
]:
    print(experiment)
    operator = make_operator(experiment.geo)
    op_norm = operator_norm(operator)
    step_size = 1.0 / op_norm**2

    if params["testing"]:
        data = experiment.get_testing_dataset()
        split = "test"
    else:
        data = experiment.get_validation_dataset()
        split = "val"
    dataloader = torch.utils.data.DataLoader(data, 1, shuffle=False)

    psnrs = []
    ssims = []
    for i, (y, x) in enumerate(dataloader):
        y, x = y.cuda(), x.cuda()
        recon = torch.zeros_like(x)
        for it in range(100):
            data_obj, data_grad = data_obj_grad(operator, recon, y)
            reg_obj, reg_grad = model.obj_grad(recon)
            print((data_obj + reg_obj).item())
            recon = recon - step_size * (data_grad + reg_grad)
        recon = denoiser(recon.detach())
        psnrs.append(psnr(x, recon).item())
        ssims.append(my_ssim(x, recon).item())
        print(
            f"It {i + 1} / {len(dataloader)}: PSNR = {psnrs[-1]:.1f} dB, SSIM = {ssims[-1]:.3}"
        )
    psnrs, ssims = torch.tensor(psnrs), torch.tensor(ssims)
    torch.save(
        {"psnrs": psnrs, "ssims": ssims},
        os.path.join(
            params["result_path"],
            f"gs_drunet_{experiment.experiment_params.name.replace(' ', '_')}_{split}_noise_level={config['noise_level']}.pt",
        ),
    )
    print(
        f"PSNR = {psnrs.mean():.1f} +- {psnrs.std():.1f} dB, SSIM= {ssims.mean():.3f} +- {ssims.std():.3f}"
    )
