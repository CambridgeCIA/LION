import LION.experiments.ct_benchmarking_experiments as ct_benchmarking
from LION.models.LIONmodel import LIONParameter
from LION.models.PnP import DnCNN, DRUNet, GSDRUNet


import argparse
import git
import json
from kornia.augmentation import RandomCrop, RandomErasing

from math import inf
import os
import torch
import uuid
import wandb


def psnr(x, y):
    return 10 * torch.log10((x**2).max() / ((x - y) ** 2).mean())


def mean_grad_norm(model: torch.nn.Module):
    num_params = 0
    grad_sqr_norms = 0.0
    for param in model.parameters():
        num_params += param.numel()
        grad_sqr_norms += param.grad.norm() ** 2
    return (grad_sqr_norms.sqrt() / num_params).item()


parser = argparse.ArgumentParser("train_denoisers")
parser.add_argument("--model", type=str, default="dncnn")
parser.add_argument("--depth", type=int, default=20)
parser.add_argument("--n_blocks", type=int, default=4)
parser.add_argument("--channels", type=int, default=64)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--noise_level", type=float, default=0.05)
parser.add_argument("--kernel_size", type=int, default=3)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--bias_free", action="store_true")
parser.add_argument("--enforce_positivity", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--results_path", type=str, default="")


if __name__ == "__main__":
    params = vars(parser.parse_args())
    if params["debug"]:
        os.environ["WANDB_MODE"] = "offline"

    assert params["model"] in ["dncnn", "drunet", "gs_drunet"]

    print(params)
    torch.cuda.set_device(params["device"])

    commit_hash = git.Repo(
        ".", search_parent_directories=True
    ).head.reference.commit.hexsha

    if params["model"] == "dncnn":
        model_params = LIONParameter(
            in_channels=1,
            int_channels=params["channels"],
            kernel_size=(params["kernel_size"], params["kernel_size"]),
            blocks=params["depth"],
            residual=True,
            bias_free=params["bias_free"],
            act="leaky_relu",
            enforce_positivity=params["enforce_positivity"],
            batch_normalisation=True,
        )
        model = DnCNN(model_params).cuda()
        config = {
            "depth": params["depth"],
            "int_channels": params["channels"],
            "kernel_size": params["kernel_size"],
            "bias_free": params["bias_free"],
            "enforce_positivity": params["enforce_positivity"],
            "lr": params["lr"],
            "epochs": params["epochs"],
            "noise_level": params["noise_level"],
            "commit_hash": commit_hash,
        }
    elif params["model"] == "drunet":
        model_params = LIONParameter(
            in_channels=1,
            out_channels=1,
            int_channels=params["channels"],
            kernel_size=(params["kernel_size"], params["kernel_size"]),
            n_blocks=params["n_blocks"],
            use_noise_level=False,
            bias_free=params["bias_free"],
            act="leaky_relu",
            enforce_positivity=params["enforce_positivity"],
        )
        model = DRUNet(model_params).cuda()
        config = {
            "n_blocks": params["n_blocks"],
            "int_channels": params["channels"],
            "kernel_size": params["kernel_size"],
            "bias_free": params["bias_free"],
            "enforce_positivity": params["enforce_positivity"],
            "lr": params["lr"],
            "epochs": params["epochs"],
            "noise_level": params["noise_level"],
            "commit_hash": commit_hash,
        }
    elif params["model"] == "gs_drunet":
        model_params = LIONParameter(
            in_channels=1,
            out_channels=1,
            int_channels=params["channels"],
            kernel_size=(params["kernel_size"], params["kernel_size"]),
            n_blocks=params["n_blocks"],
            use_noise_level=False,
            bias_free=params["bias_free"],
            act="elu",
            enforce_positivity=params["enforce_positivity"],
        )
        model = GSDRUNet(model_params).cuda()
        config = {
            "n_blocks": params["n_blocks"],
            "int_channels": params["channels"],
            "kernel_size": params["kernel_size"],
            "bias_free": params["bias_free"],
            "lr": params["lr"],
            "epochs": params["epochs"],
            "noise_level": params["noise_level"],
            "commit_hash": commit_hash,
        }
    else:
        raise NotImplementedError(f"Model {params['model']} has not been implemented!")

    experiment_id = uuid.uuid1()
    experiment_name = f"{params['model']}"
    if params["bias_free"]:
        experiment_name += "_bias_free"
    if "dncnn" in params["model"]:
        experiment_name += f"_depth={params['depth']}"
    else:
        experiment_name += f"_n_blocks={params['n_blocks']}"
    experiment_name += f"_noise_level={params['noise_level']}_{experiment_id}"
    print(experiment_name)
    print(config)
    wandb.init(project="benchmarking_ct", config=config, name=experiment_name)

    optimiser = torch.optim.Adam(model.parameters(), lr=params["lr"], betas=(0.9, 0.9))
    random_crop = RandomCrop((256, 256))
    random_erasing = RandomErasing()
    experiment = ct_benchmarking.GroundTruthCT()

    training_data = experiment.get_training_dataset()
    validation_data = experiment.get_validation_dataset()
    testing_data = experiment.get_testing_dataset()

    print(
        f"N_train={len(training_data)}, N_val={len(validation_data)}, N_test={len(testing_data)}"
    )

    batch_size = 1

    training_dataloader = torch.utils.data.DataLoader(
        training_data, batch_size, shuffle=True
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_data, batch_size, shuffle=False
    )
    testing_dataloader = torch.utils.data.DataLoader(
        testing_data, batch_size, shuffle=False
    )

    with open("normalisation.json", "r") as fp:
        normalisation = json.load(fp)
        x_min, x_max = normalisation["x_min"], normalisation["x_max"]

    best_val_psnr = -inf

    losses = []
    val_psnrs = []
    for epoch in range(params["epochs"]):
        model.train()
        for it, x in enumerate(training_dataloader):
            x = x.cuda()
            x = (x - x_min) / (x_max - x_min)
            patches = random_erasing(
                torch.cat([random_crop(x) for _ in range(5)], dim=0)
            )
            optimiser.zero_grad()
            y = patches + params["noise_level"] * torch.randn_like(patches)
            recon = model(y)
            loss = torch.mean((recon - patches) ** 2)
            loss.backward()
            grad_norm = mean_grad_norm(model)
            losses.append(loss.item())
            optimiser.step()
            with torch.no_grad():
                y_psnr = psnr(patches, y)
                recon_psnr = psnr(patches, recon)
            print(
                f"Epoch {epoch}, it {it}: PSNR(x, y) = {y_psnr.item():.1f} dB, PSNR(x, recon) = {recon_psnr:.1f} dB, loss = {loss.item():.2e}"
            )
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "train_psnr": recon_psnr.item(),
                    "train_psnr_y": y_psnr.item(),
                    "train_psnr_offset": (recon_psnr - y_psnr).item(),
                    "grad_norm": grad_norm,
                }
            )

        psnrs = []
        y_psnrs = []
        model.eval()
        for x in validation_dataloader:
            x = x.cuda()
            x = (x - x_min) / (x_max - x_min)
            y = x + params["noise_level"] * torch.randn_like(x)
            if params["model"] not in ["gs_drunet"]:
                with torch.no_grad():
                    recon = model(y)
            else:
                recon = model(y)
            with torch.no_grad():
                psnrs.append(psnr(x, recon).item())
                y_psnrs.append(psnr(x, y).item())
        psnrs = torch.tensor(psnrs)
        y_psnrs = torch.tensor(y_psnrs)
        print(
            f"Epoch {epoch}, val PSNR(x, y) = {y_psnrs.mean():.1f} +- {y_psnrs.std():.1f} dB, val PSNR(x, recon) = {psnrs.mean():.1f} +- {psnrs.std():.1f} dB"
        )
        print(
            f"Epoch {epoch}, val PSNRs: 5%-quantile {psnrs.quantile(0.05):.1f} dB, median {psnrs.quantile(0.5):.1f}, 95%-quantile {psnrs.quantile(0.95):.1f} dB"
        )
        wandb.log({"val_psnrs": psnrs})

        if psnrs.mean() > best_val_psnr:
            best_val_psnr = psnrs.mean().item()
            torch.save(
                {"config": config, "state_dict": model.state_dict()},
                os.path.join(params["results_path"], f"{experiment_name}.pt"),
            )
        wandb.log({"best_val_psnr": best_val_psnr})
        wandb.log({"val_psnr_y": y_psnrs.mean().item()})
        wandb.log({"val_psnr_offset": best_val_psnr - y_psnrs.mean().item()})
