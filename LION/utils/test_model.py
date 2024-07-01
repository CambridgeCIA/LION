# % A small set of methods for testing trained models on unseen data

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from LION.models.LIONmodel import LIONmodel
import LION.experiments.ct_experiments as ct_experiments
from typing import Type


def test_with_img_output(
    experiment: ct_experiments.Experiment,
    model_type: Type[LIONmodel],
    model_fpath: str,
    img_fpath: str,
    n=1,
    device: str = "cpu",
):
    """Test model trained and saved at model_fpath on n pieces of test data. Outputs predictions and ground truths as images at img_fpath{n}

    Args:
        experiment (ct_experiments.Experiment): Expriment model was trained on. Must be same, otherwise results cannot be guaranteed.
        model_type (LIONmodel): Type of LIONmodel to be tested
        model_fpath (str): Filepath of trained model
        img_fpath (str): Base filepath to store resultant images at
        n (int): Number of test data to test
        device: Device to run forward pass of model on. Default=cpu

    Raises:
        ValueError: If model_type is not a valid subclass of LIONmodel
    """
    preds, gts = test_with_experiment(experiment, model_type, model_fpath, n, device)

    # put stuff on the cpu, otherwise matplotlib throws an error
    preds = preds.detach().cpu().numpy()
    gts = gts.detach().cpu().numpy()

    # split path into actual path and file extension
    split_path = img_fpath.rsplit('.', 1)
    for i in range(len(preds)):
        plt.figure()
        plt.subplot(121)
        plt.imshow(preds[i].T)
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(gts[i].T)
        plt.colorbar()
        # reconstruct filepath with suffix i
        plt.savefig(f'{split_path[0]}{i+1}.{split_path[1]}')


def test_with_experiment(
    experiment: ct_experiments.Experiment,
    model_type: Type[LIONmodel],
    model_fpath: str,
    n: int = 1,
    device: str = "cpu",
):
    """Test model trained and saved at model_fpath. Returns prediction and ground truth data

    Args:
        experiment (ct_experiments.Experiment): Expriment model was trained on. Must be same, otherwise results cannot be guaranteed.
        model_type (LIONmodel): Type of LIONmodel to be tested
        model_fpath (str): Filepath of trained model
        n (int): Number of test data to test
        device: Device to run forward pass of model on. Default=cpu

    Raises:
        ValueError: If model_type is not a valid subclass of LIONmodel
    """
    # Set device:
    dev = torch.device(device)
    torch.cuda.set_device(dev)

    # Load test data
    test_data = experiment.get_testing_dataset()
    test_dataloader = DataLoader(test_data, n, shuffle=True)

    # Load trained model
    if not issubclass(model_type, LIONmodel):
        raise ValueError("model must inherit from LIONmodel: ")
    model, _, _ = model_type.load(model_fpath)
    model = model.to(dev)

    # Sample a random batch (size n)
    data, gt = next(iter(test_dataloader))
    data = data.to(dev)
    # run model on data
    with torch.autocast(device_type=dev.type):
        pred = model(data)

    return pred, gt
