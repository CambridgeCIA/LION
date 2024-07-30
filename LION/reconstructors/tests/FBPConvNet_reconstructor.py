import torch
from torch.utils.data import DataLoader
import pathlib
from LION.models.post_processing.FBPConvNet_subclassed import FBPConvNet
import LION.models.LIONmodelSubclasses as LIONmodelSubclasses
import matplotlib.pyplot as plt
import LION.experiments.ct_experiments as ct_experiments

from LION.reconstructors.reconstructor import LIONreconstructor

from torch.utils.data import Subset

#%% First run FBPConvNet_train.py to train and save model, then run this.
# % Set device:
device = torch.device("cuda:1")
torch.cuda.set_device(device)
# Give paths to trained models
savefolder = pathlib.Path("/home/hyt35/trained_models/clinical_dose_subclassed/")
final_result_fname = savefolder.joinpath("FBPConvNet_final_iter.pt")

# set up experiment model was trained on
# the same experiment should be used, results cannot be guaranteed otherwise
experiment = ct_experiments.clinicalCTRecon()


# load trained model
model, _, _ = FBPConvNet.load(final_result_fname)  # loads statedict only
print(model.__class__)
model = LIONmodelSubclasses.Constructor(model)
model.to(device)

# reconstruct
dataset = experiment.get_testing_dataset()
reconstructor = LIONreconstructor(model, dataset, reduction="none")
metrics = reconstructor(batch_size=2, subset_size=50)
print(metrics)
