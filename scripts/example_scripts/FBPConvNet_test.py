import torch
from torch.utils.data import DataLoader
import pathlib
from LION.models.post_processing.FBPConvNet import FBPConvNet
import matplotlib.pyplot as plt
import LION.experiments.ct_experiments as ct_experiments


#%% First run FBPConvNet_train.py to train and save model, then run this.
# % Set device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Give paths to trained models
savefolder = pathlib.Path("/store/DAMTP/cs2186/trained_models/clinical_dose/")
final_result_fname = savefolder.joinpath("FBPConvNet_final_iter.pt")

# set up experiment model was trained on
# the same experiment should be used, results cannot be guaranteed otherwise
experiment = ct_experiments.clinicalCTRecon()
test_data = experiment.get_testing_dataset()
test_dataloader = DataLoader(test_data, 1, shuffle=True)

# load trained model
model, _, _ = FBPConvNet.load(final_result_fname)
model.to(device)

# sample a random batch (size 1, so really just one image, truth pair)
data, gt = next(iter(test_dataloader))
x = model(data)

# put stuff back on the cpu, otherwise matplotlib throws an error
x = x.detach().cpu().numpy()
gt = gt.detach().cpu().numpy()

plt.figure()
plt.subplot(121)
plt.imshow(x[0].T)
plt.colorbar()
plt.subplot(122)
plt.imshow(gt[0].T)
plt.colorbar()
plt.savefig("img.png")