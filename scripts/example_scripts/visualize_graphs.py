from LION.CTtools.ct_utils import make_operator
from LION.classical_algorithms.fdk import fdk
from LION.models.CNNs.MSDNets.MSDNet import MSD_Params, MSDNet
from LION.experiments.ct_experiments import clinicalCTRecon
from torchviz import make_dot
from torch.utils.data import DataLoader

experiment = clinicalCTRecon()
op = make_operator(experiment.geo)
dataset = experiment.get_training_dataset()
dataloader = DataLoader(dataset, 1, True)

sino, gt = next(iter(dataloader))
sino = sino.to("cuda:2")
gt = gt.to("cuda:2")
recon = fdk(sino, op)
recon = recon.to("cuda:2")

model = MSDNet(MSD_Params(1, 1, 2, [1, 2])).to("cuda:2")
yhat = model(recon)
make_dot(yhat, params=dict(list(model.named_parameters()))).render(
    "msdpytorch_graph", format="png"
)
