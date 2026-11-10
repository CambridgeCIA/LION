# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Your own pieces: a plain torch model, and a forward operator that isn't CT.

LION's pieces are plain torch underneath, and each can be your own:

- **a model:** any ``torch.nn.Module`` becomes a LION model with
  ``TorchModel(YourClass, arguments)``: it is then saved, loaded and recorded
  like LION's. (For a model that uses an operator, or to add it to LION,
  derive from ``LIONmodel``: see ``demos/d13_extending_lion.py``.)
- **an operator:** an ``Operator`` (forward, adjoint, shapes) and the
  parameters that make it, an ``OperatorParameters`` (its ``make_operator``).
  Solvers, models and reconstructors take the parameters, as they take a CT
  geometry. Here: a Gaussian blur, so the problem is deblurring.
- **data:** any torch ``Dataset`` of (measurements, target) pairs. Only LION's
  datasets (``LIONDataset``) are recorded with the model; a plain one is
  trained on with a warning that it isn't.

The images are the LIDC-IDRI slices of LION's experiments, blurred here.

Needs: a GPU (the experiment's dataset simulates its sinograms, which are
not used here), the processed LIDC-IDRI data in ``LION_DATA_PATH``, and
``LION_RESULTS_PATH``.
"""

# %% Imports
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from LION.experiments import ct_experiments
from LION.metrics.psnr import PSNR
from LION.models.torch_model import TorchModel
from LION.operators.Operator import Operator
from LION.optimizers.SupervisedSolver import SupervisedSolver
from LION.reconstructors.ForwardReconstructor import ForwardReconstructor
from LION.utils.parameter import OperatorParameters
from LION.utils.paths import results_path
from LION.utils.device import default_device

# %% Settings
EPOCHS = 5
TRAINING_SLICES = 200
NOISE = 0.01  # Gaussian noise added to the blurred images
# The GPU to use: LION's models, data and operators all follow it
GPU = 0
if torch.cuda.is_available():
    torch.cuda.set_device(GPU)
# where this script moves samples to use them: LION's device, that GPU
device = default_device()
verbose = 2  # 0: nothing; 1: one progress bar; 2: also each epoch's bars and messages

save_folder = results_path("examples/your_own_pieces")
save_folder.mkdir(parents=True, exist_ok=True)


# %% An operator: a Gaussian blur
def blurred_image(y, op):
    """The blur's default reconstruction: the blurred image as it is (what an
    image-to-image model starts from)."""
    return y


class Blur(Operator):
    """A Gaussian blur of images (1, H, W), by convolution with zero padding:
    the kernel is symmetric, so the blur is its own adjoint."""

    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        size, sigma = parameters.kernel_size, parameters.sigma
        t = torch.arange(size) - size // 2
        g = torch.exp(-(t**2) / (2 * sigma**2))
        self.kernel = (g[:, None] * g[None, :] / g.sum() ** 2)[None, None]

    def __call__(self, x, out=None):
        return self.forward(x)

    def forward(self, x):
        batched = x.dim() == 4  # (B, 1, H, W), or one image (1, H, W)
        x = x if batched else x[None]
        kernel = self.kernel.to(x)
        y = F.conv2d(x, kernel, padding=self.parameters.kernel_size // 2)
        return y if batched else y[0]

    def adjoint(self, y):
        return self.forward(y)

    @property
    def domain_shape(self):
        return tuple(self.parameters.image_shape)

    @property
    def range_shape(self):
        return tuple(self.parameters.image_shape)

    def autograd(self):
        # Only torch operations: torch differentiates it as it is
        return self, self.adjoint

    def default_reconstruction_fn(self):
        return blurred_image


class BlurParameters(OperatorParameters):
    """The blur's parameters, which make it (and are recorded)."""

    def make_operator(self):
        return Blur(self)


# %% Data: the experiment's images, blurred
class Deblurring(Dataset):
    """(blurred noisy image, image) pairs, from a dataset of (anything, image);
    each image's noise is the same every epoch."""

    def __init__(self, images, operator):
        self.images, self.operator = images, operator

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = self.images[i][1]
        noise = torch.randn(image.shape, generator=torch.Generator().manual_seed(i))
        return self.operator(image) + NOISE * noise.to(image), image


experiment = ct_experiments.LowDoseCTRecon()
image_shape = experiment.operator_params.image_shape  # [1, 512, 512]
blur = BlurParameters(kernel_size=9, sigma=2.0, image_shape=list(image_shape))
slices = experiment.get_dataset("train")
n = len(slices) if TRAINING_SLICES is None else min(TRAINING_SLICES, len(slices))
training = Deblurring(torch.utils.data.Subset(slices, range(n)), blur.make_operator())
training_loader = DataLoader(training, batch_size=4, shuffle=True)


# %% A model: a plain torch module
class SmallResNet(torch.nn.Module):
    """A few convolutions that learn the residual."""

    def __init__(self, channels=32, layers=5):
        super().__init__()
        modules = [torch.nn.Conv2d(1, channels, 3, padding=1), torch.nn.ReLU()]
        for _ in range(layers - 2):
            modules += [torch.nn.Conv2d(channels, channels, 3, padding=1)]
            modules += [torch.nn.ReLU()]
        modules += [torch.nn.Conv2d(channels, 1, 3, padding=1)]
        self.layers = torch.nn.Sequential(*modules)

    def forward(self, x):
        return x + self.layers(x)


model = TorchModel(SmallResNet, {"channels": 32, "layers": 5})

# %% Train, with LION's solver and the blur's parameters
solver = SupervisedSolver(
    model,
    torch.optim.Adam(model.parameters(), lr=1e-3),
    torch.nn.MSELoss(),
    operator_params=blur,
    verbose=verbose,
)
# (Two warnings: the dataset isn't a LIONDataset, so it isn't recorded; and a
# class defined in this script can't be made again from the record, so the
# initial weights are saved instead of the random state they came from.)
solver.set_training(training_loader)
solver.set_saving(save_folder, "SmallResNet")
solver.train(EPOCHS)
solver.save_final_results()

# Loaded again: classes defined in this script are allowed by name ("__main__.")
model, _, _ = TorchModel.load(save_folder / "SmallResNet", allow=["__main__."])

# %% Deblur a test image
blurred, image = Deblurring(experiment.get_dataset("test"), blur.make_operator())[0]
# The reconstructor gives the model the blur's default reconstruction (the
# blurred image) as its solver did
model.eval()
deblurred = ForwardReconstructor(blur, model).reconstruct(blurred.to(device)).cpu()
# The same by hand: the model on the blurred image, without gradients
# with torch.no_grad():
#     deblurred = model(blurred.unsqueeze(0).to(device))[0].cpu()
psnr = PSNR()
print(
    f"PSNR: blurred {psnr(blurred.cpu(), image.cpu(), batched=False):.2f} dB, "
    f"deblurred {psnr(deblurred, image.cpu(), batched=False):.2f} dB"
)
# Plain torch, if you prefer: the model is a torch.nn.Module, and the loop
#   for y, x in training_loader:
#       loss = torch.nn.functional.mse_loss(model(y), x) ...
# is yours to write. LION's solver adds the records, checkpoints and validation.
