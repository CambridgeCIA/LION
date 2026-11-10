# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Extending LION
==============

Each piece of LION is a small base class over plain torch, and each can be
your own: an operator, a model, a dataset, a reconstructor, a solver. Written
as LION's are, yours get what LION's have: records, saving and loading,
evaluation, publishing.

This tutorial builds a problem LION doesn't have, deblurring: a blur operator,
a dataset of blurred images, an unrolled model for it, and a reconstructor,
then trains and uses them with LION's solver. To add them to LION itself, see
the contributor guide.

Previous: :doc:`d12_publishing`.

Needs a GPU and the processed LIDC-IDRI data in ``LION_DATA_PATH`` (the images
to blur).

.. note::

   The models trained in these tutorials are toys: a few epochs on a few
   slices, to show how LION is used, so their results are poor (often worse
   than FDK). Trained as their papers train them (many epochs, all the data),
   they do what the papers report.
"""

# %%
# An operator
# -----------
# An operator derives from ``Operator`` and defines ``forward`` (and
# ``__call__``, which calls it), ``adjoint``, its ``domain_shape`` and
# ``range_shape``, and, if it can be differentiated, ``autograd()``; if it has
# a usual reconstruction, ``default_reconstruction_fn()``. It is made by its
# parameters, an ``OperatorParameters``, which is what models, solvers and
# records take (as they take a CT geometry).

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from LION.operators.Operator import Operator
from LION.utils.parameter import LIONParameter, OperatorParameters


def blurred_image(y, op):
    """The blur's default reconstruction: the blurred image as it is."""
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
        y = F.conv2d(x, self.kernel.to(x), padding=self.parameters.kernel_size // 2)
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
        # only torch operations: torch differentiates it as it is
        return self, self.adjoint

    def default_reconstruction_fn(self):
        return blurred_image


class BlurParameters(OperatorParameters):
    """The blur's parameters, which make it."""

    def make_operator(self):
        return Blur(self)


blur = BlurParameters(kernel_size=9, sigma=2.0, image_shape=[1, 512, 512])

# %%
# A dataset
# ---------
# A dataset derives from ``LIONDataset`` (a torch ``Dataset``) and keeps its
# constructor's arguments as attributes of the same names, so its record can
# say how to make it again. Listing its samples (``sample_ids()``) and their
# groups (``sample_groups()``, e.g. patients, which must stay in one split)
# lets records and evaluations check what was trained and tested on. Its
# random numbers come from ``seeded(seed, sample)``: each sample is the same
# every time it is read.

from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
from LION.data_loaders.LIONDataset import LIONDataset
from LION.utils.rng import seeded


class BlurredLIDC(LIONDataset):
    """LIDC-IDRI slices blurred by the blur of ``operator_params``, with
    Gaussian noise of standard deviation ``parameters.noise``: (blurred image,
    image) pairs. ``mode``: "train", "validation" or "test"."""

    def __init__(self, mode, operator_params, parameters=None):
        self.mode = mode
        self.operator_params = operator_params
        self.parameters = (
            parameters if parameters is not None else self.default_parameters()
        )
        self.images = LIDC_IDRI(mode)
        self.op = operator_params.make_operator()

    @staticmethod
    def default_parameters():
        return LIONParameter(noise=0.01, seed=0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index][1]
        with seeded(self.parameters.seed, self.sample_ids()[index]):
            noise = torch.randn(image.shape)
        return self.op(image) + self.parameters.noise * noise.to(image), image

    def sample_ids(self):
        return self.images.sample_ids()

    def sample_groups(self):
        return self.images.sample_groups()


training = torch.utils.data.Subset(BlurredLIDC("train", blur), range(40))
validation = torch.utils.data.Subset(BlurredLIDC("validation", blur), range(8))
training_loader = DataLoader(training, batch_size=4, shuffle=True)
validation_loader = DataLoader(validation, batch_size=4)

# %%
# A model
# -------
# A model derives from ``LIONmodel``. One that uses an operator takes its
# parameters first and its own second, and makes the operator (with
# ``autograd=True`` to differentiate through it: ``self.A``, ``self.AT``).
# ``default_parameters()`` lists every setting, with ``model_input_type``:
# what it takes. A model from a paper defines ``cite()``.
#
# Here, learned gradient descent: K steps x ← x - CNN(x, Aᵀ(Ax - y)), from
# the operator's default reconstruction.

from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType


class LearnedGradientDescent(LIONmodel):
    """``n_steps`` learned gradient steps, each a small CNN of the image and
    the data fit's gradient."""

    def __init__(self, operator_params, model_parameters=None):
        super().__init__(model_parameters, operator_params)
        self._make_operator(autograd=True)
        channels = self.model_parameters.channels
        self.steps = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Conv2d(2, channels, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(channels, 1, 3, padding=1),
            )
            for _ in range(self.model_parameters.n_steps)
        )

    @staticmethod
    def default_parameters():
        parameters = LIONModelParameter()
        parameters.model_input_type = ModelInputType.SINOGRAM  # the measurements
        parameters.n_steps = 5
        parameters.channels = 32
        return parameters

    def forward(self, y):
        x = self.initial_reconstruction(y, self.op)
        for step in self.steps:
            gradient = self.AT(self.A(x) - y)
            x = x - step(torch.cat([x, gradient], dim=1))
        return x


model = LearnedGradientDescent(blur)
print(model.model_parameters)

# %%
# Training it
# -----------
# LION's solver trains it as it trains LION's models: the operator's
# parameters come from the model.

from LION.metrics.psnr import PSNR
from LION.optimizers.SupervisedSolver import SupervisedSolver

folder = Path(tempfile.mkdtemp())
solver = SupervisedSolver(
    model, torch.optim.Adam(model.parameters()), torch.nn.MSELoss()
)
solver.set_training(training_loader)
solver.set_validation(validation_loader, 1, PSNR(), validation_fname="best")
solver.set_saving(folder, "LGD")
solver.train(3)
solver.save_final_results()

# %%
# Its file records the model, the blur and the data, as for LION's. Files only
# make LION's classes again unless told otherwise: classes of your own are
# allowed by their module's name (here, this script's: ``__main__``).

model, options, _ = LIONmodel.load(folder / "LGD", allow=["__main__."])
print(options.model_name, type(options.operator_params).__name__)

# %%
# A reconstructor
# ---------------
# A reconstructor derives from ``LIONReconstructor`` and defines
# ``default_parameters()`` and ``reconstruct_sample(y, x0, verbose)``, reading
# its settings from ``self.parameters``. Here: the model's image, made
# consistent with the data by a few conjugate gradient steps on
# ||Ax - y||² + λ||x - model(y)||².

from LION.classical_algorithms.conjugate_gradient import conjugate_gradient
from LION.reconstructors.LIONreconstructor import LIONReconstructor


class DataConsistent(LIONReconstructor):
    """The model's reconstruction, made consistent with the measurements."""

    @staticmethod
    def default_parameters():
        return LIONParameter(lam=0.1, cg_iterations=20)

    def reconstruct_sample(self, y, x0=None, verbose=0):
        lam = self.parameters.lam
        with torch.no_grad():
            x_model = self.model(y[None])[0]
            return conjugate_gradient(
                lambda x: self.op.adjoint(self.op(x)) + lam * x,
                self.op.adjoint(y) + lam * x_model,
                x_model,
                self.parameters.cg_iterations,
                1e-6,
            )


reconstructor = DataConsistent(blur, model)
blurred, image = validation[0]
deblurred = reconstructor.reconstruct(blurred)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (title, x) in zip(
    axes, [("target", image), ("blurred", blurred), ("reconstructed", deblurred)]
):
    ax.imshow(x[0].cpu(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")

# %%
# A solver
# --------
# A solver derives from ``LIONsolver`` and defines ``mini_batch_step(y,
# target)``, the loss of one mini-batch (adding ``self._regularized(...)``
# for a model regularisation), and ``default_parameters()``, its own settings.
# E.g. a supervised solver with an L1 and a gradient-matching loss:
#
# .. code-block:: python
#
#    class EdgeSolver(LIONsolver):
#        @staticmethod
#        def default_parameters():
#            return LIONParameter(edge_weight=0.1)
#
#        def mini_batch_step(self, y, target):
#            output = self._forward(y)
#            edges = lambda x: x[..., 1:, :] - x[..., :-1, :]
#            loss = self.loss_fn(output, target) + self.solver_params.edge_weight * (
#                edges(output) - edges(target)
#            ).abs().mean()
#            return self._regularized(loss, y, output, target, self._forward)
#
# The contributor guide has the rules, and a checklist for each piece, to add
# it to LION itself.
