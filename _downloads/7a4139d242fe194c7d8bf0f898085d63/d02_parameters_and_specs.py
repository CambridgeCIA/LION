# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Parameters and specs
====================

Every piece of LION is set by parameters: the geometry, a model's layers, a
solver's settings, an experiment's noise. They are ``LIONParameter``\\ s, plain
namespaces you read, change and save. And every object LION records (an
optimizer, a loss, a model) is saved as a **spec**: its class and the
arguments that build it again. Together they are how LION knows, from a file
alone, how anything was made.

Previous: :doc:`d01_operators`. Next: :doc:`d03_data`.
"""

# %%
# LIONParameter
# -------------
# A namespace of settings: set any attribute, nest them, print them.

import tempfile
from pathlib import Path

from LION.utils.parameter import LIONParameter

parameters = LIONParameter(learning_rate=1e-3, epochs=10)
parameters.optimizer = LIONParameter(name="Adam", betas=[0.9, 0.99])
parameters.epochs = 20
print(parameters)

# %%
# They are saved as JSON, and read back:

folder = Path(tempfile.mkdtemp())
parameters.save(folder / "parameters.json")
print((folder / "parameters.json").read_text())

again = LIONParameter()
again.load(folder / "parameters.json")
print(again.optimizer.betas)

# %%
# default_parameters()
# --------------------
# Every class with settings has ``default_parameters()``: every setting it
# has, with its default (for a method from a paper, the paper's). Change what
# you need and give them back.

from LION.models.iterative_unrolled.LPD import LPD
from LION.optimizers.Noise2InverseSolver import Noise2InverseSolver
from LION.reconstructors.PnP import PnP

print(LPD.default_parameters())
print(Noise2InverseSolver.default_parameters())
print(PnP.default_parameters("ADMM"))

# %%
# A model is made from its operator's parameters (for CT, the geometry) and its
# own; it keeps them, so the model is always what its parameters say.

from LION.operators.ct import Geometry

lpd_parameters = LPD.default_parameters()
lpd_parameters.n_iters = 5
model = LPD(Geometry.default_parameters(), lpd_parameters)
print(model.model_parameters.n_iters, model.operator_params.dso)

# %%
# The geometry is a ``LIONParameter`` too (an ``OperatorParameters``: one
# that makes an operator), so it saves and loads the same way.

geometry = Geometry.default_parameters()
geometry.save(folder / "geometry.json")
loaded = Geometry()
loaded.load(folder / "geometry.json")
print(loaded.dsd, loaded.image_shape)

# %%
# Specs
# -----
# What LION saves of an object is its spec: a dict with the class
# (``_target_``) and the arguments that build it. ``to_spec`` makes one,
# ``build`` makes the object again.

import torch

from LION.utils import spec

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
print(spec.to_spec(optimizer))  # its hyperparameters
print(spec.to_spec(torch.nn.L1Loss(reduction="sum")))
print(spec.to_spec(geometry)["dso"])

# %%
# An optimizer is built again for a model's parameters, given when building:

new_optimizer = spec.build(spec.to_spec(optimizer), params=model.parameters())
print(new_optimizer.defaults["lr"])

# %%
# A model's spec is its class and parameters: building it gives the same
# architecture with new random weights (the weights are saved separately).

model_spec = spec.to_spec(model)
print(model_spec["_target_"])
same_architecture = spec.build(model_spec)
print(type(same_architecture).__name__, same_architecture.model_parameters.n_iters)

# %%
# Your own classes
# ----------------
# LION reads an object's arguments from its attributes of the same names (the
# "same-name rule"): keep every constructor argument as an attribute and your
# class is saved like LION's. (If it can't, define ``spec_kwargs()``, which
# returns them.)


class WeightedL1(torch.nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight  # same name as the argument

    def forward(self, output, target):
        return self.weight * (output - target).abs().mean()


loss_spec = spec.to_spec(WeightedL1(weight=0.5))
print(loss_spec)

# %%
# Building only imports LION, torch, numpy and a few builtins: a file can't run
# other code. Your own modules are allowed by name:

try:
    spec.build(loss_spec)
except Exception as error:
    print(type(error).__name__, ":", str(error)[:80], "...")
loss = spec.build(loss_spec, allow=["__main__."])
print(loss.weight)

# %%
# Specs are what LION's records are made of: the data a model saw, how it was
# trained, how it was tested. :doc:`d09_saving_and_records` reads
# them.
