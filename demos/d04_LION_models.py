# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================

import numpy as np
import os

# LION imports
import LION.CTtools.ct_geometry as ctgeo
import LION.CTtools.ct_geometry as ct


#%%
# LION comes with several ML models implemented for you.
# LION models are basically a specific class of torch models that contain many helper functions to make your life easier.
# This demo shows how the models behave, and how to make a new LIONmodel with your torch models.

# NOTE: This demo does not fully execute.
# %% 1- Example of LION models implemented in LION
# e.g. the Learned Primal Dual model
from LION.models.iterative_unrolled.LPD import LPD
from LION.utils.parameter import LIONParameter

# Get a geometry
geo = ctgeo.Geometry.default_parameters()

# All LIONmodels will have as optional input "model_parameters", these are parameters that define the model. If not give, the default (i.e. the papers) parameters will be used.
# Some models also use the CT operators, and for those ones, like the LPD, the geometry of the operator has to be input.
model_lpd = LPD(geometry_parameters=geo, model_parameters=None)

# There are several models curently in LION, but lets not exhaustively showcase them here.
# This is another example of a model, one that does not need geometry as input:
from LION.models.CNNs.MS_D import MS_D

model_msd = MS_D()

# %% 2- What is a LIONmodel
# A LIONmodel is just an abstract class for a torch.nn.Module.
# This means, its just your standard torch.nn.Module that you are used to, but it comes with extra things on top,
# that will help you write and use these models in a more reliable way. So, think of them as a torch.nn.Module, but easier to use!

#%% 3-Methods of a LIONmodel
# Lets showcase what methods a class that derives from a LIONmodel will always have (and of course, you can add more things to the model)
# For that, lets use LPD that we defined above, to showcase.

# Static methods:
# --------------
# These are methods that do not require an instance (model_lpd) to work. A LIONmodel should always have:

# - default_parameters() Gives you the original papers parameters.
model_params = LPD.default_parameters()

# - cite(format)  Provides the MLA or bibtex reference to the paper originating this model
LPD.cite("MLA")
LPD.cite("bib")

# - load(filename)  Loads a saved model from disk (see save() function later on)
# You generally only need to capture the first output, the other two are optional.
model, options, data = LPD.load("some_file.pt")

# - load_checkpoint(filename). Loads a saved checkoint. Currently very similar to load()
# You generally only need to capture the first output, the other two are optional.
model, options, data = LPD.load_checkpoint("some_file.pt")

# - final_file_exists(filename,stop_code=False). Tells you if the file exists, and optionally terminates the code
does_it_exist = LPD.final_file_exists("some_file.pt")
# Use it as this to terminate the code if its final model exists.
LPD.final_file_exists("some_file.pt", stop_code=True)

# - load_checkpoint_if_exists(pattern, model, optimiser, total_loss,). Loads checkpoint if it does exist, otherwise it doesn't. Returns either loaded, or untouched values
# The optimizer is the return value of a torch.optim call.
# total_loss is just the np.array to track the loss
model, optimiser, start_epoch, total_loss, data = LPD.load_checkpoint_if_exists(
    "LPD_checkpoint_*.pt", optimiser, total_loss
)
# This function is quite useful when you have checkpointed code, as putting it just before the training loop will ensure you always load the latests checkpoint.
# Make sure to use "start_epoch" in your training loop for the range.

# Instance methods:
# --------------
# These are methods that do require an instance (model_lpd) to work.

# - save(filename) Saves a LIONmodel to memory.
# Save has optinal keyword arguments that are heavily suggested as input, to improve reproducibility. These are loaded on "options" in load()
model_lpd.save("some_file.pt", dataset=dataset_params, training=training_params)
# if your model does not containg geometry info in itself, but you are using it for CT, please also add that:
model_msd.save(
    "some_file.pt",
    dataset=dataset_params,
    training=training_params,
    geometry=geo_params,
)

# - save_checkpoint(filename, epoch, loss, optimizer, training_param,**kwargs). This fucntion is a wrapper of save(), but with mandatory parameters
model_lpd.save_checkpoint("some_file.py", epoch, loss, optimiser, training_param)
# and with full optional parameters:
model_msd.save_checkpoint(
    "some_file.py",
    epoch,
    loss,
    optimiser,
    training_param,
    dataset=dataset_params,
    geometry=geo_params,
)

# - get_parameter(). Returns parameter objects
model_param, geo = model_lpd.get_parameter()

# ---------------------------------------------------------------
#%% 4- Developers methods
# This functions are not designed to be used by the user, please only use them if you really need to

# Static methods:
# --------------

# - _load_data() Loads data, essentially a fancy wrap around torch.load()
data = LPD._load_data("some_file.pt")

# - _load_parameter_file() Loads parameter file that contains the data for the geometry and model, so it can be initialized appropiately
param = LPD._load_parameter_file("some_parameter.json")

# - _current_file()  Returns filename of current model. A class method, not static method
filename = LPD._current_file()

# Instance methods:
# -----------------

# - _make_operator(). uses self.geo to make a pytorch compatible operator and saves it in self.A and self.AT
# ONLY USE INSIDE A CLASS! __init__ functions of classes, like LPD, define it arleady. Calling it again will replace attributes of the model.
model_lpd._make_operator()
# now
model_lpd.A(image)
# and
model_lpd.AT(sinogram)
# exist, and pytorch will treat them as adjoint for autograd purposes.

# %% 5- Making your own LIONmodel
# Lets make a dummy LIONmodel to show

from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType
import torch


class DummyModel(LIONmodel):
    def __init__(
        self,
        geometry_parameters: ctgeo.Geometry,
        model_parameters: LIONModelParameter = None,
    ):

        # If the model prameters are not given, the parent class will initialize them by calling default_parameters()
        # and it will put the parameters in self.model_parameters

        # If a geometry is given, then self.geometry is set, and allows you to use the operator and autograd it.
        super().__init__(model_parameters, geometry_parameters)  # initialize LIONmodel

        # you may want to e.g. have pytorch compatible operators
        self._make_operator()  # done!
        # Now self.A and self.AT are available for use, and they will be autogradable

        self.layer = torch.nn.Conv2d(
            self.model_parameters.channel_in,
            self.model_parameters.channel_out,
            3,
            padding=1,
            bias=False,
        )

    # You must define this method
    @staticmethod
    def default_parameters():
        param = LIONModelParameter()
        # LIONModelParameters have this. It helps the training code know that it needs to do a recon (or not) to train the model.
        # it can also be ModelInputType.IMAGE
        param.model_input_type = ModelInputType.SINOGRAM
        param.channel_in = 1
        param.channel_out = 1
        return param

    def forward(self, sino):
        sino_conv = self.layer(sino)
        out = self.AT(
            sino_conv  # pytorch will know how to autograd this, because of _make_operator()
        )
        return out
