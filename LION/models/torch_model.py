# This file is part of LION library
# License : GPL-3
# =============================================================================

import inspect
from typing import Optional

import torch

from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType
from LION.classical_algorithms.fdk import fdk
from LION.utils.parameter import LIONParameter
from LION.utils.spec import import_path

#: How torch denoisers take their noise level, by the import path of their
#: class: "channel" (an extra input channel, constant at the noise level) or
#: the name of the keyword argument it goes in. Add yours here (or give
#: TorchModel's ``noise_level``) when it can't be told from the class's
#: ``forward``, e.g. for a denoiser that takes it as a channel.
NOISE_LEVEL = {}

#: Keyword arguments recognised as the noise level in a class's ``forward``.
NOISE_LEVEL_KEYWORDS = ("noise_level", "sigma")


def noise_level_input(module_class) -> Optional[str]:
    """How instances of ``module_class`` take the noise level: from
    ``NOISE_LEVEL``, else a keyword of ``NOISE_LEVEL_KEYWORDS`` that its
    ``forward`` has; None if neither (it takes none, as far as LION knows)."""
    path = import_path(module_class)
    if path in NOISE_LEVEL:
        return NOISE_LEVEL[path]
    parameters = inspect.signature(module_class.forward).parameters
    for name in NOISE_LEVEL_KEYWORDS:
        if name in parameters:
            return name
    return None


class TorchModel(LIONmodel):
    """A plain torch module as a LIONmodel: ``module_class(**arguments)``.

    The class and its arguments are recorded, so the model can be saved, loaded
    and made again like any LIONmodel, and used where LION needs one (e.g. as
    GSD's denoiser)::

        model = TorchModel(MyNet, {"channels": 32})
        model.save("my_net")
        model = TorchModel.load("my_net", allow=["my_package."])

    ``module_class`` must be importable where the model is loaded: a class
    from ``torch.nn`` or LION loads as it is; one from your own module needs
    ``allow=["your_module."]`` when loading (one defined in the script that
    loads it: ``allow=["__main__."]``). ``arguments`` must be values a spec can
    hold (numbers, strings, lists, dicts, LION parameters, ...), not tensors.

    ``model_parameters``: LION's own (``model_input_type``, IMAGE by default).

    ``noise_level``: how the module takes the noise level, for a denoiser:
    "channel" (an extra input channel, constant at the noise level) or the name
    of its keyword argument (e.g. "sigma"). None: as ``noise_level_input``
    finds (``NOISE_LEVEL``, then the module's ``forward``). The model is called
    as every LION model taking a noise level is, ``model(x, noise_level=...)``,
    and gives it to the module that way. Other arguments go to the module as
    they are.
    """

    def __init__(
        self,
        module_class,
        arguments: Optional[dict] = None,
        model_parameters: LIONParameter = None,
        noise_level: Optional[str] = None,
    ):
        super().__init__(model_parameters)
        self.module_class = module_class
        self.arguments = dict(arguments or {})
        self.noise_level = noise_level
        self.module = module_class(**self.arguments)
        self._noise_level_input = (
            noise_level if noise_level is not None else noise_level_input(module_class)
        )

    @staticmethod
    def default_parameters():
        params = LIONModelParameter()
        params.model_input_type = ModelInputType.IMAGE
        # how its input is made from measurements (set_initial_reconstruction)
        params.initial_reconstruction = fdk
        return params

    def takes_noise_level(self) -> bool:
        return self._noise_level_input is not None

    def forward(self, x, *args, noise_level=None, **kwargs):
        how = self._noise_level_input
        if noise_level is None or how is None:
            return self.module(x, *args, **kwargs)
        if how == "channel":
            level = torch.ones_like(x[:, :1]) * noise_level
            return self.module(torch.cat((x, level), dim=1), *args, **kwargs)
        return self.module(x, *args, **{how: noise_level}, **kwargs)
