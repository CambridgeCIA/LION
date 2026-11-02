# This file is part of LION library
# License : GPL-3
# =============================================================================

from typing import Optional

from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType
from LION.utils.parameter import LIONParameter


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
    ``forward`` passes everything to the module.
    """

    def __init__(
        self,
        module_class,
        arguments: Optional[dict] = None,
        model_parameters: LIONParameter = None,
    ):
        super().__init__(model_parameters)
        self.module_class = module_class
        self.arguments = dict(arguments or {})
        self.module = module_class(**self.arguments)

    @staticmethod
    def default_parameters():
        params = LIONModelParameter()
        params.model_input_type = ModelInputType.IMAGE
        return params

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
