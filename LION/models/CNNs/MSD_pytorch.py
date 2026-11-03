# This file is part of LION library
# License : GPL-3
#
# Author  : See ./MS-D. Daniel Pelt et al
# Modifications: Ander Biguri
# =============================================================================


from LION.models import LIONmodel
from LION.models.LIONmodel import LIONModelParameter, ModelInputType
from LION.utils.parameter import LIONParameter
from LION.utils.citation import citation


def _import_msd_pytorch():
    """msd_pytorch is not a LION dependency (it needs its own CUDA build), so
    it is imported only when this model is made."""
    try:
        import msd_pytorch
    except ImportError as error:
        raise ImportError(
            "MSD_pytorch needs `msd_pytorch` (https://github.com/ahendriksen/"
            "msd_pytorch), which is not installed. LION's own MSDNet "
            "(LION.models.CNNs.MSDNet) needs nothing extra."
        ) from error
    return msd_pytorch


class MSD_pytorch(LIONmodel.LIONmodel):
    def __init__(self, model_parameters=None):

        super().__init__(model_parameters)
        msd = _import_msd_pytorch()

        if self.model_parameters.type == "regression":
            model = msd.MSDRegressionModel(
                self.model_parameters.c_in,
                self.model_parameters.c_out,
                self.model_parameters.depth,
                self.model_parameters.width,
                dilations=self.model_parameters.dilations,
                loss="L2",
            )

        elif self.model_parameters.type == "segmentation":
            if self.model_parameters.num_labels is None:
                raise ValueError(
                    "For a segmentation network, please set the model_parameters.num_labels vairable to the number of labels in the training set"
                )
            model = msd.MSDSegmentationModel(
                self.model_parameters.c_in,
                self.model_parameters.num_labels,
                self.model_parameters.depth,
                self.model_parameters.width,
                dilations=self.model_parameters.dilations,
            )
        # We don't want MS-D to define our optimizer.
        model.optimizer = None

        self.net = model.net

    @staticmethod
    def default_parameters(mode="regression"):
        param = LIONModelParameter()
        param.model_input_type = ModelInputType.IMAGE
        if mode == "regression":
            param.type = "regression"
            param.c_in = 1
            param.c_out = 1
            param.depth = 100
            param.width = 1
            param.dilations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        elif mode == "segmentation":
            param.type = "segmentation"
            param.c_in = 1
            param.num_labels = None
            param.depth = 100
            param.width = 1
            param.dilations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return param

    @staticmethod
    def cite(cite_format="bib"):
        """Where MSD_pytorch comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{pelt2018mixed,
            title={A mixed-scale dense convolutional neural network for image analysis},
            author={Pelt, Dani{\\"e}l M and Sethian, James A},
            journal={Proceedings of the National Academy of Sciences},
            volume={115},
            number={2},
            pages={254--259},
            year={2018},
            publisher={National Acad Sciences}
            }
            """,
            mla="""
            Pelt, Daniël M., and James A. Sethian.
            "A mixed-scale dense convolutional neural network for image analysis."
            Proceedings of the National Academy of Sciences  
            115.2 (2018): 254-259.
            """,
        )

    def forward(self, x):
        return self.net(x)
