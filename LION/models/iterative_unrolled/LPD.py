# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType

from LION.utils.math import power_method
from LION.utils.parameter import LIONParameter, OperatorParameters


import torch
import torch.nn as nn
from LION.utils.citation import citation


class dataProximal(nn.Module):
    """The dual (data) block of LPD: 3x3 convolutions with PReLUs through the
    ``channels`` listed, the last a 1x1 convolution; optional instance
    normalisation."""

    def __init__(self, layers, channels, conv_bias, instance_norm=False):

        super().__init__()
        # imput parsing
        if len(channels) != layers + 1:
            raise ValueError(
                "Second input (channels) should have as many elements as layers your network has"
            )
        if layers < 1:
            raise ValueError("At least one layer required")
        # convolutional layers
        layer_list = []
        for ii in range(layers):
            if instance_norm:
                layer_list.append(nn.InstanceNorm2d(channels[ii]))
            # PReLUs and 3x3 kernels all the way except the last
            if ii < layers - 1:
                layer_list.append(
                    nn.Conv2d(
                        channels[ii], channels[ii + 1], 3, padding=1, bias=conv_bias
                    )
                )
                layer_list.append(nn.PReLU())
            else:
                layer_list.append(
                    nn.Conv2d(
                        channels[ii], channels[ii + 1], 1, padding=0, bias=conv_bias
                    )
                )
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        """The block's update of the dual variables, from ``x`` (the dual, the
        projected primal and the measurements)."""
        return self.block(x)


class RegProximal(nn.Module):
    """The primal (regularisation) block of LPD: as ``dataProximal``, on the
    primal side."""

    def __init__(self, layers, channels, conv_bias, instance_norm=False):
        super().__init__()
        if len(channels) != layers + 1:
            raise ValueError(
                "Second input (channels) should have as many elements as layers your network has"
            )
        if layers < 1:
            raise ValueError("At least one layer required")

        layer_list = []
        for ii in range(layers):
            if instance_norm:
                layer_list.append(nn.InstanceNorm2d(channels[ii]))
            # PReLUs and 3x3 kernels all the way except the last
            if ii < layers - 1:
                layer_list.append(
                    nn.Conv2d(
                        channels[ii], channels[ii + 1], 3, padding=1, bias=conv_bias
                    )
                )
                layer_list.append(nn.PReLU())
            else:
                layer_list.append(
                    nn.Conv2d(
                        channels[ii], channels[ii + 1], 1, padding=0, bias=conv_bias
                    )
                )
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        """The block's update of the primal variables, from ``x`` (the primal
        and the backprojected dual)."""
        return self.block(x)


def _check_channels(params):
    """Number of primal/dual channels, from the output of the blocks.

    The dual block sees [h, A f, g] (n_dual + 2 channels) and the primal block
    [f, A^T h] (n_primal + 1 channels); both output an update of their variable.
    """
    n_dual, n_primal = params.data_channels[-1], params.reg_channels[-1]
    if params.data_channels[0] != n_dual + 2 or params.reg_channels[0] != n_primal + 1:
        raise ValueError(
            "data_channels must start with data_channels[-1] + 2 channels and "
            "reg_channels with reg_channels[-1] + 1, got "
            f"data_channels={params.data_channels}, reg_channels={params.reg_channels}"
        )
    # channel of the primal variable projected in the dual step: 1 as in Adler &
    # Oktem (the second primal channel)
    channel = params.dual_step_primal_channel
    if not 0 <= channel < n_primal:
        raise ValueError(
            f"dual_step_primal_channel={channel} needs at least {channel + 1} primal channels"
        )
    return n_primal, n_dual, channel


class LPD(LIONmodel):
    """Learned Primal-Dual (Adler and Öktem, 2018): ``n_iters`` iterations,
    each a dual step (a CNN of the dual, the projection of the primal and the
    measurements) and a primal step (a CNN of the primal and the backprojection
    of the dual). The primal starts from the initial reconstruction (the
    operator's default, FDK for CT, unless set). Works with any operator with
    ``autograd()``. Takes the measurements.

    ``model_parameters`` (``default_parameters()``):

    - ``n_iters``: the iterations (10);
    - ``data_channels``: the dual block's channels, a list: from ``n_dual + 2``
      to ``n_dual`` ([7, 32, 32, 5]);
    - ``reg_channels``: the primal block's channels: from ``n_primal + 1`` to
      ``n_primal`` ([6, 32, 32, 5]);
    - ``learned_step``: learn each iteration's step size (False);
    - ``step_size``: its initial value (None: 1 / ||A||, by the power method);
    - ``step_positive``: keep the learned step positive (False);
    - ``instance_norm``: instance normalisation in the blocks (False);
    - ``conv_bias``: biases in the convolutions (True);
    - ``dual_step_primal_channel``: the primal channel projected in the dual
      step (1, as in the paper)."""

    def __init__(
        self,
        operator_params: OperatorParameters,
        model_parameters: LIONParameter = None,
    ):

        if operator_params is None:
            raise ValueError(
                "operator_params required: the parameters of the operator (for CT, the geometry)"
            )

        super().__init__(model_parameters, operator_params)

        # Create layers per iteration
        for i in range(self.model_parameters.n_iters):
            primal, dual = self._iteration_blocks()
            self.add_module(f"{i}_primal", primal)
            self.add_module(f"{i}_dual", dual)

        _check_channels(self.model_parameters)

        # Create pytorch compatible operators and send them to autograd
        self._make_operator(autograd=True)

        # Define step size
        if self.model_parameters.step_size is None:
            # compute step size
            self.model_parameters.step_size = float(1 / power_method(self.op))
        # Step size of each iteration: the primal update adds step * A^T(h).
        # (A lambda_primal used to be created too, but was never used.)
        n_iters = self.model_parameters.n_iters
        step_size = self.model_parameters.step_size
        if self.model_parameters.learned_step:
            self.lambda_dual = nn.ParameterList(
                [nn.Parameter(torch.ones(1) * step_size) for _ in range(n_iters)]
            )
        else:
            self.register_buffer(
                "lambda_dual", torch.ones(n_iters) * step_size, persistent=False
            )

    def _iteration_blocks(self):
        """The primal and dual blocks of one iteration: CNNs for LPD (cLPD's
        are continuous)."""
        params = self.model_parameters
        primal = RegProximal(
            layers=len(params.reg_channels) - 1,
            channels=params.reg_channels,
            conv_bias=params.conv_bias,
            instance_norm=params.instance_norm,
        )
        dual = dataProximal(
            layers=len(params.data_channels) - 1,
            channels=params.data_channels,
            conv_bias=params.conv_bias,
            instance_norm=params.instance_norm,
        )
        return primal, dual

    @staticmethod
    def default_parameters():
        """The paper's LPD: 10 iterations of 3-layer blocks of 32 channels, 5
        primal and 5 dual channels (see the class)."""
        LPD_params = LIONModelParameter()
        LPD_params.n_iters = 10
        LPD_params.data_channels = [7, 32, 32, 5]
        LPD_params.reg_channels = [6, 32, 32, 5]
        LPD_params.learned_step = False
        LPD_params.step_size = None
        LPD_params.step_positive = False
        LPD_params.instance_norm = False
        LPD_params.conv_bias = True
        # project the second primal channel in the dual step, as in Adler & Oktem
        LPD_params.dual_step_primal_channel = 1
        LPD_params.model_input_type = ModelInputType.SINOGRAM
        return LPD_params

    @staticmethod
    def continous_LPD_paper():
        """The LPD that cLPD's paper (Runkel et al., 2024) compares with: 5
        iterations of 4-layer blocks, learned positive steps, instance
        normalisation, no biases."""
        LPD_params = LIONModelParameter()
        LPD_params.n_iters = 5
        LPD_params.data_channels = [7, 32, 32, 32, 5]
        LPD_params.reg_channels = [6, 32, 32, 32, 5]
        LPD_params.learned_step = True
        LPD_params.step_size = None
        LPD_params.step_positive = True
        LPD_params.instance_norm = True
        LPD_params.conv_bias = False
        # project the second primal channel in the dual step, as in Adler & Oktem
        LPD_params.dual_step_primal_channel = 1
        LPD_params.model_input_type = ModelInputType.SINOGRAM
        return LPD_params

    @staticmethod
    def __dual_step(g, h, f, module):
        x = torch.cat((h, f, g), dim=1)
        out = module(x)
        return h + out

    @staticmethod
    def __primal_step(f, update, module):
        x = torch.cat((f, update), dim=1)
        out = module(x)
        return f + out

    @staticmethod
    def cite(cite_format="bib"):
        """Where LPD comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{adler2018learned,
            title={Learned primal-dual reconstruction},
            author={Adler, Jonas and {\\"O}ktem, Ozan},
            journal={IEEE transactions on medical imaging},
            volume={37},
            number={6},
            pages={1322--1332},
            year={2018},
            publisher={IEEE}
            }
            """,
            mla="""
            Adler, Jonas, and Öktem, Ozan.
            "Learned primal-dual reconstruction."
            IEEE transactions on medical imaging 
            37.6 (2018): 1322-1332.
            """,
        )

    def forward(self, g):
        """The reconstructions of the measurements ``g`` (B, 1, ...), of the
        operator's range shape."""
        B, C, W, H = g.shape

        if C != 1:
            raise NotImplementedError("Only 2D CT images supported")

        if tuple(g.shape[1:]) != tuple(self.op.range_shape):
            raise ValueError(
                f"measurements of shape {tuple(g.shape[1:])} don't match the "
                f"operator's range {tuple(self.op.range_shape)}"
            )

        n_primal, n_dual, c = _check_channels(self.model_parameters)

        # initialize parameters
        h = g.new_zeros(B, n_dual, W, H)
        f_primal = g.new_zeros(B, n_primal, *self.op.domain_shape[1:])
        # every primal channel starts from the initial reconstruction (the operator's default, FDK for CT, unless set)
        f_primal[:] = self.initial_reconstruction(g[:, :1], self.op)

        for i in range(self.model_parameters.n_iters):
            primal_module = getattr(self, f"{i}_primal")
            dual_module = getattr(self, f"{i}_dual")
            f_dual = self.A(f_primal[:, c : c + 1])
            h = self.__dual_step(g, h, f_dual, dual_module)

            step = self.lambda_dual[i]
            if self.model_parameters.step_positive:
                step = step.abs()  # learned steps stay positive
            update = step * self.AT(h[:, :1])
            f_primal = self.__primal_step(f_primal, update, primal_module)

        return f_primal[:, 0:1]
