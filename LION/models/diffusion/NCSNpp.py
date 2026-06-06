# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modified from Song, Yang, Jascha Sohl-Dickstein, Diederik P. Kingma,
# Abhishek Kumar, Stefano Ermon, and Ben Poole. ‘Score-Based Generative
# Modeling through Stochastic Differential Equations’. arXiv:2011.13456.
# Preprint, arXiv, 10 February 2021. https://doi.org/10.48550/arXiv.2011.13456.
# and their codebase at https://github.com/yang-song/score_sde_pytorch/models/ncsnpp.py


from LION.models.diffusion.NCSNpp_helpers import utils, layers, layerspp, normalization
from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType
from LION.utils.parameter import LIONParameter
from typing import Optional
import LION.CTtools.ct_geometry as ct
import torch.nn as nn
import functools
import torch
import numpy as np
import warnings

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


class NCSNpp(LIONmodel):
    """NCSN++ model"""

    @staticmethod
    def default_parameters(mode="PaDIS-ddpmpp") -> LIONModelParameter:
        # These are the default parameters for the PaDIS implementation of NCSN++, which
        # falls back to the DDPM++ version.

        # # Patch options
        # patch_params = LIONParameter()
        # patch_params.p_largest_patch = 0.5   # The probability of using the largest patch size during training

        model_params = LIONModelParameter()

        # Model options
        if mode == "PaDIS-ddpmpp":
            model_params.embedding_type = "positional"
            model_params.channel_mult_noise = 1
            model_params.model_channels = 128
            model_params.channel_mult = [1, 2, 2, 2]
            model_params.dropout = 0.05

            # Resample filter
            model_params.fir = False  # Whether to use finite impuse response (FIR) filters for up/downsampling. If false, then use a nearest-neighbour 2x resize for upsampling and for downsampling, average pooling or a stride-2 conv if resample_with_conv is True.
            model_params.fir_kernel = [1, 1]  # Ignored here because fir=False

            model_params.skip_rescale = True  # Whether to rescale skip connections in residual blocks by sqrt(2).
            model_params.progressive = "none"  # Decoder type (standard -> none)
            model_params.progressive_input = (
                "none"  # Encoder type (standard -> none, residual -> residual)
            )
            model_params.progressive_combine = "sum"
            model_params.init_scale = 1e-5
            model_params.fourier_scale = 16
            model_params.scale_by_sigma = False
            model_params.centered = True

            # Can't remember where these came from??
            # model_params.augment = 0.12             # probability of performing augmentations
            # model_params.implicit_mlp = False       # Do not train an implict MLP to encode coords before sending to convolutions
            # model_params.padding = True             # Whether to zero-pad images for training
            # model_params.four_channels = 1          # The number of Fourier embedding frequencies
            # model_params.hash_channels = 1          # The number of hash embedding frequencies
            # model_params.pad_width = 64             # The width on all sides of zero padding
            # model_params.use_fp16 = False           # Whether to use mixed precision training (fp16). This can speed up training and reduce memory usage, but may cause instability for some models.
            # model_params.cudnn_benchmark = True     # Whether to use cudnn benchmark mode. This can speed up training for fixed input sizes, but may cause instability for some models.

            model_params.channel_mult_emb = 4
            model_params.nonlinearity = "swish"
            model_params.conditional = True  # Whether or not to use timestep / noise level embedding conditioning
            model_params.num_res_blocks = 4
            model_params.attn_resolutions = [16]

            model_params.resblock_type = "biggan"
            model_params.resamp_with_conv = True

            model_params.sigma_min = 0.002
            model_params.sigma_max = 40
            model_params.num_scales = 1000

        # if mode == "ddpmpp":
        #   model_params.embedding_type = "positional"
        #   model_params.encoder_type = "standard"
        #   model_params.decoder_type = "standard"
        #   model_params.channel_mult_noise = 1
        #   model_params.resample_filter = [1,1]
        #   model_params.model_channels = 128
        #   model_params.channel_mult = [2,2,2]
        #   model_params.dropout = 0.05

        #   model_params.fir = False              # Whether to use finite impuse response (FIR) filters for up/downsampling. If false, then use a nearest-neighbour 2x resize for upsampling and for downsampling, average pooling or a stride-2 conv if resample_with_conv is True.
        #   model_params.fir_kernel = [1, 3, 3, 1]# Ignored here because fir=False
        #   model_params.skip_rescale = False      # Whether to rescale skip connections in residual blocks by sqrt(2).
        # elif mode == "ncsnpp":
        #   model_params.embedding_type = "fourier"
        #   model_params.encoder_type = "residual"
        #   model_params.decoder_type = "standard"
        #   model_params.channel_mult_noise = 2
        #   model_params.resample_filter = [1,3,3,1]
        #   model_params.model_channels = 128
        #   model_params.channel_mult = [2,2,2]
        #   model_params.dropout = 0.05
        # else:
        #   raise ValueError(f'Mode {mode} not recognized.')
        # model_params.augment = 0.12             # probability of performing augmentations
        # model_params.implicit_mlp = False       # Do not train an implict MLP to encode coords before sending to convolutions
        # model_params.padding = True             # Whether to zero-pad images for training
        # model_params.four_channels = 1          # The number of Fourier embedding frequencies
        # model_params.hash_channels = 1          # The number of hash embedding frequencies
        # model_params.pad_width = 64             # The width on all sides of zero padding
        # model_params.use_fp16 = False           # Whether to use mixed precision training (fp16). This can speed up training and reduce memory usage, but may cause instability for some models.
        # model_params.cudnn_benchmark = True     # Whether to use cudnn benchmark mode. This can speed up training for fixed input sizes, but may cause instability for some models.
        # model_params.channel_mult_emb = 4
        # model_params.nonlinearity = "swish"
        # model_params.conditional = True          # Whether or not to use timestep / noise level embedding conditioning
        # model_params.num_res_blocks = 4
        # model_params.attn_resolutions = [16]

        # model_params.resblock_type = "biggan"
        # model_params.resamp_with_conv = True

        # model_params

        # model = LIONModelParameter()
        # if mode = "ddpm"
        #   # training.sde = 'vpsde'
        #   # training.continuous = False
        #   # training.reduce_mean = True
        #   # sampling.method = 'pc'
        #   # sampling.predictor = 'ancestral_sampling'
        #   # sampling.corrector = 'none'
        #   # data.centered = True

        #   model.name = 'ncsnpp'
        #   model.scale_by_sigma = False
        #   model.ema_rate = 0.9999
        #   model.normalization = 'GroupNorm'
        #   model.nonlinearity = 'swish'
        #   model.nf = 128
        #   model.ch_mult = (1, 2, 2, 2)
        #   model.num_res_blocks = 4
        #   model.attn_resolutions = (16,)
        #   model.resamp_with_conv = True
        #   model.conditional = True
        #   model.fir = False
        #   model.fir_kernel = [1, 3, 3, 1]
        #   model.skip_rescale = True
        #   model.resblock_type = 'biggan'
        #   model.progressive = 'none'
        #   model.progressive_input = 'none'
        #   model.progressive_combine = 'sum'
        #   model.attention_type = 'ddpm'
        #   model.init_scale = 0.0
        #   model.embedding_type = 'positional'
        #   model.fourier_scale = 16
        #   model.conv_size = 3
        # elif mode == "ncsnpp":
        #   training.sde = 'vesde'
        #   training.continuous = False
        #   sampling.method = 'pc'
        #   sampling.predictor = 'reverse_diffusion'
        #   sampling.corrector = 'langevin'

        #   model.name = 'ncsnpp'
        #   model.scale_by_sigma = True
        #   model.ema_rate = 0.999
        #   model.normalization = 'GroupNorm'
        #   model.nonlinearity = 'swish'
        #   model.nf = 128
        #   model.ch_mult = (1, 2, 2, 2)
        #   model.num_res_blocks = 4
        #   model.attn_resolutions = (16,)
        #   model.resamp_with_conv = True
        #   model.conditional = True
        #   model.fir = True
        #   model.fir_kernel = [1, 3, 3, 1]
        #   model.skip_rescale = True
        #   model.resblock_type = 'biggan'
        #   model.progressive = 'none'
        #   model.progressive_input = 'residual'
        #   model.progressive_combine = 'sum'
        #   model.attention_type = 'ddpm'
        #   model.init_scale = 0.0
        #   model.embedding_type = 'positional'
        #   model.conv_size = 3

        return model_params

        # # Training parameters (README)
        # train_params = LIONParameter()
        # train_params.cond = 0
        # train_params.arch = "ddpmpp"
        # train_params.batch_size = 16
        # train_params.batch_size_gpu = None # maximum batch size per gpu. None = no limit.
        # train_params.lr = 1e-4
        # train_params.dropout = 0.05
        # train_params.augment = 0
        # train_params.real_p = 0.5
        # train_params.padding = 1
        # train_params.tick = 2
        # train_params.snap = 10
        # train_params.pad_width = 64
        # train_params.precond = "pedm" # others could be [vp, ve, edm, pedm]
        # train_params.ls = 1 # loss scaling factor
        # train_params.seed = 33

        # # Reconstruction parameters (README)
        # reconstruction_params = LIONParameter()
        # reconstruction_params.image_size = 256
        # reconstruction_params.views = 20
        # reconstruction_params.name = "ct_parbeam"
        # reconstruction_params.steps = 100
        # reconstruction_params.sigma_min = 0.003
        # reconstruction_params.sigma_max = 10
        # reconstruction_params.zeta = 0.3
        # reconstruction_params.pad = 24
        # reconstruction_params.psize = 56

    def __init__(
        self, params: Optional[LIONModelParameter], geometry: Optional[ct.Geometry]
    ):
        super().__init__(params, geometry)
        self.params = params
        self.act = get_act(params)
        act = self.act
        positional_encoding = params.embedding_type in ["positional", "fourier"]
        input_channels = geometry.image_shape[0] + 2 * positional_encoding
        noise_channels = params.model_channels * params.channel_mult_noise
        emb_channels = params.model_channels * params.channel_mult_emb

        # next task is to figure out the sigmas
        self.register_buffer("sigmas", torch.tensor(utils.get_sigmas(params)))

        channel_mult = params.channel_mult
        num_res_blocks = params.num_res_blocks
        attn_resolutions = params.attn_resolutions
        dropout = params.dropout
        resamp_with_conv = params.resamp_with_conv
        num_resolutions = len(channel_mult)
        all_resolutions = [
            geometry.image_shape[1] // (2**i) for i in range(num_resolutions)
        ]

        conditional = params.conditional  # noise-conditional
        fir = params.fir
        fir_kernel = params.fir_kernel
        skip_rescale = params.skip_rescale
        resblock_type = params.resblock_type.lower()
        progressive = params.progressive.lower()
        progressive_input = params.progressive_input.lower()
        embedding_type = params.embedding_type.lower()
        init_scale = params.init_scale
        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional"]
        combine_method = params.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            warnings.warn(
                "Note that Fourier features are only used for continuous training.",
                UserWarning,
                stacklevel=2,
            )

            modules.append(
                layerspp.GaussianFourierProjection(
                    embedding_size=noise_channels // 2, scale=params.fourier_scale
                )
            )
            embed_dim = noise_channels

        elif embedding_type == "positional":
            embed_dim = noise_channels

        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        if conditional:
            modules.append(nn.Linear(embed_dim, emb_channels))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(emb_channels, emb_channels))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(
            layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale
        )

        Upsample = functools.partial(
            layerspp.Upsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive == "output_skip":
            self.pyramid_upsample = layerspp.Upsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive == "residual":
            pyramid_upsample = functools.partial(
                layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        Downsample = functools.partial(
            layerspp.Downsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive_input == "input_skip":
            self.pyramid_downsample = layerspp.Downsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(
                layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=emb_channels,
            )

        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=emb_channels,
            )

        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        # Downsampling block

        output_channels = getattr(params, "output_channels", input_channels)
        if progressive_input != "none":
            input_pyramid_ch = input_channels

        modules.append(conv3x3(input_channels, params.model_channels))
        hs_c = [params.model_channels]

        in_ch = params.model_channels
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = params.model_channels * channel_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == "input_skip":
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2

                elif progressive_input == "residual":
                    modules.append(
                        pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch)
                    )
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = params.model_channels * channel_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            conv3x3(in_ch, output_channels, init_scale=init_scale)
                        )
                        pyramid_ch = output_channels
                    elif progressive == "residual":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            conv3x3(
                                in_ch, output_channels, bias=True, init_scale=init_scale
                            )
                        )
                        pyramid_ch = output_channels
                    elif progressive == "residual":
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name")

            if i_level != 0:
                if resblock_type == "ddpm":
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != "output_skip":
            modules.append(
                nn.GroupNorm(
                    num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
                )
            )
            modules.append(conv3x3(in_ch, output_channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond):
        # timestep/noise_level embedding; only for continuous training
        params = self.params
        modules = self.all_modules
        num_resolutions = len(params.channel_mult)
        num_res_blocks = params.num_res_blocks
        attn_resolutions = params.attn_resolutions
        resblock_type = params.resblock_type.lower()
        progressive = params.progressive.lower()
        progressive_input = params.progressive_input.lower()
        embedding_type = params.embedding_type.lower()
        skip_rescale = params.skip_rescale
        m_idx = 0
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif embedding_type == "positional":
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, params.model_channels)

        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        if params.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not params.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.0

        # Downsampling block
        input_pyramid = None
        if progressive_input != "none":
            input_pyramid = x

        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if progressive_input == "input_skip":
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-1] in attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif progressive == "residual":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:
                    if progressive == "output_skip":
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif progressive == "residual":
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f"{progressive} is not a valid name")

            if i_level != 0:
                if resblock_type == "ddpm":
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1

        assert not hs

        if progressive == "output_skip":
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)
        if params.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        return h
