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


def score_from_denoiser(
    noisy_image_patch: torch.Tensor, denoised_patch: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    while sigma.ndim < noisy_image_patch.ndim:
        sigma = sigma.unsqueeze(-1)
    return (denoised_patch - noisy_image_patch) / sigma.square()


class NCSNpp(LIONmodel):
    """NCSN++ model"""

    @staticmethod
    def default_parameters(mode="padis-paper-ct-256") -> LIONModelParameter:
        model_params = LIONModelParameter()

        if mode in ("padis-paper-ct-256", "PaDIS-ddpmpp"):
            model_params.embedding_type = "positional"
            model_params.channel_mult_noise = 1
            model_params.model_channels = 128
            model_params.channel_mult = [1, 2, 2, 2]
            model_params.dropout = 0.05
            model_params.largest_patch_size = 56
            model_params.pad_width = 24
            model_params.patch_sizes = [16, 32, 56]
            model_params.patch_probabilities = [0.2, 0.3, 0.5]
        elif mode == "padis-paper-ct-512":
            model_params.embedding_type = "positional"
            model_params.channel_mult_noise = 1
            model_params.model_channels = 128
            model_params.channel_mult = [1, 2, 2, 2]
            model_params.dropout = 0.05
            model_params.largest_patch_size = 64
            model_params.pad_width = 64
            model_params.patch_sizes = [16, 32, 64]
            model_params.patch_probabilities = [0.2, 0.3, 0.5]
        else:
            raise ValueError(f"Mode {mode} not recognized.")

        model_params.fir = False
        model_params.fir_kernel = [1, 1]
        model_params.skip_rescale = True
        model_params.progressive = "none"
        model_params.progressive_input = "none"
        model_params.progressive_combine = "sum"
        model_params.init_scale = 1e-5
        model_params.fourier_scale = 16
        model_params.scale_by_sigma = False
        model_params.centered = True
        model_params.noise_label_type = "identity"
        model_params.channel_mult_emb = 4
        model_params.nonlinearity = "swish"
        model_params.conditional = True
        model_params.num_res_blocks = 4
        model_params.attn_resolutions = [16]
        model_params.resblock_type = "biggan"
        model_params.resamp_with_conv = True
        model_params.sigma_min = 0.002
        model_params.sigma_max = 40
        model_params.num_scales = 1000
        model_params.input_position_channels = 2
        model_params.model_input_type = ModelInputType.IMAGE

        return model_params

    def __init__(
        self, params: Optional[LIONModelParameter], geometry: Optional[ct.Geometry]
    ):
        super().__init__(params, geometry)
        params = self.model_parameters
        self.params = params
        self.act = get_act(params)
        act = self.act
        if geometry is None:
            raise ValueError("NCSNpp requires geometry to infer image channels.")
        input_channels = int(geometry.image_shape[0]) + int(
            getattr(params, "input_position_channels", 0)
        )
        self.image_channels = int(geometry.image_shape[0])
        self.input_channels = input_channels
        noise_channels = params.model_channels * params.channel_mult_noise
        emb_channels = params.model_channels * params.channel_mult_emb

        self.register_buffer("sigmas", torch.tensor(utils.get_sigmas(params)))

        channel_mult = params.channel_mult
        num_res_blocks = params.num_res_blocks
        attn_resolutions = params.attn_resolutions
        dropout = params.dropout
        resamp_with_conv = params.resamp_with_conv
        num_resolutions = len(channel_mult)
        model_input_size = int(
            getattr(params, "largest_patch_size", geometry.image_shape[1])
        )
        patch_sizes = getattr(params, "patch_sizes", [model_input_size])
        self.attention_at_level = [
            any(
                (int(patch_size) // (2**i)) in attn_resolutions
                for patch_size in patch_sizes
            )
            for i in range(num_resolutions)
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

        output_channels = getattr(params, "output_channels", self.image_channels)
        self.output_channels = int(output_channels)
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

                if self.attention_at_level[i_level]:
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

            if self.attention_at_level[i_level]:
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

    def forward(self, x, time_cond, class_labels=None, augment_labels=None):
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
            timesteps = time_cond
            if not torch.is_floating_point(timesteps):
                used_sigmas = self.sigmas[timesteps.long()]
                timesteps = timesteps.float()
            else:
                used_sigmas = timesteps
                noise_label_type = getattr(params, "noise_label_type", "identity")
                if noise_label_type == "log_sigma":
                    timesteps = torch.log(timesteps.clamp_min(1e-12))
                elif noise_label_type == "log_sigma_over_four":
                    timesteps = torch.log(timesteps.clamp_min(1e-12)) / 4
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
                if self.attention_at_level[i_level]:
                    attn = modules[m_idx]
                    if h.shape[-1] in attn_resolutions:
                        h = attn(h)
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

            if self.attention_at_level[i_level]:
                attn = modules[m_idx]
                if h.shape[-1] in attn_resolutions:
                    h = attn(h)
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
