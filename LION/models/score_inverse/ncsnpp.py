"""
This module implements the score-based model (NCSN++).

Author: Tianzhen Peng

References
----------
.. [Song2021] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., 
   Ermon, S., & Poole, B. (2021). "Score-Based Generative Modeling 
   through Stochastic Differential Equations." ICLR. https://openreview.net/forum?id=PxTIG12RRHS

.. [Song2022] Song, Y., Shen, L., Xing, L., & Ermon, S. (2022). 
   "Solving Inverse Problems in Medical Imaging with Score-Based 
   Generative Models." ICLR. https://openreview.net/forum?id=vaRCHVj0uGI
"""

import torch
import torch.nn as nn
from .layer import default_num_groups, GaussianFourierProjection, SelfAttn, Conv3x3, Combine, Upsample, Downsample, ResnetBlock
from .sde import SimpleForwardSDE

class NCSNpp(nn.Module):
    """
    NCSN++ architecture for score matching.
    
    Uses a U-Net structure with multi-resolution ResNet blocks, self-attention layers, FIR up/down-sampling, and Gaussian Fourier embeddings for noise-level.
    
    Args:
        image_resolution (int): Height/width of square input images.
        num_channels (int): Input image channels (typically 1 for CT).
        nf (int): Base feature map channels.
        ch_mult (tuple): Channel multiplier per resolution stage.
        num_res_blocks (int): Number of ResNet blocks per stage.
        attn_resolutions (tuple): Resolutions at which self-attention is applied.
        fourier_scale (float): Scaling factor for Gaussian Fourier embeddings for noise levels.
        fir (bool): Enable FIR up/down-sampling.
        act (nn.Module): Activation function.
        skip_rescale (bool): Rescale skip connection outputs by 1/sqrt(2).
        init_scale (float): Scale factor for initialization of convolutional weights.
        dropout (float): Dropout probability.
        scale_by_sigma (bool): If True, normalize outputs by current noise levels (sigma).
    """
    def __init__(self,
                 image_resolution=512,
                 num_channels=1,
                 nf=16,
                 ch_mult=(1, 2, 4, 8, 16, 32, 32),
                 num_res_blocks=1,
                 attn_resolutions=(16,),
                 fourier_scale=16,
                 fir=True,
                 act=nn.SiLU(),
                 skip_rescale=True,
                 init_scale=0.,
                 dropout=0.,
                 scale_by_sigma=True):
        super().__init__()
        self.act = act
        self.nf = nf
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.fir = fir
        self.skip_rescale = skip_rescale
        self.scale_by_sigma = scale_by_sigma
        
        resnetblock = lambda in_ch, out_ch, up=False, down=False: ResnetBlock(in_ch, out_ch, temb_dim=nf * 4, dropout=dropout, act=act, fir=fir, init_scale=init_scale, skip_rescale=skip_rescale, up=up, down=down)
        selfattn = lambda ch: SelfAttn(ch, init_scale=init_scale, skip_rescale=skip_rescale)
        gnorm = lambda ch: nn.GroupNorm(num_groups=default_num_groups(ch), num_channels=ch)
        self.pyramid_upsample = Upsample(ch=num_channels, fir=fir)
        self.pyramid_downsample = Downsample(ch=num_channels, fir=fir)
        
        self.temb_gfp = GaussianFourierProjection(embedding_size=nf, scale=fourier_scale)
        self.temb_mlp = nn.Sequential(nn.Linear(nf * 2, nf * 4), act, nn.Linear(nf * 4, nf * 4))
        
        self.input_conv = Conv3x3(num_channels, nf)
        chs = [nf]
        self.horizontal_rbs_down = nn.ModuleList()
        self.horizontal_sas_down = nn.ModuleList()
        self.rbs_down = nn.ModuleList()
        self.combs_down = nn.ModuleList()
        for i_level, mult in enumerate(ch_mult):
            for i_block in range(num_res_blocks):
                out_ch = nf * mult
                self.horizontal_rbs_down.append(resnetblock(chs[-1], out_ch))
                if image_resolution in attn_resolutions:
                    self.horizontal_sas_down.append(selfattn(out_ch))
                chs.append(out_ch)

            if i_level != len(ch_mult) - 1:
                self.rbs_down.append(resnetblock(chs[-1], chs[-1], down=True))
                self.combs_down.append(Combine(skip_ch=num_channels, main_ch=chs[-1]))
                chs.append(chs[-1])
                image_resolution //= 2
        
        self.bottom_res1 = resnetblock(chs[-1], chs[-1])
        self.bottom_sa = selfattn(chs[-1])
        self.bottom_res2 = resnetblock(chs[-1], chs[-1])
        
        in_ch = chs[-1]
        self.horizontal_rbs_up = nn.ModuleList()
        self.horizontal_sas_up = nn.ModuleList()
        self.pyramid_convs_up = nn.ModuleList()
        self.rbs_up = nn.ModuleList()
        for i_level_r, mult in enumerate(reversed(ch_mult)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * mult
                self.horizontal_rbs_up.append(resnetblock(in_ch + chs.pop(), out_ch))
                if image_resolution in attn_resolutions:
                    self.horizontal_sas_up.append(selfattn(out_ch))
                in_ch = out_ch

            self.pyramid_convs_up.append(nn.Sequential(gnorm(in_ch), act, Conv3x3(in_ch, num_channels, init_scale=init_scale)))

            if i_level_r != len(ch_mult) - 1:
                self.rbs_up.append(resnetblock(in_ch, in_ch, up=True))
            image_resolution *= 2
                
    def forward(self, x, noise_level):
        """
        Forward pass of the model.
        
        Args:
            x: Tensor of shape (B, C, H, W), the input image.
            noise_level: Tensor of shape (B,), the noise level for each sample in the batch
        
        Returns:
            Tensor of shape (B, C, H, W), the output score function.
        """
        temb = self.temb_gfp(torch.log(noise_level))
        temb = self.temb_mlp(temb)
        
        input_pyramid = x
        sas_iter = iter(self.horizontal_sas_down)
        hs = [self.input_conv(x)]
        for i_level in range(len(self.ch_mult)):
            for i_block in range(self.num_res_blocks):
                h = self.horizontal_rbs_down[i_level * self.num_res_blocks + i_block](hs[-1], temb)
                if h.shape[2] in self.attn_resolutions:
                    h = next(sas_iter)(h)
                hs.append(h)

            if i_level != len(self.ch_mult) - 1:
                h = self.rbs_down[i_level](hs[-1], temb)
                input_pyramid = self.pyramid_downsample(input_pyramid)
                h = self.combs_down[i_level](input_pyramid, h)
                hs.append(h)
        
        h = self.bottom_res1(hs[-1], temb)
        h = self.bottom_sa(h)
        h = self.bottom_res2(h, temb)
        
        sas_iter = iter(self.horizontal_sas_up)
        for i_level_r in range(len(self.ch_mult)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.horizontal_rbs_up[i_level_r * (self.num_res_blocks + 1) + i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if h.shape[2] in self.attn_resolutions:
                    h = next(sas_iter)(h)

            if i_level_r == 0:
                pyramid = self.pyramid_convs_up[i_level_r](h)
            else:
                pyramid = self.pyramid_upsample(pyramid)
                pyramid = pyramid + self.pyramid_convs_up[i_level_r](h)

            if i_level_r != len(self.ch_mult) - 1:
                h = self.rbs_up[i_level_r](h, temb)

        assert not hs
        h = pyramid
        if self.scale_by_sigma:
            h = h / noise_level.view(-1, 1, 1, 1)
        return h

def get_score_fn(model: nn.Module, sde: SimpleForwardSDE):
    """
    Returns a function that computes the score for a given input and time.
    
    Args:
        model: the score-based model, which accepts (x, noise_level) as input and outputs the score function.
        sde: the forward SDE to be used for training.
    
    Returns:
        A function that takes (x, t) as input and returns the score function.
    """
    def score_fn(x, t):
        return model(x, sde.beta(t))
    return score_fn