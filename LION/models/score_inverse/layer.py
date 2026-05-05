"""
This module implements the necessary building blocks for the score-based model (NCSN++).

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
import torch.nn.functional as F
import math

def default_init(module, scale=1.):
    if hasattr(module, 'weight') and module.weight is not None:
        if scale == 0.:
            nn.init.zeros_(module.weight)
        else:
            nn.init.xavier_uniform_(module.weight, gain=math.sqrt(scale))
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)

def default_num_groups(ch):
    return max(min(ch // 4, 32), 1)

class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embeddings for noise levels.
    """
    def __init__(self, embedding_size=256, scale=1.):
        super().__init__()
        self.register_buffer('W', torch.randn(embedding_size) * scale)

    def forward(self, x):
        """
        Forward pass for Gaussian Fourier projection.
        Args:
            x: Tensor of shape (B,) containing log noise levels.
        Returns:
            Tensor of shape (B, 2 * embedding_size) containing Fourier features.
        """
        x_proj = x[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Conv1x1(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding='same', bias=True, dilation=1, init_scale=1.):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=padding, bias=bias, dilation=dilation)
        default_init(self.conv, init_scale)

    def forward(self, x):
        return self.conv(x)

class Conv3x3(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding='same', bias=True, dilation=1, init_scale=1.):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=padding, bias=bias, dilation=dilation)
        default_init(self.conv, init_scale)

    def forward(self, x):
        return self.conv(x)

class Combine(nn.Module):
    """
    Combine information from skip connections. Passes the skip connection through a 1x1 convolution and then sums it with the main path.
    """
    def __init__(self, skip_ch, main_ch):
        """
        Args:
            skip_ch: number of channels in the skip connection.
            main_ch: number of channels in the main path.
        """
        super().__init__()
        self.conv = Conv1x1(skip_ch, main_ch)

    def forward(self, x, y):
        """
        Args:
            x: Tensor of shape (B, skip_ch, H, W) from the skip connection.
            y: Tensor of shape (B, main_ch, H, W) from the main path.
        Returns:
            Tensor of shape (B, main_ch, H, W) combining x and y.
        """
        h = self.conv(x)
        return h + y

class SelfAttn(nn.Module):
    def __init__(self, ch, skip_rescale=False, init_scale=0.):
        super().__init__()
        self.skip_rescale = skip_rescale
        self.init_scale = init_scale
        self.norm = nn.GroupNorm(num_groups=default_num_groups(ch), num_channels=ch)
        self.w_q = Conv1x1(ch, ch, init_scale=0.1)
        self.w_k = Conv1x1(ch, ch, init_scale=0.1)
        self.w_v = Conv1x1(ch, ch, init_scale=0.1)
        self.w_h = Conv1x1(ch, ch, init_scale=init_scale)
    
    def forward(self, x, use_einsum=True):
        B, C, H, W = x.shape
        N = H * W
        
        h = self.norm(x)
        if use_einsum:
            q = self.w_q(h)
            k = self.w_k(h)
            v = self.w_v(h)
            w = torch.einsum('bchw,bcHW->bhwHW', q, k) / math.sqrt(C)
            w = w.view(B, H, W, N)
            w = F.softmax(w, dim=-1)
            w = w.view(B, H, W, H, W)
            h = torch.einsum('bhwHW,bcHW->bchw', w, v)
        else:
            q_t = self.w_q(h).reshape(B, C, N).transpose(1, 2).contiguous()  # (B, N, C)
            k = self.w_k(h).reshape(B, C, N).contiguous()  # (B, C, N)
            v_t = self.w_v(h).reshape(B, C, N).transpose(1, 2).contiguous()  # (B, N, C)
            w = torch.bmm(q_t, k) / math.sqrt(C)  # (B, N, N)
            w = F.softmax(w, dim=-1)  # (B, N, N)
            h_t = torch.bmm(w, v_t)  # (B, N, C)
            h = h_t.transpose(1, 2).contiguous().reshape(B, C, H, W)  # (B, C, H, W)
        
        h = self.w_h(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / math.sqrt(2.)

def get_fir_kernel(factor=1.):
    k = torch.tensor([1, 3, 3, 1], dtype=torch.float32)
    k = torch.outer(k, k)
    k /= k.sum()
    k *= factor**2
    return k.view(1, 1, 4, 4)

class Upsample(nn.Module):
    def __init__(self, ch, with_conv=False, fir=False):
        super().__init__()
        self.ch = ch
        self.with_conv = with_conv
        self.fir = fir
        if self.fir:
            self.register_buffer('fir_kernel', get_fir_kernel(factor=2.).repeat(ch, 1, 1, 1))
        if with_conv:
            if not self.fir:
                self.conv = Conv3x3(ch, ch)
            else:
                self.convt = nn.ConvTranspose2d(ch, ch, kernel_size=3, stride=2, padding=0)
                default_init(self.convt)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.ch
        if not self.fir:
            x = F.interpolate(x, scale_factor=2., mode='nearest')
            if self.with_conv:
                x = self.conv(x)
        else:
            if not self.with_conv:
                x = F.conv_transpose2d(x, self.fir_kernel, stride=2, padding=1, groups=C)
            else:
                x = self.convt(x)
                x = F.conv2d(x, self.fir_kernel, stride=1, padding=1, groups=C)

        assert x.shape == (B, C, 2 * H, 2 * W)
        return x

class Downsample(nn.Module):
    def __init__(self, ch, with_conv=False, fir=False):
        super().__init__()
        self.ch = ch
        self.with_conv = with_conv
        self.fir = fir
        if self.fir:
            self.register_buffer('fir_kernel', get_fir_kernel().repeat(ch, 1, 1, 1))
        if with_conv:
            self.conv = Conv3x3(ch, ch, stride=2, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.ch
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1, 0, 1), mode='constant', value=0.)
                x = self.conv(x)
            else:
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
        else:
            if not self.with_conv:
                x = F.conv2d(x, self.fir_kernel, stride=2, padding=1, groups=C)
            else:
                x = F.pad(x, (2, 2, 2, 2), mode='constant', value=0.)
                x = F.conv2d(x, self.fir_kernel, stride=1, padding=0, groups=C)
                x = self.conv(x)

        assert x.shape == (B, C, H // 2, W // 2)
        return x

class ResnetBlock(nn.Module):
    """
    ResNet block from BigGAN.
    """
    def __init__(self, in_ch, out_ch=None, up=False, down=False, act=F.silu, dropout=0.1, fir=False, temb_dim=None, skip_rescale=True, init_scale=0.):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch if out_ch is not None else in_ch
        self.up = up
        self.down = down
        self.act = act
        self.skip_rescale = skip_rescale

        self.norm1 = nn.GroupNorm(num_groups=default_num_groups(in_ch), num_channels=in_ch)
        if self.up:
            self.resizer = Upsample(self.in_ch, fir=fir)
        elif self.down:
            self.resizer = Downsample(self.in_ch, fir=fir)
        self.conv1 = Conv3x3(self.in_ch, self.out_ch)
        if temb_dim is not None:
            self.temb_linear = nn.Linear(temb_dim, self.out_ch)
            default_init(self.temb_linear)
        self.norm2 = nn.GroupNorm(num_groups=default_num_groups(self.out_ch), num_channels=self.out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = Conv3x3(self.out_ch, self.out_ch, init_scale=init_scale)
        if self.in_ch != self.out_ch or self.up or self.down:
            self.conv3 = Conv1x1(self.in_ch, self.out_ch)
    
    def forward(self, x, temb=None):
        h = self.act(self.norm1(x))
        if self.up or self.down:
            h = self.resizer(h)
            x = self.resizer(x)
        h = self.conv1(h)
        if temb is not None:
            h += self.temb_linear(self.act(temb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.conv3(x)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / math.sqrt(2.)