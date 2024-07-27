import math
from functools import lru_cache, reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from torchvision.io import write_video, write_png
import sys
sys.path.append('.')
sys.path.append('..')
from .patchnet import *
import torchvision.transforms as T
import torchvision
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .CLIP_backbone import  build_CLIPmodel_basedadapter_cls


mean = torch.FloatTensor([123.675, 116.28, 103.53])
std = torch.FloatTensor([58.395, 57.12, 57.375])
unnormalize = T.Normalize(mean=-mean / std, std=1.0 / std)
normalize_clip = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
mean, std = np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375])

def fragment_infos(D, H, W, fragments=7, device="cuda"):
    m = torch.arange(fragments).unsqueeze(-1).float()
    m = (m + m.t() * fragments).reshape(1, 1, 1, fragments, fragments)
    m = F.interpolate(m.to(device), size=(D, H, W)).permute(0, 2, 3, 4, 1)
    return m.long()


@lru_cache(maxsize=None)
def global_position_index(
    D,
    H,
    W,
    fragments=(1, 7, 7),
    window_size=(8, 7, 7),
    shift_size=(0, 0, 0),
    device="cuda",
):
    frags_d = torch.arange(fragments[0])
    frags_h = torch.arange(fragments[1])
    frags_w = torch.arange(fragments[2])
    frags = torch.stack(
        torch.meshgrid(frags_d, frags_h, frags_w)
    ).float()  # 3, Fd, Fh, Fw
    coords = (
        torch.nn.functional.interpolate(frags[None].to(device), size=(D, H, W))
        .long()
        .permute(0, 2, 3, 4, 1)
    )
    # print(shift_size)
    coords = torch.roll(
        coords, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3)
    )
    window_coords = window_partition(coords, window_size)
    relative_coords = (
        window_coords[:, None, :] - window_coords[:, :, None]
    )  # Wd*Wh*Ww, Wd*Wh*Ww, 3
    return relative_coords  # relative_coords


@lru_cache(maxsize=None)
def get_adaptive_window_size(
    base_window_size, input_x_size, base_x_size,
):
    tw, hw, ww = base_window_size
    tx_, hx_, wx_ = input_x_size
    tx, hx, wx = base_x_size
    print((tw * tx_) // tx, (hw * hx_) // hx, (ww * wx_) // wx)
    return (tw * tx_) // tx, (hw * hx_) // hx, (ww * wx_) // wx


class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0],
        window_size[0],
        H // window_size[1],
        window_size[1],
        W // window_size[2],
        window_size[2],
        C,
    )
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(-1, reduce(mul, window_size), C)
    )
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        frag_bias=False,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1)
                * (2 * window_size[1] - 1)
                * (2 * window_size[2] - 1),
                num_heads,
            )
        )  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        if frag_bias:
            self.fragment_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * window_size[0] - 1)
                    * (2 * window_size[1] - 1)
                    * (2 * window_size[2] - 1),
                    num_heads,
                )
            )

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(
            torch.meshgrid(coords_d, coords_h, coords_w)
        )  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (
            2 * self.window_size[2] - 1
        )
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, fmask=None, resized_window_size=None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        # print(x.shape)
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if resized_window_size is None:
            rpi = self.relative_position_index[:N, :N]
        else:
            relative_position_index = self.relative_position_index.reshape(
                *self.window_size, *self.window_size
            )
            d, h, w = resized_window_size

            rpi = relative_position_index[:d, :h, :w, :d, :h, :w]
        relative_position_bias = self.relative_position_bias_table[
            rpi.reshape(-1)
        ].reshape(
            N, N, -1
        )  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        if hasattr(self, "fragment_position_bias_table"):
            fragment_position_bias = self.fragment_position_bias_table[
                rpi.reshape(-1)
            ].reshape(
                N, N, -1
            )  # Wd*Wh*Ww,Wd*Wh*Ww,nH
            fragment_position_bias = fragment_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww

        ### Mask Position Bias
        if fmask is not None:
            # fgate = torch.where(fmask - fmask.transpose(-1, -2) == 0, 1, 0).float()
            fgate = fmask.abs().sum(-1)
            nW = fmask.shape[0]
            relative_position_bias = relative_position_bias.unsqueeze(0)
            fgate = fgate.unsqueeze(1)
            # print(fgate.shape, relative_position_bias.shape)
            if hasattr(self, "fragment_position_bias_table"):
                relative_position_bias = (
                    relative_position_bias * fgate
                    + fragment_position_bias * (1 - fgate)
                )

            attn = attn.view(
                B_ // nW, nW, self.num_heads, N, N
            ) + relative_position_bias.unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        else:
            attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(2, 7, 7),
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        jump_attention=False,
        frag_bias=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.jump_attention = jump_attention
        self.frag_bias = frag_bias

        assert (
            0 <= self.shift_size[0] < self.window_size[0]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[1] < self.window_size[1]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[2] < self.window_size[2]
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            frag_bias=frag_bias,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward_part1(self, x, fragment_size,mask_matrix, resized_window_size=None):
        B, D, H, W, C = x.shape
        # print(x.shape)
        window_size, shift_size = get_window_size(
            (D, H, W),
            self.window_size if resized_window_size is None else resized_window_size,
            self.shift_size,
        )

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        if False:  # not hasattr(self, 'finfo_windows'):
            finfo = fragment_infos(Dp, Hp, Wp)

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3),
            )
            if False:  # not hasattr(self, 'finfo_windows'):
                shifted_finfo = torch.roll(
                    finfo,
                    shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                    dims=(1, 2, 3),
                )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            if False:  # not hasattr(self, 'finfo_windows'):
                shifted_finfo = finfo
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        if False:  # not hasattr(self, 'finfo_windows'):
            self.finfo_windows = window_partition(shifted_finfo, window_size)
        # W-MSA/SW-MSA
        # print(shift_size)
        fragment_size= H #96 /8 =12
        gpi = global_position_index(
            Dp,
            Hp,
            Wp,
            fragments=(1,) + window_size[1:],
            window_size=window_size,
            shift_size=shift_size,
            device=x.device,
        )
        attn_windows = self.attn(
            x_windows,
            mask=attn_mask,
            fmask=gpi,
            resized_window_size=window_size
            if resized_window_size is not None
            else None,
        )  # self.finfo_windows)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(
            attn_windows, window_size, B, Dp, Hp, Wp
        )  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, fragment_size,mask_matrix, resized_window_size=None):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if not self.jump_attention:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(
                    self.forward_part1, x,fragment_size, mask_matrix, resized_window_size
                )
            else:
                x = self.forward_part1(x, fragment_size,mask_matrix, resized_window_size)
            x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# cache each stage results
@lru_cache(maxsize=None)
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for h in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for w in (
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=(1, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        jump_attention=False,
        frag_bias=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # print(window_size)
        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    jump_attention=jump_attention,
                    frag_bias=frag_bias,
                )
                for i in range(depth)
            ]
        )

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x,fragment_size, resized_window_size=None):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape

        window_size, shift_size = get_window_size(
            (D, H, W),
            self.window_size if resized_window_size is None else resized_window_size,
            self.shift_size,
        )
        # print(window_size)
        x = rearrange(x, "b c d h w -> b d h w c")
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, fragment_size,attn_mask, resized_window_size=resized_window_size)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x

def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        #trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
        _no_grad_trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

def _no_grad_trunc_normal_(tensor, mean, std, a, b) :
    # Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # Modified from
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [lower, upper], then translate
        # to [2lower-1, 2upper-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()
        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def build_norm_layer(norm_cfg,embed_dims):
    assert norm_cfg['type'] == 'LN'
    norm_layer = nn.LayerNorm(embed_dims)
    return norm_cfg['type'],norm_layer



class Semantic_Transformation2(nn.Module):
    def __init__(self, inChannels):
        super(Semantic_Transformation2, self).__init__()
        #self.conv1 = nn.Conv2d(inChannels, inChannels, 3, padding=1, stride=1)
        #self.conv2 = nn.Conv2d(inChannels, inChannels, 3, padding=1, stride=1)
       
        self.conv_gama = nn.Conv2d(inChannels, 1, 1, padding=0, stride=1)
        self.conv_beta = nn.Conv2d(inChannels, 1, 1, padding=0, stride=1)
      
        self.act = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,input): #input: B,hw,c
        #out = self.act(self.conv1(x))
        #out = self.act(self.conv2(out))
        gama = self.sigmoid(self.conv_gama(x))
        beta = self.conv_beta(x)
       
        return gama*input+beta
    
class Semantic_Transformation4(nn.Module):
    def __init__(self, inChannels):
        super(Semantic_Transformation4, self).__init__()
        #self.conv1 = nn.Conv2d(inChannels, inChannels, 3, padding=1, stride=1)
        #self.conv2 = nn.Conv2d(inChannels, inChannels, 3, padding=1, stride=1)
        
        self.get_gamma = nn.Linear(inChannels, inChannels)
        self.get_beta = nn.Linear(inChannels, inChannels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.conv_gama = nn.Conv2d(inChannels, inChannels, 1, padding=0, stride=1)
        #self.conv_beta = nn.Conv2d(inChannels, inChannels, 1, padding=0, stride=1)
      
        self.act = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
    def stdpool(self,x):
            """2D global standard variation pooling"""
            #B,C,THW
            BT,C,H,W=x.shape
            x=x.view(BT,C,H*W)
            return torch.std(x,
                            dim=2, keepdim=True)
    def forward(self, x,input):
        bt,c,h,w=input.shape
        #out = self.act(self.conv1(x))
        #out = self.act(self.conv2(x))
       
        out_mean = self.avgpool(x).view(x.shape[0],x.shape[1])
        out_std= self.stdpool(x).view(x.shape[0],x.shape[1])   
        gama = self.sigmoid(self.get_gamma(out_std))
        beta = self.get_beta(out_mean)
        return gama.unsqueeze(2).unsqueeze(3)*input+ beta.unsqueeze(2).unsqueeze(3)


class Semantic_Transformation6(nn.Module):
    def __init__(self, inChannels):
        super(Semantic_Transformation6, self).__init__()
        #self.conv1 = nn.Conv2d(inChannels, inChannels, 3, padding=1, stride=1)
        #self.conv2 = nn.Conv2d(inChannels, inChannels, 3, padding=1, stride=1)
        #self.conv3 = nn.Conv2d(inChannels, inChannels, 3, padding=1, stride=1)
        #self.conv4 = nn.Conv2d(inChannels, inChannels, 3, padding=1, stride=1)
       
        self.conv_gama = nn.Conv2d(inChannels, 1, 1, padding=0, stride=1)
        self.conv_beta = nn.Conv2d(inChannels, 1, 1, padding=0, stride=1)
        self.get_gamma = nn.Linear(inChannels, inChannels)
        self.get_beta = nn.Linear(inChannels, inChannels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def stdpool(self,x):
            """2D global standard variation pooling"""
            #B,C,THW
            BT,C,H,W=x.shape
            x=x.view(BT,C,H*W)
            return torch.std(x,
                            dim=2, keepdim=True)
    def forward(self, x,input):
        bt,c,h,w=input.shape
        #out = self.act(self.conv1(x))
        #out = self.act(self.conv2(out))
        sgama = self.sigmoid(self.conv_gama(x))
        sbeta = self.conv_beta(x)

        input1 = sgama*input+sbeta

        out_mean = self.avgpool(x).view(x.shape[0],x.shape[1])
        out_std= self.stdpool(x).view(x.shape[0],x.shape[1])   
        sgama = self.sigmoid(self.conv_gama(x))
        sbeta = self.conv_beta(x)
        cgama = self.sigmoid(self.get_beta(out_mean))
        cbeta = self.get_beta(out_std)

        return cgama.unsqueeze(2).unsqueeze(3)*input1+cbeta.unsqueeze(2).unsqueeze(3)

    
    
class Semantic_Transformation8(nn.Module):
    def __init__(self, inChannels):
        super(Semantic_Transformation8, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, inChannels, 3, padding=1, stride=1)
        #self.conv2 = nn.Conv2d(inChannels, inChannels, 3, padding=1, stride=1)
       
        self.conv_gama = nn.Conv2d(inChannels, 1, 1, padding=0, stride=1)
        self.conv_beta = nn.Conv2d(inChannels, 1, 1, padding=0, stride=1)
      
        self.act = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,input): #input: B,hw,c
        out = self.act(self.conv1(x))
        #out = self.act(self.conv2(out))
        gama = self.sigmoid(self.conv_gama(out))
        beta = self.conv_beta(out)
       
        return gama*input+beta
    
    
class Dist_Transformation3(nn.Module):
    def __init__(self, inChannels):
        super(Dist_Transformation3, self).__init__()
        #self.conv1 = nn.Conv3d(inChannels, inChannels, 3, padding=1, stride=1)#
        #self.conv2 = nn.Conv3d(inChannels, inChannels, 3, padding=1, stride=1)#
        self.avgpool = nn.AdaptiveAvgPool3d(1)
 
        self.get_gamma = nn.Linear(inChannels, inChannels)
        self.get_beta = nn.Linear(inChannels, inChannels)
        #self.feat_ext = FeatureExtractor(inChannels, num_resblocks=4, kSize=3)
        self.act = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
    def stdpool(self,x):
            """3D global standard variation pooling"""
            #B,C,THW
            B,C,T,H,W=x.shape
            x=x.view(B,C,T*H*W)
            return torch.std(x,dim=2, keepdim=True)
    def forward(self, x,input):
        b,thw,c=input.shape
        #out = self.act(self.conv1(x))
        #out = self.act(self.conv2(out))
        out_mean = self.avgpool(x).view(x.shape[0],x.shape[1])
        out_std= self.stdpool(x).view(x.shape[0],x.shape[1])   
        gama = self.sigmoid(self.get_gamma(out_std))
        beta = self.get_beta(out_mean)
        return gama.unsqueeze(1)*input+beta.unsqueeze(1)

class Dist_Transformation5(nn.Module):
    def __init__(self, inChannels):
        super(Dist_Transformation5, self).__init__()
        #self.conv1 = nn.Conv3d(inChannels, inChannels, 3, padding=1, stride=1)#
        #self.conv2 = nn.Conv3d(inChannels, inChannels, 3, padding=1, stride=1)#

        self.conv_gama = nn.Conv3d(inChannels, 1, 3, padding=1, stride=1)
        self.conv_beta = nn.Conv3d(inChannels, 1, 3, padding=1, stride=1)
        #self.feat_ext = FeatureExtractor(inChannels, num_resblocks=4, kSize=3)
        self.act = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
    def stdpool(self,x):
            """3D global standard variation pooling"""
            #B,C,THW
            B,C,T,H,W=x.shape
            x=x.view(B,C,T*H*W)
            return torch.std(x,
                            dim=2, keepdim=True)
    def forward(self, x,input):
        b,thw,c=input.shape
        #out = self.act(self.conv1(x))
        #out = self.act(self.conv2(out))
        gama = self.sigmoid(self.conv_gama(x)).view(x.shape[0],1,thw)
        beta = self.conv_beta(x).view(x.shape[0],1,thw)
        input1=  gama*input.permute(0,2,1)+beta #b,c,thw
        return input1.permute(0,2,1)

class Dist_Transformation7(nn.Module):
    def __init__(self, inChannels):
        super(Dist_Transformation7, self).__init__()
        #self.conv1 = nn.Conv3d(inChannels, inChannels, 3, padding=1, stride=1)#
        #self.conv2 = nn.Conv3d(inChannels, inChannels, 3, padding=1, stride=1)#

        self.conv_gama = nn.Conv3d(inChannels, 1, 3, padding=1, stride=1)
        self.conv_beta = nn.Conv3d(inChannels, 1, 3, padding=1, stride=1)
        self.get_gamma = nn.Linear(inChannels, inChannels)
        self.get_beta = nn.Linear(inChannels, inChannels)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        #self.feat_ext = FeatureExtractor(inChannels, num_resblocks=4, kSize=3)
        self.act = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
    def stdpool(self,x):
            """3D global standard variation pooling"""
            #B,C,THW
            B,C,T,H,W=x.shape
            x=x.view(B,C,T*H*W)
            return torch.std(x, dim=2, keepdim=True)
    def forward(self, x,input):
        b,thw,c=input.shape
        #out = self.act(self.conv1(x))
        #out = self.act(self.conv2(out))
        sgama = self.sigmoid(self.conv_gama(x)).view(x.shape[0],1,thw)
        sbeta = self.conv_beta(x).view(x.shape[0],1,thw)
        input1=  sgama*input.permute(0,2,1)+sbeta
        input1= input1.permute(0,2,1) #b,thw,c
        out_mean = self.avgpool(x).view(x.shape[0],x.shape[1])
        out_std= self.stdpool(x).view(x.shape[0],x.shape[1])   
        cgama = self.sigmoid(self.get_gamma(out_std))
        cbeta = self.get_beta(out_mean)
        return cgama.unsqueeze(1)*input1+cbeta.unsqueeze(1)

        
class KSVQE(nn.Module):

    def __init__(
        self,
        pretrained='/data2/luyt/KSVQE/pretrained_checkpoint/swin_tiny_patch244_window877_kinetics400_1k.pth',
        pretrained2d=False,
        num_samples=500,
        sample_type = 'topkpertubation',
        CLIP_location=10,
        cls_use=True,
        #cls_pat ='11',
        tuning_stage=2,
        a1=1,
        a2=0,
        
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        frozen_stages=-1,
        use_checkpoint=True,
        jump_attention=[False, False, False, False],
        frag_biases=[True, True, True, False],
        base_x_size=(32, 224, 224),
    ):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        self.base_x_size = base_x_size
        self.N_key =5
        
        self.CLIP_tool = build_CLIPmodel_basedadapter_cls(CLIP_location=CLIP_location,cls_use=cls_use)
        
        encoder = get_network('resnet50', pretrained=False)
        model = CONTRIQUE_model( encoder, 2048)
        #model.load_state_dict(torch.load(args.model_path))
        self.distortion_tool=model 
        self.distortion_tool.load_state_dict(torch.load('/data2/luyt/KSVQE/pretrained_checkpoint/CONTRIQUE_checkpoint25.tar'))
        self.dist_adapter = nn.Sequential(
                                        nn.Linear(128, 128// 4),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(128// 4, 128),
                                        nn.ReLU(inplace=True),
                                        ) 
        for param in self.distortion_tool.parameters():
            param.requires_grad = False
        
        
        self.spa_patchnet = RegionNet_CLIP( k=7*7,anchor_size=32,stride=1,num_samples=num_samples,sample_type=sample_type)
       
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        sigma=0.5

        self.sigma_max = sigma
        self.sigma = sigma
        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer]
                if isinstance(window_size, list)
                else window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
                jump_attention=jump_attention[i_layer],
                frag_bias=frag_biases[i_layer],
            )
            self.layers.append(layer)
        
        #self.qlslast_attn = crossattention1(dim=int(embed_dim * 2 ** (self.num_layers - 1)),num_heads=16)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)
    
        self.tuning_stage=tuning_stage
        #cross attn 作为模块1，spatial modulation2, style modulation3

        
        from torch.nn import ModuleList 
        self.semantic_adapter = ModuleList()
        self.distortion_adapter = ModuleList()
        self.semantic_mod = ModuleList()
        self.distortion_mod = ModuleList()
        self.semantic_cross = ModuleList()
        self.distortion_cross = ModuleList()
        self.distortion_self = ModuleList()
    
        
        from torch.nn import ModuleList 
       
        self.a1=nn.Parameter(0 * torch.ones((len(depths)-self.tuning_stage,1))+a1, requires_grad=True)
        self.a2=nn.Parameter(0 * torch.ones((len(depths)-self.tuning_stage,1))+a2, requires_grad=True)
       
        for i in range(self.tuning_stage,len(depths)):
            #
            if i+1>len(depths)-1:
                i=len(depths)-2
            semantic_transform2 = Semantic_Transformation2(inChannels=int(embed_dim * 2 ** (i+1)))
            distortion_transform3 = Dist_Transformation3(inChannels=int(embed_dim * 2 ** (i+1)))
            semantic_transform4 = Semantic_Transformation4(inChannels=int(embed_dim * 2 ** (i+1)))
            distortion_transform5 = Dist_Transformation5(inChannels=int(embed_dim * 2 ** (i+1)))
            semantic_transform6 = Semantic_Transformation6(inChannels=int(embed_dim * 2 ** (i+1)))
            distortion_transform7 = Dist_Transformation7(inChannels=int(embed_dim * 2 ** (i+1)))
            semantic_transform8 = Semantic_Transformation8(inChannels=int(embed_dim * 2 ** (i+1)))

            attn_cross = crossattention1(dim=int(embed_dim * 2 ** (i+1)), num_heads=num_heads[i])
            attn_cross1 = crossattention1(dim=int(embed_dim * 2 ** (i+1)), num_heads=num_heads[i])
            attn_self = Attention(dim=int(embed_dim * 2 ** (i+1)), heads=num_heads[i])
            semantic_adapter = nn.Sequential(
                                    nn.Linear(768, 768// 4),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(768// 4, int(embed_dim * 2 ** (i+1))),
                                    nn.ReLU(inplace=True),
                                    ) 
            distortion_adapter = nn.Sequential(
                                    nn.Linear(128, 128// 4),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(128// 4, int(embed_dim * 2 ** (i+1))),
                                    nn.ReLU(inplace=True),
                                    )
            
            self.semantic_adapter.append(semantic_adapter)
            self.distortion_adapter.append(distortion_adapter)
            self.semantic_mod.append(semantic_transform2)
            self.distortion_mod.append(distortion_transform3)
            self.semantic_cross.append(attn_cross)
            self.distortion_cross.append(attn_cross1)
            self.distortion_self.append(attn_self)
            
      
        self._freeze_stages()

        self.init_weights()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location="cpu")
        state_dict = checkpoint["model"]

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [
            k for k in state_dict.keys() if "relative_position_index" in k
        ]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict["patch_embed.proj.weight"] = (
            state_dict["patch_embed.proj.weight"]
            .unsqueeze(2)
            .repeat(1, 1, self.patch_size[0], 1, 1)
            / self.patch_size[0]
        )

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [
            k for k in state_dict.keys() if "relative_position_bias_table" in k
        ]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            wd = self.window_size[0]
            if nH1 != nH2:
                print(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(
                            1, nH1, S1, S1
                        ),
                        size=(
                            2 * self.window_size[1] - 1,
                            2 * self.window_size[2] - 1,
                        ),
                        mode="bicubic",
                    )
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(
                        nH2, L2
                    ).permute(
                        1, 0
                    )
            state_dict[k] = relative_position_bias_table_pretrained.repeat(
                2 * wd - 1, 1
            )

        msg = self.load_state_dict(state_dict, strict=False)
        print(msg)
        print(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def load_swin(self, load_path, strict=False):
        print("loading swin lah")
        from collections import OrderedDict

        model_state_dict = self.state_dict()
        state_dict = torch.load(load_path)["state_dict"]

        clean_dict = OrderedDict()
        for key, value in state_dict.items():
            if "backbone" in key:
                clean_key = key[9:]
                clean_dict[clean_key] = value
                if "relative_position_bias_table" in clean_key:
                    forked_key = clean_key.replace(
                        "relative_position_bias_table", "fragment_position_bias_table"
                    )
                    if forked_key in clean_dict:
                        print("load_swin_error?")
                    else:
                        clean_dict[forked_key] = value

        ## Clean Mismatched Keys
        for key, value in model_state_dict.items():
            if key in clean_dict:
                if value.shape != clean_dict[key].shape:
                    print(key)
                    clean_dict.pop(key)

        print('load swin from kk:',self.load_state_dict(clean_dict, strict=strict))

    def init_weights(self, pretrained=None):
        print(self.pretrained, self.pretrained2d)
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
        
            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights()
            else:
                # Directly load 3D model.
                self.load_swin(self.pretrained, strict=False)  # , logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")
    def obtain_keyframes(self,x):
       
        b, c,t,h,w = x.shape
        x=x.permute(0,2,1,3,4)#b,t,c,h,w
        device=x.device
         
        key_frame=torch.cat((x[:,0,...]\
                             .unsqueeze(1),x[:,t//4-1,...]\
                             .unsqueeze(1),x[:,t//2-1,...]\
                             .unsqueeze(1),x[:,t*3//4-1,...]\
                             .unsqueeze(1)),dim=1)
        group_idx=x.new_zeros((b,t))
        for i in range(b):
            group_id=0
            for j in range(t):#30
                    
                    if j==t//4-1:
                        group_id=group_id+1
                    elif j==t//2-1:
                        group_id=group_id+1
                    elif j==t*3//4-1:
                        group_id=group_id+1
                    group_idx[i,j]=torch.tensor(group_id).to(device) #0001122333 #00111111
                  
        return group_idx,key_frame
    
    def extend_fullcls_attn(self,cls_attn,group_id):
        
        B,N_key,L = cls_attn.shape
        B,T = group_id.shape
        full_cls_attn=cls_attn.new_zeros((B,T,L))
        for i in range(B):
            for j in range(T):
                    full_cls_attn[i,j] = cls_attn[i,int(group_id[i,j].item())]
        return full_cls_attn
        

    def forward(self, x, multi=False, layer=-1, adaptive_window_size=False):
       
        #print('current a1:',self.a1,'a2',self.a2)
        """Forward function."""
        if adaptive_window_size:
            resized_window_size = get_adaptive_window_size(
                self.window_size, x.shape[2:], self.base_x_size
            )
        else:
            resized_window_size = None
        revideo= x['resize_video']
        fragment= x['fragment']
        dis_label= x['dis_label']

        group_id,key_frame=self.obtain_keyframes(revideo) #B,N_key,h,w 
        b,N_key,c,h1,w1=key_frame.shape
        key_frame=key_frame.view(b*N_key,c,h1,w1)
        cls_attn1,cls_token,patch_token_list = self.CLIP_tool(key_frame) # B*N_key, h*w, num_layer*B*hw*D

        cls_attn1 = cls_attn1.view(b,N_key,-1)
        
        cls_token=cls_token.view(b,N_key,-1)
        
        num_layer,_,gridnum,_=patch_token_list.shape
        patch_token_list = patch_token_list.contiguous().view(num_layer*b,N_key,-1)
        full_patch_token_list=self.extend_fullcls_attn(patch_token_list,group_id)

        _,_,t,_,_=fragment.shape
        full_patch_token_list=full_patch_token_list.view(num_layer,b,t,gridnum,-1)
        
        #QRS
        x_sel_ori= self.spa_patchnet(fragment,cls_attn1,self.sigma,group_id) 
        print(x_sel_ori.shape)
        x_sel = self.patch_embed(x_sel_ori)
       
        #obtain distortion feature
        dist_token_ori = self.distortion_tool(x_sel_ori.detach()[:,:,::2,...])
        dist_token_ori = 0.2*self.dist_adapter(dist_token_ori)+0.8*dist_token_ori
        #obtain distortion contrastive loss
        dis_contrast_loss = distortion_contrastive_supervised(dist_token_ori,dis_label)
        
        x_sel = self.pos_drop(x_sel)
        feats = [x_sel]
        fragment_size=fragment.size(-1)//32
        for l, mlayer in enumerate(self.layers):
            #print(l)
            x_sel = mlayer(x_sel.contiguous(),fragment_size, resized_window_size)
            if l>=self.tuning_stage:
                    #CDM (modulation)
                    
                    #semantic modulation
                    x_sel_fors = x_sel
                    if full_patch_token_list.shape[0]>1:
                       print('pat cls num:',full_patch_token_list.shape[0])
                       patch_token= full_patch_token_list[l-self.tuning_stage]
                    else:
                       patch_token= full_patch_token_list[0]
                    
                    #projection
                    b,t,num_grid,_ = patch_token.shape
                    patch_token = patch_token[:,::2,...]
                    patch_token= patch_token.view(b*t//2,num_grid,_)
                    patch_token=self.semantic_adapter[l-self.tuning_stage](patch_token)
                    n,c,t,h,w = x_sel_fors.shape
                    x_sel_fors = x_sel_fors.contiguous().permute(0,2,1,3,4).contiguous().view(n*t,c,h*w).permute(0,2,1) #n*t,h*w,c
                    #semantic cross attention
                    patch_token_enhanced,_ =self.semantic_cross[l-self.tuning_stage](x_sel_fors,patch_token)
                    
                    patch_token_enhanced = patch_token_enhanced.permute(0,2,1).view(n*t,c,h,w)
                    x_sel_fors = x_sel_fors.permute(0,2,1).view(n*t,c,h,w)
                    x_sel_fors = self.semantic_mod[l-self.tuning_stage](patch_token_enhanced,x_sel_fors) ##n*t,h*w
                    x_sel_fors = x_sel_fors.contiguous().view(n,t,c,h,w).permute(0,2,1,3,4)

                    #distortion modulation 
                    x_sel_ford = x_sel
                    #projection
                    b,t,num_grid,_ = dist_token_ori.shape
                    dist_token_ori= dist_token_ori.view(b*t,num_grid,_)
                    dist_token=self.distortion_adapter[l-self.tuning_stage](dist_token_ori) #B,T,num_grid,C
                    dist_token_ori= dist_token_ori.view(b,t,num_grid,_)
                    dist_token=dist_token.contiguous().view(b*t,num_grid,-1)
                    n,c,t,h,w = x_sel_ford.shape
                    x_sel_ford = x_sel_ford.contiguous().permute(0,2,1,3,4).contiguous().view(n*t,c,h*w).permute(0,2,1) #n*t,h*w,c
                    #distortion cross 2D attention
                    dist_token_enhanced,_ =self.distortion_cross[l-self.tuning_stage](x_sel_ford,dist_token) #n*t,h*w,c
                    dist_token_enhanced =dist_token_enhanced.contiguous().view(n,t,h*w,c).permute(0,2,1,3).contiguous().view(n*h*w,t,c)
                    dist_token_enhanced =self.distortion_self[l-self.tuning_stage](dist_token_enhanced)
                    dist_token_enhanced = dist_token_enhanced.contiguous().view(n,h*w,t,c).permute(0,3,2,1).contiguous().view(n,c,t,h,w)
                    #semantci modulation
                    x_sel_ford = x_sel_ford.contiguous().view(n,t*h*w,c) 
                    x_sel_ford = self.distortion_mod[l-self.tuning_stage](dist_token_enhanced,x_sel_ford) ##n,c
                    x_sel_ford = x_sel_ford.contiguous().view(n,t,h,w,c).permute(0,4,1,2,3)
                    
                    x_sel=(self.a1[l-self.tuning_stage]*x_sel_ford+self.a2[l-self.tuning_stage]*x_sel_fors)/2
                    #x_sel=(self.a1[l-self.tuning_stage]*x_sel_ford+self.a2[l-self.tuning_stage]*x_sel_fors+x_sel)#/3
 
            feats += [x_sel]

        x_sel = rearrange(x_sel, "n c d h w -> n d h w c")
        x_sel = self.norm(x_sel)
        x_sel = rearrange(x_sel, "n d h w c -> n c d h w")
        if multi:
            shape = x_sel.shape[2:]
            return torch.cat(
                [F.interpolate(xi, size=shape, mode="trilinear") for xi in feats[:-1]],
                1,
            )
        elif layer > -1:
            print("something", len(feats))
            return feats[layer]
        else:
            return x_sel,dis_contrast_loss

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(KSVQE, self).train(mode)
        self._freeze_stages()


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
       
       
        dim_head=dim//heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) #if project_out else nn.Identity()

    def forward(self, x,mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        '''
        if mask is not None:
            dots_mask = rearrange(~mask, 'b i -> b 1 i 1') * rearrange(~mask, 'b j -> b 1 1 j')
            dots_mask=  repeat(dots_mask,'b () n n1 -> b h n n1',h=dots.shape[1])
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(~dots_mask, mask_value)
        '''
        if mask is not None:
            #dots_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(mask, 'b j -> b 1 1 j')
            mask=rearrange(mask, 'b i -> b 1 1 i')
            #print(dots_mask[0])
            dots_mask=  repeat(mask,'b 1 () n1 -> b 1 n n1',n=dots.shape[2])
            dots_mask=  repeat(dots_mask,'b () n n1 -> b h n n1',h=dots.shape[1])
            mask_value = -torch.finfo(dots.dtype).max
            #print(~dots_mask[0])
            dots = dots.masked_fill(~dots_mask, mask_value)
        attn = dots.softmax(dim=-1)
        #print(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class crossattention1(nn.Module):
    def __init__(self, dim, num_heads, ln=False):
             #ref :MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        super(crossattention1, self).__init__()
        self.dim_V = dim
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim, dim)
        self.fc_k = nn.Linear(dim, dim)
        self.fc_v = nn.Linear(dim, dim)
       
       
    def forward(self, Q, K):
        #Q=Q.permute(1,0,2)
        #K=K.permute(1,0,2)
        B,N_r,C=K.shape
      
        Q = self.fc_q(Q)
        B,N_p,C=Q.shape
        K, V = self.fc_k(K), self.fc_v(K)
        #alpha = self.alpha_chooser()
        B,N,C=K.shape
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0) #batch_size*4,num_inds,dim_split
        K_ = torch.cat(K.split(dim_split, 2), 0) # batchsize*4,num_patch,dim_split
        V_ = torch.cat(V.split(dim_split, 2), 0) #batchsize*4,num_patch,dim_split
        #A = entmax_bisect(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V),alpha[0],dim=2)
        A = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V) #batch_size*4,num_inds,num_patch
        
        A = torch.softmax(A,2)

        O = torch.cat((A.bmm(V_)).split(Q.size(0), 0), 2)
        
        A = A.view(B,self.num_heads,N_p,N).mean(dim=1)
        return O,A
    
def swin_3d_tiny(**kwargs):
    ## Original Swin-3D Tiny with reduced windows
    return SwinTransformer3D(depths=[2, 2, 6, 2], frag_biases=[0, 0, 0, 0], **kwargs)


def swin_3d_small(**kwargs):
    # Original Swin-3D Small with reduced windows
    return SwinTransformer3D(depths=[2, 2, 18, 2], frag_biases=[0, 0, 0, 0], **kwargs)


class SwinTransformer2D(nn.Sequential):
    def __init__(self):
        ## Only backbone for Swin Transformer 2D
        from timm.models import swin_tiny_patch4_window7_224

        super().__init__(*list(swin_tiny_patch4_window7_224().children())[:-2])




def get_network(name, pretrained=False):
    network = {
        "VGG16": torchvision.models.vgg16(pretrained=pretrained),
        "VGG16_bn": torchvision.models.vgg16_bn(pretrained=pretrained),
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet34": torchvision.models.resnet34(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
	"resnet101": torchvision.models.resnet101(pretrained=pretrained),
	"resnet152": torchvision.models.resnet152(pretrained=pretrained),
    }
    if name not in network.keys():
        raise KeyError(f"{name} is not a valid network architecture")
    return network[name]
        
class CONTRIQUE_model(nn.Module):
    # resnet50 architecture with projector
    def __init__(self,  encoder, n_features,anchor_size=32, \
                 patch_dim = (2,2), normalize = True, projection_dim = 128):
        super(CONTRIQUE_model, self).__init__()
        self.anchor_size=anchor_size
        self.normalize = normalize
        self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.n_features = n_features
        self.patch_dim = patch_dim
        
       
        # MLP for projector
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.BatchNorm1d(self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
        )
        
    def forward(self, x):
        # fragment patch
        b,c,t,h,w=x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w', b=b, h=h)
        x = (x.contiguous()
            .view(b*t,c,h//self.anchor_size,self.anchor_size,w//self.anchor_size,self.anchor_size)
            .permute(0,2,4,1,3,5) 
            .contiguous()
            .view(b*t*h//self.anchor_size* (w//self.anchor_size),c,self.anchor_size,self.anchor_size))
        num_grid = h//self.anchor_size* (w//self.anchor_size)
        h = self.encoder(x)
       
        h = h.view(-1, self.n_features)
       
        if self.normalize:
            h = nn.functional.normalize(h, dim=1)
            
        
        # global projections
        z = self.projector(h)
        z =z.view(b,t,num_grid,-1)
       
        return z 
def distortion_contrastive_supervised( distortion_feature,dis_label):
      #aassert(len(dis_label)==1)
      b,t,num_grid,_ = distortion_feature.shape
      distortion_feature= distortion_feature.view(b*t*num_grid,-1)
      #distortion class
      dist_label = (dis_label.unsqueeze(1).repeat(1,b)==dis_label).to(torch.float32).to(distortion_feature.device)#torch.eye(b)
      #expend to fragment
      #print(dist_label)
      dist_labels = dist_label.repeat(1,t*num_grid).view(b*t*num_grid,-1)
      #dist_label = dist_label.to(distortion_feature.device)
      #calculate similaroty and divide by temperature parameter
      z=nn.functional.normalize(distortion_feature,p=2,dim=1)
      sim = torch.mm(z,z.T)/0.1 
      #dist_labels=dist_labels.cpu()
      positive_mask = torch.mm(dist_labels.to_sparse(), dist_labels.T)
      positive_mask = positive_mask.fill_diagonal_(0).to(sim.device)
    
      N=b*t*num_grid
      zero_diag = torch.ones((N, N)).fill_diagonal_(0).to(sim.device)

      # calculate normalized cross entropy value
      positive_sum = torch.sum(positive_mask, dim=1)
      denominator = torch.sum(torch.exp(sim)*zero_diag,dim=1)
      loss = torch.mean(torch.log(denominator) - \
                          (torch.sum(sim * positive_mask, dim=1)/positive_sum))
      return loss
            
def distortion_contrastive( distortion_feature):
      b,t,num_grid,_ = distortion_feature.shape
      distortion_feature= distortion_feature.view(b*t*num_grid,-1)
      #distortion class
      dist_label = torch.eye(b)
      #expend to fragment
     
      dist_labels = dist_label.repeat(1,t*num_grid).view(b*t*num_grid,-1)
      #dist_label = dist_label.to(distortion_feature.device)
      #calculate similaroty and divide by temperature parameter
      z=nn.functional.normalize(distortion_feature,p=2,dim=1)
      sim = torch.mm(z,z.T)/0.1 
      #dist_labels=dist_labels.cpu()
      positive_mask = torch.mm(dist_labels.to_sparse(), dist_labels.T)
      positive_mask = positive_mask.fill_diagonal_(0).to(sim.device)
    
      N=b*t*num_grid
      zero_diag = torch.ones((N, N)).fill_diagonal_(0).to(sim.device)

      # calculate normalized cross entropy value
      positive_sum = torch.sum(positive_mask, dim=1)
      denominator = torch.sum(torch.exp(sim)*zero_diag,dim=1)
      loss = torch.mean(torch.log(denominator) - \
                          (torch.sum(sim * positive_mask, dim=1)/positive_sum))

      return loss
          
    
      