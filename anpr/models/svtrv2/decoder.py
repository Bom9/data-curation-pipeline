"""RCTC decoder for SVTRv2.

Extracted from OpenOCR (https://github.com/Topdu/OpenOCR).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from anpr.models.svtrv2.common import Mlp


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RCTCDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 6625,
        return_feats: bool = False,
        **kwargs: object,
    ):
        super().__init__()
        self.char_token = nn.Parameter(
            torch.zeros([1, 1, in_channels], dtype=torch.float32),
            requires_grad=True,
        )
        trunc_normal_(self.char_token, mean=0, std=0.02)
        self.fc = nn.Linear(in_channels, out_channels, bias=True)
        self.fc_kv = nn.Linear(in_channels, 2 * in_channels, bias=True)
        self.w_atten_block = Block(
            dim=in_channels, num_heads=in_channels // 32, mlp_ratio=4.0, qkv_bias=False
        )
        self.out_channels = out_channels
        self.return_feats = return_feats

    def forward(
        self, x: torch.Tensor, data: object = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        x = (
            self.w_atten_block(x.permute(0, 2, 3, 1).reshape(-1, W, C))
            .reshape(B, H, W, C)
            .permute(0, 3, 1, 2)
        )

        x_kv = self.fc_kv(x.flatten(2).transpose(1, 2)).reshape(B, H * W, 2, C).permute(2, 0, 3, 1)
        x_k, x_v = x_kv.unbind(0)

        char_token = self.char_token.tile([B, 1, 1])
        attn_ctc2d = char_token @ x_k
        attn_ctc2d = attn_ctc2d.reshape([-1, 1, H, W])
        attn_ctc2d = F.softmax(attn_ctc2d, 2)
        attn_ctc2d = attn_ctc2d.permute(0, 3, 1, 2)
        x_v = x_v.reshape(B, C, H, W)
        feats = attn_ctc2d @ x_v.permute(0, 3, 2, 1)
        feats = feats.squeeze(2)

        predicts = self.fc(feats)

        if self.return_feats:
            return feats, predicts

        if not self.training:
            predicts = F.softmax(predicts, dim=2)

        return predicts
