"""SVTRv2 encoder (SVTRv2LNConvTwo33).

Extracted from OpenOCR (https://github.com/Topdu/OpenOCR).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_

from anpr.models.svtrv2.common import DropPath, Identity, Mlp


class ConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        groups: int = 1,
        act: type = nn.GELU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


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
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.dim)
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
        eps: float = 1e-6,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop_path(self.mixer(x)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


class FlattenBlockRe2D(Block):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = super().forward(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
        eps: float = 1e-6,
        num_conv: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer = nn.Sequential(
            *[
                nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=num_heads)
                for _ in range(num_conv)
            ]
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C, H, W = x.shape[1:]
        x = x + self.drop_path(self.mixer(x))
        x = self.norm1(x.flatten(2).transpose(1, 2))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        x = x.transpose(1, 2).reshape(-1, C, H, W)
        return x


class FlattenTranspose(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)


class SubSample2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: list[int] | None = None):
        super().__init__()
        if stride is None:
            stride = [2, 1]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, sz: list[int]) -> tuple[torch.Tensor, list[int]]:
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        x = x.transpose(1, 2).reshape(-1, C, H, W)
        return x, [H, W]


class SubSample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: list[int] | None = None):
        super().__init__()
        if stride is None:
            stride = [2, 1]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, sz: list[int]) -> tuple[torch.Tensor, list[int]]:
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(-1, C, sz[0], sz[1])
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, [H, W]


class IdentitySize(nn.Module):
    def forward(self, x: torch.Tensor, sz: list[int]) -> tuple[torch.Tensor, list[int]]:
        return x, sz


class SVTRStage(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        out_dim: int = 256,
        depth: int = 3,
        mixer: list[str] | None = None,
        kernel_sizes: list[int] | None = None,
        sub_k: list[int] | None = None,
        num_heads: int = 2,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path: list[float] | None = None,
        norm_layer: type = nn.LayerNorm,
        act: type = nn.GELU,
        eps: float = 1e-6,
        num_conv: list[int] | None = None,
        downsample: bool | None = None,
        **kwargs: object,
    ):
        super().__init__()
        if mixer is None:
            mixer = ["Local"] * 3
        if kernel_sizes is None:
            kernel_sizes = [3] * 3
        if sub_k is None:
            sub_k = [2, 1]
        if drop_path is None:
            drop_path = [0.1] * 3
        if num_conv is None:
            num_conv = [2] * 3

        self.dim = dim
        self.blocks = nn.Sequential()

        for i in range(depth):
            if mixer[i] == "Conv":
                self.blocks.append(
                    ConvBlock(
                        dim=dim,
                        kernel_size=kernel_sizes[i],
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        act_layer=act,
                        drop_path=drop_path[i],
                        norm_layer=norm_layer,
                        eps=eps,
                        num_conv=num_conv[i],
                    )
                )
            else:
                if mixer[i] == "Global":
                    block = Block
                elif mixer[i] == "FGlobal":
                    block = Block
                    self.blocks.append(FlattenTranspose())
                elif mixer[i] == "FGlobalRe2D":
                    block = FlattenBlockRe2D
                else:
                    block = Block
                self.blocks.append(
                    block(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        act_layer=act,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path[i],
                        norm_layer=norm_layer,
                        eps=eps,
                    )
                )

        if downsample:
            if mixer[-1] in ("Conv", "FGlobalRe2D"):
                self.downsample = SubSample2D(dim, out_dim, stride=sub_k)
            else:
                self.downsample = SubSample1D(dim, out_dim, stride=sub_k)
        else:
            self.downsample = IdentitySize()

    def forward(self, x: torch.Tensor, sz: list[int]) -> tuple[torch.Tensor, list[int]]:
        for blk in self.blocks:
            x = blk(x)
        x, sz = self.downsample(x, sz)
        return x, sz


class ADDPosEmbed(nn.Module):
    def __init__(self, feat_max_size: list[int] | None = None, embed_dim: int = 768):
        super().__init__()
        if feat_max_size is None:
            feat_max_size = [8, 32]
        pos_embed = torch.zeros(
            [1, feat_max_size[0] * feat_max_size[1], embed_dim], dtype=torch.float32
        )
        trunc_normal_(pos_embed, mean=0, std=0.02)
        self.pos_embed = nn.Parameter(
            pos_embed.transpose(1, 2).reshape(1, embed_dim, feat_max_size[0], feat_max_size[1]),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sz = x.shape[2:]
        x = x + self.pos_embed[:, :, : sz[0], : sz[1]]
        return x


class POPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        feat_max_size: list[int] | None = None,
        embed_dim: int = 768,
        use_pos_embed: bool = False,
        flatten: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        if feat_max_size is None:
            feat_max_size = [8, 32]
        self.patch_embed = nn.Sequential(
            ConvBNLayer(
                in_channels=in_channels,
                out_channels=embed_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias=bias,
            ),
            ConvBNLayer(
                in_channels=embed_dim // 2,
                out_channels=embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias=bias,
            ),
        )
        if use_pos_embed:
            self.patch_embed.append(ADDPosEmbed(feat_max_size, embed_dim))
        if flatten:
            self.patch_embed.append(FlattenTranspose())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        sz = x.shape[2:]
        x = self.patch_embed(x)
        return x, [sz[0] // 4, sz[1] // 4]


class Feat2D(nn.Module):
    def forward(self, x: torch.Tensor, sz: list[int]) -> tuple[torch.Tensor, list[int]]:
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(-1, C, sz[0], sz[1])
        return x, sz


class LastStage(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, last_drop: float, out_char_num: int = 0
    ):
        super().__init__()
        self.last_conv = nn.Linear(in_channels, out_channels, bias=False)
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=last_drop)

    def forward(self, x: torch.Tensor, sz: list[int]) -> tuple[torch.Tensor, list[int]]:
        x = x.reshape(-1, sz[0], sz[1], x.shape[-1])
        x = x.mean(1)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        return x, [1, sz[1]]


class SVTRv2LNConvTwo33(nn.Module):
    def __init__(
        self,
        max_sz: list[int] | None = None,
        in_channels: int = 3,
        out_channels: int = 192,
        depths: list[int] | None = None,
        dims: list[int] | None = None,
        mixer: list[list[str]] | None = None,
        use_pos_embed: bool = True,
        sub_k: list[list[int]] | None = None,
        num_heads: list[int] | None = None,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        last_drop: float = 0.1,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: type = nn.LayerNorm,
        act: type = nn.GELU,
        last_stage: bool = False,
        feat2d: bool = False,
        eps: float = 1e-6,
        num_convs: list[list[int]] | None = None,
        kernel_sizes: list[list[int]] | None = None,
        pope_bias: bool = False,
        **kwargs: object,
    ):
        super().__init__()
        if max_sz is None:
            max_sz = [32, 128]
        if depths is None:
            depths = [3, 6, 3]
        if dims is None:
            dims = [64, 128, 256]
        if mixer is None:
            mixer = [["Conv"] * 3, ["Conv"] * 3 + ["Global"] * 3, ["Global"] * 3]
        if sub_k is None:
            sub_k = [[1, 1], [2, 1], [1, 1]]
        if num_heads is None:
            num_heads = [2, 4, 8]
        if num_convs is None:
            num_convs = [[2] * 3, [2] * 3 + [3] * 3, [3] * 3]
        if kernel_sizes is None:
            kernel_sizes = [[3] * 3, [3] * 3 + [3] * 3, [3] * 3]

        num_stages = len(depths)
        self.num_features = dims[-1]

        feat_max_size = [max_sz[0] // 4, max_sz[1] // 4]
        self.pope = POPatchEmbed(
            in_channels=in_channels,
            feat_max_size=feat_max_size,
            embed_dim=dims[0],
            use_pos_embed=use_pos_embed,
            flatten=mixer[0][0] != "Conv",
            bias=pope_bias,
        )

        dpr = np.linspace(0, drop_path_rate, sum(depths)).tolist()

        self.stages = nn.ModuleList()
        for i_stage in range(num_stages):
            ks = (
                kernel_sizes[i_stage]
                if len(kernel_sizes[i_stage]) == len(mixer[i_stage])
                else [3] * len(mixer[i_stage])
            )
            nc = (
                num_convs[i_stage]
                if len(num_convs[i_stage]) == len(mixer[i_stage])
                else [2] * len(mixer[i_stage])
            )
            stage = SVTRStage(
                dim=dims[i_stage],
                out_dim=dims[i_stage + 1] if i_stage < num_stages - 1 else 0,
                depth=depths[i_stage],
                mixer=mixer[i_stage],
                kernel_sizes=ks,
                sub_k=sub_k[i_stage],
                num_heads=num_heads[i_stage],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]) : sum(depths[: i_stage + 1])],
                norm_layer=norm_layer,
                act=act,
                downsample=i_stage != num_stages - 1,
                eps=eps,
                num_conv=nc,
            )
            self.stages.append(stage)

        self.out_channels = self.num_features
        self.last_stage = last_stage
        if last_stage:
            self.out_channels = out_channels
            self.stages.append(LastStage(self.num_features, out_channels, last_drop))
        if feat2d:
            self.stages.append(Feat2D())
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 5:
            x = x.flatten(0, 1)
        x, sz = self.pope(x)
        for stage in self.stages:
            x, sz = stage(x, sz)
        return x
