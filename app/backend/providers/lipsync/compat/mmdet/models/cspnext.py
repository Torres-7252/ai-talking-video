"""CSPNeXt layers required by MuseTalk's bundled RTMPose checkpoint.

The architecture matches MMDetection 3.1.0 while avoiding unrelated
MMCV compiled detection operators that are unavailable on Python 3.12/Windows.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS


class ChannelAttention(BaseModule):
    def __init__(self, channels: int, init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            output = self.global_avgpool(value)
        return value * self.act(self.fc(output))


class CSPNeXtBlock(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = False,
        kernel_size: int = 5,
        conv_cfg=None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="SiLU"),
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = conv(
            in_channels,
            hidden_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv2 = DepthwiseSeparableConvModule(
            hidden_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.conv2(self.conv1(value))
        return output + value if self.add_identity else output


class CSPLayer(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        use_depthwise: bool = False,
        use_cspnext_block: bool = False,
        channel_attention: bool = False,
        conv_cfg=None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="Swish"),
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if not use_cspnext_block:
            raise ValueError("The MuseTalk compatibility layer requires CSPNeXt blocks")
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels, mid_channels, 1, conv_cfg=conv_cfg,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.short_conv = ConvModule(
            in_channels, mid_channels, 1, conv_cfg=conv_cfg,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.blocks = nn.Sequential(*[
            CSPNeXtBlock(
                mid_channels,
                mid_channels,
                1.0,
                add_identity,
                use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            for _ in range(num_blocks)
        ])
        self.attention = ChannelAttention(2 * mid_channels) if channel_attention else None
        self.final_conv = ConvModule(
            2 * mid_channels, out_channels, 1, conv_cfg=conv_cfg,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = torch.cat((self.blocks(self.main_conv(value)), self.short_conv(value)), dim=1)
        if self.attention is not None:
            output = self.attention(output)
        return self.final_conv(output)


class SPPBottleneck(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (5, 9, 13),
        conv_cfg=None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="Swish"),
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(
            in_channels, mid_channels, 1, conv_cfg=conv_cfg,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2)
            for size in kernel_sizes
        ])
        self.conv2 = ConvModule(
            mid_channels * (len(kernel_sizes) + 1), out_channels, 1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv1(value)
        with torch.amp.autocast("cuda", enabled=False):
            value = torch.cat([value] + [pool(value) for pool in self.poolings], dim=1)
        return self.conv2(value)


@MODELS.register_module()
class CSPNeXt(BaseModule):
    arch_settings = {
        "P5": [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 3, False, True]],
        "P6": [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]],
    }

    def __init__(
        self,
        arch: str = "P5",
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        arch_ovewrite=None,
        spp_kernel_sizes: Sequence[int] = (5, 9, 13),
        channel_attention: bool = True,
        conv_cfg=None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="SiLU"),
        norm_eval: bool = False,
        init_cfg: dict = dict(
            type="Kaiming", layer="Conv2d", a=math.sqrt(5),
            distribution="uniform", mode="fan_in", nonlinearity="leaky_relu"
        ),
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        settings = arch_ovewrite or self.arch_settings[arch]
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        first_channels = int(settings[0][0] * widen_factor)
        self.stem = nn.Sequential(
            ConvModule(3, first_channels // 2, 3, padding=1, stride=2,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(first_channels // 2, first_channels // 2, 3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(first_channels // 2, first_channels, 3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.layers = ["stem"]
        for index, (in_channels, out_channels, blocks, identity, use_spp) in enumerate(settings):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            stage = [conv(
                in_channels, out_channels, 3, stride=2, padding=1,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
            )]
            if use_spp:
                stage.append(SPPBottleneck(
                    out_channels, out_channels, spp_kernel_sizes,
                    conv_cfg, norm_cfg, act_cfg
                ))
            stage.append(CSPLayer(
                out_channels,
                out_channels,
                num_blocks=max(round(blocks * deepen_factor), 1),
                add_identity=identity,
                use_depthwise=use_depthwise,
                use_cspnext_block=True,
                expand_ratio=expand_ratio,
                channel_attention=channel_attention,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ))
            name = f"stage{index + 1}"
            self.add_module(name, nn.Sequential(*stage))
            self.layers.append(name)

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            for index in range(self.frozen_stages + 1):
                module = getattr(self, self.layers[index])
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, _BatchNorm):
                    module.eval()

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = []
        for index, name in enumerate(self.layers):
            value = getattr(self, name)(value)
            if index in self.out_indices:
                outputs.append(value)
        return tuple(outputs)
