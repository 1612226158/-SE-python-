import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet34, resnet101, resnet18
from torchvision.models import resnet101
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision import datasets, transforms
from PIL import Image
import os
import re


class SEBlock(nn.Module):
    """
    多区域 Squeeze-and-Excitation 模块

    支持两种模式：
    - regions=[1]: 标准SE模块（全局平均池化）
    - regions=[1,2]: 多区域SE（全局 + 四象限）
    - regions=[1,2,4]: 空间金字塔SE（全局 + 四象限 + 16区域）

    参数:
        in_channels: 输入通道数
        reduction: 降维比例，默认16
        regions: 区域配置列表，默认[1,2]表示1x1全局+2x2四象限
    """

    def __init__(self, in_channels, reduction=16, regions=None):
        super(SEBlock, self).__init__()

        # 默认使用多区域配置：全局 + 四象限
        if regions is None:
            regions = [1, 2, 4]

        self.in_channels = in_channels
        self.regions = regions
        self.reduction = reduction

        # ============ 多区域池化层 ============
        # 为每个区域级别创建自适应平均池化
        self.region_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(output_size=(r, r))
            for r in regions
        ])

        # 计算区域总数和拼接后的特征维度
        # regions=[1,2] -> 1×1 + 2×2 = 5个区域
        self.num_regions = sum([r * r for r in regions])
        self.total_features = in_channels * self.num_regions

        # ============ 激励网络（MLP） ============
        # 计算中间层维度，确保能被整除且不会太小
        reduced_channels = max(in_channels // reduction, 8)

        # 根据是否多区域选择不同的MLP结构
        if len(regions) == 1 and regions[0] == 1:
            # 标准SE：输入维度 = in_channels
            self.fc1 = nn.Linear(in_channels, reduced_channels, bias=False)
            self.fc2 = nn.Linear(reduced_channels, in_channels, bias=False)
        else:
            # 多区域SE：输入维度 = in_channels × num_regions
            # 使用更合理的中间维度
            mid_features = max(self.total_features // reduction, reduced_channels)
            self.fc1 = nn.Linear(self.total_features, mid_features, bias=False)
            self.fc2 = nn.Linear(mid_features, in_channels, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # 打印配置信息（初始化时）
        # print(f"SEBlock 配置: regions={regions}, "
        #       f"num_regions={self.num_regions}, "
        #       f"total_features={self.total_features}")

    def forward(self, x):
        batch_size, channels, H, W = x.size()

        # ============ Squeeze阶段：多区域池化 ============
        region_features = []

        for pool in self.region_pools:
            # 池化：(B, C, H, W) -> (B, C, r, r)
            pooled = pool(x)
            # 展平：(B, C, r, r) -> (B, C × r × r)
            flattened = pooled.view(batch_size, -1)
            region_features.append(flattened)

        # 拼接所有区域特征
        # regions=[1,2]: (B, C×1) + (B, C×4) -> (B, C×5)
        if len(region_features) == 1:
            y = region_features[0]
        else:
            y = torch.cat(region_features, dim=1)

        # ============ Excitation阶段：通道权重学习 ============
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)

        # ============ Scale阶段：重新校准 ============
        # (B, C) -> (B, C, 1, 1)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)

        # 通道加权
        return x * y

    def extra_repr(self):
        """打印模块信息"""
        return (f'in_channels={self.in_channels}, '
                f'reduction={self.reduction}, '
                f'regions={self.regions}')
