import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (resnet50, resnet34,
                                resnet101, resnet18,
                                resnext101_32x8d,
                                resnext50_32x4d,
                                wide_resnet101_2)
from torchvision.models import resnet101
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision import datasets, transforms
from PIL import Image
import os
import re

from src_v2 import merged_dict
from src_v2.SE_Block import SEBlock as NewSEBlock


def get_train_folder_len():
    path = '../train'
    items = os.listdir(path)
    # 过滤出文件夹
    folders = [item for item in items if os.path.isdir(os.path.join(path, item))]
    # print(folders)
    delete = []
    for idx, key in enumerate(folders):
        if not key.isdigit():
            delete.append(idx)

    for index in sorted(delete, reverse=True):  # 从后往前删除，避免索引变化
        del folders[index]

    num_classes = len(folders)

    # print(folders)
    return num_classes


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        assert in_channels % reduction == 0, "in_channels must be divisible by reduction."

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        # 全局平均池化并压缩形状
        y = self.global_avg_pool(x).flatten(1)  # 形状: [batch_size, channels]

        # 两层全连接操作
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)

        # 注意力权重并调整维度
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)

        # 通道加权
        return x * y


# 注意力机制的主模块
class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)  # 权重
        return weights * x


class ResNetTransformer(nn.Module):
    def __init__(self, transformer_layers=3,
                 d_model=512, nhead=8,
                 id_to_main_class=None,
                 renew_class_to_index=None,
                 regions=[1, 2],       # 多区域SE开关: [1,2]=多区域 / [1]=标准SE / None=无SE(Identity)
                 use_decouple=True,     # 简化解耦层开关: False=直通
                 ):
        super().__init__()

        # self.id_to_main_class = id_to_main_class
        self.renew_class_to_index = renew_class_to_index

        # 存储映射关系
        self.id_to_main_class = id_to_main_class
        # self.id_to_child_class = id_to_child_class
        # self.main_class_to_index = main_class_to_index
        # self.child_class_to_index = child_class_to_index
        # self.index_to_main_class = {idx: cls for cls, idx in main_class_to_index.items()}
        self.merged_dict = merged_dict or {}
        # 计算父类数量
        # num_parent_classes = len(main_class_to_index)

        # 计算父类数量
        num_parent_classes = len(renew_class_to_index)
        # 加载预训练的 ResNet-34 并移除分类头
        # backbone = resnet34(pretrained=True)
        # 尝试使用 ResNet-50
        backbone = resnext50_32x4d(pretrained=True)
        # 尝试使用 ResNet-18
        # backbone = resnet18(pretrained=True)
        # 尝试使用 ResNet-101
        # backbone = resnet101(pretrained=True)
        # 尝试使用 Wide-ResNet-101
        # backbone = wide_resnet101_2(pretrained=True)
        # self.resnet_backbone = nn.Sequential(*list(backbone.children())[:-2])
        in_features = backbone.fc.in_features

        self.resnet_backbone = nn.Sequential(*list(backbone.children())[:-2])
        # 多区域SE 开关：regions=None 时用 Identity 直通（消融 G-NoSE / G-PureBB）
        self.se_block = NewSEBlock(
            in_channels=in_features,
            reduction=16,
            regions=regions
        ) if regions else nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(in_features, d_model)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=nhead,
                                                dropout=0.3)
        # Transformer 开关：layers=0 时直通（消融 G-NoTF / G-PureBB）
        self.transformer = TransformerEncoder(encoder_layer,
                                              num_layers=transformer_layers) if transformer_layers > 0 else nn.Identity()
        # 父类分类器
        self.parent_classifier = nn.Linear(d_model, num_parent_classes)

        # # 特征解耦层
        # self.disentangle_layer = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),  # 降维压缩
        #     nn.BatchNorm1d(d_model // 2),
        #     nn.GELU(),
        #     nn.Dropout(0.6),
        #     nn.Linear(d_model // 2, d_model),  # 恢复原始维度
        #     nn.BatchNorm1d(d_model),
        #     nn.GELU(),
        #     nn.Dropout(0.5)
        # )
        # 简化解耦层（use_decouple=False 时直通：消融 G-PureBB）
        self.disentangle_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.5),  # 统一 dropout
            nn.Linear(d_model // 2, d_model),
        ) if use_decouple else nn.Identity()
        self.freeze_backbone()

    def forward(self, x):
        # ResNet 特征提取
        x = self.resnet_backbone(x)
        x = self.se_block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 投影到 Transformer 的输入维度
        features = self.projection(x).unsqueeze(1)
        features = self.dropout1(features)  # 添加dropout

        # Transformer 编码
        transformer_output = self.transformer(features.permute(1, 0, 2))

        # 全局特征
        global_feature = transformer_output.mean(dim=0)
        global_feature = self.dropout2(global_feature)  # 添加dropout
        disentangled_feature = self.disentangle_layer(global_feature)

        # 父类预测
        parent_output = self.parent_classifier(disentangled_feature)

        # # 子类预测（为每个父类预测其子类）
        # child_outputs = {}
        # for parent_idx, child_classifier in self.child_classifiers.items():
        #     # print(f"{self.child_classifiers}")
        #     # print(f"{child_classifier=}")
        #     child_outputs[parent_idx] = child_classifier(global_feature)

        # return parent_output, child_outputs
        # 只返回父类
        return parent_output

    def freeze_backbone(self):
        for name, param in self.resnet_backbone.named_parameters():
            param.requires_grad = False

    # 解冻阶段2
    def unfreeze_layer4(self):
        for p in self.resnet_backbone[7].parameters():
            p.requires_grad = True
        # for name, module in self.resnet_backbone.named_children():
        #     if "layer4" in name:
        #         for p in module.parameters():
        #             p.requires_grad = True

    # 解冻阶段3
    def unfreeze_all(self):
        for p in self.resnet_backbone.parameters():
            p.requires_grad = True

    def get_grouped_params(self, backbone_lr, head_lr):
        """
        根据 self.resnet_backbone 自动区分参数组
        """
        # 1. 记录 Backbone 中所有参数的内存地址 (ID)
        backbone_ids = [id(p) for p in self.resnet_backbone.parameters()]

        backbone_params = []
        head_params = []

        # 2. 遍历整个模型的所有参数
        for param in self.parameters():
            if param.requires_grad:  # 只处理解冻的参数
                if id(param) in backbone_ids:
                    # 如果 ID 在 backbone 列表里，归为 backbone 组
                    backbone_params.append(param)
                else:
                    # 其他所有参数（Head, BN层, 自定义的FC等）归为 Head 组
                    head_params.append(param)

        # 3. 返回给优化器的格式
        return [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr}
        ]


# 解决图像透明或图像读取异常的类
class ConvertToRGBError:
    # 解决带有透明图片的问题
    """
    UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
    """

    def __call__(self, img):
        if img.mode in ("P", "PA"):  # 检查是否为调色板模式
            img = img.convert("RGBA")  # 转为 RGBA 模式
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            # 如果是 RGBA 或者带透明度的图片，转换为 RGB
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'PA':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[3])  # 使用 alpha 通道作为 mask
            return background
        return img.convert('RGB')  # 其他情况直接转换为 RGB


if __name__ == '__main__':
    """
    ResNetTransformer 就可以结合 ResNet-101 的强大特征提取能力以及 Transformer 的建模能力，适用于复杂分类任务。
    """
    # 创建一个ResNet18模型
    # model = SimpleResNet(ResNetBlock, [2, 2, 2, 2], num_classes=get_train_folder_len())

    d_model = 512
    transformer_layers = 6
    nhead = 8
    model = ResNetTransformer(transformer_layers=transformer_layers,
                              d_model=d_model,
                              nhead=nhead)
    print(model)
    print(get_train_folder_len())
    print(sum(len(lis) for lis in merged_dict.values()) - len(merged_dict))
