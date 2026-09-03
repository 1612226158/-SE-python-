# -*- coding: utf-8 -*-
"""
ResNetTransformer.py —— 修正版 Transformer 分类模型（独立路线，当前【暂不使用】）

【这个文件是什么】
    把 ResNet.py 里的 ResNetTransformer 复制过来，改正了其中"Transformer 退化"的问题：
    - 原版（ResNet.py）：backbone → avgpool(压成 1×1) → projection → unsqueeze(1) → Transformer。
      喂给 Transformer 的序列长度 = 1，自注意力退化成"恒等 + 输出投影"，等价于一个花哨 MLP，
      与后面的解耦层/分类头重复，增益 ≈ 0。
    - 本版（ResNetTransformerV2）：backbone → SE → 展平成 H*W 个 patch token（7×7=49）→ projection
      → +可学习位置编码 → Transformer（真正的多 token 自注意力）→ 平均池化 → 解耦层 → 分类头。
      Transformer 现在能建模 49 个空间小块之间的细腻关系，契合"垃圾分类要关注小块细节"的动机。

【当前状态：先放着不用，别改、别删】
    - 正在跑的 12 个 G 系列 + 之后 2 个 Long 实验，用的都是 ResNet.py 里的旧模型（seq_len=1 那版），
      本文件【不参与】当前任何实验，train.py 也【不导入】本文件。
    - 不要删 ResNet.py，也不要动 src_v2 里任何正在跑的代码。

【之后的规划（等 12 个 G 系列 + 2 个 Long 全跑完再做，见 task_plan 的"🔬 Long 验证实验"块与决策记录）】
    1. 作为"Transformer-v2 独立路线"立项：把 train.py 里 `from ResNet import ResNetTransformer`
       改为 `from ResNetTransformer import ResNetTransformerV2`（并相应处理 PRESETS / 配置切换）。
    2. 重新做 Transformer 消融（加/不加 Transformer 的对照，此时对照才有意义），
       以及必要的 2 种子 + 可能的长轮数验证。
    3. 论文里 Transformer 的相关表述（现已决定"弱化/退场"）届时再根据新结果决定是否升级为卖点。

【参数说明】
    - transformer_layers / d_model / nhead：与原版一致（默认 3 / 512 / 8）。
    - num_patches：backbone 输出的 patch 数，ResNeXt50@224 输入 → 7×7 = 49（改输入尺寸需同步改）。
    - regions / use_decouple：SE 与解耦层开关，与原版语义一致（用于消融）。
"""
import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from src_v2 import merged_dict
from src_v2.SE_Block import SEBlock as NewSEBlock


class ResNetTransformerV2(nn.Module):
    """修正版：ViT 式多 token Transformer（替代 ResNet.py 里 seq_len=1 的退化版）。

    与原版的差异只有两点：
      1) __init__ 多一个可学习位置编码 self.pos_embed；
      2) forward 不再 avgpool，而是把 7×7 特征图展平成 49 个 token 再进 Transformer。
    其余（backbone / 多区域SE / 解耦层 / 分类头 / 渐进式解冻方法）与原版一致，保持可无缝替换。
    """
    def __init__(self, transformer_layers=3,
                 d_model=512, nhead=8,
                 id_to_main_class=None,
                 renew_class_to_index=None,
                 regions=[1, 2],       # 多区域SE开关: [1,2]=多区域 / [1]=标准SE / None=无SE(Identity)
                 use_decouple=True,     # 简化解耦层开关: False=直通
                 num_patches=49,        # ResNeXt50@224 → 7×7=49 个 patch token
                 ):
        super().__init__()

        self.renew_class_to_index = renew_class_to_index
        self.id_to_main_class = id_to_main_class
        self.merged_dict = merged_dict or {}
        num_parent_classes = len(renew_class_to_index)

        # 加载预训练 ResNeXt50 并去掉分类头（去掉 fc + avgpool，保留 7×7 特征图）
        backbone = resnext50_32x4d(pretrained=True)
        in_features = backbone.fc.in_features          # 2048
        self.resnet_backbone = nn.Sequential(*list(backbone.children())[:-2])

        # 多区域 SE（regions=None 时直通，用于消融 G-NoSE / G-PureBB）
        self.se_block = NewSEBlock(
            in_channels=in_features,
            reduction=16,
            regions=regions
        ) if regions else nn.Identity()

        # 关键修正：不做 avgpool，直接把每个空间位置当 patch token
        self.projection = nn.Linear(in_features, d_model)                    # 每个 patch: C → d_model
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)  # 可学习位置编码
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=nhead,
                                                dropout=0.3)
        # Transformer 开关：layers=0 时直通（消融用）
        self.transformer = TransformerEncoder(encoder_layer,
                                              num_layers=transformer_layers) if transformer_layers > 0 else nn.Identity()

        # 父类分类器
        self.parent_classifier = nn.Linear(d_model, num_parent_classes)

        # 简化解耦层（use_decouple=False 时直通：消融 G-PureBB）
        self.disentangle_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(d_model // 2, d_model),
        ) if use_decouple else nn.Identity()

        self.freeze_backbone()

    def forward(self, x):
        x = self.resnet_backbone(x)                      # [B, C, H, W]
        x = self.se_block(x)                             # [B, C, H, W]（多区域 SE，regions=None 则直通）
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)                 # [B, H*W, C] = N 个 patch token
        x = self.projection(x)                           # [B, N, d_model]
        x = x + self.pos_embed[:, :x.size(1)]            # 加位置编码（按实际 token 数截取，容错）
        x = self.dropout1(x)
        x = self.transformer(x.transpose(0, 1))          # [N, B, d_model] 多 token 真自注意力
        x = x.transpose(0, 1)                            # [B, N, d_model]
        # 平均池化所有 patch token 得到全局特征（也可改用 [CLS] token，此处先用 mean 简化）
        global_feature = x.mean(dim=1)                   # [B, d_model]
        global_feature = self.dropout2(global_feature)
        disentangled_feature = self.disentangle_layer(global_feature)
        parent_output = self.parent_classifier(disentangled_feature)
        return parent_output

    # ---- 渐进式解冻（与原版一致，供 train.py 复用） ----
    def freeze_backbone(self):
        for p in self.resnet_backbone.parameters():
            p.requires_grad = False

    def unfreeze_layer4(self):
        for p in self.resnet_backbone[7].parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.resnet_backbone.parameters():
            p.requires_grad = True

    def get_grouped_params(self, backbone_lr, head_lr):
        """按 backbone / head 分组返回优化器参数组（与原版一致）。"""
        backbone_ids = [id(p) for p in self.resnet_backbone.parameters()]
        backbone_params, head_params = [], []
        for param in self.parameters():
            if param.requires_grad:
                (backbone_params if id(param) in backbone_ids else head_params).append(param)
        return [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr},
        ]


if __name__ == '__main__':
    # 快速自检：构建 + 前向，验证多 token 修正后 shape 正确（不参与训练）。
    # 用法：在项目根 E:\DataSet\垃圾分类图片-2 下执行
    #   python src_v2/ResNetTransformer.py
    d_model, nhead = 512, 8
    fake_class_index = {i: i for i in range(254)}   # 254 类占位，自检只关心前向 shape
    model = ResNetTransformerV2(transformer_layers=3, d_model=d_model, nhead=nhead,
                                renew_class_to_index=fake_class_index)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print('前向输出 shape:', tuple(out.shape), '(应为 (2, 254))')
    print('位置编码可学习参数数量:', model.pos_embed.numel())
