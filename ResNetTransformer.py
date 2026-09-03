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

from src_v2 import merged_dict
from src_v2.SE_Block import SEBlock as NewSEBlock


class DropPath(nn.Module):
    """随机深度（Stochastic Depth）：训练时按概率丢弃整个残差分支，正则化防过拟合。

    评估模式恒等；训练时以 keep_prob 概率保留、并除以 keep_prob 保持期望不变。
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # x: [seq, batch, d_model] → 掩码 [1, batch, 1]，逐样本（batch）丢弃整个残差
        shape = (1, x.shape[1], 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random_tensor.floor() / keep_prob


class _TransformerBlock(nn.Module):
    """pre-LN Transformer 块：LN→多头注意力→DropPath，LN→FFN(GELU)→DropPath。

    相对 nn.TransformerEncoderLayer（post-LN、默认 relu、无 DropPath）：
      - pre-LN（norm_first）更稳、深层更鲁棒；
      - FFN 用 GELU（ViT 标准）；
      - 两个残差分支都加 DropPath（随机深度）。
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, drop_path_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.drop_path1 = DropPath(drop_path_rate)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(self, x):
        # x: [seq, batch, d_model]
        x_norm = self.norm1(x)
        x = x + self.drop_path1(self.attn(x_norm, x_norm, x_norm)[0])
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        return x


class TransformerEncoderStack(nn.Module):
    """堆叠多个 pre-LN 块组成的编码器（替代 nn.TransformerEncoder，支持随机深度）。"""
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout, drop_path_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            _TransformerBlock(d_model, nhead, dim_feedforward, dropout, drop_path_rate)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResNetTransformerV2(nn.Module):
    """修正版：ViT 式多 token Transformer（替代 ResNet.py 里 seq_len=1 的退化版）。

    与原版的核心差异：
      1) __init__ 多一个可学习位置编码 self.pos_embed；
      2) forward 不再 avgpool，而是把 7×7 特征图展平成 49 个 token 再进 Transformer。
    Transformer 部件做了三处改进（相对 nn.TransformerEncoder）：
      A) FFN 用 GELU（原默认 relu）；
      B) pre-LN（更稳、为将来加深留余地）；
      D) patch-embedding dropout 0.5→0.1 + 每层随机深度 DropPath。
    次级消融开关（做 V2 次级消融时用，见 task_plan 🧪 块）：
      E) use_cls_token=False/True → V2-Mean(mean-pool) vs V2-CLS([CLS] token 读出)；
      C) pos_embed_type='1d'/'2d' → V2-1Dpos(1D learned) vs V2-2Dpos(行列解耦)。
    其余（backbone / 多区域SE / 解耦层 / 分类头 / 渐进式解冻方法）与原版一致，保持可无缝替换。
    """
    def __init__(self, transformer_layers=3,
                 d_model=512, nhead=8,
                 id_to_main_class=None,
                 renew_class_to_index=None,
                 regions=[1, 2],       # 多区域SE开关: [1,2]=多区域 / [1]=标准SE / None=无SE(Identity)
                 use_decouple=True,     # 简化解耦层开关: False=直通
                 num_patches=49,        # ResNeXt50@224 → 7×7=49 个 patch token
                 drop_path_rate=0.1,    # 随机深度：每层残差丢弃概率
                 use_cls_token=False,   # 读出方式（次级消融 V2-Mean vs V2-CLS）：False=mean-pool，True=[CLS]
                 pos_embed_type='1d',   # 位置编码（次级消融 V2-1Dpos vs V2-2Dpos）：'1d' | '2d' 行列解耦
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

        # 位置编码（次级消融 V2-1Dpos vs V2-2Dpos）
        assert pos_embed_type in ('1d', '2d'), f'pos_embed_type 只支持 1d/2d，收到 {pos_embed_type}'
        self.pos_embed_type = pos_embed_type
        if pos_embed_type == '1d':
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)  # 1D learned（ViT-B 做法）
            self.row_emb = self.col_emb = None
        else:  # '2d' 行列解耦：位置(r,c) = row[r] + col[c]
            grid = int(round(num_patches ** 0.5))
            assert grid * grid == num_patches, f'2D 编码需要 num_patches 为平方数，收到 {num_patches}'
            self.row_emb = nn.Parameter(torch.randn(1, grid, d_model) * 0.02)
            self.col_emb = nn.Parameter(torch.randn(1, grid, d_model) * 0.02)
            self.pos_embed = None

        # [CLS] token（次级消融 V2-Mean vs V2-CLS）：use_cls_token=True 时启用
        self.use_cls_token = use_cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02) if use_cls_token else None

        self.dropout1 = nn.Dropout(0.1)     # patch-embedding dropout（ViT 标准 0.1；原 0.5 偏高）
        self.dropout2 = nn.Dropout(0.5)

        # Transformer 开关：layers=0 时直通（消融 G-NoTF / 无 Transformer 基线）
        # 改进（相对 nn.TransformerEncoder）：pre-LN + GELU FFN + 随机深度 DropPath
        self.transformer = TransformerEncoderStack(
            num_layers=transformer_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.3,
            drop_path_rate=drop_path_rate,
        ) if transformer_layers > 0 else nn.Identity()

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

        # 位置编码：1D（逐 token 加）或 2D（行列解耦 row[r]+col[c]）
        if self.pos_embed_type == '1d':
            x = x + self.pos_embed[:, :x.size(1)]
        else:
            pos2d = self.row_emb[:, :H].unsqueeze(2) + self.col_emb[:, :W].unsqueeze(1)  # [1, H, W, d_model]
            x = x + pos2d.flatten(1, 2)                  # [1, H*W, d_model] → 广播到 [B, N, d_model]

        x = self.dropout1(x)

        # 读出方式（次级消融 V2-Mean vs V2-CLS）
        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)        # [B, 1, d_model]
            x = torch.cat([cls, x], dim=1)                # [B, 1+N, d_model]

        x = self.transformer(x.transpose(0, 1))          # [1+N 或 N, B, d_model] 多 token 真自注意力
        x = x.transpose(0, 1)                            # [B, 1+N 或 N, d_model]

        if self.use_cls_token:
            global_feature = x[:, 0]                      # 取 [CLS] 输出作为全局特征
        else:
            global_feature = x.mean(dim=1)                # mean-pool 所有 patch token
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
