# -*- coding: utf-8 -*-
"""V3 空间特征分类模型：可切换骨干、多区域 SE、可选 Transformer 和解耦分类头。

注册实验的 SE 区域、骨干和解冻时点见 model_registry.py，当前均为 NoTF。
支持 ResNeXt50、ResNet18/34、ShuffleNetV2 1.0×，保留正层数 Transformer 构造。
ResNetTransformerV2 类名和 model='v2' 为现有实现接口。
"""
import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d, resnet18, resnet34, shufflenet_v2_x1_0

from src_v3 import merged_dict
from src_v3.taxonomy import EXPECTED_NUM_CLASSES
from src_v3.SE_Block import SEBlock as NewSEBlock


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
        # 分类路径不消费注意力权重；显式关闭返回可避免无用的 [B,N,N] 张量。
        x = x + self.drop_path1(self.attn(x_norm, x_norm, x_norm, need_weights=False)[0])
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
                 backbone_name='resnext50_32x4d',
                 pretrained=True,
                 backbone_bn_policy='frozen_stats',
                 pos_embed_type='1d',   # 位置编码（次级消融 V2-1Dpos vs V2-2Dpos）：'1d' | '2d' 行列解耦
                 ):
        super().__init__()

        if renew_class_to_index is None:
            raise ValueError('renew_class_to_index 不能为空；V2分类头必须显式绑定当前标签映射')
        if not isinstance(transformer_layers, int) or transformer_layers < 0:
            raise ValueError(f'transformer_layers 必须为非负整数，收到 {transformer_layers!r}')
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f'd_model 必须为正整数，收到 {d_model!r}')
        if not isinstance(nhead, int) or nhead <= 0 or d_model % nhead != 0:
            raise ValueError(f'nhead 必须为正整数且整除 d_model，收到 d_model={d_model}, nhead={nhead}')
        if not isinstance(num_patches, int) or num_patches <= 0:
            raise ValueError(f'num_patches 必须为正整数，收到 {num_patches!r}')

        self.renew_class_to_index = renew_class_to_index
        self.id_to_main_class = id_to_main_class
        self.merged_dict = merged_dict or {}
        num_parent_classes = len(renew_class_to_index)
        builders = {'resnext50_32x4d': resnext50_32x4d, 'resnet18': resnet18,
                    'resnet34': resnet34, 'shufflenet_v2_x1_0': shufflenet_v2_x1_0}
        if backbone_name not in builders:
            raise ValueError(f'不支持的骨干: {backbone_name}')
        if backbone_bn_policy != 'frozen_stats':
            raise ValueError('当前对照统一使用frozen_stats骨干BN策略')
        self.backbone_name = backbone_name
        self.backbone_bn_policy = backbone_bn_policy
        self.transformer_layers = transformer_layers
        self.d_model = d_model
        self.nhead = nhead
        self.num_patches = num_patches
        self.regions = list(regions) if regions is not None else None
        self.use_decouple = bool(use_decouple)
        self.model_impl_version = 4  # 区域池化采用FP32累加，拒绝混用修复前断点。
        self.ffn_dim = 4 * d_model
        self.transformer_dropout = 0.3
        self.patch_dropout = 0.1
        self.head_dropout = 0.5
        self.drop_path_rate = float(drop_path_rate)

        # 保留骨干空间特征，去掉原分类读出。
        backbone = builders[backbone_name](pretrained=pretrained)
        in_features = backbone.fc.in_features
        self.feature_dim = in_features
        if backbone_name == 'shufflenet_v2_x1_0':
            # Shuffle 的 conv5 位于 fc 前，必须保留其 1024 通道输出。
            self.resnet_backbone = nn.Sequential(
                backbone.conv1, backbone.maxpool, backbone.stage2,
                backbone.stage3, backbone.stage4, backbone.conv5,
            )
        else:
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
        grid = int(round(num_patches ** 0.5))
        self.expected_grid = (grid, grid) if grid * grid == num_patches else None
        if pos_embed_type == '1d':
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)  # 1D learned（ViT-B 做法）
            self.row_emb = self.col_emb = None
        else:  # '2d' 行列解耦：位置(r,c) = row[r] + col[c]
            assert grid * grid == num_patches, f'2D 编码需要 num_patches 为平方数，收到 {num_patches}'
            self.row_emb = nn.Parameter(torch.randn(1, grid, d_model) * 0.02)
            self.col_emb = nn.Parameter(torch.randn(1, grid, d_model) * 0.02)
            self.pos_embed = None

        # [CLS] token（次级消融 V2-Mean vs V2-CLS）：use_cls_token=True 时启用
        self.use_cls_token = use_cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02) if use_cls_token else None

        self.dropout1 = nn.Dropout(self.patch_dropout)     # patch-embedding dropout（原 0.5 偏高）
        self.dropout2 = nn.Dropout(self.head_dropout)

        # Transformer 开关：layers=0 时直通（消融 G-NoTF / 无 Transformer 基线）
        # 改进（相对 nn.TransformerEncoder）：pre-LN + GELU FFN + 随机深度 DropPath
        self.transformer = TransformerEncoderStack(
            num_layers=transformer_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=self.ffn_dim,
            dropout=self.transformer_dropout,
            drop_path_rate=self.drop_path_rate,
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
        actual_patches = H * W
        if actual_patches != self.num_patches:
            raise RuntimeError(
                f'V2 token数量不匹配：配置要求 {self.num_patches}，backbone实际输出 '
                f'{H}x{W}={actual_patches}。请同步检查输入尺寸、backbone和num_patches。')
        if self.expected_grid is not None and (H, W) != self.expected_grid:
            raise RuntimeError(
                f'V2空间网格不匹配：配置要求 {self.expected_grid[0]}x{self.expected_grid[1]}，'
                f'实际为 {H}x{W}。禁止仅凭token总数相同静默改变二维布局。')
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

    def architecture_metadata(self):
        """返回可写入checkpoint/results的稳定结构元数据，不含运行态对象。"""
        return {
            'architecture': type(self).__name__,
            'model_impl_version': self.model_impl_version,
            'backbone': self.backbone_name,
            'feature_dim': self.feature_dim,
            'backbone_bn_policy': self.backbone_bn_policy,
            'se_parameters': sum(p.numel() for p in self.se_block.parameters()),
            'projection_parameters': sum(p.numel() for p in self.projection.parameters()),
            'num_classes': self.parent_classifier.out_features,
            'regions': self.regions,
            'transformer_layers': self.transformer_layers,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'ffn_dim': self.ffn_dim,
            'num_patches': self.num_patches,
            'pos_embed_type': self.pos_embed_type,
            'readout': 'cls' if self.use_cls_token else 'mean',
            'use_decouple': self.use_decouple,
            'transformer_norm': 'pre_ln_without_final_norm',
            'ffn_activation': 'gelu',
            'transformer_dropout': self.transformer_dropout,
            'patch_dropout': self.patch_dropout,
            'head_dropout': self.head_dropout,
            'drop_path_rate': self.drop_path_rate,
        }

    def train(self, mode=True):
        super().train(mode)
        # 所有实验统一固定骨干BN运行统计；头部BN仍正常训练。
        for module in self.resnet_backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

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

    def get_grouped_params(self, backbone_lr, head_lr, include_frozen=False):
        """按 backbone / head 分组返回优化器参数组（与原版一致）。"""
        backbone_ids = [id(p) for p in self.resnet_backbone.parameters()]
        backbone_params, head_params = [], []
        for param in self.parameters():
            if include_frozen or param.requires_grad:
                (backbone_params if id(param) in backbone_ids else head_params).append(param)
        return [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr},
        ]


if __name__ == '__main__':
    # 快速自检：构建 + 前向，验证多 token 修正后 shape 正确（不参与训练）。
    # 用法：在项目根目录执行
    #   python src_v3/ResNetTransformer.py
    d_model, nhead = 512, 8
    fake_class_index = {i: i for i in range(EXPECTED_NUM_CLASSES)}
    model = ResNetTransformerV2(transformer_layers=0, pretrained=False, d_model=d_model, nhead=nhead,
                                renew_class_to_index=fake_class_index)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert tuple(out.shape) == (1, EXPECTED_NUM_CLASSES)
    assert tuple(model.pos_embed.shape) == (1, 49, d_model)
    print('V2_SELFTEST_OK', {
        'output_shape': tuple(out.shape),
        'pos_embed_shape': tuple(model.pos_embed.shape),
        'metadata': model.architecture_metadata(),
    })
