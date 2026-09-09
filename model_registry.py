# -*- coding: utf-8 -*-
"""
model_registry.py —— 模型"配置/名字/元数据"唯一注册表（2026-09-09 会话9 建立）

【为什么存在】此前模型定义散落 4 处、新增一个模型要改 4 个文件：
  - train.py 的 PRESETS（训练参数）          → 现为 MODEL_PARAMS 的别名导入
  - predict_gui_snoprog.py 的 MODEL_REGISTRY（推理参数+短说明）/ CN_NAMES（中文全称）/ TARGET_EPOCHS
  - run_queue.py 的 QUEUE 注释里的基线数字      → 队列顺序仍留 run_queue.py（本文件只管"模型是什么"）
  - ReportChart 各脚本对模型英文名的使用         → chart_data.py 经本文件校验/查中文名

【以后怎么改】
  - 改模型参数 / 中文名 / 家族 / 目标轮数 / 新增模型 → 只改本文件（在 MODELS 里加/改一个条目）
  - 选择"跑哪些模型、什么顺序" → 去 run_queue.py 改 QUEUE（引用的 id 必须是本文件注册过的）
  - ReportChart / GUI / train.py 均自动跟随（按 id 索引本文件），无需再各自登记
  - 新增模型 = MODELS 加条目（params 必须能被 train.py 用）+ run_queue QUEUE 加一行，即完成

【命名族谱（与 train.py / run_queue.py 语义一致，别混）】
  G- = 渐进式解冻（state1→2 @30轮、state2→3 @60轮 固定触发）；S- = 全解冻（unfreeze='none'）
  -CAWR = state2/3 或全程用 CAWR 调度（T_0=5, T_mult=2, eta_min=1e-6）；无后缀 = RLRP 旧时代（已过时仅留档）
  V2- = 修正版 Transformer 头（ResNetTransformerV2：7×7=49 patch token + pos_embed + patch-emb dropout 0.1，
        替代 v1 头 avgpool 1×1 单 token + dropout 0.5 + 退化 seq_len=1 注意力的组合）
  SE 开关：regions=[1,2]=多区域SE / None=无SE；transformer_layers=3（有栈）/ 0（去栈，V2-NoTF）
  ★ key 大小写规范：注册表与 runs/ 文件名一律用规范 key（如 'S-NoProg' 小写 g）；
    论文/叙述里的 "S-NoProG"（大写 G）只是显示写法，经 ALIASES 归一化到规范 key。

【当前状态基线（results JSON 权威，254 类/CAWR 口径）】
  S-NoProg 82.63@74 / S-NoSE 82.53@73（SE≈0.10pp）/ G-Full-CAWR 80.96@92（调度器修复后干净重跑）
  G-Full-CAWR-Long 83.565@132（160轮）/ V2-Full 83.353@74（vs S-NoProG +0.72pp，n=1 初步）
  V2-NoTF 跑中（09-09）/ G-V2 立项排队（2×2 第四格）
"""
from __future__ import annotations

# =====================================================================
# MODELS —— 唯一事实来源。字段：
#   params:        训练参数 dict（= 原 train.py PRESETS 条目，train.py 直接引用）
#   arch:          'v1'=ResNetTransformer（seq_len=1 退化）/ 'v2'=ResNetTransformerV2（49-token 真注意力）
#                  （留空则由 params['model']=='v2' 推导，默认 v1）
#   short:         短说明（GUI 报错/提示用）
#   cn:            中文全称（GUI 说明行 / 文档 / 汇报用）
#   target_epochs: 完整训练轮数目标（Long=160，其余 100；GUI 进度显示用）
#   note:          补充说明（状态/成绩，仅注释性质）
# =====================================================================
MODELS = {
    # —— G 系列：渐进式解冻（30/60 固定触发）——
    'G-Full': {
        'params': dict(regions=[1, 2], transformer_layers=3, use_decouple=True, unfreeze='progressive'),
        'short': '渐进式·RLRP旧基线',
        'cn': '渐进式解冻·RLRP调度·含多区域SE（历史基线）',
        'target_epochs': 100,
        'note': 'RLRP 时代主结果（56.88 过时，仅留档）',
    },
    'G-NoSE': {
        'params': dict(regions=None, transformer_layers=3, use_decouple=True, unfreeze='progressive'),
        'short': '渐进式·无SE·RLRP',
        'cn': '渐进式解冻·RLRP调度·无多区域SE（历史基线）',
        'target_epochs': 100,
        'note': 'RLRP 旧消融（已完成留档，勿再跑）',
    },
    'G-Full-CAWR': {
        'params': dict(regions=[1, 2], transformer_layers=3, use_decouple=True, unfreeze='progressive', scheduler='cawr'),
        'short': '渐进式+CAWR调度',
        'cn': '渐进式解冻·CAWR调度·含多区域SE',
        'target_epochs': 100,
        'note': '调度器修复后干净重跑 80.96@92（旧脏 70.82 已存档）',
    },
    'G-Full-CAWR-Long': {
        'params': dict(regions=[1, 2], transformer_layers=3, use_decouple=True, unfreeze='progressive', scheduler='cawr'),
        'short': '渐进式+CAWR·Long160轮',
        'cn': '渐进式解冻·CAWR调度·Long160轮·含多区域SE',
        'target_epochs': 160,
        'note': '100→160 断点续跑；best 83.565@132',
    },
    # —— S 系列：全解冻（unfreeze='none'，标准微调）——
    'S-NoProg': {
        'params': dict(regions=[1, 2], transformer_layers=3, use_decouple=True, unfreeze='none'),
        'short': '全程全解冻·含多区域SE',
        'cn': '全程全解冻·含多区域SE·CAWR调度',
        'target_epochs': 100,
        'note': '主基线 82.63@74（显示写法常作 S-NoProG，key 规范为小写 g）',
    },
    'S-NoSE': {
        'params': dict(regions=None, transformer_layers=3, use_decouple=True, unfreeze='none'),
        'short': '全程全解冻·无多区域SE',
        'cn': '全程全解冻·无多区域SE·CAWR调度（SE消融）',
        'target_epochs': 100,
        'note': 'SE 消融 82.53@73（SE 增益≈0.10pp）',
    },
    # —— V2 路线：修正版 49-token Transformer（v2 架构）——
    'V2-Full': {
        'params': dict(model='v2', regions=[1, 2], transformer_layers=3, use_decouple=True, unfreeze='none'),
        'arch': 'v2',
        'short': '修正版49-token注意力',
        'cn': 'V2修正Transformer·全程全解冻·含多区域SE（49-token真注意力）',
        'target_epochs': 100,
        'note': 'best 83.353@74（100轮预算当前最高；vs S-NoProg +0.72pp n=1）',
    },
    'V2-NoTF': {
        'params': dict(model='v2', regions=[1, 2], transformer_layers=0, use_decouple=True, unfreeze='none'),
        'arch': 'v2',
        'short': 'V2架构去Transformer',
        'cn': 'V2架构去Transformer·全程全解冻·含多区域SE（纯注意力消融对照）',
        'target_epochs': 100,
        'note': 'V2-Full − V2-NoTF = 纯注意力增益（跑中 09-09）',
    },
    'G-V2': {
        'params': dict(model='v2', regions=[1, 2], transformer_layers=3, use_decouple=True, unfreeze='progressive', scheduler='cawr'),
        'arch': 'v2',
        'short': '渐进式+V2修正头',
        'cn': 'V2修正Transformer·渐进式解冻30/60·含多区域SE（2×2第四格）',
        'target_epochs': 100,
        'note': '2026-09-09 立项：2×2 网格第四格（头 v1/V2 × 解冻 全/渐进），队列排 V2-NoTF 后',
    },
    # —— 已弃用/暂缓（不再进队列；保留条目仅为历史推理入口/文档）——
    # 'G-NoSE-CAWR': 渐进式无SE(弃用：SE消融须对准全解冻主模型)
    # 'G-NoTF':      v1 头去Transformer(弃用：v1 头已整体退场，由 V2-NoTF 方案取代)
    # 'G-SingleSE'/'G-PureBB': 单尺度SE/纯骨干（暂缓）
}

# 规范顺序（GUI 下拉无成绩项、文档展示按此序）
ORDER = ['G-Full', 'G-NoSE', 'S-NoProg', 'S-NoSE', 'G-Full-CAWR',
         'G-Full-CAWR-Long', 'V2-Full', 'V2-NoTF', 'G-V2']

# 显示写法 → 规范 key（论文/口头常用 S-NoProG，文件与注册表用 S-NoProg）
ALIASES = {'S-NoProG': 'S-NoProg'}

# 推理/预测 GUI 下拉默认模型
DEFAULT_MODEL = 'V2-Full'

# ---------------------------------------------------------------------
# 派生字典（供旧代码原位替换：train.py 用 MODEL_PARAMS(原 PRESETS)，
# GUI 用 MODEL_CN / MODEL_SHORT / MODEL_TARGET_EPOCHS）
# ---------------------------------------------------------------------
MODEL_PARAMS = {k: v['params'] for k, v in MODELS.items()}
MODEL_CN = {k: v['cn'] for k, v in MODELS.items()}
MODEL_SHORT = {k: v['short'] for k, v in MODELS.items()}
MODEL_ARCH = {k: v.get('arch', 'v2' if v['params'].get('model') == 'v2' else 'v1') for k, v in MODELS.items()}
MODEL_TARGET_EPOCHS = {k: v['target_epochs'] for k, v in MODELS.items()}


# ---------------------------------------------------------------------
# 工具函数（消费方不要直接 import 上面大字典，用这些；取不到时优雅回退）
# ---------------------------------------------------------------------
def normalize(name: str) -> str:
    """显示写法 → 规范 key（未知名称原样返回，由调用方自行报错）。"""
    return ALIASES.get(name, name)


def known_models() -> list:
    """按规范顺序返回全部已注册模型 id。"""
    return list(ORDER)


def is_known(name: str) -> bool:
    return normalize(name) in MODELS


def params_of(name: str) -> dict:
    """某模型训练参数（= 原 PRESETS 条目）。"""
    return dict(MODELS[normalize(name)]['params'])


def cn_of(name: str) -> str:
    """中文全称（GUI 说明行/文档）。"""
    cid = normalize(name)
    return MODEL_CN.get(cid, cid)


def short_of(name: str) -> str:
    """短说明。"""
    cid = normalize(name)
    return MODEL_SHORT.get(cid, cid)


def arch_of(name: str) -> str:
    """模型家族：'v1' | 'v2'。"""
    cid = normalize(name)
    return MODEL_ARCH.get(cid, 'v1')


def regions_of(name: str):
    return params_of(name).get('regions')


def tf_of(name: str) -> int:
    return params_of(name).get('transformer_layers', 0)


def dec_of(name: str) -> bool:
    return params_of(name).get('use_decouple', False)


def target_of(name: str) -> int:
    cid = normalize(name)
    return MODEL_TARGET_EPOCHS.get(cid, 100)


if __name__ == '__main__':
    # 自检：打印注册表总览（只读，无 torch/GPU 依赖）
    print(f'[model_registry 自检] 共 {len(MODELS)} 个模型，默认 {DEFAULT_MODEL}')
    for cid in ORDER:
        m = MODELS[cid]
        print(f'  {cid:22s} arch={MODEL_ARCH[cid]:<3s} tf={m["params"].get("transformer_layers",0)} '
              f'regions={m["params"].get("regions")} dec={m["params"].get("use_decouple")} '
              f'target={m["target_epochs"]}轮\n'
              f'      cn : {m["cn"]}\n'
              f'      note: {m["note"]}')
