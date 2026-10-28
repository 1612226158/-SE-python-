# -*- coding: utf-8 -*-
"""
run_queue.py —— 训练协议 v2 正式实验队列（G / S / V2 系列自动排队器）

【本版定位（2026-09-26 重写）】
  已切换为「协议 v2 全量重跑」模式。旧协议（AMP 梯度裁剪前未 unscale）的全部产物已移出
  runs/ 根目录、归档至 runs/DIRTY_unscale_bug_20260926/。完成项跳过，有断点的未完成项续跑，
  新任务从 epoch 0 开始；完成判定同时核验正式结果身份和完整指标轮次。

用法（确认 PyCharm 没有手动训练在跑，二者互斥；必须用带 torch 的训练解释器）：
    D:\\Python\\Python3.10.7\\python.exe E:\\DataSet\\垃圾分类图片-2\\src_v2\\run_queue.py

它会按 QUEUE 顺序依次处理每个 (CONFIG_ID, SEED)：
  1. 正式 results 身份/协议正确，且指标覆盖完整目标轮次 → 跳过
     文件不可读、轮次冲突或指标已齐但缺 results → 停队报告，不自动重跑
  2. 正在训练（checkpoint_<tag>_last.pth 最近 15 分钟还在写）→ 等待 5 分钟重查
  3. Long 配置且自己无断点 → 从源配置断点复制续跑（ensure_long_resume，省重跑前 100 轮）
  4. 有断点但已停 → 自动 RESUME=true 续跑；无断点 → RESUME=false 全新跑
  5. 运行 train.py（阻塞等待自然结束）；失败自动用 RESUME=true 重试一次，再失败则停队
进度/报错写入 src_v2/runs/queue_log.txt；任务切换前经 temp_guard 温度守护。

【启动前置校验（本次新增）】_preflight() 在开跑前硬性检查 train.py 是否含
  `scaler.unscale_(optimizer)`（协议 v2 的唯一代码判别标志）；缺失即拒绝启动。
  这正是 2026-09 AMP 事故的护城河——防止代码回退后又产出一批不能进论文的脏数据。
"""
import glob
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import temp_guard  # 温度守护：任务切换前检测 CPU/GPU 温度，过高自动休息 10 分钟
import model_registry
from ReportChart.chart_data import all_metrics_files, DIRTY_RECORDS
from taxonomy import (
    EXPECTED_NUM_CLASSES,
    LABEL_SCHEMA_ID,
    LABEL_SCHEMA_DEFINITION_HASH,
    LABEL_SCHEMA_VERSION,
    load_current_label_schema,
)

SRC_V2 = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(SRC_V2, "runs")
CONFIG = os.path.join(SRC_V2, "experiment_config.json")
TRAIN = os.path.join(SRC_V2, "train.py")
LOG = os.path.join(RUNS, "queue_log.txt")
PYTHON = sys.executable  # 用启动本脚本的同一个 Python（保证 torch 1.11 环境）
PROJECT_ROOT = os.path.dirname(SRC_V2)
_, _, CURRENT_LABEL_MAPPING_HASH = load_current_label_schema(PROJECT_ROOT)

# —— 执行队列总览（2026-09-26 重写：协议 v2 全量重跑）——
# 模型"参数/中文名/家族/目标轮数"唯一真源 = src_v2\model_registry.py（train.py PRESETS / GUI / ReportChart 共用）；
# 本文件只维护 QUEUE 顺序与 Long 机制——"选队列/换顺序"改这里，"改模型定义/新增模型"改 model_registry.py。
#
# 【为什么要全部重跑】2026-09-25 发现 train.py 的 AMP 路径在 clip_grad_norm_ 之前漏了 scaler.unscale_()：
#   裁剪发生在缩放空间 ⇒ 每个 epoch 都被以 0.5/scale（≈1.9e-6~3.1e-5）的阈值裁剪，
#   比论文声明的 max_norm=0.5 小 4~5 个数量级。实测 runs 内 91/91 个 epoch 的 gradient_norm
#   恰好等于 0.5/2^n（2^14~2^18），指纹完全命中 ⇒ 修复前产出的所有结果都属「旧协议」，
#   与论文方法描述不符、不可复现，故整体重做。旧产物已移至
#   runs\DIRTY_unscale_bug_20260926\（内含 README_脏数据说明.txt 与 manifest）。
#
# 【命名速查（与 train.py PRESETS 顶部族谱一致）】
#   G- = 渐进式解冻（state1→2 @30轮、state2→3 @60轮）；S- = 全解冻（unfreeze='none'，标准微调）
#   -CAWR = CAWR 调度（T_0=5, T_mult=2, eta_min=1e-6）；无后缀 = RLRP 旧时代（已过时仅留档）
#   V2- = 修正版 Transformer 头（ResNetTransformerV2：7×7=49 patch token + pos_embed + patch-emb dropout 0.1，
#        替代 v1 头 avgpool 1×1 单 token + dropout 0.5 + 退化 seq_len=1 注意力的组合）
#   SE 消融 = S-NoSE（无SE）/ S-StdSE（标准全局SE，regions=[1]）/ S-NoProg（多区域SE，regions=[1,2]）
#
# 【队列顺序的设计依据】按"论文依赖强度"排序，而不是按字母或历史习惯：
#   ① 块 1（SE 三点链）——论文标题级归因。先跑主基线 S-NoProg（它是 V2/G 两条线的共同对照，
#      先出来才能解锁后面所有比较），再补 S-NoSE / S-StdSE。三点仅 regions 不同，
#      故可干净隔离"多区域描述符"相对"标准全局 SE"的净增益。
#   ② 块 2（V2 头归因）——V2-Full 在前、V2-NoTF 紧随：V2-Full − V2-NoTF = 纯注意力栈增益。
#   ③ 块 3（解冻策略对撞）——G-Full-CAWR（渐进式 + v1 头），对照 S-NoProg 回答"渐进式 vs 全解冻"。
#   ④ 块 4（训练预算）——G-Full-CAWR-Long **必须排在块 3 之后**：它靠 ensure_long_resume 复制
#      G-Full-CAWR 的 100 轮断点再续跑到 160 轮，省去重跑前 100 轮。
#      train.py 会校验训练/采样/标签协议，但不校验 CONFIG_ID；同一 252 类标签协议下的跨配置
#      Long 续跑仍可用。若源断点属于旧 254 类标签协议，会在加载模型前被明确拒绝。
#   ⑤ 块 5（2×2 第四格）——G-V2（V2 头 × 渐进式），补齐"头 v1/V2 × 解冻 全/渐进"网格。
#   ⑥ 块 6（seed2 复现组）——放最后：先把 seed1 的完整故事跑齐（解锁论文写作），
#      再用 seed2 把 n=1 升级为 mean±std（论文 seed2 列目前全为【待补】）。
#
# 【已退场，不再排】G-Full / G-NoSE（RLRP 时代旧基线，已被 G-Full-CAWR 取代）；
#   G-NoSE-CAWR（SE 消融须对准全解冻主模型）；G-NoTF（v1 头已整体退场，由 V2-NoTF 取代）；
#   G-SingleSE / G-PureBB（暂缓）。条目保留在 model_registry.py 中供历史推理入口使用。
#
# 【实测成本（834 s/轮）】块1~5 共 8 个 run ≈ 7.3 天连续 GPU（其中 Long 多跑 60 轮 ≈ +13.9 h）；
#   连块 6 的 7 个 seed2 共 15 个 run ≈ 14.2 天。
QUEUE = [
    # ===== 块 1：SE 三点归因链（仅 regions 不同；论文标题级结论）=====
    ("S-NoProg", 1),          # 全解冻·多区域SE（[1,2]）—— 主基线，V2/G 两条线的共同对照
    ("S-NoSE",   1),          # 全解冻·无SE（None）
    ("S-StdSE",  1),          # 全解冻·标准全局SE（[1]）—— 隔离"多区域"净增益的关键中间点

    # ===== 块 2：V2 修正头归因 =====
    ("V2-Full",  1),          # V2 头（49-token 真注意力）+ 多区域SE + 全解冻
    ("V2-NoTF",  1),          # V2 架构去 Transformer（transformer_layers=0）
                              #    ⇒ V2-Full − 本项 = 纯注意力栈增益
                              #    早期观测：V2-NoTF 拟合速度 ≈ V2-Full >> v1 头(S-NoProg)，
                              #    提示初期提速主要来自 V2 侧头部修正，注意力净增益待本项收官确认

    # ===== 块 3：解冻策略对撞 =====
    ("G-Full-CAWR", 1),       # 渐进式解冻(30/60) + CAWR + 多区域SE（v1 头）—— 对照 S-NoProg

    # ===== 块 4：训练预算（依赖块 3，顺序不可调换）=====
    # ("G-Full-CAWR-Long", 1),  # 160 轮预算；从 G-Full-CAWR 的 100 轮断点续跑（ensure_long_resume）

    # ===== 块 5：2×2 网格第四格 =====
    ("G-V2", 1),              # V2 修正头 × 渐进式解冻30/60 —— 补齐 头(v1/V2) × 解冻(全/渐进) 网格

    # ===== 块 6：seed2 复现组（n=1 → mean±std）=====
    ("S-NoProg", 2),
    ("S-NoSE", 2),
    ("S-StdSE", 2),
    ("V2-Full", 2),
    ("V2-NoTF", 2),
    ("G-Full-CAWR", 2),
    ("G-V2", 2),

    # ===== 已退场，不再排（保留注释仅为追溯；勿取消注释）=====
    # ("G-Full", 1),                           # RLRP 时代旧基线，已被 G-Full-CAWR 取代
    # ("G-NoSE", 1), ("G-NoSE", 2),            # RLRP 旧消融，已被 S-NoSE 取代
    # ("G-NoSE-CAWR", 1), ("G-NoSE-CAWR", 2),  # 渐进式无SE（SE 消融须对准全解冻主模型）
    # ("G-NoTF", 1), ("G-NoTF", 2),            # v1 头 Transformer 消融（v1 头已整体退场）
    # ("G-SingleSE", 1), ("G-SingleSE", 2),    # 单尺度 SE（暂缓）
    # ("G-PureBB", 1), ("G-PureBB", 2),        # 纯骨干（暂缓）
]

# 旧协议产物归档目录（2026-09-26 建立）；仅用于 _preflight() 的现场报告，不被本脚本读写
LEGACY_DIR = os.path.join(RUNS, "DIRTY_unscale_bug_20260926")
# 协议 v2 的唯一代码判别标志：有此调用才算"裁剪前已 unscale"
AMP_FIX_MARKER = "scaler.unscale_(optimizer)"
# 逐 epoch 采样协议的代码判别标志：同一实验 seed/绝对 epoch 的采样序列可重建。
EPOCH_SAMPLER_MARKER = "_derive_sampler_seed(SEED, epoch)"


# Long 配置：目标轮数（默认 100）。这类配置跑到 SNAPSHOT_EPOCH 轮时，train.py 会自动把该轮 checkpoint/metrics 复制存档。
LONG_CONFIGS = {
    "G-Full-CAWR-Long": 160,
}
SNAPSHOT_EPOCH = 100  # Long 配置跑到该轮时，自动存档 100 轮快照

# Long 配置 → 其续跑的"源配置"：从源配置的 100 轮断点续跑，省去重跑前 100 轮
LONG_BASE = {
    "G-Full-CAWR-Long": "G-Full-CAWR",
}

FRESH_SECONDS = 15 * 60  # checkpoint 最近 15 分钟有写 = 训练进行中
# 队列侧完成守卫；公共绘图工具不承担严格完成判定。
TRAINING_PROTOCOL_VERSION = 2
SAMPLING_PROTOCOL_VERSION = 1
SAMPLER_SEED_STRIDE = 1_000_000
_metrics_cache_signature = None
_metrics_cache = {}


class CompletionCheckError(RuntimeError):
    """产物状态未知或矛盾，必须停队，不能当作未完成自动开跑。"""


def _integer(value):
    if isinstance(value, bool):
        raise ValueError('布尔值不是整数身份字段')
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f'不是有限整数: {value!r}')
    return int(number)


def _matches(record, expected):
    """JSON数值和CSV字符串共用身份校验。"""
    for key, value in expected.items():
        actual = record.get(key)
        try:
            if isinstance(value, bool):
                if not isinstance(actual, bool):
                    return False
            elif isinstance(value, int):
                actual = _integer(actual)
        except (TypeError, ValueError, OverflowError):
            return False
        if actual != value:
            return False
    return True


def _label_fields():
    return {'label_schema_version': LABEL_SCHEMA_VERSION,
            'label_mapping_hash': CURRENT_LABEL_MAPPING_HASH,
            'num_classes': EXPECTED_NUM_CLASSES}


def _metrics_signature(files):
    try:
        signature = []
        for path in files:
            stat = os.stat(path)
            signature.append((path, stat.st_mtime_ns, stat.st_size))
        return tuple(signature)
    except OSError as error:
        raise CompletionCheckError(f'无法检查指标文件状态：{error}') from error


def _metrics_index():
    """按文件变化刷新索引；复用公共文件白名单/作废登记，不重复拼读取规则。"""
    global _metrics_cache_signature, _metrics_cache
    files = all_metrics_files()
    signature = _metrics_signature(files)
    if signature == _metrics_cache_signature:
        return _metrics_cache
    index = {}
    required = {'config_id', 'seed', 'epoch', 'sampler_seed', 'val_acc', *_label_fields()}
    for path in files:
        try:
            with open(path, encoding='utf-8-sig', newline='') as file:
                reader = csv.DictReader(file, strict=True)
                if not required.issubset(reader.fieldnames or []):
                    raise CompletionCheckError(f'{path} 缺少完成校验所需列')
                dirty = DIRTY_RECORDS.get(os.path.basename(path), set())
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError('CSV行列数不一致')
                    key = row['config_id'], _integer(row['seed'])
                    if key in dirty or not _matches(row, _label_fields()):
                        continue
                    epoch = _integer(row['epoch'])
                    if epoch < 0 or _integer(row['sampler_seed']) != key[1] * SAMPLER_SEED_STRIDE + epoch:
                        raise ValueError(f'{key} epoch={epoch} 的采样身份不一致')
                    accuracy = float(row['val_acc'])
                    if not math.isfinite(accuracy) or not 0 <= accuracy <= 100:
                        raise ValueError('验证准确率无效')
                    # 重复保存同一条指标可接受；同epoch的不同指标不能静默覆盖。
                    payload = {k: v for k, v in row.items() if k != 'save_time'}
                    entry = index.setdefault(key, {})
                    if epoch in entry and entry[epoch] != payload:
                        raise ValueError(f'{key} epoch={epoch} 有冲突记录')
                    entry[epoch] = payload
        except (OSError, UnicodeError, csv.Error, ValueError, TypeError) as error:
            raise CompletionCheckError(f'指标读取失败 {path}：{error}') from error
    if _metrics_signature(files) != signature or all_metrics_files() != files:
        raise CompletionCheckError('核验期间指标文件变化，请重试状态检查')
    _metrics_cache_signature, _metrics_cache = signature, index
    return index


def _result_record(config_id, seed, target_epoch):
    path = os.path.join(RUNS, f'results_{config_id}_seed{seed}.json')
    try:
        with open(path, encoding='utf-8-sig') as file:
            result = json.load(file)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompletionCheckError(f'结果读取失败 {path}：{error}') from error
    preset = model_registry.params_of(config_id)
    expected = {**_label_fields(), 'config_id': config_id, 'seed': seed,
                'label_schema_id': LABEL_SCHEMA_ID,
                'label_schema_definition_hash': LABEL_SCHEMA_DEFINITION_HASH,
                'training_protocol_version': TRAINING_PROTOCOL_VERSION,
                'sampling_protocol_version': SAMPLING_PROTOCOL_VERSION,
                'sampler_seed_scheme': f'run_seed*{SAMPLER_SEED_STRIDE}+epoch',
                'last_sampler_seed': seed * SAMPLER_SEED_STRIDE + target_epoch,
                'amp_gradient_clipping': 'unscale_before_clip',
                'unfreeze_mode': preset['unfreeze']}
    if not isinstance(result, dict) or not _matches(result, expected):
        raise CompletionCheckError(f'{path} 的实验身份、训练/采样/标签协议不匹配')
    try:
        start, end = map(_integer, result['epochs_run'])
        best_epoch = _integer(result['best_val_epoch'])
        best_accuracy = float(result['best_val_acc'])
        if not (0 <= start <= end == target_epoch and 0 <= best_epoch <= end
                and math.isfinite(best_accuracy) and 0 <= best_accuracy <= 100):
            raise ValueError('结果轮次或best指标无效')
        # 旧v1结果没有结构快照，不伪造恢复；已保存的快照必须与注册配置一致。
        if 'model_metadata' in result:
            metadata = result['model_metadata']
            expected_model = {k: preset[k] for k in ('regions', 'transformer_layers', 'use_decouple')}
            expected_model.update(num_classes=EXPECTED_NUM_CLASSES,
                                  architecture='ResNetTransformerV2' if preset.get('model') == 'v2'
                                  else 'ResNetTransformer')
            if not isinstance(metadata, dict) or not _matches(metadata, expected_model):
                raise ValueError('模型结构快照与注册配置不匹配')
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise CompletionCheckError(f'结果校验失败 {path}：{error}') from error
    return result


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _preflight():
    """开跑前校验：train.py 必须已是训练协议 v2（AMP 梯度裁剪前显式 unscale）。

    背景：2026-09 的事故根因就是 clip_grad_norm_ 之前缺 scaler.unscale_()，梯度在缩放空间
    被裁剪，实际阈值被缩放 2^14~2^18 倍。若代码回退到旧写法，本函数直接拒绝启动队列，
    避免又产出一批不能进论文正式结果矩阵的脏数据。
    返回 True 允许开跑；False 拒绝启动。
    """
    try:
        with open(TRAIN, encoding="utf-8") as f:
            src = f.read()
    except OSError as e:
        log(f"✖ 前置校验失败：读不到 {TRAIN}（{e}）")
        return False

    if AMP_FIX_MARKER not in src:
        log("✖ 前置校验失败：train.py 中未找到 `scaler.unscale_(optimizer)`。")
        log("  当前代码不是训练协议 v2；用它跑出的结果不得进论文正式结果矩阵。")
        log("  请先恢复 2026-09-25 的 AMP 修复（参见 毕业设计/2026-09-26_协议v2重跑与旧数据归档方案.md）后再启动队列。")
        return False
    log(f"✔ 前置校验通过：train.py 含 `{AMP_FIX_MARKER}`（训练协议 v2）")

    if EPOCH_SAMPLER_MARKER not in src:
        log("✖ 前置校验失败：train.py 未启用逐 epoch sampler seed。")
        log("  续跑会重放 WeightedRandomSampler 随机流开头，拒绝启动正式队列。")
        return False
    log("✔ 前置校验通过：WeightedRandomSampler 使用实验 seed + 绝对 epoch 独立播种")
    log(f"✔ 标签协议：{LABEL_SCHEMA_ID} v{LABEL_SCHEMA_VERSION} / "
        f"{EXPECTED_NUM_CLASSES}类 / hash={CURRENT_LABEL_MAPPING_HASH}")

    # 现场报告（不阻断，仅提示）：确认旧协议产物确已移出 runs\ 根目录
    n_res = len(glob.glob(os.path.join(RUNS, "results_*.json")))
    n_ckpt = len(glob.glob(os.path.join(RUNS, "checkpoint_*.pth")))
    n_mr = len(all_metrics_files())
    n_legacy = len(glob.glob(os.path.join(LEGACY_DIR, "*"))) if os.path.isdir(LEGACY_DIR) else 0
    log(f"  现场：runs\\ 根目录 results={n_res} / checkpoint={n_ckpt} / metrics_record={n_mr}"
        f"；归档档 {os.path.basename(LEGACY_DIR)} 内 {n_legacy} 项")
    if n_res or n_ckpt:
        log("  提示：done() 核验结果身份/协议及完整轮次，读取异常会停队，")
        log("     train.py 也会拒绝续跑旧 254 类 checkpoint。")
    return True


def done(config_id, seed):
    """True=可信完成，False=未完成；未知/矛盾状态抛异常，process负责停队。"""
    if not model_registry.is_known(config_id):
        raise CompletionCheckError(f'未知模型配置：{config_id}')
    target_epoch = model_registry.target_of(config_id) - 1
    index = _metrics_index()
    rows = dict(index.get((config_id, seed), {}))
    # Long的0—99允许沿用经严格核验的源实验；不要求续跑段重新生成前100轮。
    base = LONG_BASE.get(config_id)
    if base and rows and min(rows) >= SNAPSHOT_EPOCH and done(base, seed):
        rows = {**index.get((base, seed), {}), **rows}
    result = _result_record(config_id, seed, target_epoch)
    expected_epochs = set(range(target_epoch + 1))
    if result is None and target_epoch not in rows:
        return False
    if set(rows) != expected_epochs:
        missing = sorted(expected_epochs - rows.keys())
        extra = sorted(rows.keys() - expected_epochs)
        raise CompletionCheckError(f'{config_id} seed{seed} 轮次不完整/越界：缺{missing}，多{extra}')
    if result is None:
        raise CompletionCheckError(f'{config_id} seed{seed} 指标已齐但缺正式results，请核查训练收尾；不自动重跑')
    best_row = rows[_integer(result['best_val_epoch'])]
    reported_best = float(result['best_val_acc'])
    measured_best = max(float(row['val_acc']) for row in rows.values())
    if (abs(float(best_row['val_acc']) - reported_best) > 0.00006
            or abs(measured_best - reported_best) > 0.00006):
        raise CompletionCheckError(f'{config_id} seed{seed} results best与完整指标矛盾')
    return True


def in_progress(config_id, seed):
    p = os.path.join(RUNS, f"checkpoint_{config_id}_seed{seed}_last.pth")
    if not os.path.exists(p):
        return False
    return (time.time() - os.path.getmtime(p)) < FRESH_SECONDS


def write_config(config_id, seed, resume):
    max_epochs = LONG_CONFIGS.get(config_id, 100)
    cfg = {
        "CONFIG_ID": config_id,
        "SEED": seed,
        "SMOKE": False,
        "MAX_EPOCHS": max_epochs,
        "RESUME": resume,
        "AUTO": True,
        "UNFREEZE1_EPOCH": 30,
        "UNFREEZE2_EPOCH": 60,
    }
    # Long 配置：跑到 SNAPSHOT_EPOCH 轮时，train.py 会自动把该轮 checkpoint/metrics 复制存档
    if config_id in LONG_CONFIGS:
        cfg["SNAPSHOT_EPOCH"] = SNAPSHOT_EPOCH
        log(f"{config_id} 为 Long 配置：MAX_EPOCHS={max_epochs}，跑到第 {SNAPSHOT_EPOCH} 轮自动存档 100 轮快照")
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def run_once(config_id, seed, resume):
    write_config(config_id, seed, resume)
    log(f"▶ 启动 {config_id} seed{seed}（RESUME={resume}）")
    r = subprocess.run([PYTHON, TRAIN], cwd=SRC_V2)
    return r.returncode


def ensure_long_resume(config_id, seed):
    """Long 配置：若自己没有断点、但源配置有断点，则复制过来续跑（省去重跑前 100 轮）。"""
    base = LONG_BASE.get(config_id)
    if base is None:
        return
    long_pth = os.path.join(RUNS, f"checkpoint_{config_id}_seed{seed}_last.pth")
    if os.path.exists(long_pth):
        return  # 自己已有断点，不覆盖
    src_pth = os.path.join(RUNS, f"checkpoint_{base}_seed{seed}_last.pth")
    if os.path.exists(src_pth):
        shutil.copy(src_pth, long_pth)
        log(f" {config_id} seed{seed} 从 {base} seed{seed} 的断点续跑（已复制 checkpoint）")


def process(config_id, seed):
    try:
        return _process(config_id, seed)
    except CompletionCheckError as error:
        log(f"✖ 完成状态无法确认，停止队列：{error}")
        return "failed"


def _process(config_id, seed):
    if done(config_id, seed):
        log(f"跳过已完成 {config_id} seed{seed}")
        return "done"
    if in_progress(config_id, seed):
        log(f"{config_id} seed{seed} 正在训练，等待 5 分钟")
        return "running"

    ensure_long_resume(config_id, seed)  # Long 配置：从源配置断点续跑
    resume = os.path.exists(os.path.join(RUNS, f"checkpoint_{config_id}_seed{seed}_last.pth"))
    rc = run_once(config_id, seed, resume)
    if rc == 0 and done(config_id, seed):
        log(f"完成 {config_id} seed{seed}")
        return "done"

    log(f"⚠ {config_id} seed{seed} 异常（退出码 {rc}），改用 RESUME=true 重试一次")
    rc2 = run_once(config_id, seed, True)
    if rc2 == 0 and done(config_id, seed):
        log(f"RESUME 完成 {config_id} seed{seed}")
        return "done"

    log(f"✖ {config_id} seed{seed} 二次失败，停止队列，请人工检查后重启本脚本")
    return "failed"


def main():
    if not _preflight():
        return 1
    log("=" * 60)
    log("run_queue 启动；队列（协议 v2 全量重跑）：")
    for c, s in QUEUE:
        log(f"    {c} seed{s}")
    log("=" * 60)

    i = 0
    while i < len(QUEUE):
        config_id, seed = QUEUE[i]
        st = process(config_id, seed)
        if st == "done":
            i += 1
            # 一个任务跑完、下一个开始前：温度过高则休息 10 分钟
            if i < len(QUEUE):
                _nxt_c, _nxt_s = QUEUE[i]
                temp_guard.check_temps_and_rest(
                    tag=f" 切换前(下一项 {_nxt_c} seed{_nxt_s})",
                    log_info=log, log_warn=log,
                )
        elif st == "running":
            time.sleep(300)  # 5 分钟后再查
        else:  # failed
            return 1
    log("队列全部完成")
    return 0


if __name__ == "__main__":
    print("提示：请另开【管理员】cmd 运行 disk_temp_sampler.py（盘温采样）；")
    print("      队列在任务切换前会经 temp_guard 读取 runs\\disk_temp.log 判断是否需要休息。")
    time.sleep(5)
    sys.exit(main())
