# -*- coding: utf-8 -*-
"""V3六项NoTF实验队列；--check只读检查，直接执行则运行队列。"""
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
from pathlib import Path

_source_dir = Path(__file__).resolve().parent
for _path in (str(_source_dir), str(_source_dir.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import temp_guard  # 温度守护：任务切换前检测 CPU/GPU 温度，过高自动休息 10 分钟
import model_registry
from launch_checks import handle_cli
from run_paths import run_directory, result_path, checkpoint_path, load_config, normalization_signature
from src_v3.result_tools.chart_data import all_metrics_files, DIRTY_RECORDS
from taxonomy import (
    EXPECTED_NUM_CLASSES,
    LABEL_SCHEMA_ID,
    LABEL_SCHEMA_DEFINITION_HASH,
    LABEL_SCHEMA_VERSION,
    load_current_label_schema,
)

SRC_V2 = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(SRC_V2, "runs")
CONFIG = os.environ.get("V3_CONFIG_FILE", os.path.join(SRC_V2, "experiment_config.json"))
TRAIN = os.path.join(SRC_V2, "train.py")
LOG = os.path.join(RUNS, "queue_log.txt")
PYTHON = sys.executable  # 用启动本脚本的同一个 Python（保证 torch 1.11 环境）
PROJECT_ROOT = os.path.dirname(SRC_V2)
CURRENT_LABEL_MAPPING_HASH = None

# 执行队列：每行是 (模型配置名, seed)，严格按列表顺序运行。
# 增删任务、调整顺序或seed直接改这里；模型结构仍在model_registry.py定义。
# 当前六项均100轮、seed1；追加其他seed时另写一行，不自动展开注册表。
QUEUE = [
    # NoTF解冻基线
    ("FullUnfreeze", 1),
    ("ProgressiveUnfreeze", 1),          # 30/60解冻

    # 解冻时点对照
    ("ProgressiveUnfreeze-E15-40", 1),
    ("ProgressiveUnfreeze-E45-75", 1),

    # 轻骨干对照
    ("ResNet18-FullUnfreeze", 1),
    ("ResNet34-FullUnfreeze", 1),
]
AMP_FIX_MARKER = "scaler.unscale_(optimizer)"
EPOCH_SAMPLER_MARKER = "_derive_sampler_seed(SEED, epoch)"
# 本批不支持长预算或跨配置续跑，兼容辅助分支保持为空。
LONG_CONFIGS = {}
LONG_BASE = {}
SNAPSHOT_EPOCH = 100

FRESH_SECONDS = 15 * 60  # checkpoint 最近 15 分钟有写 = 训练进行中
# 队列侧完成守卫；公共绘图工具不承担严格完成判定。
TRAINING_PROTOCOL_VERSION = 3
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
    global CURRENT_LABEL_MAPPING_HASH
    if CURRENT_LABEL_MAPPING_HASH is None:
        _, _, CURRENT_LABEL_MAPPING_HASH = load_current_label_schema(PROJECT_ROOT)
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
    path = str(result_path(config_id, seed, runs_root=RUNS))
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
                'unfreeze_mode': preset['unfreeze'],
                'backbone': preset['backbone'], 'unfreeze_epochs': preset['unfreeze_epochs'],
                'experiment_signature': model_registry.experiment_signature_of(config_id),
                'normalization_signature': normalization_signature(load_config())}
    if not isinstance(result, dict) or not _matches(result, expected):
        raise CompletionCheckError(f'{path} 的实验身份、训练/采样/标签协议不匹配')
    try:
        start, end = map(_integer, result['epochs_run'])
        best_epoch = _integer(result['best_val_epoch'])
        best_accuracy = float(result['best_val_acc'])
        if not (0 <= start <= end == target_epoch and 0 <= best_epoch <= end
                and math.isfinite(best_accuracy) and 0 <= best_accuracy <= 100):
            raise ValueError('结果轮次或best指标无效')
        # 协议3正式结果必须包含实际骨干和结构快照。
        metadata = result['model_metadata']
        expected_model = {k: preset[k] for k in ('regions', 'transformer_layers', 'use_decouple')}
        expected_model.update(num_classes=EXPECTED_NUM_CLASSES, backbone=preset['backbone'],
                              feature_dim=model_registry.BACKBONE_FEATURE_DIMS[preset['backbone']],
                              architecture='ResNetTransformerV2')
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
    """开跑前校验：train.py 必须已是训练协议 v3（AMP 梯度裁剪前显式 unscale）。

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
        log("  当前代码不是训练协议 v3；用它跑出的结果不得进论文正式结果矩阵。")
        log("  请先恢复 2026-09-25 的 AMP 修复（参见 毕业设计/2026-09-26_协议v2重跑与旧数据归档方案.md）后再启动队列。")
        return False
    log(f"✔ 前置校验通过：train.py 含 `{AMP_FIX_MARKER}`（训练协议 v3）")

    if EPOCH_SAMPLER_MARKER not in src:
        log("✖ 前置校验失败：train.py 未启用逐 epoch sampler seed。")
        log("  续跑会重放 WeightedRandomSampler 随机流开头，拒绝启动正式队列。")
        return False
    log("✔ 前置校验通过：WeightedRandomSampler 使用实验 seed + 绝对 epoch 独立播种")
    log(f"✔ 标签协议：{LABEL_SCHEMA_ID} v{LABEL_SCHEMA_VERSION} / "
        f"{EXPECTED_NUM_CLASSES}类 / hash={CURRENT_LABEL_MAPPING_HASH}")

    # 现场报告（不阻断，仅提示）：确认旧协议产物确已移出 runs\ 根目录
    n_res = len(glob.glob(os.path.join(RUNS, "*", "results_*.json")))
    n_ckpt = len(glob.glob(os.path.join(RUNS, "*", "checkpoint_*.pth")))
    n_mr = len(all_metrics_files())
    log(f"  现场：results={n_res} / checkpoint={n_ckpt} / metrics={n_mr}")
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
    p = str(checkpoint_path(config_id, seed, runs_root=RUNS))
    if not os.path.exists(p):
        return False
    return (time.time() - os.path.getmtime(p)) < FRESH_SECONDS


def write_config(config_id, seed, resume):
    # 不在队列中复制30/60或骨干默认值，全部从同一注册配置生成。
    cfg = model_registry.config_for_run(config_id, seed, resume=resume)
    cfg['NORMALIZATION'] = load_config()['NORMALIZATION']
    with open(CONFIG, 'w', encoding='utf-8') as file:
        json.dump(cfg, file, ensure_ascii=False, indent=2)


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
    resume = checkpoint_path(config_id, seed, runs_root=RUNS).is_file()
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
    if handle_cli('queue', queue_tasks=QUEUE):
        return 0

    print("提示：盘温采样可单独运行 disk_temp_sampler.py。")
    _label_fields()
    if not _preflight():
        return 1
    log("=" * 60)
    log("V3首批NoTF配对队列：")
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
    sys.exit(main())
