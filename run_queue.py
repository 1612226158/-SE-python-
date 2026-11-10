# -*- coding: utf-8 -*-
"""显式队列：默认正式训练；--smoke固定3轮；--check只读预览选定模式。"""
import glob
import csv
from copy import deepcopy
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_source_dir = Path(__file__).resolve().parent
# 脚本入口和注册表查询共用同一份Long声明。
sys.modules.setdefault('run_queue', sys.modules[__name__])
for _path in (str(_source_dir), str(_source_dir.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import temp_guard  # 温度守护：任务切换前检测 CPU/GPU 温度，过高自动休息 10 分钟
import model_registry
from launch_checks import handle_cli
from run_paths import result_path, checkpoint_path, run_directory, load_config, normalization_signature, METRICS_COLUMNS
from src_v3.result_tools.chart_data import all_metrics_files, DIRTY_RECORDS
from taxonomy import (
    EXPECTED_NUM_CLASSES,
    LABEL_SCHEMA_ID,
    LABEL_SCHEMA_DEFINITION_HASH,
    LABEL_SCHEMA_VERSION,
    load_current_label_schema,
)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(SRC_DIR, "runs")
CONFIG = str(Path(os.environ.get("V3_CONFIG_FILE", os.path.join(SRC_DIR, "experiment_config.json"))).resolve())
TRAIN = os.path.join(SRC_DIR, "train.py")
LOG = os.path.join(RUNS, "queue_log.txt")
PYTHON = sys.executable  # 用启动本脚本的同一个 Python（保证 torch 1.11 环境）
PROJECT_ROOT = os.path.dirname(SRC_DIR)
CURRENT_LABEL_MAPPING_HASH = None
_queue_config = None
_logger = logging.getLogger('src_v3.run_queue')

# 执行队列：每行是 (模型配置名, seed)，严格按列表顺序运行。
# 未注释的行才执行；结构在model_registry.py定义，不自动展开模型或seed。
QUEUE = [
    # 第一阶段：当前增强（AA+RE）下的SE[1]轻骨干对比，100轮。
    ("ResNet18-SE1-FullUnfreeze", 1),
    ("ShuffleNetV2x1.0-SE1-FullUnfreeze", 1),

    # 配对重复：已注册，初轮检查后按预算逐行取消注释。
    # ("ResNet18-SE1-FullUnfreeze", 2),
    # ("ShuffleNetV2x1.0-SE1-FullUnfreeze", 2),
    # ("ResNet18-SE1-FullUnfreeze", 3),
    # ("ShuffleNetV2x1.0-SE1-FullUnfreeze", 3),

    # SE[1,2]与其他骨干：已注册；RegNet仍待接入，不可直接启用。
    # ("ResNet18-FullUnfreeze", 1),
    # ("ShuffleNetV2x1.0-FullUnfreeze", 1),
    # ("ResNet34-FullUnfreeze", 1),
    # ("RegNetX400MF-FullUnfreeze", 1),  # 待骨干适配

    # 原ResNeXt50解冻对照：已注册，保留后续使用，当前不运行。
    # ("FullUnfreeze", 1),
    # ("ProgressiveUnfreeze", 1),          # 30/60解冻
    # ("ProgressiveUnfreeze-E15-40", 1),
    # ("ProgressiveUnfreeze-E45-75", 1),

    # 第二阶段：增强四格。以下ID仅规划，尚未注册/实现，不能直接取消注释。
    # AugBase=共同裁剪/归一化；AugAA=仅AA；AugRE=仅RE。
    # AA+RE是上方无Aug后缀的当前基线，不重复创建同参数任务。
    # 先ResNet18筛选，再按结果选Shuffle的对照；不要求全矩阵都运行。
    # ("ResNet18-SE1-FullUnfreeze-AugBase", 1),
    # ("ResNet18-SE1-FullUnfreeze-AugAA", 1),
    # ("ResNet18-SE1-FullUnfreeze-AugRE", 1),
    # ("ResNet18-SE1-FullUnfreeze-AugBase", 2),
    # ("ResNet18-SE1-FullUnfreeze-AugAA", 2),
    # ("ResNet18-SE1-FullUnfreeze-AugRE", 2),
    # ("ResNet18-SE1-FullUnfreeze-AugBase", 3),
    # ("ResNet18-SE1-FullUnfreeze-AugAA", 3),
    # ("ResNet18-SE1-FullUnfreeze-AugRE", 3),
    # ("ShuffleNetV2x1.0-SE1-FullUnfreeze-AugBase", 1),
    # ("ShuffleNetV2x1.0-SE1-FullUnfreeze-AugAA", 1),
    # ("ShuffleNetV2x1.0-SE1-FullUnfreeze-AugRE", 1),
    # ("ShuffleNetV2x1.0-SE1-FullUnfreeze-AugBase", 2),
    # ("ShuffleNetV2x1.0-SE1-FullUnfreeze-AugAA", 2),
    # ("ShuffleNetV2x1.0-SE1-FullUnfreeze-AugRE", 2),
    # ("ShuffleNetV2x1.0-SE1-FullUnfreeze-AugBase", 3),
    # ("ShuffleNetV2x1.0-SE1-FullUnfreeze-AugAA", 3),
    # ("ShuffleNetV2x1.0-SE1-FullUnfreeze-AugRE", 3),

    # 参数/数据尚未确定的候选，只列方向，不虚构运行ID：
    # 裁剪强度、擦除强度、保比例几何、颜色策略、后期减弱增强；
    # 增强×SE区域/头宽度、增强×采样、MixUp/CutMix；
    # 固定252类的真实补图对照；RegNet增强迁移。
    # 调度/等参数SE及Long等旧计划仍按各自方案另定，不自动恢复全部预算。
]
# Long只延长同seed已完成实验的总轮数；登记后仍需在QUEUE逐行添加任务。
LONG_CONFIGS = {
    # "FullUnfreeze-Long200": {"source": "FullUnfreeze", "epochs": 200},
}
_long_entries = {}

# 队列侧完成守卫；公共绘图工具不承担严格完成判定。
TRAINING_PROTOCOL_VERSION = 3
SAMPLING_PROTOCOL_VERSION = 1
SAMPLER_SEED_STRIDE = 1_000_000


class CompletionCheckError(RuntimeError):
    """产物状态未知或矛盾，必须停队，不能当作未完成自动开跑。"""


def long_entry(config_id):
    """将本文件的Long声明转换为注册表条目，不创建运行产物。"""
    if config_id not in LONG_CONFIGS:
        raise KeyError(f'未注册的V3实验ID: {config_id!r}')
    if config_id in _long_entries:
        return _long_entries[config_id]
    declaration = LONG_CONFIGS[config_id]
    source, epochs = declaration['source'], declaration['epochs']
    if (config_id in model_registry.MODELS or not config_id
            or any(c in config_id for c in '/\\:')
            or source not in model_registry.MODELS
            or type(epochs) is not int or epochs <= model_registry.target_of(source)):
        raise ValueError(f'Long必须从普通实验延长总轮数，且使用独立配置名：{config_id}')
    entry = deepcopy(model_registry.MODELS[source])
    entry.update(target_epochs=epochs, phase='long',
                 short=entry['short'] + f' Long{epochs}',
                 cn=entry['cn'] + f' Long{epochs}',
                 long_source={'config_id': source, 'epochs': [0, model_registry.target_of(source) - 1],
                              'experiment_signature': model_registry.experiment_signature_of(source)})
    _long_entries[config_id] = entry
    return entry


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


def _metrics_index(config_id, seed):
    """只读取目标run当前指标，保留行身份、采样和重复轮次校验。"""
    path = run_directory(config_id, seed, runs_root=RUNS) / 'metrics_record.csv'
    index = {}
    if not path.is_file():
        return index
    try:
        with open(path, encoding='utf-8-sig', newline='') as file:
            reader = csv.DictReader(file, strict=True)
            if reader.fieldnames != list(METRICS_COLUMNS):
                raise CompletionCheckError(f'{path} 不符合当前20列指标格式')
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
    return index


def _result_record(config_id, seed, target_epoch, config=None):
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
                'normalization_signature': normalization_signature(config or _queue_config)}
    source = model_registry.long_source_of(config_id, seed)
    if source:
        expected['long_source'] = source
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


def log(msg, level=logging.INFO):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    _logger.log(level, msg)


def _preflight():
    """检查配置和显式队列，报告标签协议及已有产物。"""
    global _queue_config
    try:
        _queue_config = load_config(CONFIG)
        for config_id in LONG_CONFIGS:
            long_entry(config_id)
        if len(QUEUE) != len(set(QUEUE)):
            raise ValueError('QUEUE存在重复的(模型名, seed)任务')
        for config_id, seed in QUEUE:
            model_registry.config_for_run(config_id, seed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        log(f'✖ 队列/配置校验失败：{error}', logging.ERROR)
        return False
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


def done(config_id, seed, config=None):
    """True=可信完成，False=未完成；未知/矛盾状态抛异常，process负责停队。"""
    if not model_registry.is_known(config_id):
        raise CompletionCheckError(f'未知模型配置：{config_id}')
    target_epoch = model_registry.target_of(config_id) - 1
    index = _metrics_index(config_id, seed)
    rows = dict(index.get((config_id, seed), {}))
    result = _result_record(config_id, seed, target_epoch, config)
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


def write_config(config_id, seed, resume, smoke=False, *, smoke_epochs=None):
    # 骨干、时点及正式/冒烟预算均由注册表生成。
    cfg = model_registry.config_for_run(config_id, seed, resume=resume, smoke=smoke,
                                        smoke_epochs=smoke_epochs)
    cfg['NORMALIZATION'] = _queue_config['NORMALIZATION']
    destination = os.path.join(RUNS, '_smoke', 'queue_config.json') if smoke else CONFIG
    if smoke:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as file:
        json.dump(cfg, file, ensure_ascii=False, indent=2)
    os.replace(temporary, destination)
    return destination


def run_once(config_id, seed, resume, smoke=False, *, smoke_epochs=None):
    config_path = write_config(config_id, seed, resume, smoke=smoke, smoke_epochs=smoke_epochs)
    log(f"▶ 启动 {config_id} seed{seed}（RESUME={resume}）")
    environment = os.environ.copy()
    environment['V3_CONFIG_FILE'] = config_path
    r = subprocess.run([PYTHON, TRAIN], cwd=SRC_DIR, env=environment)
    return r.returncode


def ensure_long_resume(config_id, seed, config=None):
    """首次Long继承同seed来源的断点和完整指标；已有目标交给正常续跑。"""
    source = model_registry.long_source_of(config_id, seed)
    if source is None:
        return False
    config = config or _queue_config or load_config(CONFIG)
    directory = run_directory(config_id, seed, runs_root=RUNS)
    if directory.exists() and any(directory.iterdir()):
        # 首次初始化最后写config.json；缺它表示未完成初始化，不能自动续跑。
        snapshot = directory / 'config.json'
        if not snapshot.is_file():
            raise CompletionCheckError(f'Long目录缺少初始化配置，保留现场：{directory}')
        saved = json.loads(snapshot.read_text(encoding='utf-8'))
        if saved.get('LONG_SOURCE') != source:
            raise CompletionCheckError(f'Long目录来源不一致：{directory}')
        if not checkpoint_path(config_id, seed, runs_root=RUNS).is_file():
            raise CompletionCheckError(f'Long目录缺少last，保留现场：{directory}')
        if not checkpoint_path(config_id, seed, 'best', runs_root=RUNS).is_file():
            raise CompletionCheckError(f'Long目录缺少best，保留现场：{directory}')
        return False
    base = source['config_id']
    if not done(base, seed, config):
        raise CompletionCheckError(f'Long来源尚未正式完成：{base} seed{seed}')
    source_dir = run_directory(base, seed, runs_root=RUNS)
    result = json.loads(result_path(base, seed, runs_root=RUNS).read_text(encoding='utf-8'))
    with (source_dir / 'metrics_record.csv').open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        rows = list(reader)
    if ([int(row['epoch']) for row in rows] != list(range(source['epochs'][1] + 1))
            or any(row['config_id'] != base or int(row['seed']) != seed for row in rows)):
        raise CompletionCheckError('Long来源metrics_record.csv必须是本实验完整且有序的记录')
    import torch
    checkpoints = {which: torch.load(checkpoint_path(base, seed, which, runs_root=RUNS),
                                      map_location='cpu') for which in ('last', 'best')}
    id_to_main_class, renew_class_to_index, _ = load_current_label_schema(PROJECT_ROOT)
    metadata = result['model_metadata']
    model_signature = hashlib.sha256(json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    expected = {**_label_fields(), 'config_id': base, 'seed': seed,
                'label_schema_id': LABEL_SCHEMA_ID,
                'label_schema_definition_hash': LABEL_SCHEMA_DEFINITION_HASH,
                'id_to_main_class': id_to_main_class, 'renew_class_to_index': renew_class_to_index,
                'training_protocol_version': TRAINING_PROTOCOL_VERSION,
                'sampling_protocol_version': SAMPLING_PROTOCOL_VERSION,
                'experiment_signature': source['experiment_signature'],
                'normalization_signature': normalization_signature(config),
                'model_metadata': metadata, 'model_signature_sha256': model_signature,
                'backbone': model_registry.backbone_of(base),
                'unfreeze_epochs': model_registry.unfreeze_epochs_of(base),
                'training_policy': model_registry.training_policy_of(base)}
    for which, checkpoint in checkpoints.items():
        epoch = source['epochs'][1] if which == 'last' else int(result['best_val_epoch'])
        if (not _matches(checkpoint, expected) or checkpoint.get('epoch') != epoch
                or checkpoint.get('sampler_seed') != seed * SAMPLER_SEED_STRIDE + epoch
                or checkpoint.get('best_epoch') != result['best_val_epoch']
                or abs(float(checkpoint['best_val_acc']) - result['best_val_acc']) > 0.00006):
            raise CompletionCheckError(f'Long来源{which}身份或轮次/best与正式结果不一致')
        if which == 'last':
            for key in ('state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
                        'scaler_state_dict', 'state', 'train_acc_at_best'):
                if key not in checkpoint:
                    raise CompletionCheckError(f'Long来源last缺少{key}')
        elif 'state_dict' not in checkpoint:
            raise CompletionCheckError('Long来源best缺少state_dict')
    directory.mkdir(parents=True, exist_ok=True)
    target_signature = model_registry.experiment_signature_of(config_id)
    with (directory / 'metrics_record.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(dict(row, config_id=config_id) for row in rows)
    for which, checkpoint in checkpoints.items():
        checkpoint.update(config_id=config_id, experiment_signature=target_signature, long_source=source)
        torch.save(checkpoint, checkpoint_path(config_id, seed, which, runs_root=RUNS))
    snapshot = model_registry.config_for_run(config_id, seed)
    snapshot.update(NORMALIZATION=config['NORMALIZATION'], RUN_DIRECTORY=str(directory))
    (directory / 'config.json').write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return True


def process(config_id, seed):
    try:
        return _process(config_id, seed)
    except (CompletionCheckError, OSError, ValueError, KeyError) as error:
        log(f"✖ 完成状态无法确认，停止队列：{error}", logging.ERROR)
        return "failed"


def _process(config_id, seed):
    if done(config_id, seed):
        log(f"跳过已完成 {config_id} seed{seed}")
        return "done"
    if ensure_long_resume(config_id, seed):
        base = LONG_CONFIGS[config_id]['source']
        log(f" {config_id} seed{seed} 从 {base} seed{seed} 的断点续跑（已复制 checkpoint）")
    resume = checkpoint_path(config_id, seed, runs_root=RUNS).is_file()
    rc = run_once(config_id, seed, resume)
    if rc == 0 and done(config_id, seed):
        log(f"完成 {config_id} seed{seed}")
        return "done"

    if not checkpoint_path(config_id, seed, runs_root=RUNS).is_file():
        log(f"✖ {config_id} seed{seed} 未产生last断点（退出码 {rc}），停止队列；保留现场，不伪装续跑或覆盖重跑", logging.ERROR)
        return "failed"
    log(f"⚠ {config_id} seed{seed} 异常（退出码 {rc}），从已有last断点重试一次", logging.WARNING)
    rc2 = run_once(config_id, seed, True)
    if rc2 == 0 and done(config_id, seed):
        log(f"RESUME 完成 {config_id} seed{seed}")
        return "done"

    log(f"✖ {config_id} seed{seed} 二次失败，停止队列，请人工检查后重启本脚本", logging.ERROR)
    return "failed"


def main():
    args = handle_cli('queue', queue_tasks=QUEUE)
    if args.check or args.list:
        return 0

    log_path = os.path.join(RUNS, '_smoke', 'queue_log.txt') if args.smoke else LOG
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    _logger.addHandler(handler)

    print("提示：盘温采样可单独运行 disk_temp_sampler.py。")
    _label_fields()
    if not _preflight():
        return 1
    log("=" * 60)
    log("V3首批NoTF配对队列：")
    if args.smoke:
        smoke_epochs = args.smoke_epochs or model_registry.DEFAULT_SMOKE_EPOCHS
        log(f'模式：各任务{smoke_epochs}轮冒烟；不续跑，不转入正式100轮。')
    for c, s in QUEUE:
        log(f"    {c} seed{s}")
    log("=" * 60)

    i = 0
    while i < len(QUEUE):
        config_id, seed = QUEUE[i]
        if args.smoke:
            rc = run_once(config_id, seed, False, smoke=True, smoke_epochs=smoke_epochs)
            st = 'done' if rc == 0 else 'failed'
            log(f'冒烟 {config_id} seed{seed} 退出码 {rc}',
                logging.INFO if rc == 0 else logging.ERROR)
        else:
            st = process(config_id, seed)
        if st == "done":
            i += 1
            # 一个任务跑完、下一个开始前：温度过高则休息 10 分钟
            if i < len(QUEUE):
                _nxt_c, _nxt_s = QUEUE[i]
                temp_guard.check_temps_and_rest(
                    tag=f" 切换前(下一项 {_nxt_c} seed{_nxt_s})",
                    log_info=log, log_warn=lambda msg: log(msg, logging.WARNING),
                )
        else:  # failed
            return 1
    log("队列全部完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
