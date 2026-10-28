import os as _entry_os
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_dir = _EntryPath(__file__).resolve().parent
for _entry_path in (str(_entry_dir), str(_entry_dir.parent)):
    if _entry_path not in _entry_sys.path:
        _entry_sys.path.insert(0, _entry_path)
if __name__ == '__main__':
    from launch_checks import handle_cli as _handle_cli
    if _handle_cli('train'):
        raise SystemExit(0)
    from run_paths import configure_process as _configure_process
    _configure_process()
import random
import sys
import time
from datetime import datetime
import warnings
import logging
import os
import json
import hashlib

import temp_guard  # 温度守护：关键节点检测 CPU/GPU 温度，过高自动休息 10 分钟
from functools import partial

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from tqdm import tqdm
import psutil
# import pillow_avif
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import cv2  # 通常与 Albumentations 配合使用

from ResNet import ResNetTransformer
from ResNetTransformer import ResNetTransformerV2  # 修正版 ViT 式多 token Transformer（独立路线；配置加 model='v2' 时启用）
from calculate import (calculate_mean_and_std,
                       SafeImageFolder,
                       FocalLoss, add_record_metrics, add_record_metrics_v2,
                       get_classnames,
                       generate_mappings, get_class_proportion,
                       get_output_class_sample_counts,
                       get_class_weight, hierarchical_collate_fn)
from calculate_update import SafeImageFolder as NewSafeImageFolder
from src_v3 import (device, DynamicMeanPad,
                    pth_file_name, save_model_after_epoch,
                    extensions_suffix, merged_dict, ACC_STOP, ERROR_RATE,
                    TRAIN_NUM_WORKERS, VAL_NUM_WORKERS)
from src_v3.taxonomy import (
    EXPECTED_NUM_CLASSES,
    LABEL_SCHEMA_DEFINITION_HASH,
    LABEL_SCHEMA_ID,
    LABEL_SCHEMA_VERSION,
    mapping_fingerprint,
)

# from batch_size import adjust_batch_size

# ==================== 实验配置：编辑同目录 experiment_config.json（无需改本文件） ====================
CONFIG_FILE = os.environ.get('V3_CONFIG_FILE', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_config.json'))
_DEFAULTS = {
    'CONFIG_ID': 'FullUnfreeze',  # 默认首批全解冻NoTF基线
    'SEED': 1,               # 随机种子 0/1/2，每配置跑 3 次取 mean±std
    'SMOKE': True,           # True=冒烟只跑 3 轮验证管线；正式训练 False
    'MAX_EPOCHS': 100,
    'RESUME': False,         # True=从 runs 下断点续跑（支持中途暂停后继续）
    'AUTO': True,            # 无人值守：跳过所有交互输入
    'UNFREEZE1_EPOCH': 30,   # 固定解冻触发：state1→2
    'UNFREEZE2_EPOCH': 60,   # state2→3
    'SNAPSHOT_EPOCH': None,  # Long 配置跑到该轮时自动存档 100 轮快照（如 100）
    'ALLOW_LEGACY_PROTOCOL_RESUME': False,  # 仅诊断旧实验时手动开启；正式实验禁止混合 AMP 协议
}

# temp_guard_lis = (25, 50, 75, 99)
# temp_guard_lis = (10, 20, 30, 40, 50, 60, 70, 80, 90)
temp_guard_lis = [i for i in range(100)]


_cfg = {}
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            _cfg = json.load(f)
        logging.info(f'[配置] 已加载 {CONFIG_FILE}: {_cfg}')
    except Exception as e:
        logging.warning(f'[配置] 配置文件读取失败({e})，使用内置默认值')
# 显式赋值（不用 globals() 动态注入，保证 IDE 可解析、无黄字）
CONFIG_ID = _cfg.get('CONFIG_ID', _DEFAULTS['CONFIG_ID'])
SEED = _cfg.get('SEED', _DEFAULTS['SEED'])
SMOKE = _cfg.get('SMOKE', _DEFAULTS['SMOKE'])
if SMOKE:
    MAX_EPOCHS = 3
    RESUME = False
else:
    MAX_EPOCHS = _cfg.get('MAX_EPOCHS', _DEFAULTS['MAX_EPOCHS'])
    RESUME = _cfg.get('RESUME', _DEFAULTS['RESUME'])
AUTO = _cfg.get('AUTO', _DEFAULTS['AUTO'])
SNAPSHOT_EPOCH = _cfg.get('SNAPSHOT_EPOCH', _DEFAULTS['SNAPSHOT_EPOCH'])
ALLOW_LEGACY_PROTOCOL_RESUME = _cfg.get('ALLOW_LEGACY_PROTOCOL_RESUME',
                                        _DEFAULTS['ALLOW_LEGACY_PROTOCOL_RESUME'])

# 注册表为骨干、解冻时点和共同训练策略的唯一来源。
import model_registry
PRESETS = model_registry.MODEL_PARAMS
model_registry.validate_config_binding(CONFIG_ID, _cfg)
PRESET = model_registry.params_of(CONFIG_ID)
unfreeze_mode = PRESET['unfreeze']
BACKBONE_NAME = PRESET['backbone']
UNFREEZE_EPOCHS = model_registry.unfreeze_epochs_of(CONFIG_ID)
UNFREEZE1_EPOCH, UNFREEZE2_EPOCH = UNFREEZE_EPOCHS if UNFREEZE_EPOCHS else (None, None)
TRAINING_POLICY = model_registry.training_policy_of(CONFIG_ID)
EXPERIMENT_SIGNATURE = model_registry.experiment_signature_of(CONFIG_ID)
NORMALIZATION_SIGNATURE = os.environ.get('V3_NORMALIZATION_SIGNATURE')
SCHEDULER_MODE = PRESET['scheduler']
BACKBONE_LR = TRAINING_POLICY['backbone_lr']
HEAD_LR = TRAINING_POLICY['head_lr']
if ALLOW_LEGACY_PROTOCOL_RESUME:
    raise ValueError('V3实验不允许绕过协议校验接续旧训练')


def _make_optimizer(model):
    # 从开始就保留全部参数组；冻结参数无梯度，解冻时不重建优化器。
    return optim.AdamW(model.get_grouped_params(BACKBONE_LR, HEAD_LR, include_frozen=True),
                       weight_decay=TRAINING_POLICY['weight_decay'])


def _advance_unfreeze_stage(model, state, epoch):
    """按注册时点切换梯度；不重建优化器或重启调度器。"""
    target = 3 if unfreeze_mode == 'none' else 1 + sum(epoch >= e for e in UNFREEZE_EPOCHS)
    if target < state:
        raise RuntimeError('解冻状态与当前epoch不一致')
    if target != state:
        model.freeze_backbone()
        if target == 2:
            model.unfreeze_layer4()
        elif target == 3:
            model.unfreeze_all()
        logging.info(f'解冻状态 {state}->{target} 于epoch {epoch}；optimizer/lr状态保持')
    return target


# 路径全部基于本文件定位（与 PyCharm 工作目录无关）
SRC_V2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_V2_DIR)
RUNS_DIR = os.environ.get('V3_RUN_DIR', os.path.join(SRC_V2_DIR, 'runs'))
if __name__ == '__main__':
    os.makedirs(RUNS_DIR, exist_ok=True)
TRAIN_ROOT = os.path.join(PROJECT_ROOT, 'train')
VAL_ROOT = os.path.join(PROJECT_ROOT, 'val')

RUN_TAG = f'{CONFIG_ID}_seed{SEED}'
BEST_PTH = os.path.join(RUNS_DIR, f'checkpoint_{RUN_TAG}_best.pth')
LAST_PTH = os.path.join(RUNS_DIR, f'checkpoint_{RUN_TAG}_last.pth')
# 协议3：骨干/时点绑定、统一优化器/全局调度/BN及实验签名；完整启动仍受未就绪限制。
TRAINING_PROTOCOL_VERSION = 3
# 采样协议版本 1：每个绝对 epoch 使用可独立重建的 sampler seed，续跑不再重放随机流开头。
SAMPLING_PROTOCOL_VERSION = 1
SAMPLER_SEED_STRIDE = 1_000_000


def _label_metadata(id_to_main_class, renew_class_to_index):
    """生成写入 checkpoint/results/CSV 的当前标签协议元数据。"""
    return {
        'label_schema_id': LABEL_SCHEMA_ID,
        'label_schema_version': LABEL_SCHEMA_VERSION,
        'label_schema_definition_hash': LABEL_SCHEMA_DEFINITION_HASH,
        'label_mapping_hash': mapping_fingerprint(id_to_main_class, renew_class_to_index),
        'num_classes': len(renew_class_to_index),
        'id_to_main_class': id_to_main_class,
        'renew_class_to_index': renew_class_to_index,
    }


def _check_label_schema(checkpoint, id_to_main_class, renew_class_to_index):
    """恢复前强校验标签体系；254 类旧权重不能接到当前 252 类输出头。"""
    expected = _label_metadata(id_to_main_class, renew_class_to_index)
    actual_version = checkpoint.get('label_schema_version')
    actual_hash = checkpoint.get('label_mapping_hash')
    actual_num_classes = checkpoint.get('num_classes')
    if (actual_version != LABEL_SCHEMA_VERSION or
            actual_hash != expected['label_mapping_hash'] or
            actual_num_classes != expected['num_classes'] or
            checkpoint.get('renew_class_to_index') != renew_class_to_index or
            checkpoint.get('id_to_main_class') != id_to_main_class):
        raise RuntimeError(
            '断点标签体系与当前训练不兼容，拒绝续跑。'
            f'断点: version={actual_version}, num_classes={actual_num_classes}, hash={actual_hash}; '
            f'当前: version={LABEL_SCHEMA_VERSION}, num_classes={expected["num_classes"]}, '
            f'hash={expected["label_mapping_hash"]}。'
            '本次已把“蔬菜/菜根菜叶→菜叶、插头电线/充电线→插头充电线”纳入合并，'
            '输出由254类变为252类，必须从 epoch 0 重新训练；ALLOW_LEGACY_PROTOCOL_RESUME 不绕过标签校验。')


def _derive_sampler_seed(run_seed, epoch):
    """由实验 seed 与绝对 epoch 唯一确定本轮 WeightedRandomSampler seed。"""
    run_seed = int(run_seed)
    epoch = int(epoch)
    if run_seed < 0 or epoch < 0:
        raise ValueError(f'run_seed 与 epoch 必须为非负整数，实际为 run_seed={run_seed}, epoch={epoch}')
    return run_seed * SAMPLER_SEED_STRIDE + epoch


def seed_all(seed):
    """固定所有随机源（torch / numpy / python random）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _restore_best_state(checkpoint, epoch_start):
    """恢复全局 best；兼容旧断点（旧格式缺字段时从历史 metrics CSV 回查）。"""
    if checkpoint is not None:
        required = ('best_val_acc', 'best_epoch', 'train_acc_at_best')
        if all(key in checkpoint for key in required):
            return (float(checkpoint['best_val_acc']),
                    int(checkpoint['best_epoch']),
                    float(checkpoint['train_acc_at_best']))

    if epoch_start <= 0:
        return -1.0, -1, -1.0

    history = []
    for name in os.listdir(RUNS_DIR):
        if not (name.startswith('metrics_record') and name.endswith('.csv')):
            continue
        path = os.path.join(RUNS_DIR, name)
        try:
            df = pd.read_csv(path)
            required_columns = {'config_id', 'seed', 'epoch', 'val_acc', 'train_acc'}
            if not required_columns.issubset(df.columns):
                continue
            matched = df[(df['config_id'].astype(str) == str(CONFIG_ID)) &
                         (pd.to_numeric(df['seed'], errors='coerce') == SEED) &
                         (pd.to_numeric(df['epoch'], errors='coerce') < epoch_start)].copy()
            if not matched.empty:
                history.append(matched)
        except Exception as exc:
            logging.warning(f'回查历史 best 时跳过 {name}: {exc}')

    if not history:
        logging.warning('旧断点不含全局 best，且 metrics 中未找到历史记录；best 将从当前续跑段重新计算')
        return -1.0, -1, -1.0

    merged = pd.concat(history, ignore_index=True)
    merged['val_acc'] = pd.to_numeric(merged['val_acc'], errors='coerce')
    merged['epoch'] = pd.to_numeric(merged['epoch'], errors='coerce')
    merged['train_acc'] = pd.to_numeric(merged['train_acc'], errors='coerce')
    merged = merged.dropna(subset=['val_acc', 'epoch', 'train_acc'])
    if merged.empty:
        logging.warning('历史 metrics 中 best 字段无法解析；best 将从当前续跑段重新计算')
        return -1.0, -1, -1.0

    row = merged.loc[merged['val_acc'].idxmax()]
    restored = float(row['val_acc']), int(row['epoch']), float(row['train_acc'])
    logging.info(f'已从历史 metrics 恢复全局 best: epoch {restored[1]}, val_acc {restored[0]:.4f}%')
    return restored

# 只关闭特定的警告
warnings.filterwarnings("ignore", message=".*Palette images with Transparency expressed in bytes.*")
# 设置日志配置：文件 + 控制台双写（两者都输出全部 INFO 日志）。
# 背景：PyCharm「Python 控制台」运行时会先给 root logger 预挂 handler，使 basicConfig 变 no-op，
#       导致 FileHandler 不生效、日志文件 0 字节。这里改为显式清空 root handler 再挂两个 handler，
#       无论从 PyCharm 控制台 / 普通 Run / 终端 / 子进程启动，都能同时：
#         1) 写入 runs\logging_<RUN_TAG>.log（UTF-8，事后定位用）；
#         2) 打印到 sys.stdout（PyCharm 控制台 / 终端实时可见）。
def _setup_logging():
    # 只在主进程调用（见 if __name__ == '__main__'）；DataLoader 的 spawn worker 会重新
    # import 本模块，若在模块层调用则每个 worker 都执行一遍、重复打印。
    logger = logging.getLogger()  # 根 logger；train.py 全程用 logging.info() 走这里
    logger.setLevel(logging.INFO)
    # 清掉已有 handler（含 PyCharm 控制台预挂的、以及重复运行残留的），避免重复打印或 basicConfig 失效
    for _h in list(logger.handlers):
        logger.removeHandler(_h)
        try:
            _h.close()
        except Exception:
            pass
    _fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # 文件 handler：INFO 及以上全落盘（事后定位用）
    _fh = logging.FileHandler(os.path.join(RUNS_DIR, f'logging_{RUN_TAG}.log'), mode='a', encoding='utf-8')
    _fh.setLevel(logging.INFO)
    _fh.setFormatter(_fmt)
    # 控制台 handler：只输出 WARNING 及以上（异常/报错），INFO 不进控制台，避免刷屏；
    # tqdm 进度条直接写 stderr、不经 logging，不受影响，控制台保持和原本一样只有进度条 + 报错。
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setLevel(logging.WARNING)
    _sh.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_sh)

batch_size = 64
sleep = 5


class DynamicCachedDataset(Dataset):
    def __init__(self, dataset, max_memory_usage=0.7):
        """
        :param dataset: 原始 PyTorch 数据集
        :param max_memory_usage: 最大内存使用比率（0-1之间），默认70%
        """
        self.dataset = dataset
        self.max_memory_usage = max_memory_usage

        # 估算每个数据项的内存大小，并计算最大缓存大小
        self.data_size_per_item = self._compute_data_size_per_item()
        self.max_cache_size = self._compute_max_cache_size()

        self.cache = [None] * len(dataset)  # 创建一个缓存结构
        self.cached_count = 0  # 当前缓存数量

    def _compute_data_size_per_item(self):
        # 获取一个数据项
        sample = self.dataset[0]

        # 假设每个数据项是一个图像
        image_tensor = sample[0]  # 假设数据项的第一个元素是图像张量
        # 获取图像大小：height x width x channels
        height, width, channels = image_tensor.shape

        # 计算数据类型的大小
        data_type_size = image_tensor.element_size()  # 返回每个元素占用的字节数

        # 计算每个数据项的大小
        return height * width * channels * data_type_size

    def _compute_max_cache_size(self):
        # 获取系统总内存
        # total_memory = torch.cuda.get_device_properties(
        #     0).total_memory if torch.cuda.is_available() else 16 * 1024 ** 3  # 默认16GB
        available_memory = psutil.virtual_memory().available

        # 计算最大可用内存
        max_available_memory = available_memory * self.max_memory_usage

        # 计算最多可以缓存多少数据项
        return max_available_memory // self.data_size_per_item

    def __getitem__(self, index):
        # 如果缓存区已满，则不再缓存
        if self.cached_count >= self.max_cache_size:
            return self.dataset[index]

        if self.cache[index] is None:  # 如果没有缓存，加载并缓存
            self.cache[index] = self.dataset[index]
            self.cached_count += 1
        return self.cache[index]

    def __len__(self):
        return len(self.dataset)


def validate(model, val_loader, parent_criterion, alpha=1.0, beta=0.0):
    """
    :param model: 模型
    :param val_loader: 验证集
    # :param criterion: 损失函数
    # :param id_to_main_class: 合并分类的映射元素的字典
    :return:
    """
    model.eval()  # 设置模型为评估模式
    # correct = 0
    total = 0
    val_loss = 0.0

    parent_correct = 0
    # y_true = []
    # y_pred = []

    y_true_parent = []
    y_pred_parent = []

    with torch.no_grad():  # 不计算梯度
        # for inputs, (parent_labels, child_labels) in val_loader:
        for inputs, parent_labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)  # 将数据转移到 GPU

            parent_labels = parent_labels.to(device, non_blocking=True).long()
            # child_labels = child_labels.to(device).long()

            # parent_output, child_output = model(inputs)
            parent_output = model(inputs)

            # 父类损失
            parent_loss = parent_criterion(parent_output, parent_labels)
            child_loss = 0
            loss = alpha * parent_loss + beta * child_loss
            val_loss += loss.item()

            # 父类预测
            _, parent_predicted = torch.max(parent_output, 1)
            parent_correct += (parent_predicted == parent_labels).sum().item()

            # 收集父类和子类真实和预测标签用于计算精确率等指标
            y_true_parent.extend(parent_labels.cpu().numpy())
            y_pred_parent.extend(parent_predicted.cpu().numpy())
            total += inputs.size(0)

    # total_elements_parent = total
    # total_elements_child = child_total

    parent_accuracy = 100 * parent_correct / total

    # child_accuracy = 100 * child_correct / child_total if child_total > 0 else 0
    # val_loss /= len(val_loader.dataset)
    val_loss /= len(val_loader)
    s = f'Validation Accuracy Parent: {parent_accuracy:.5f}%'
    print(s, file=fp)
    logging.info(s)

    # 父类和子类精确率计算
    parent_precision = precision_score(y_true_parent, y_pred_parent, average='weighted', zero_division=0)
    parent_recall = recall_score(y_true_parent, y_pred_parent, average='weighted', zero_division=0)
    parent_f1 = f1_score(y_true_parent, y_pred_parent, average='weighted', zero_division=0)
    parent_f1_macro = f1_score(y_true_parent, y_pred_parent, average='macro', zero_division=0)

    time.sleep(sleep)

    child_accuracy = 0.0
    child_precision = 0.0
    child_recall = 0.0
    child_f1 = 0.0

    return (val_loss,
            (parent_accuracy, child_accuracy),
            (parent_precision, child_precision),
            (parent_recall, child_recall),
            (parent_f1, child_f1),
            parent_f1_macro)


def train_and_validate(model,
                       train_loader,
                       val_loader,
                       # criterion,
                       parent_criterion,
                       # child_criterion,
                       scheduler,
                       num_epochs,
                       epoch_start=0,
                       alpha=0.5,
                       beta=0.5,
                       state=1,
                       optimizer=None,
                       resume_checkpoint=None,
                       ):
    """
    :param model: 模型
    :param train_loader: 训练集
    :param val_loader: 验证集
    :param parent_criterion: 公共类损失函数
    :param scheduler: 调度器
    :param num_epochs: 运行多少轮
    :param epoch_start: 从第几轮开始
    :return:
    """
    alpha = 1.0
    beta = 0.0
    sleep_all_spend = 0
    EPOCH_PERIOD = 50
    EPOCH_PERIOD_time = 600
    best_val_acc, best_epoch, train_acc_at_best = _restore_best_state(resume_checkpoint, epoch_start)
    # num_epochs 即目标总轮数：续跑时自动"补到目标轮"，而非"再跑 MAX_EPOCHS 轮"
    for epoch in range(epoch_start, num_epochs):
        # 休息一下吧
        if epoch % EPOCH_PERIOD == 0 and (epoch - epoch_start) >= EPOCH_PERIOD:
            print(f"休息一下~\n现在是{datetime.now()}")
            time.sleep(EPOCH_PERIOD_time)
            sleep_all_spend += EPOCH_PERIOD_time
            print("开始继续工作！")
            time.sleep(5)  # 防止print信息串行
        state = _advance_unfreeze_stage(model, state, epoch)

        # 每个绝对 epoch 独立播种采样器：无论中途重启多少次，同一 run_seed/epoch
        # 都会得到相同的 WeightedRandomSampler 索引序列；persistent_workers 保持不变。
        sampler_seed = _derive_sampler_seed(SEED, epoch)
        sampler_generator = getattr(train_loader.sampler, 'generator', None)
        if sampler_generator is None:
            raise RuntimeError('训练采样器缺少独立 generator，无法应用逐 epoch sampler seed')
        sampler_generator.manual_seed(sampler_seed)
        logging.info(f'epoch {epoch} sampler_seed={sampler_seed}')

        TV__time_spend_start = time.time()
        epoch_start_time = time.time()
        model.train()  # 设置模型为训练模式
        running_loss = 0.0

        # correct = 0
        total = 0

        parent_correct = 0

        # 训练过程
        with tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]", unit="batch") as tq:
            # for inputs, (parent_labels, child_labels) in tq:
            for inputs, parent_labels in tq:
                inputs = inputs.to(device, non_blocking=True)  # 将数据转移到 GPU

                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    print("Input contains NaN/Inf!")
                if torch.isnan(parent_labels).any() or torch.isinf(parent_labels).any():
                    print("Labels contain NaN/Inf!")

                parent_labels = parent_labels.to(device, non_blocking=True).long()

                # child_labels = child_labels.to(device).long()
                optimizer.zero_grad()  # 清除梯度

                # 使用 autocast 进行前向传播（混合精度）
                with autocast():  # 自动选择精度（FP16 或 FP32）
                    # parent_output, child_output = model(inputs)
                    parent_output = model(inputs)
                    if torch.isnan(parent_output).any():
                        print("Model output contains NaN!")

                    parent_loss = parent_criterion(parent_output, parent_labels)
                    child_loss = 0
                    loss = alpha * parent_loss + beta * child_loss

                    # 更新子类正确数和总数
                    # child_correct += child_correct_batch
                    # child_total += child_total_batch

                scaler.scale(loss).backward()  # 反向传播

                # AMP 下 backward 得到的是缩放后的梯度；必须先还原，再按真实范数裁剪。
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                scaler.step(optimizer)  # 更新权重
                scaler.update()  # 更新 GradScaler

                running_loss += loss.item()

                _, parent_predicted = torch.max(parent_output, 1)
                parent_correct += (parent_predicted == parent_labels).sum().item()
                total += inputs.size(0)

                tq.set_postfix(loss=running_loss / (tq.n + 1))  # 显示平均损失

            s = f"Epoch [{epoch}/{num_epochs - 1}], Loss: {running_loss / len(train_loader)}"
            print(s, file=fp)
            logging.info(s)

        # 在每个 epoch 结束时，计算总的子类准确率
        # if child_total > 0:
        #     train_child_acc = 100 * child_correct / child_total
        # else:
        #     train_child_acc = 0.0

        train_loss = running_loss / len(train_loader)

        # total_elements_parent = total

        train_parent_acc = 100 * parent_correct / total

        s = f'Train Accuracy Parent: {train_parent_acc:.5f}%'
        print(s, file=fp)
        # print(s)
        logging.info(s)

        # s = f'Train Accuracy Child: {train_child_acc:.5f}%'
        # print(s, file=fp)
        # print(s)
        # print(f"train_child, number", f"{child_correct=}, ", f"{child_total=}")
        # logging.info(s)

        gradient_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]))

        learning_rate = optimizer.param_groups[0]['lr']

        # 每个 epoch 后验证
        val_loss, val_accuracy, precision, recall, f1, f1_macro = validate(model,
                                                                           val_loader,
                                                                           parent_criterion,
                                                                           alpha=1.0,
                                                                           beta=0.0)
        # 实时保存 best（只存 state_dict，刷新才覆盖；最后用 best.pth 测测试集）
        # 冒烟（SMOKE）不存任何 pth：避免留下断点让 run_queue 误判"中断需续跑"；metrics 记录已足够验证管线
        if val_accuracy[0] > best_val_acc and not SMOKE:
            best_val_acc = val_accuracy[0]
            best_epoch = epoch
            train_acc_at_best = train_parent_acc
            torch.save({'epoch': epoch, 'config_id': CONFIG_ID, 'seed': SEED,
                        'arch': type(model).__name__,
                        'model_metadata': _model_metadata(model),
                        'model_signature_sha256': _model_identity(model)[1],
                        'experiment_signature': EXPERIMENT_SIGNATURE,
                        'normalization_signature': NORMALIZATION_SIGNATURE,
                        'backbone': BACKBONE_NAME, 'unfreeze_epochs': UNFREEZE_EPOCHS,
                        'training_policy': TRAINING_POLICY,
                        'state_dict': model.state_dict(), 'val_acc': val_accuracy[0],
                        'best_val_acc': best_val_acc,
                        'best_epoch': best_epoch,
                        'train_acc_at_best': train_acc_at_best,
                        'training_protocol_version': TRAINING_PROTOCOL_VERSION,
                        'sampling_protocol_version': SAMPLING_PROTOCOL_VERSION,
                        'sampler_seed': sampler_seed,
                        **_label_metadata(id_to_main_class, renew_class_to_index)},
                       BEST_PTH)
            logging.info(f'best 刷新: epoch {epoch}, val_acc {val_accuracy[0]:.4f}% -> {BEST_PTH}')
        # scheduler.step(val_loss)
        print(val_accuracy)
        # 调度器 step：CAWR 的 step(epoch=None) 无参递增；RLRP 才需要传 val 指标。
        # ⚠️ 2026-09-07 修复：原代码 state2/3 一律 step(val_accuracy[0])，对 RLRP 正确，
        #    但对 CAWR 会把 val_acc 误当 epoch → CAWR 永不重启、lr 贴地板（G-Full-CAWR 70.82 被污染）。
        if unfreeze_mode == 'none' or SCHEDULER_MODE == 'cawr':
            scheduler.step()
        elif state in [2, 3]:
            scheduler.step(val_accuracy[0])
        else:
            scheduler.step()
        epoch_end = time.time()
        epoch_time = (epoch_end - epoch_start_time - sleep)
        minute = epoch_time // 60
        second = epoch_time % 60

        s = f"第{epoch}轮, 花费时常:{minute}分{second:.2f}秒!"
        print(s, file=fp)
        logging.info(s)
        # 如果有任意ACC超过了预定值, 就会把这次模型强制保存, 且只会保存一次, 以防后续模型过拟合, 留样
        if val_accuracy[0] > ACC_STOP or val_accuracy[1] > ACC_STOP:
            save_model_after_epoch(epoch=epoch,
                                   model=model,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   renew_class_to_index=renew_class_to_index,
                                   state=state,
                                   config_id=CONFIG_ID,
                                   seed=SEED,
                                   sampler_seed=sampler_seed,
                                   sampling_protocol_version=SAMPLING_PROTOCOL_VERSION,
                                   id_to_main_class=id_to_main_class,
                                   training_protocol_version=TRAINING_PROTOCOL_VERSION,
                                   scaler_state_dict=scaler.state_dict(),
                                   other_pth_file_name=os.path.join(RUNS_DIR, f'ACC_STOP-VAL-{ACC_STOP}.pth'))
        # if train_parent_acc > ACC_STOP or train_child_acc > ACC_STOP:
        #     save_model_after_epoch(epoch=epoch,
        #                            model=model,
        #                            optimizer=optimizer,
        # state = state,
        #                            other_pth_file_name=f'ACC_STOP-TRAIN-{ACC_STOP}.pth')

        error_rate_num = ACC_STOP * (1 - ERROR_RATE)
        if (train_parent_acc > ACC_STOP) and \
                (val_accuracy[0] < error_rate_num and val_accuracy[0] < error_rate_num):
            save_model_after_epoch(epoch=epoch,
                                   model=model,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   renew_class_to_index=renew_class_to_index,
                                   state=state,
                                   config_id=CONFIG_ID,
                                   seed=SEED,
                                   sampler_seed=sampler_seed,
                                   sampling_protocol_version=SAMPLING_PROTOCOL_VERSION,
                                   id_to_main_class=id_to_main_class,
                                   training_protocol_version=TRAINING_PROTOCOL_VERSION,
                                   scaler_state_dict=scaler.state_dict(),
                                   other_pth_file_name=os.path.join(RUNS_DIR, f'ACC_STOP-OVERFIT-{ACC_STOP}.pth'))

        if save_model_after_epoch_flag:
            save_model_after_epoch(epoch=epoch,
                                   model=model,
                                   renew_class_to_index=renew_class_to_index,
                                   state=state,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   config_id=CONFIG_ID,
                                   seed=SEED,
                                   sampler_seed=sampler_seed,
                                   sampling_protocol_version=SAMPLING_PROTOCOL_VERSION,
                                   id_to_main_class=id_to_main_class,
                                   training_protocol_version=TRAINING_PROTOCOL_VERSION,
                                   scaler_state_dict=scaler.state_dict())

        TV__time_spend_end = time.time()

        if save_model_indicators:
            save_epoch = add_record_metrics(epoch=epoch,
                                            train_acc_parent=train_parent_acc,
                                            # train_acc_child=train_child_acc,

                                            val_acc_parent=val_accuracy[0],
                                            # val_acc_child=val_accuracy[1],

                                            train_loss=train_loss,
                                            val_loss=val_loss,

                                            precision_parent=precision[0],
                                            # precision_child=precision[1],

                                            recall_parent=recall[0],
                                            # recall_child=recall[1],

                                            f1_parent=f1[0],
                                            # f1_child=f1[1],

                                            gradient_norm=gradient_norm,
                                            learning_rate=learning_rate,
                                            time_spend=TV__time_spend_end - TV__time_spend_start - sleep,
                                            state=state
                                            )
            s = f"模型指标已经保存, 第{save_epoch}轮! 当前state状态为: {state}"
            logging.info(s)
            print(s)

        if save_model_indicators:
            add_record_metrics_v2(epoch=epoch,
                                  config_id=CONFIG_ID,
                                  seed=SEED,
                                  sampler_seed=sampler_seed,
                                  label_schema_version=LABEL_SCHEMA_VERSION,
                                  label_mapping_hash=mapping_fingerprint(id_to_main_class, renew_class_to_index),
                                  num_classes=len(renew_class_to_index),
                                  train_acc=train_parent_acc,
                                  train_loss=train_loss,
                                  val_acc=val_accuracy[0],
                                  val_loss=val_loss,
                                  val_macro_f1=f1_macro,
                                  val_weighted_f1=f1[0],
                                  lr=learning_rate,
                                  gradient_norm=gradient_norm.item(),
                                  time_spend=TV__time_spend_end - TV__time_spend_start - sleep,
                                  state=state)

        # 每轮保存断点（state_dict + optimizer + scheduler），支持中途暂停后续跑
        # 冒烟（SMOKE）跳过：不留 checkpoint_*_last.pth，run_queue 才不会被误导而 RESUME
        if not SMOKE:
            torch.save({'epoch': epoch, 'config_id': CONFIG_ID, 'seed': SEED,
                        'arch': type(model).__name__,
                        'model_metadata': _model_metadata(model),
                        'model_signature_sha256': _model_identity(model)[1],
                        'experiment_signature': EXPERIMENT_SIGNATURE,
                        'normalization_signature': NORMALIZATION_SIGNATURE,
                        'backbone': BACKBONE_NAME, 'unfreeze_epochs': UNFREEZE_EPOCHS,
                        'training_policy': TRAINING_POLICY,
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_val_acc': best_val_acc,
                        'best_epoch': best_epoch,
                        'train_acc_at_best': train_acc_at_best,
                        'training_protocol_version': TRAINING_PROTOCOL_VERSION,
                        'sampling_protocol_version': SAMPLING_PROTOCOL_VERSION,
                        'sampler_seed': sampler_seed,
                        'state': state,
                        **_label_metadata(id_to_main_class, renew_class_to_index)},
                       LAST_PTH)

        # Long 配置：跑到 SNAPSHOT_EPOCH 轮时，把该轮的 checkpoint + metrics 单独复制存档（100 轮快照）
        if SNAPSHOT_EPOCH is not None and epoch == SNAPSHOT_EPOCH - 1:
            try:
                import shutil
                _snap_last = os.path.join(RUNS_DIR, f'checkpoint_{RUN_TAG}_epoch{SNAPSHOT_EPOCH}.pth')
                _snap_best = os.path.join(RUNS_DIR, f'checkpoint_{RUN_TAG}_best_epoch{SNAPSHOT_EPOCH}.pth')
                _snap_metrics = os.path.join(RUNS_DIR, f'metrics_record_{RUN_TAG}_epoch{SNAPSHOT_EPOCH}.csv')
                shutil.copy(LAST_PTH, _snap_last)
                if os.path.exists(BEST_PTH):
                    shutil.copy(BEST_PTH, _snap_best)
                if os.path.exists(os.path.join(RUNS_DIR, 'metrics_record.csv')):
                    shutil.copy(os.path.join(RUNS_DIR, 'metrics_record.csv'), _snap_metrics)
                logging.info(f'[Long快照] 第 {SNAPSHOT_EPOCH} 轮快照已存档：checkpoint/metrics 各复制一份')
            except Exception as _e:
                logging.warning(f'[Long快照] 存档失败({_e})，不影响训练')

        # 温度守护：关键 epoch（10-90，0 起算）后检测 CPU/GPU 温度，过高则休息 10 分钟
        if epoch in temp_guard_lis:
            temp_guard.check_temps_and_rest(tag=f" epoch{epoch}")

        time.sleep(sleep)
        sleep_all_spend += sleep * 2

    # 训练结束：写本运行的结果汇总 JSON（一次性数据放这里，不塞逐轮 CSV）
    _results = {
        'config_id': CONFIG_ID,
        'seed': SEED,
        'num_classes': len(renew_class_to_index),
        'params': sum(p.numel() for p in model.parameters()),
        'batch_size': batch_size,
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'epochs_run': [epoch_start, num_epochs - 1],
        'best_val_epoch': best_epoch,
        'best_val_acc': round(best_val_acc, 4),
        'train_acc_at_best': round(train_acc_at_best, 4),
        'training_protocol_version': TRAINING_PROTOCOL_VERSION,
        'sampling_protocol_version': SAMPLING_PROTOCOL_VERSION,
        'label_schema_id': LABEL_SCHEMA_ID,
        'label_schema_version': LABEL_SCHEMA_VERSION,
        'label_schema_definition_hash': LABEL_SCHEMA_DEFINITION_HASH,
        'label_mapping_hash': mapping_fingerprint(id_to_main_class, renew_class_to_index),
        'sampler_seed_scheme': f'run_seed*{SAMPLER_SEED_STRIDE}+epoch',
        'last_sampler_seed': _derive_sampler_seed(SEED, num_epochs - 1),
        'amp_gradient_clipping': 'unscale_before_clip',
        'peak_gpu_mb': round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1) if torch.cuda.is_available() else 0,
        'unfreeze_mode': unfreeze_mode,
        'model_metadata': _model_metadata(model),
        'model_signature_sha256': _model_identity(model)[1],
        'experiment_signature': EXPERIMENT_SIGNATURE,
                        'normalization_signature': NORMALIZATION_SIGNATURE,
        'backbone': BACKBONE_NAME, 'unfreeze_epochs': UNFREEZE_EPOCHS,
        'training_policy': TRAINING_POLICY,
        'save_time': datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RUNS_DIR, f'results_{RUN_TAG}.json'), 'w', encoding='utf-8') as f:
        json.dump(_results, f, ensure_ascii=False, indent=2)
    logging.info(f'结果汇总已保存: results_{RUN_TAG}.json -> {_results}')

    return num_epochs, sleep_all_spend


def main(num_epochs, epoch_start=0, checkpoint=None, optimizer_scheduler_YN=False, optimizer=None):
    """
    :param num_epochs: epoch几轮
    :param epoch_start: 从多少轮开始, 默认从0开始时, 不为0时一般是继续训练
    :return:
    """
    # # 使用 Adam 优化器
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    #
    # 使用交叉熵损失函数（适用于分类任务）
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # criterion = FocalLoss(gamma=2.0, alpha=None).to(device)

    # class_proportion = get_class_proportion()
    # class_proportion_parent = class_proportion[0].to(device)
    # class_proportion_child = class_proportion[1].to(device)

    weight = get_class_weight(
        renew_class_to_index=renew_class_to_index,
        # main_class_to_index=main_class_to_index,
        #                       id_to_child_class=id_to_child_class,
        merged_dict_user=merged_dict,
        id_to_main_class=id_to_main_class
    )

    # print(f"CrossEntropyLoss权重-parent:{len(torch.tensor(weight[1]))}")
    # print(f"CrossEntropyLoss权重-child:{len(torch.tensor(weight[0]))}")

    # parent_criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=torch.tensor(weight[1], device=device)).to(device)
    # 旧的
    # parent_criterion = nn.CrossEntropyLoss(label_smoothing=0.1,
    #                                        weight=torch.tensor(weight, device=device)
    #                                        ).to(device)
    # 新的 保留采样器，去掉损失权重， 2025_12_11
    parent_criterion = nn.CrossEntropyLoss(label_smoothing=0.1,
                                           ).to(device)
    # child_criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)  # 移除权重参数避免维度不匹配

    mean, std, _ = calculate_mean_and_std(file, retrieve_file=True)


    # 设置数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # DynamicMeanPad(target_size=299),  # 对小图片补边
        # transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),

        # 以下这些是暂时屏蔽的
        # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        # transforms.RandomHorizontalFlip(p=0.3),
        # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        # # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
        # transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        # 以上这些是暂时屏蔽的

        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),  # 随机擦除, 且应该在归一化之后
    ])

    # 在main函数中调用验证
    # train_dataset = SafeImageFolder(root=file, transform=train_transform, train=True,
    #                                 id_to_main_class=id_to_main_class,
    #                                 id_to_child_class=id_to_child_class,
    #                                 main_class_to_index=main_class_to_index)
    # verify_label_mappings(train_dataset, id_to_main_class, id_to_child_class, main_class_to_index)
    # sys.exit(0)

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    """
    train_transform = A.Compose([
        # 1. Resize 和 RandomResizedCrop
        # Albumentations 的 RandomResizedCrop 已经包含了缩放功能，可以一步到位。
        # 替换方案：先缩放，再随机裁剪，效果等价
        A.Resize(height=256, width=256),
        A.RandomCrop(height=224, width=224),        # 2. AutoAugment 的替代方案
        # Albumentations 没有直接的 AutoAugment，但提供了更灵活的组合工具。
        # 这里我们用一些强大的增强来模拟其效果，例如 OneOf, SomeOf 等。
        # 这是一个非常强大的组合示例：
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.RandomRotate90(p=1),
        ], p=0.5),  # 50% 的概率执行其中一种翻转或旋转

        A.OneOf([
            A.MotionBlur(p=1),
            A.GaussNoise(p=1),
            A.ISONoise(p=1)
        ], p=0.5),  # 50% 的概率施加一种噪声或模糊

        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.HueSaturationValue(p=1),
        ], p=0.5),  # 50% 的概率调整亮度/对比度或色相/饱和度
        # 3. Normalize (归一化)
        # 注意：Normalize 必须在 ToTensorV2 之前！
        A.Normalize(mean=mean, std=std),
        # 4. RandomErasing 的等效实现
        # CoarseDropout 是 Albumentations 中与 RandomErasing 功能最接近的。
        # max_holes=1 确保只有一个擦除区域。
        # min/max_height/width 设置为图像尺寸的百分比，模拟 scale=(0.02, 0.2)
        A.CoarseDropout(max_holes=1, max_height=int(224*0.2), max_width=int(224*0.2), p=0.5),
        # 5. 转换为 Tensor
        # 必须使用 ToTensorV2
        ToTensorV2(),
    ])
    # 验证集的数据增强
    val_transforms = A.Compose([
        # 先缩放到短边为 256
        A.SmallestMaxSize(max_size=256),
        # 然后中心裁剪到 224x224
        A.CenterCrop(height=224, width=224),
        # 归一化
        A.Normalize(mean=mean, std=std),
        # 转换为 Tensor
        ToTensorV2(),
    ])
    """


    # 加载数据集
    train_dataset = SafeImageFolder(root=file, transform=train_transform,
                                    train=True,
                                    id_to_main_class=id_to_main_class,
                                    renew_class_to_index=renew_class_to_index,
                                    # id_to_child_class=id_to_child_class,
                                    # main_class_to_index=main_class_to_index,
                                    # child_class_to_index=child_class_to_index
                                    )
    val_dataset = SafeImageFolder(root=VAL_ROOT, transform=val_transforms,
                                  train=False,
                                  id_to_main_class=id_to_main_class,
                                  renew_class_to_index=renew_class_to_index,
                                  # id_to_child_class=id_to_child_class,
                                  # main_class_to_index=main_class_to_index,
                                  # child_class_to_index=child_class_to_index
                                  )
    """
    # 新的
    train_dataset = NewSafeImageFolder(root=file, transform=train_transform,
                                       train=True,
                                       id_to_main_class=id_to_main_class,
                                       renew_class_to_index=renew_class_to_index,
                                       # id_to_child_class=id_to_child_class,
                                       # main_class_to_index=main_class_to_index,
                                       # child_class_to_index=child_class_to_index
                                       )
    val_dataset = NewSafeImageFolder(root='../val', transform=val_transforms,
                                     train=False,
                                     id_to_main_class=id_to_main_class,
                                     renew_class_to_index=renew_class_to_index,
                                     # id_to_child_class=id_to_child_class,
                                     # main_class_to_index=main_class_to_index,
                                     # child_class_to_index=child_class_to_index
                                     )
    """

    # cached_dataset = DynamicCachedDataset(train_dataset)
    labels = [s[1] for s in train_dataset.samples]  # 永远与 samples 同步
    # labels = [train_dataset.targets[i] for i in range(len(train_dataset))]  # 获取所有样本的标签
    # print(f"train_dataset's labels {labels=}")
    # 逆频加权的口径 = **合并后的输出类**（当前 252 类），不是 265 个原始文件夹：
    # 合并组内各成员文件夹取同一个权重（= 1 / 该输出类总样本数），否则"按类别平衡"
    # 不成立（例：蔬菜与菜根菜叶并为「菜叶」后，两者仍按各自文件夹规模加权）。
    # 详见 calculate.get_output_class_sample_counts 的说明。
    class_sample_counts = get_output_class_sample_counts()
    weights = 1. / class_sample_counts.clone().detach().float()
    # print(weights)
    sample_weights = weights[labels]  # labels 是每个样本的类别标签

    sampler_generator = torch.Generator()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),
                                    replacement=True, generator=sampler_generator)
    #

    # subset_indices = random.sample(range(len(train_dataset)), int(0.2 * len(train_dataset)))
    # train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    # train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        # cached_dataset,
        batch_size=batch_size,
        # shuffle=True,
        # worker 数集中在 src_v2/__init__.py 的 TRAIN_NUM_WORKERS（2026-09-26：8→6 压提交内存峰值）
        num_workers=TRAIN_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        sampler=sampler,
        prefetch_factor=4,
        collate_fn=hierarchical_collate_fn,
        generator=torch.Generator().manual_seed(SEED)
    )
    # ⚠️ persistent_workers=True 不可省（2026-09-26 修）：
    #   若 val_loader 是"非持久"的，每轮验证都会在 8 个常驻训练 worker 存活的同时
    #   新建 4 个 worker；Windows 下新建 worker 需要重新加载整套模块链（提交内存峰值出现在
    #   这一刻），一旦第 4 个加载失败（WinError 1455），它会在向父进程注册前就死掉，
    #   主进程随即永久阻塞在启动握手 —— 2026-09-26 01:43 的停摆正卡在
    #   「Epoch 0 训练结束 → 首次 iter(val_loader)」这一步（日志停在 Train Accuracy Parent，
    #   之后 Validation Accuracy Parent 再无输出）。
    #   常驻后 worker 只在首次验证创建一次，后续每轮复用，既消除峰值也省掉重复的启动开销。
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            # worker 数集中在 src_v2/__init__.py 的 VAL_NUM_WORKERS（2026-09-26：4→2）
                            num_workers=VAL_NUM_WORKERS,
                            pin_memory=True,
                            persistent_workers=True,
                            collate_fn=hierarchical_collate_fn,
                            generator=torch.Generator().manual_seed(SEED)
                            )

    # 定义总步数和热身步数
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * 0.1)  # 热身占 10%

    # 调度器
    # scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.8, mode='min', threshold=0.01)

    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_steps
    # )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=TRAINING_POLICY['t_0'], T_mult=TRAINING_POLICY['t_mult'],
        eta_min=TRAINING_POLICY['eta_min'])
    if optimizer_scheduler_YN:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info('已恢复 scheduler 状态（续跑无 LR 跳变）')
        # ⚠️ 2026-09-22 移除"续跑时把 lr 抬到 ≥1e-5"的补丁（src 时代遗留）：
        #    它存在的理由是当时 scheduler 状态不恢复（见 src/train.py 第 697 行被注释），
        #    续跑只能靠抬 lr 以避免永久卡在 eta_min=1e-6 地板；v2 已能恢复 scheduler 状态
        #    （上一行），根因消除，而该补丁会在续跑后第 1 个 epoch 静默覆盖刚恢复的调度值，
        #    与论文声明的 CAWR(eta_min=1e-6) 不一致（上一句还打印"续跑无 LR 跳变"）。
        #    核查依据：被抬过的轮次 lr 恰好等于 1e-05，可当指纹；全量 13 份 metrics 归档中
        #    仅 5 处命中，且全部落在已废弃/已排除的 run（脏跑 G-Full-CAWR、废弃的 Long 第 1 次、
        #    RLRP 时代 G-NoSE seed2）——论文用到的每个 run 都未执行过该分支，
        #    故本次移除不改变任何已得结果（详见 findings.md §十七）。
        # current_lr = optimizer.param_groups[0]['lr']
        # if current_lr < 1e-5:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = max(param_group['lr'], 1e-5)
        #         param_group['lr'] = min(param_group['lr'], 1e-3)

    # raw_dataset = SafeImageFolder(root=file, transform=transform)

    # 包裹成缓存数据集

    # train_loader_len = len(train_loader)
    start = time.time()

    epoch, sleep_all_spend = train_and_validate(model,
                                                train_loader,
                                                val_loader,
                                                parent_criterion,
                                                scheduler,
                                                num_epochs,
                                                epoch_start,
                                                state=state,
                                                optimizer=optimizer,
                                                resume_checkpoint=checkpoint
                                                )

    end = time.time()

    # print(f"花费时常:{(end - start - (sleep * 2 * num_epochs)) // 60}分钟!", file=fp)
    # s = f"花费时常:{(end - start - sleep_all_spend) // 60}分钟!"
    # print(s, file=fp)
    # logging.info(s)

    if not SMOKE and not save_model_after_epoch_flag:
        # 这里是全部运行完, 如果没有设置每轮保存, 则在此保存, 若开启每轮保存, 则无需再进行保存
        # train_and_validate 返回的是目标总轮数；共享模型的 epoch 元数据应写最后一个零基 epoch。
        final_epoch = num_epochs - 1
        save_model_after_epoch(epoch=final_epoch,
                               model=model,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               renew_class_to_index=renew_class_to_index,
                               state=state,
                               config_id=CONFIG_ID,
                               seed=SEED,
                               sampler_seed=_derive_sampler_seed(SEED, final_epoch),
                               sampling_protocol_version=SAMPLING_PROTOCOL_VERSION,
                               id_to_main_class=id_to_main_class,
                               training_protocol_version=TRAINING_PROTOCOL_VERSION,
                               scaler_state_dict=scaler.state_dict(),
                               )

    # checkpoint = {
    #     'epoch': epoch,
    #     'model': model,  # 保存整个模型对象
    #     'model_state_dict': model.state_dict(),  # 模型权重
    #     'optimizer_state_dict': optimizer.state_dict(),
    # }
    #
    # torch.save(checkpoint, pth_file_name)


def _check_arch(checkpoint, model):
    """自适应 arch 校验：断点记录的架构必须与当前模型一致，防止 RESUME 静默跑错模型。

    - 断点有 arch 且与当前模型不同 → 直接报错拒绝续跑；
    - 断点无 arch（旧格式历史断点）→ 警告并放行（按当前模型加载）。
    arch 值来自 type(model).__name__，无需硬编码，切 V2 后自动适配。
    """
    ck_arch = checkpoint.get('arch')
    cur_arch = type(model).__name__
    if ck_arch is None:
        logging.warning(f'[arch] 断点无 arch 标记（旧格式），按当前模型 {cur_arch} 加载')
    elif ck_arch != cur_arch:
        raise SystemExit(f'[arch] 断点架构 {ck_arch} ≠ 当前模型 {cur_arch}，拒绝续跑（防静默跑错模型）')


def _model_metadata(model):
    """生成稳定、JSON可序列化的结构元数据；用于断点身份校验和结果复核。"""
    if hasattr(model, 'architecture_metadata'):
        return model.architecture_metadata()

    transformer = getattr(model, 'transformer', None)
    layers = getattr(transformer, 'layers', None)
    se_block = getattr(model, 'se_block', None)
    classifier = getattr(model, 'parent_classifier', None)
    return {
        'architecture': type(model).__name__,
        'backbone': 'resnext50_32x4d',
        'num_classes': getattr(classifier, 'out_features', None),
        'regions': list(getattr(se_block, 'regions', [])) if hasattr(se_block, 'regions') else None,
        'transformer_layers': len(layers) if layers is not None else 0,
        'd_model': getattr(classifier, 'in_features', None),
        'nhead': None,
        'num_patches': 1,
        'pos_embed_type': None,
        'readout': 'single_token',
        'use_decouple': not isinstance(getattr(model, 'disentangle_layer', None), nn.Identity),
    }


def _model_identity(model):
    metadata = _model_metadata(model)
    payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    signature = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    return metadata, signature


def _check_model_identity(checkpoint, model):
    """新格式断点必须与当前精确结构一致；旧断点缺签名时保留兼容警告。"""
    if NORMALIZATION_SIGNATURE and checkpoint.get('normalization_signature') != NORMALIZATION_SIGNATURE:
        raise RuntimeError('断点归一化参数与当前配置不一致，拒绝续跑')
    if checkpoint.get('experiment_signature') != EXPERIMENT_SIGNATURE:
        raise RuntimeError('断点骨干/解冻时点/共同训练策略与当前实验不一致，拒绝续跑')
    expected_metadata, expected_signature = _model_identity(model)
    actual_signature = checkpoint.get('model_signature_sha256')
    if actual_signature is None:
        raise RuntimeError('V3断点缺少模型结构签名，拒绝续跑')
    if actual_signature != expected_signature:
        raise RuntimeError(
            '断点模型结构签名与当前配置不一致，拒绝续跑。'
            f'断点={actual_signature}，当前={expected_signature}；'
            f'断点结构={checkpoint.get("model_metadata")}，当前结构={expected_metadata}')


def _validate_model_before_training(model, preset):
    """正式开跑前的轻量门禁；V2额外执行真实224输入的CPU零样本前向。"""
    expected_v2 = preset.get('model') == 'v2'
    actual_v2 = isinstance(model, ResNetTransformerV2)
    if expected_v2 != actual_v2:
        raise RuntimeError(
            f'配置与模型实现不一致：preset model={preset.get("model", "v1")}, '
            f'实际模型={type(model).__name__}')

    metadata, signature = _model_identity(model)
    if metadata['backbone'] != preset['backbone']:
        raise RuntimeError('实际骨干与注册配置不一致')
    if metadata['feature_dim'] != model_registry.BACKBONE_FEATURE_DIMS[preset['backbone']]:
        raise RuntimeError('骨干输出通道与注册配置不一致')
    if metadata['num_classes'] != EXPECTED_NUM_CLASSES:
        raise RuntimeError(
            f'模型输出头应为 {EXPECTED_NUM_CLASSES} 类，实际 {metadata["num_classes"]} 类')

    if expected_v2:
        expected_layers = int(preset['transformer_layers'])
        if metadata['transformer_layers'] != expected_layers:
            raise RuntimeError(
                f'V2 Transformer层数不一致：配置={expected_layers}，模型={metadata["transformer_layers"]}')
        was_training = model.training
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 224, 224))
        model.train(was_training)
        if tuple(output.shape) != (1, EXPECTED_NUM_CLASSES):
            raise RuntimeError(
                f'V2启动前向输出形状错误：预期 (1,{EXPECTED_NUM_CLASSES})，实际 {tuple(output.shape)}')
        logging.info(f'[V2_PRECHECK] 224输入/49-token/252类前向通过；signature={signature}')
    else:
        logging.info(f'[MODEL_IDENTITY] signature={signature}')
    return metadata, signature


def _build_model(preset, transformer_layers, d_model, nhead, id_to_main_class, renew_class_to_index):
    """按当前注册骨干构造同一NoTF空间头；不回退到其他模型。"""
    if preset.get('model') != 'v2':
        raise ValueError('当前V3注册仅支持空间模型实现')
    return ResNetTransformerV2(
        transformer_layers=transformer_layers, d_model=d_model, nhead=nhead,
        id_to_main_class=id_to_main_class, renew_class_to_index=renew_class_to_index,
        regions=preset['regions'], use_decouple=preset['use_decouple'],
        backbone_name=preset['backbone'], pretrained=preset['pretrained'],
        backbone_bn_policy=TRAINING_POLICY['backbone_bn_policy'])


if __name__ == '__main__':
    _setup_logging()  # 主进程才配置日志；spawn worker 不进入此块，不会重复打印
    preset = PRESETS[CONFIG_ID]
    seed_all(SEED)
    state = 1
    d_model = 512
    transformer_layers = preset['transformer_layers']
    nhead = 8
    num_epochs = 3 if SMOKE else MAX_EPOCHS
    # fp = open("logging.txt", mode='a')
    fp = sys.stdout
    file = TRAIN_ROOT
    epoch_start = 0
    num_epochs_user_input = 0

    # logging_demand_flag = True
    print(f"模型训练使用device:{device}", file=fp)
    s = f"模型训练使用device:{device}"
    logging.info(s)

    key = 'n'
    checkpoint = None

    # id_to_main_class, id_to_child_class, main_class_to_index, child_class_to_index = generate_mappings(
    #     merged_dict=merged_dict)
    id_to_main_class, renew_class_to_index = generate_mappings(merged_dict=merged_dict)
    if len(renew_class_to_index) != EXPECTED_NUM_CLASSES:
        raise RuntimeError(
            f'当前标签映射应生成 {EXPECTED_NUM_CLASSES} 类，实际 {len(renew_class_to_index)} 类，拒绝训练')
    logging.info(
        f'[标签协议] {LABEL_SCHEMA_ID} v{LABEL_SCHEMA_VERSION}, '
        f'{len(renew_class_to_index)} 类, hash={mapping_fingerprint(id_to_main_class, renew_class_to_index)}')

    # print(len(id_to_main_class), len(id_to_child_class), len(get_classnames()))
    # print(main_class_to_index, len(main_class_to_index))
    # sys.exit()
    # print(id_to_main_class, id_to_child_class, sep='\n' + '=' * 100 + '\n')
    # sys.exit()

    # print("父类索引映射:", main_class_to_index)
    # print("子类索引映射:", id_to_child_class)
    # print(id_to_main_class, id_to_child_class, main_class_to_index, sep="\n\n\n")
    # print(len(id_to_main_class), len(id_to_child_class), len(main_class_to_index), sep="\n\n\n")
    use_old_model = False
    if RESUME:
        _resume_path = LAST_PTH if os.path.exists(LAST_PTH) else pth_file_name
        if os.path.exists(_resume_path):
            use_old_model = True
            checkpoint = torch.load(_resume_path)
            _check_label_schema(checkpoint, id_to_main_class, renew_class_to_index)
            checkpoint_protocol = checkpoint.get('training_protocol_version', 1)
            if (checkpoint_protocol != TRAINING_PROTOCOL_VERSION and
                    not ALLOW_LEGACY_PROTOCOL_RESUME):
                raise RuntimeError(
                    f'断点训练协议 v{checkpoint_protocol} 与当前 v{TRAINING_PROTOCOL_VERSION} 不兼容：'
                    '旧断点在 AMP 梯度裁剪前未 unscale。为避免混合协议，正式实验请先归档旧产物并从 epoch 0 重跑；'
                    '仅为诊断而明确接受混合协议时，才在 experiment_config.json 设置 '
                    'ALLOW_LEGACY_PROTOCOL_RESUME=true。')
            checkpoint_sampling_protocol = checkpoint.get('sampling_protocol_version', 0)
            if (checkpoint_sampling_protocol != SAMPLING_PROTOCOL_VERSION and
                    not ALLOW_LEGACY_PROTOCOL_RESUME):
                raise RuntimeError(
                    f'断点采样协议 v{checkpoint_sampling_protocol} 与当前 v{SAMPLING_PROTOCOL_VERSION} 不兼容：'
                    '旧断点使用进程级连续随机流，当前实现使用逐 epoch sampler seed。'
                    '为避免混合采样协议，正式实验请从 epoch 0 重跑；仅诊断时才允许 legacy resume。')
            if checkpoint_sampling_protocol == SAMPLING_PROTOCOL_VERSION:
                checkpoint_seed = int(checkpoint.get('seed', SEED))
                if checkpoint_seed != int(SEED):
                    raise RuntimeError(f'断点 seed={checkpoint_seed} 与当前实验 seed={SEED} 不一致，拒绝续跑')
                checkpoint_epoch = int(checkpoint.get('epoch', -1))
                expected_sampler_seed = _derive_sampler_seed(checkpoint_seed, checkpoint_epoch)
                actual_sampler_seed = checkpoint.get('sampler_seed')
                if actual_sampler_seed is None or int(actual_sampler_seed) != expected_sampler_seed:
                    raise RuntimeError(
                        f'断点 sampler_seed={actual_sampler_seed} 与预期 {expected_sampler_seed} 不一致，拒绝续跑')
            state = checkpoint.get('state', state)
            epoch_start = checkpoint.get('epoch', -1) + 1
            if 'model' in checkpoint:
                # 旧格式：整个模型对象
                model = checkpoint['model']
                _check_arch(checkpoint, model)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 新格式：state_dict 断点文件
                model = _build_model(preset, transformer_layers, d_model, nhead,
                                     id_to_main_class, renew_class_to_index)
                _check_arch(checkpoint, model)
                _check_model_identity(checkpoint, model)
                model.load_state_dict(checkpoint['state_dict'])
            logging.info(f'RESUME: 从 {_resume_path} 继续, epoch {epoch_start}, state={state}')
        else:
            logging.warning('RESUME=True 但未找到断点文件，改为全新训练')
    if not use_old_model:
        # 从头开始训练（按 PRESETS 预设开关）
        model = _build_model(preset, transformer_layers, d_model, nhead,
                             id_to_main_class, renew_class_to_index)

    model_metadata, model_signature_sha256 = _validate_model_before_training(model, preset)

    # 设置优化器
    print(f"设置优化器===")
    for i, m in enumerate(model.resnet_backbone):
        print(i, type(m))
    # 续跑修复：先把模型解冻到断点记录的 state，再建优化器。
    # 否则 state=2/3 续跑时 get_grouped_params 拿到的是空 backbone 组，
    # optimizer.load_state_dict 会报 "parameter group doesn't match the size"。
    if unfreeze_mode == 'progressive':
        if state >= 3:
            model.unfreeze_all()
        elif state == 2:
            model.unfreeze_layer4()
    # state
    if unfreeze_mode == 'none':
        model.unfreeze_all()
        state = 3
    optimizer = _make_optimizer(model)
    if key in ['y', 'yes']:
        optimizer_scheduler_YN = True
    else:
        optimizer_scheduler_YN = use_old_model
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.8)

    scaler = GradScaler()
    if use_old_model:
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logging.info('已恢复 GradScaler 状态')
        else:
            logging.warning('旧断点不含 GradScaler 状态，将以新的缩放状态继续')
        if checkpoint.get('training_protocol_version') != TRAINING_PROTOCOL_VERSION:
            logging.warning('当前断点来自旧训练协议；修复 AMP 后继续会混合两种协议，正式实验应从 epoch 0 重跑')
        if checkpoint.get('sampling_protocol_version', 0) != SAMPLING_PROTOCOL_VERSION:
            logging.warning('当前断点来自旧采样协议；续跑会混合进程级随机流与逐 epoch sampler seed')
    model = model.to(device)  # 将模型转移到 GPU
    for name, module in model.named_children():
        print(name, " ---> ", type(module))
    if AUTO:
        num_epochs_user_input = num_epochs
        save_model_after_epoch_flag = False
        save_model_indicators = True
        print(f"AUTO 模式: CONFIG={CONFIG_ID}, SEED={SEED}, SMOKE={SMOKE}, "
              f"epochs={num_epochs_user_input}, resume={use_old_model}")
    else:
        try:
            num_epochs_user_input = int(input(f"请指定训练轮数[默认{num_epochs}]:").strip())
        except ValueError:
            num_epochs_user_input = num_epochs
        finally:
            if num_epochs_user_input <= 0:
                num_epochs_user_input = num_epochs

        save_model_after_epoch_flag = input("是否要每一轮epoch保存一次模型[默认n]?").strip()
        save_model_after_epoch_flag = save_model_after_epoch_flag in ['yes', 'y']

        save_model_indicators = input("是否要每一轮epoch保存模型指标[默认y]?").strip()
        save_model_indicators = save_model_indicators not in ['n', 'no']

        print("请务必再次确认:")
        print(f"使用历史模型继续训练:{use_old_model}")
        print(f"指定训练轮数为:{num_epochs_user_input}")
        print(f"每轮保存模型:{save_model_after_epoch_flag}")
        print(f"每轮保存模型指标:{save_model_indicators}")
        again_enable = input(f'如果您确认, 请在此输入"y/yes":')
        if not again_enable in ['y', 'yes']:
            sys.exit(0)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # ===== 正式全新训练前：把上次产物带时间戳归档（防误操作覆盖，绝不删除） =====
    # 触发条件：非冒烟 且 非续跑；冒烟/续跑不归档
    if not SMOKE and not use_old_model:
        _ts = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
        for _name in ['garbage_classification_model.pth', 'metrics.csv', 'metrics_record.csv', 'last_config.json',
                      f'checkpoint_{RUN_TAG}_best.pth', f'checkpoint_{RUN_TAG}_last.pth',
                      f'results_{RUN_TAG}.json']:
            _p = os.path.join(RUNS_DIR, _name)
            if os.path.exists(_p):
                _stem, _ext = os.path.splitext(_name)
                os.rename(_p, os.path.join(RUNS_DIR, f'{_stem}_archive_{_ts}{_ext}'))
                logging.info(f'已归档上次产物: {_name} -> {_stem}_archive_{_ts}{_ext}')
        # 保存本次运行配置快照（配置变了就归档旧的、开新的）
        with open(os.path.join(RUNS_DIR, 'last_config.json'), 'w', encoding='utf-8') as f:
            json.dump({'CONFIG_ID': CONFIG_ID, 'SEED': SEED, 'SMOKE': SMOKE,
                       'MAX_EPOCHS': MAX_EPOCHS, 'RESUME': RESUME, 'AUTO': AUTO,
                       'UNFREEZE1_EPOCH': UNFREEZE1_EPOCH, 'UNFREEZE2_EPOCH': UNFREEZE2_EPOCH,
                       'SAMPLING_PROTOCOL_VERSION': SAMPLING_PROTOCOL_VERSION,
                       'SAMPLER_SEED_STRIDE': SAMPLER_SEED_STRIDE,
                       'LABEL_SCHEMA_ID': LABEL_SCHEMA_ID,
                       'LABEL_SCHEMA_VERSION': LABEL_SCHEMA_VERSION,
                       'LABEL_SCHEMA_DEFINITION_HASH': LABEL_SCHEMA_DEFINITION_HASH,
                       'LABEL_MAPPING_HASH': mapping_fingerprint(id_to_main_class, renew_class_to_index),
                       'NUM_CLASSES': len(renew_class_to_index),
                       'MODEL_METADATA': model_metadata,
                       'MODEL_SIGNATURE_SHA256': model_signature_sha256},
                      f, ensure_ascii=False, indent=2)
        logging.info('已保存本次配置快照: runs/last_config.json')

    print(f"从{str(datetime.now())}运行", file=fp)
    s = f"从{str(datetime.now())}运行"
    logging.info(s)

    print(f"从第{epoch_start}轮继续, 本次目标轮数:{num_epochs_user_input}")

    main(num_epochs_user_input, epoch_start, checkpoint=checkpoint, optimizer_scheduler_YN=optimizer_scheduler_YN,
         optimizer=optimizer)

    print(f"以{str(datetime.now())}结束", file=fp)
    s = f"以{str(datetime.now())}结束"
    logging.info(s)
    # 如果fp为文件, 则需要关闭IO
    # fp.close()

    # 如果需要运行完定时关机, 则可以打开
    # os.system("shutdown -s -t 3600")
