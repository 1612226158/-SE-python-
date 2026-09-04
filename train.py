import random
import sys
import time
from datetime import datetime
import warnings
import logging
import os
import json

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
from calculate import (calculate_mean_and_std,
                       SafeImageFolder,
                       FocalLoss, add_record_metrics, add_record_metrics_v2,
                       get_classnames,
                       generate_mappings, get_class_proportion,
                       get_class_weight, hierarchical_collate_fn,
                       should_update_simple)
from calculate_update import SafeImageFolder as NewSafeImageFolder
from src_v2 import (device, DynamicMeanPad,
                    pth_file_name, save_model_after_epoch,
                    extensions_suffix, merged_dict, ACC_STOP, ERROR_RATE, UPDATE_N)

# from batch_size import adjust_batch_size

# ==================== 实验配置：编辑同目录 experiment_config.json（无需改本文件） ====================
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_config.json')
_DEFAULTS = {
    'CONFIG_ID': 'G-Full',   # G-Full / G-NoTF / G-NoSE / G-SingleSE / G-PureBB / S-NoProg
    'SEED': 0,               # 随机种子 0/1/2，每配置跑 3 次取 mean±std
    'SMOKE': True,           # True=冒烟只跑 3 轮验证管线；正式训练 False
    'MAX_EPOCHS': 100,
    'RESUME': False,         # True=从 runs 下断点续跑（支持中途暂停后继续）
    'AUTO': True,            # 无人值守：跳过所有交互输入
    'UNFREEZE1_EPOCH': 30,   # 固定解冻触发：state1→2
    'UNFREEZE2_EPOCH': 60,   # state2→3
}

# temp_guard_lis = (25, 50, 75, 99)
temp_guard_lis = (10, 20, 30, 40, 50, 60, 70, 80, 90)


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
    MAX_EPOCHS = 100
    RESUME = False
else:
    MAX_EPOCHS = _cfg.get('MAX_EPOCHS', _DEFAULTS['MAX_EPOCHS'])
    RESUME = _cfg.get('RESUME', _DEFAULTS['RESUME'])
AUTO = _cfg.get('AUTO', _DEFAULTS['AUTO'])
UNFREEZE1_EPOCH = _cfg.get('UNFREEZE1_EPOCH', _DEFAULTS['UNFREEZE1_EPOCH'])
UNFREEZE2_EPOCH = _cfg.get('UNFREEZE2_EPOCH', _DEFAULTS['UNFREEZE2_EPOCH'])

PRESETS = {
    'G-Full':     dict(regions=[1, 2], transformer_layers=3, use_decouple=True,  unfreeze='progressive'),
    'G-Full-CAWR':dict(regions=[1, 2], transformer_layers=3, use_decouple=True,  unfreeze='progressive', scheduler='cawr'),
    'G-NoTF':     dict(regions=[1, 2], transformer_layers=0, use_decouple=True,  unfreeze='progressive'),
    'G-NoSE':     dict(regions=None,    transformer_layers=3, use_decouple=True,  unfreeze='progressive'),
    'G-SingleSE': dict(regions=[1],     transformer_layers=3, use_decouple=True,  unfreeze='progressive'),
    'G-PureBB':   dict(regions=None,    transformer_layers=0, use_decouple=False, unfreeze='progressive'),
    'S-NoProg':   dict(regions=[1, 2], transformer_layers=3, use_decouple=True,  unfreeze='none'),
}
unfreeze_mode = PRESETS[CONFIG_ID]['unfreeze']
# state2/3 的调度器：'cawr'=余弦退火重启（G-Full-CAWR 用，与 S-NoProg 同类型，隔离"RLRP 地板"混淆）；缺省='rlrp'（原行为）
SCHEDULER_MODE = PRESETS[CONFIG_ID].get('scheduler', 'rlrp')
# 渐进式解冻的 backbone/head 学习率：G-Full-CAWR 用 1e-4（对齐 S-NoProg，隔离 lr 幅度混淆）；其余默认 5e-6/5e-5
BACKBONE_LR = 1e-4 if SCHEDULER_MODE == 'cawr' else 5e-6
HEAD_LR = 1e-4 if SCHEDULER_MODE == 'cawr' else 5e-5


def _make_state_scheduler(optimizer):
    """构建 state2/3 的调度器：SCHEDULER_MODE='cawr' 时用 CAWR，否则用 RLRP（原行为不变）。"""
    if SCHEDULER_MODE == 'cawr':
        return CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,
                             threshold=0.1, min_lr=1e-6, verbose=True)

# 路径全部基于本文件定位（与 PyCharm 工作目录无关）
SRC_V2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_V2_DIR)
RUNS_DIR = os.path.join(SRC_V2_DIR, 'runs')
os.makedirs(RUNS_DIR, exist_ok=True)
TRAIN_ROOT = os.path.join(PROJECT_ROOT, 'train')
VAL_ROOT = os.path.join(PROJECT_ROOT, 'val')

RUN_TAG = f'{CONFIG_ID}_seed{SEED}'
BEST_PTH = os.path.join(RUNS_DIR, f'checkpoint_{RUN_TAG}_best.pth')
LAST_PTH = os.path.join(RUNS_DIR, f'checkpoint_{RUN_TAG}_last.pth')


def seed_all(seed):
    """固定所有随机源（torch / numpy / python random）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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


def validate(model, val_loader, parent_criterion, alpha=0.5, beta=0.5):
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
    best_val_acc = -1.0
    best_epoch = -1
    train_acc_at_best = -1.0
    # num_epochs 即目标总轮数：续跑时自动"补到目标轮"，而非"再跑 MAX_EPOCHS 轮"
    for epoch in range(epoch_start, num_epochs):
        # 休息一下吧
        if epoch % EPOCH_PERIOD == 0 and (epoch - epoch_start) >= EPOCH_PERIOD:
            print(f"休息一下~\n现在是{datetime.now()}")
            time.sleep(EPOCH_PERIOD_time)
            sleep_all_spend += EPOCH_PERIOD_time
            print("开始继续工作！")
        if unfreeze_mode == 'progressive':
            # 固定 epoch 触发解冻（替换原 should_update_simple 自适应判据，保证消融可比）
            if state == 1 and epoch >= UNFREEZE1_EPOCH:
                model.unfreeze_layer4()
                optimizer = optim.AdamW(model.get_grouped_params(backbone_lr=BACKBONE_LR, head_lr=HEAD_LR),
                                        weight_decay=1e-2,
                                        )
                scheduler = _make_state_scheduler(optimizer)
                state += 1
                logging.info(f'state1 -> state2 于 epoch {epoch}（固定触发）')
            elif state == 2 and epoch >= UNFREEZE2_EPOCH:
                model.unfreeze_all()
                optimizer = optim.AdamW(model.get_grouped_params(backbone_lr=BACKBONE_LR, head_lr=HEAD_LR),
                                        weight_decay=1e-2,
                                        )
                scheduler = _make_state_scheduler(optimizer)
                state += 1
                logging.info(f'state2 -> state3 于 epoch {epoch}（固定触发）')
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
                                                                           parent_criterion)
        # 实时保存 best（只存 state_dict，刷新才覆盖；最后用 best.pth 测测试集）
        if val_accuracy[0] > best_val_acc:
            best_val_acc = val_accuracy[0]
            best_epoch = epoch
            train_acc_at_best = train_parent_acc
            torch.save({'epoch': epoch, 'config_id': CONFIG_ID, 'seed': SEED,
                        'arch': type(model).__name__,
                        'state_dict': model.state_dict(), 'val_acc': val_accuracy[0]},
                       BEST_PTH)
            logging.info(f'best 刷新: epoch {epoch}, val_acc {val_accuracy[0]:.4f}% -> {BEST_PTH}')
        # scheduler.step(val_loss)
        print(val_accuracy)
        if unfreeze_mode == 'none':
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
                                   other_pth_file_name=os.path.join(RUNS_DIR, f'ACC_STOP-OVERFIT-{ACC_STOP}.pth'))

        if save_model_after_epoch_flag:
            save_model_after_epoch(epoch=epoch,
                                   model=model,
                                   renew_class_to_index=renew_class_to_index,
                                   state=state,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   config_id=CONFIG_ID,
                                   seed=SEED)

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
        torch.save({'epoch': epoch, 'config_id': CONFIG_ID, 'seed': SEED,
                    'arch': type(model).__name__,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'state': state,
                    'renew_class_to_index': renew_class_to_index},
                   LAST_PTH)

        # 温度守护：关键 epoch（30/50/80，0 起算）后检测 CPU/GPU 温度，过高则休息 10 分钟
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
        'peak_gpu_mb': round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1) if torch.cuda.is_available() else 0,
        'unfreeze_mode': unfreeze_mode,
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
    class_sample_counts = get_class_proportion(pq='q')  # 每个类别的样本数
    weights = 1. / class_sample_counts.clone().detach().float()
    # print(weights)
    sample_weights = weights[labels]  # labels 是每个样本的类别标签

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),
                                    replacement=True, generator=torch.Generator().manual_seed(SEED))
    #

    # subset_indices = random.sample(range(len(train_dataset)), int(0.2 * len(train_dataset)))
    # train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    # train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        # cached_dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        sampler=sampler,
        prefetch_factor=4,
        collate_fn=hierarchical_collate_fn,
        generator=torch.Generator().manual_seed(SEED)
    )
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
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
    if unfreeze_mode == 'none' or state == 1:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,  # 每 5 轮重启一次
            T_mult=2,  # 周期递增
            eta_min=1e-6  # 最低学习率
        )
    else:
        # 续跑且 state>=2 时：建对应类型的调度器（G-Full-CAWR 用 CAWR，其余 RLRP），load_state_dict 才能对上
        scheduler = _make_state_scheduler(optimizer)
    if optimizer_scheduler_YN:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info('已恢复 scheduler 状态（续跑无 LR 跳变）')
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 1e-5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], 1e-5)
                param_group['lr'] = min(param_group['lr'], 1e-3)

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
                                                optimizer=optimizer
                                                )

    end = time.time()

    # print(f"花费时常:{(end - start - (sleep * 2 * num_epochs)) // 60}分钟!", file=fp)
    # s = f"花费时常:{(end - start - sleep_all_spend) // 60}分钟!"
    # print(s, file=fp)
    # logging.info(s)

    if not SMOKE and not save_model_after_epoch_flag:
        # 这里是全部运行完, 如果没有设置每轮保存, 则在此保存, 若开启每轮保存, 则无需再进行保存
        save_model_after_epoch(epoch=epoch,
                               model=model,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               renew_class_to_index=renew_class_to_index,
                               state=state,
                               config_id=CONFIG_ID,
                               seed=SEED,
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
            renew_class_to_index = checkpoint.get('renew_class_to_index', renew_class_to_index)
            state = checkpoint.get('state', state)
            epoch_start = checkpoint.get('epoch', -1) + 1
            if 'model' in checkpoint:
                # 旧格式：整个模型对象
                model = checkpoint['model']
                _check_arch(checkpoint, model)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 新格式：state_dict 断点文件
                model = ResNetTransformer(transformer_layers=transformer_layers,
                                          d_model=d_model,
                                          id_to_main_class=id_to_main_class,
                                          renew_class_to_index=renew_class_to_index,
                                          nhead=nhead,
                                          regions=preset['regions'],
                                          use_decouple=preset['use_decouple'])
                _check_arch(checkpoint, model)
                model.load_state_dict(checkpoint['state_dict'])
            logging.info(f'RESUME: 从 {_resume_path} 继续, epoch {epoch_start}, state={state}')
        else:
            logging.warning('RESUME=True 但未找到断点文件，改为全新训练')
    if not use_old_model:
        # 从头开始训练（按 PRESETS 预设开关）
        model = ResNetTransformer(transformer_layers=transformer_layers,
                                  d_model=d_model,
                                  id_to_main_class=id_to_main_class,
                                  renew_class_to_index=renew_class_to_index,
                                  nhead=nhead,
                                  regions=preset['regions'],
                                  use_decouple=preset['use_decouple'])

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
        # S-NoProg：全程全解冻 + 单一 AdamW（常规训练对照）
        model.unfreeze_all()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    elif state == 1:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=1e-4,
                                # weight_decay=1e-4,
                                weight_decay=5e-4,
                                )
    else:
        optimizer = optim.AdamW(model.get_grouped_params(backbone_lr=BACKBONE_LR, head_lr=HEAD_LR),
                                weight_decay=1e-2,
                                )
    if key in ['y', 'yes']:
        optimizer_scheduler_YN = True
    else:
        optimizer_scheduler_YN = use_old_model
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.8)

    scaler = GradScaler()
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
        for _name in ['garbage_classification_model.pth', 'metrics.csv', 'metrics_record.csv', 'last_config.json']:
            _p = os.path.join(RUNS_DIR, _name)
            if os.path.exists(_p):
                _stem, _ext = os.path.splitext(_name)
                os.rename(_p, os.path.join(RUNS_DIR, f'{_stem}_archive_{_ts}{_ext}'))
                logging.info(f'已归档上次产物: {_name} -> {_stem}_archive_{_ts}{_ext}')
        # 保存本次运行配置快照（配置变了就归档旧的、开新的）
        with open(os.path.join(RUNS_DIR, 'last_config.json'), 'w', encoding='utf-8') as f:
            json.dump({'CONFIG_ID': CONFIG_ID, 'SEED': SEED, 'SMOKE': SMOKE,
                       'MAX_EPOCHS': MAX_EPOCHS, 'RESUME': RESUME, 'AUTO': AUTO,
                       'UNFREEZE1_EPOCH': UNFREEZE1_EPOCH, 'UNFREEZE2_EPOCH': UNFREEZE2_EPOCH},
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
