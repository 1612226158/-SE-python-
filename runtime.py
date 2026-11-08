import logging
import os.path
from datetime import datetime

import torch
from torchvision import transforms
import numpy as np

from src_v3.taxonomy import (
    LABEL_SCHEMA_DEFINITION_HASH,
    LABEL_SCHEMA_ID,
    LABEL_SCHEMA_VERSION,
    mapping_fingerprint,
    merged_dict,
)

# 如果模型准确度超过预定的值, 将会强制保存
ACC_STOP = 98
ERROR_RATE = 0.1

# DataLoader worker 数集中配置；Windows spawn 会在子进程重新导入模块。
# 配合持久 worker 使用，调整数量时需考虑各进程的内存开销。
TRAIN_NUM_WORKERS = 6   # 训练 loader 常驻 worker（配合 persistent_workers=True + prefetch_factor）
VAL_NUM_WORKERS = 2     # 验证 loader 常驻 worker（必须 persistent_workers=True，否则首轮验证新建 worker 会复现 1455 死锁）

# 输出目录优先使用 V3_RUN_DIR，默认 src_v3/runs，与工作目录无关。
_runs_dir = os.environ.get('V3_RUN_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'))

pth_file_name = os.path.join(_runs_dir, 'garbage_classification_model.pth')
metrics_file = os.path.join(_runs_dir, "metrics_record.csv")
extensions_suffix = ('jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'avif', '.jfif')

# `merged_dict` 由 taxonomy.py 导入；请只在那里修改训练标签体系。
class DynamicMeanPad:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        # 转换为 NumPy 数组
        img_array = np.array(img)

        # 计算每个通道的均值
        mean_values = img_array.mean(axis=(0, 1))  # (R_mean, G_mean, B_mean)

        # 计算图片尺寸和填充大小
        W, H = img.size
        target_height, target_width = self.target_size, self.target_size

        padding_top = (target_height - H) // 2
        padding_bottom = target_height - H - padding_top
        padding_left = (target_width - W) // 2
        padding_right = target_width - W - padding_left

        # 填充图片
        padding = (padding_left, padding_top, padding_right, padding_bottom)
        img = transforms.Pad(padding, fill=tuple(mean_values.astype(int)))(img)

        return img


def save_model_after_epoch(epoch,
                           model,
                           optimizer,
                           scheduler,
                           renew_class_to_index,
                           state,
                           other_pth_file_name=None,
                           config_id='',
                           seed=None,
                           sampler_seed=None,
                           sampling_protocol_version=None,
                           id_to_main_class=None,
                           training_protocol_version=None,
                           scaler_state_dict=None):
    os.makedirs(_runs_dir, exist_ok=True)
    today = datetime.strftime(datetime.now(), '%Y-%m-%d')
    # 执行原始的训练函数
    checkpoint = {
        'epoch': epoch,
        'config_id': config_id,  # 实验标识（归档后可溯源）
        'seed': seed,
        'sampler_seed': sampler_seed,
        'sampling_protocol_version': sampling_protocol_version,
        'training_protocol_version': training_protocol_version,
        'label_schema_id': LABEL_SCHEMA_ID,
        'label_schema_version': LABEL_SCHEMA_VERSION,
        'label_schema_definition_hash': LABEL_SCHEMA_DEFINITION_HASH,
        'label_mapping_hash': mapping_fingerprint(id_to_main_class, renew_class_to_index),
        'num_classes': len(renew_class_to_index),
        'arch': type(model).__name__,  # 架构标记（与 train.py 的 _check_arch 配合，防静默跑错模型）
        'model': model,  # 保存整个模型对象
        'model_state_dict': model.state_dict(),  # 模型权重
        'scheduler_state_dict': scheduler.state_dict(),  # 修复：续跑恢复调度器状态
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler_state_dict,
        'renew_class_to_index': renew_class_to_index,
        'id_to_main_class': id_to_main_class,
        'today': today,
        'state': state
    }
    from .model_registry import experiment_signature_of, params_of, training_policy_of, long_source_of
    preset = params_of(config_id)
    checkpoint.update(experiment_signature=experiment_signature_of(config_id),
                      normalization_signature=os.environ.get('V3_NORMALIZATION_SIGNATURE'),
                      backbone=preset['backbone'], unfreeze_epochs=preset['unfreeze_epochs'],
                      training_policy=training_policy_of(config_id))
    source = long_source_of(config_id, seed)
    if source:
        checkpoint['long_source'] = source
    metadata = model.architecture_metadata()
    import hashlib, json
    checkpoint.update(model_metadata=metadata, model_signature_sha256=hashlib.sha256(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest())
    # *****************************************************
    # sorted_items = sorted(renew_class_to_index.items(), key=lambda x: int(x[1]))
    # print("Saving mapping (first 20):")
    # for k, v in sorted_items[:20]:
    #     print(v, "<-", k)
    # *****************************************************
    saved = False
    if other_pth_file_name is None:
        torch.save(checkpoint, pth_file_name)
        saved = True
    else:
        if not os.path.isfile(other_pth_file_name):
            torch.save(checkpoint, other_pth_file_name)
            saved = True
            s = f"特殊模型{other_pth_file_name}已保存, epoch {epoch} 完成!"
            print(s)
            logging.info(s)
    s = f"模型已保存, epoch {epoch} 完成!{today=}"
    print(s)
    if saved:
        logging.info(s)
    else:
        logging.info('目标模型已存在，跳过保存：%s，epoch=%s', other_pth_file_name, epoch)


# transforms_setting = transforms.Compose([
#     transforms.Resize([96, 96]),  # 设置图片大小
#     # 数据增强
#     transforms.RandomRotation(45),  # 随机旋转
#     transforms.CenterCrop(64),  # 从中心开始裁剪
#     transforms.RandomHorizontalFlip(p=0.5),  # 随机概率水平翻转, p为概率
#     transforms.RandomVerticalFlip(p=0.5),  # 随机概率垂直翻转, p为概率
#     transforms.ColorJitter(
#         brightness=0.2,  # 亮度
#         contrast=0.1,  # 对比度
#         saturation=0.1,  # 饱和度
#         hue=0.1,  # 色相
#     ),
#     transforms.RandomGrayscale(p=0.025),  # 概率转换为灰度图
#
#     # 这两件事必须做
#     transforms.ToTensor(),  # 转换为张量类型
#     transforms.Normalize([0.485, 0.456, 0.409], [0.229, 0.224, 0.225]),  # 均值, 标准差
# ])

# 首次获取 device 时按进程角色选择并缓存。
# 主进程优先使用可用 CUDA；数据加载 worker 使用 CPU，不探测 CUDA。
import multiprocessing as _mp

def __getattr__(name):
    """首次获取device时按进程角色选择设备，之后复用。"""
    if name != 'device':
        raise AttributeError(f'{__name__}没有属性{name!r}')
    global device
    if _mp.current_process().name == 'MainProcess':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    return device
