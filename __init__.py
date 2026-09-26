import logging
import os.path
from datetime import datetime

import torch
from torchvision import transforms
import numpy as np

from src_v2.taxonomy import (
    LABEL_SCHEMA_DEFINITION_HASH,
    LABEL_SCHEMA_ID,
    LABEL_SCHEMA_VERSION,
    mapping_fingerprint,
    merged_dict,
)

# 如果模型准确度超过预定的值, 将会强制保存
ACC_STOP = 96
ERROR_RATE = 0.1

# 每n轮检查模型指标, 以判断是否需要更新模型层
UPDATE_N = 10

# ===== DataLoader 常驻 worker 数（集中配置，train.py 只从这里读取）=====
# 2026-09-26 修 WinError 1455：Windows spawn worker 会重新 import 本包，
# 每个常驻 worker 都占一份提交内存（实测即使 worker 内 device=cpu 仍 ~4.5 GB/个）。
# 常驻合计 = TRAIN + VAL，此前 8+4 峰值 ~108.7 GB 曾撞 CommitLimit（101.11 GB），
# 降到 6+2 后峰值 ~90 GB，留 ~10 GB 漂移余量。调整时两处一起权衡。
TRAIN_NUM_WORKERS = 6   # 训练 loader 常驻 worker（配合 persistent_workers=True + prefetch_factor）
VAL_NUM_WORKERS = 2     # 验证 loader 常驻 worker（必须 persistent_workers=True，否则首轮验证新建 worker 会复现 1455 死锁）

# ===== 输出路径：全部落到 src_v2/runs/（与工作目录无关） =====
_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
os.makedirs(_runs_dir, exist_ok=True)

pth_file_name = os.path.join(_runs_dir, 'garbage_classification_model.pth')
metrics_file = os.path.join(_runs_dir, "metrics.csv")
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
    # *****************************************************
    # sorted_items = sorted(renew_class_to_index.items(), key=lambda x: int(x[1]))
    # print("Saving mapping (first 20):")
    # for k, v in sorted_items[:20]:
    #     print(v, "<-", k)
    # *****************************************************
    if other_pth_file_name is None:
        torch.save(checkpoint, pth_file_name)
    else:
        if not os.path.isfile(other_pth_file_name):
            torch.save(checkpoint, other_pth_file_name)
            s = f"特殊模型{other_pth_file_name}已保存, epoch {epoch} 完成!"
            print(s)
            logging.info(s)
    s = f"模型已保存, epoch {epoch} 完成!{today=}"
    print(s)
    logging.info(s)


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

# ⚠️ device 必须按**进程角色**分支，不能无条件探测 CUDA（2026-09-26 修）。
#
# 背景：Windows 下 DataLoader 用 spawn 派生 worker，**每个 worker 子进程都会重新执行本模块**。
#   若在模块级调用 torch.cuda.is_available()，就会在每个只负责"读图/解码/增强"的子进程里
#   初始化 CUDA 运行时并加载整套 CUDA/cuDNN/cuBLAS DLL —— 实测单个 worker 提交内存约
#   4.55 GB（torch_cuda_cu.dll 828 MB + cudnn_cnn_infer 626 MB + cudnn_ops_infer 271 MB
#   + cusparse/cusolver/cufft/cublas/curand/nvrtc 等）。
#   于是：8 个训练 worker ≈ 36 GB + 主进程 ≈ 18 GB，到首次验证再新建 4 个 val worker ≈ 18 GB，
#   合计约 70 GB ⇒ 撞上 Windows 提交内存上限，抛
#     OSError [WinError 1455] 页面文件太小 … Error loading "…torch\lib\cublas64_11.dll"
#   且失败发生在 worker 向父进程注册**之前** ⇒ 主进程永久阻塞在 DataLoader 启动握手，
#   表现为"0 CPU、日志停更、GPU 空转却占着显存"（2026-09-26 01:43 的停摆即由此而起）。
#
# 只有主进程需要 CUDA；worker 全程在 CPU 侧工作，给 cpu 即可（单 worker 从 ~4.55 GB 降到数百 MB）。
import multiprocessing as _mp

if _mp.current_process().name == 'MainProcess':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    # DataLoader worker（spawn 子进程）：不触碰 CUDA，避免无谓的提交内存开销
    device = torch.device('cpu')
