# -*- coding: utf-8 -*-
"""
predict_gui_snoprog.py —— 多模型通用单图预测 GUI（按 run_queue QUEUE 配置表驱动，2026-09-08 会话8 重写）
UI 细节（会话9 改，GUI 引擎 = PySide6/Qt6，自 tkinter 迁移）：
① 模型下拉只显示"短名+成绩"，中文全称移到选框下方说明行；
② 图片预览固定 300×300 窗口，等比缩放居中（不铺满不裁切，小图原尺寸居中），下方标注原图像素；
③ Qt 深色/浅色自适应由样式表控制（现代卡片风）。

【来源】src/eval2.py → src_v2 适配版（原版写死 S-NoProg，本版改为多模型可选）：
  - src_v2 的 best.pth 是"新格式精简 state_dict"（仅 epoch/config_id/seed/arch/state_dict/val_acc，
    无整模型、无 renew 映射）→ 按所选配置重建对应模型（v1 ResNetTransformer / v2 ResNetTransformerV2）
    后 load_state_dict(best.pth['state_dict'])。
【配置表】镜像 run_queue.py QUEUE 全集 + train.py PRESETS 模型参数（改队列/预设时两处同步维护）；
  模型结构参数缺失会直接 size mismatch 报错，是最可靠的自检。

【用法】
  GUI：      D:\\Python\\Python3.10.7\\python.exe E:\\DataSet\\垃圾分类图片-2\\src_v2\\predict_gui_snoprog.py
  命令行：   ... predict_gui_snoprog.py <图片路径>                  # 默认配置（S-NoProg）
             ... predict_gui_snoprog.py <配置ID> <图片路径>          # 指定配置（如 G-Full-CAWR-Long / V2-Full）
  下拉中某配置若还没有 best checkpoint（如 V2-NoTF 未跑、seed2 排队中）→ 状态栏提示"暂无模型"。

【运行前提】
  1. 训练/排队器占用 GPU 时 eval 可共存（~5GB），但建议串行；
  2. 首次运行会读 train/ 目录算 mean/std（有缓存则秒回）。
"""
import glob
import os
import sys
import csv

# Windows 控制台用 UTF-8 输出（避免中文乱码）
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import torchvision.transforms as transforms
from PIL import Image

# —— 路径注入：保证 `from src_v2 import ...` / `from ResNet import ...` 可用（等效 PyCharm Sources Root）——
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ResNet import ResNetTransformer
from ResNetTransformer import ResNetTransformerV2
from calculate import calculate_mean_and_std, generate_mappings, get_classnames
from src_v2 import merged_dict

# ==================== 模型配置表（镜像 run_queue.py QUEUE 全集 + train.py PRESETS） ====================
# 键 = QUEUE 中的 CONFIG_ID；值 = (模型家族, regions, transformer_layers, use_decouple, 显示说明)
#   model: 'v1' = ResNetTransformer（seq_len=1 退化版）/ 'v2' = ResNetTransformerV2（49-token 真注意力）
#   regions: [1,2]=多区域SE / None=无SE；transformer_layers: 3/0；use_decouple: True/False
# ⚠ 与 train.py PRESETS / run_queue.py QUEUE 同步维护；若某配置结构改过，这里必须跟着改。
MODEL_REGISTRY = {
    # —— 已完成（QUEUE"已完成留档"区）——
    'G-Full':         ('v1', [1, 2], 3, True,  '渐进式·RLRP旧基线'),
    'G-NoSE':         ('v1', None,   3, True,  '渐进式·无SE·RLRP'),
    'S-NoProg':       ('v1', [1, 2], 3, True,  '全程全解冻·含多区域SE'),
    'S-NoSE':         ('v1', None,   3, True,  '全程全解冻·无多区域SE'),
    'G-Full-CAWR':    ('v1', [1, 2], 3, True,  '渐进式+CAWR调度'),
    # —— 进行中/待跑（QUEUE"待跑"区；没跑出 best 时 GUI 会提示"暂无模型"）——
    'G-Full-CAWR-Long': ('v1', [1, 2], 3, True, '渐进式+CAWR·Long160轮'),
    'V2-Full':        ('v2', [1, 2], 3, True,  '修正版49-token注意力'),
    'V2-NoTF':        ('v2', [1, 2], 0, True,  'V2架构去Transformer'),
    # —— 曾出现/弃用（若想留测试入口可保留，没跑出 best 会提示暂无模型）——
    # 'G-NoSE-CAWR':  ('v1', None, 3, True, '渐进式无SE(弃用)'),
    # 'G-NoTF':       ('v1', [1, 2], 0, True, 'v1去Transformer(弃用)'),
    # 'G-SingleSE':   ('v1', [1], 3, True, '单尺度SE(暂缓)'),
    # 'G-PureBB':     ('v1', None, 0, False, '纯骨干(暂缓)'),
}
DEFAULT_CONFIG = 'V2-Full'     # 下拉默认值（与旧版行为一致）
D_MODEL = 512
NHEAD = 8
SEEDS = (1, 2)                  # 主实验约定 seed1/2（G-Full 旧 seed0 也可被 glob 命中，不限此表）

# —— 下拉列表显示用的"完整中文名称"（与论文/PRESETS 语义一致，2026-09-08 加）——
CN_NAMES = {
    'G-Full':           '渐进式解冻·RLRP调度·含多区域SE（历史基线）',
    'G-NoSE':           '渐进式解冻·RLRP调度·无多区域SE（历史基线）',
    'S-NoProg':         '全程全解冻·含多区域SE·CAWR调度',
    'S-NoSE':           '全程全解冻·无多区域SE·CAWR调度（SE消融）',
    'G-Full-CAWR':      '渐进式解冻·CAWR调度·含多区域SE',
    'G-Full-CAWR-Long': '渐进式解冻·CAWR调度·Long160轮·含多区域SE',
    'V2-Full':          'V2修正Transformer·全程全解冻·含多区域SE（49-token真注意力）',
    'V2-NoTF':          'V2架构去Transformer·全程全解冻·含多区域SE（纯注意力消融对照）',
}
# —— 完整轮数目标（镜像 run_queue LONG_CONFIGS：Long=160，其余=100）——
TARGET_EPOCHS = {'G-Full-CAWR-Long': 160}
RUNS_DIR = os.path.join(_HERE, 'runs')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== GPU 余量检测（训练保护，2026-09-08 会话8 加） ====================
# 目的：训练/run_queue 占满显存时，预测 GUI 若也加载到 GPU 会把训练挤 OOM。
# 设计：启动与每次预测前用 nvidia-smi 查询显存占用（不用 torch.cuda.mem_get_info()，
#       因为那会初始化 CUDA context、本身再占 ~几百 MB 反而挤压训练）；
#       占用 >85% → 自动降级 CPU 推理；训练结束后占用回落 → GUI 周期检查自动切回 GPU。
GPU_BUSY_THRESHOLD = 0.85


def gpu_usage():
    """返回 (used_mb, total_mb) 或 None（无 GPU / nvidia-smi 不可用）。不初始化 CUDA context。"""
    if not torch.cuda.is_available():
        return None
    try:
        import subprocess
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None
        used, total = [int(x.strip()) for x in r.stdout.strip().split(',')[:2]]
        return (used, total)
    except Exception:
        return None


def pick_device():
    """选择推理设备：GPU 有显存余量用 GPU；被训练占满(>85%)或不可用则 CPU。
    返回 (device, mode_str)，mode_str ∈ {'gpu', 'cpu-busy', 'cpu-no-gpu', 'cpu-unknown'}。"""
    u = gpu_usage()
    if u is None:
        # 无 CUDA 或查询失败 → 保守 CPU（训练保护优先；查询失败多半是驱动/权限问题）
        return torch.device('cpu'), 'cpu-unknown'
    used, total = u
    if used / total > GPU_BUSY_THRESHOLD:
        return torch.device('cpu'), 'cpu-busy'
    return torch.device('cuda'), 'gpu'


def device_hint(mode):
    """把 pick_device 的 mode 转成给用户看的一句话。"""
    return {
        'gpu': 'GPU 推理（显存充足）',
        'cpu-busy': '训练进行中，已自动降级 CPU 模式（显存占用高，避免挤爆训练）',
        'cpu-no-gpu': 'CPU 推理（无可用 CUDA GPU）',
        'cpu-unknown': 'CPU 推理（无法检测显存，保守降级）',
    }.get(mode, mode)


def list_best_checkpoints(config_id):
    """返回该配置下所有 best checkpoint 的 (路径, seed)。无则返回 []。"""
    out = []
    for p in sorted(glob.glob(os.path.join(_HERE, 'runs', f'checkpoint_{config_id}_seed*_best.pth'))):
        name = os.path.basename(p)
        try:
            seed = int(name.split('_seed')[1].split('_')[0])
        except (IndexError, ValueError):
            seed = None
        out.append((p, seed))
    return out


def pick_best(config_id, verbose=True):
    """自动挑选该配置"最已完成"的 best checkpoint：多 seed 都跑完时取 val_acc 最高者。
    返回 dict(path/seed/val_acc/epoch/config_id/arch) 或 None（暂无模型）。"""
    cands = list_best_checkpoints(config_id)
    if not cands:
        return None
    best = None
    for p, seed in cands:
        try:
            ck = torch.load(p, map_location='cpu')
            meta = {'path': p, 'seed': seed,
                    'val_acc': ck.get('val_acc'), 'epoch': ck.get('epoch'),
                    'config_id': ck.get('config_id'), 'arch': ck.get('arch')}
        except Exception as e:
            print(f'[探测] 读取 {os.path.basename(p)} 失败: {e}，跳过')
            continue
        if best is None or (meta['val_acc'] is not None and
                            (best['val_acc'] is None or meta['val_acc'] > best['val_acc'])):
            best = meta
    if verbose and best:
        s = f"[探测] {config_id} best = {os.path.basename(best['path'])}"
        if best['val_acc'] is not None:
            s += f" (val {best['val_acc']:.2f}%@epoch {best['epoch']})"
        print(s)
    return best


# ==================== 模型下拉条目构建（2026-09-08 会话8 加；会话9 短名化） ====================
# 用户要求：中文全称不再放进下拉/待选列表（太长）——列表只显示"配置短名 + 成绩状态"；
#           当前所选模型的"全称中文名"在 GUI 里显示在选框下方的说明行（CN_NAMES，见 App._select_config）。
# 格式：有模型 → "S-NoProg 82.63@74"；未跑满目标轮数 → "…72.69@17 (18/100轮)"；暂无 → "… 暂无模型"
# 排序：有 best 的按 val_acc 降序，暂无模型的排最后。


def scan_max_epoch(config_id, seed):
    """在 runs 全部 metrics_record*.csv（当前+归档）中找该 (config_id, seed) 的最大 epoch；无记录返回 None。"""
    best_ep = None
    for f in sorted(glob.glob(os.path.join(RUNS_DIR, 'metrics_record*.csv'))):
        try:
            with open(f, encoding='utf-8', newline='') as fh:
                for r in csv.DictReader(fh):
                    if r.get('config_id') == config_id and r.get('seed') == str(seed):
                        try:
                            ep = int(float(r['epoch']))
                        except (TypeError, ValueError):
                            continue
                        best_ep = ep if best_ep is None else max(best_ep, ep)
        except Exception:
            continue
    return best_ep


def has_results(config_id, seed):
    return os.path.exists(os.path.join(RUNS_DIR, f'results_{config_id}_seed{seed}.json'))


def config_complete(config_id, seed):
    """完整 = results JSON 已生成（完整跑完的标志），或 metrics 已到达目标轮数-1（epoch 索引 0 起）。"""
    if has_results(config_id, seed):
        return True
    mx = scan_max_epoch(config_id, seed)
    if mx is None:
        return False
    target = TARGET_EPOCHS.get(config_id, 100)
    return mx >= target - 1


def done_epochs(config_id, seed):
    """该 (config, seed) 已完成轮数（1-based 计数：epoch 0..mx 算 mx+1 轮）；无记录返回 0。"""
    mx = scan_max_epoch(config_id, seed)
    return (mx + 1) if mx is not None else 0


def build_combo_items():
    """返回 [(cid, 下拉显示文本), ...]：列表只显示短名+成绩（中文全称放到选框下方说明行）。
    有 best 按 val_acc 降序；暂无模型按注册顺序排最后。"""
    have, none = [], []
    for cid in MODEL_REGISTRY:
        meta = pick_best(cid, verbose=False)
        if meta is None:
            none.append((cid, f'{cid}  暂无模型'))
            continue
        val, ep, seed = meta.get('val_acc'), meta.get('epoch'), meta.get('seed')
        mid = f'{val:.2f}@{ep}' if val is not None else '?'
        if not config_complete(cid, seed):
            tgt = TARGET_EPOCHS.get(cid, 100)
            done = done_epochs(cid, seed)
            mid += f' ({done}/{tgt}轮)'
        label = f'{cid}  {mid}'
        sort_key = val if val is not None else -1.0
        have.append((cid, label, sort_key))
    have.sort(key=lambda x: x[2], reverse=True)
    return [(c, l) for c, l, _ in have] + none


class Predictor:
    """加载指定配置的 best 模型 + 254 类映射，提供单图预测。
    未跑出 best 的配置 → __init__ 抛 ModelNotReadyError（GUI 捕获后提示"暂无模型"）。"""

    def __init__(self, config_id=DEFAULT_CONFIG, verbose=True, device=None):
        self.config_id = config_id
        if config_id not in MODEL_REGISTRY:
            raise ValueError(f'未知配置 {config_id}（可选: {", ".join(MODEL_REGISTRY)}）')
        fam, regions, tf_layers, use_decouple, desc = MODEL_REGISTRY[config_id]

        meta = pick_best(config_id, verbose=verbose)
        if meta is None:
            raise ModelNotReadyError(
                f'配置 {config_id}（{desc}）暂无模型：'
                f'未找到 checkpoint_{config_id}_seed*_best.pth。\n'
                f'可能原因：该实验尚未跑完 / 尚未启动（如 V2-NoTF 排队中）。')
        self.best_pth = meta['path']

        # —— 推理设备：显存被训练占满时自动 CPU（训练保护）——
        if device is None:
            self.device, self.device_mode = pick_device()
        else:
            self.device = device
            self.device_mode = 'gpu' if device.type == 'cuda' else 'cpu-unknown'
        if verbose:
            print(f'[设备] {device_hint(self.device_mode)}')

        # —— mean/std（复用训练集自定义统计；有缓存则秒回）——
        file = os.path.join(_PROJECT_ROOT, 'train')
        self.mean, self.std, _ = calculate_mean_and_std(file, retrieve_file=True)

        # —— 254 类映射（与训练时同一 merged_dict / classname.txt / 排序逻辑）——
        self.id_to_main_class, self.renew_class_to_index = generate_mappings(merged_dict=merged_dict)
        self.index_to_name = {int(v): k for k, v in self.renew_class_to_index.items()}
        num_classes = len(self.renew_class_to_index)
        if verbose:
            print(f'[映射] 类别数 = {num_classes}（应为 254）')

        # —— 重建模型（结构须与训练时完全一致；参数取自 MODEL_REGISTRY）——
        kwargs = dict(transformer_layers=tf_layers, d_model=D_MODEL,
                      id_to_main_class=self.id_to_main_class,
                      renew_class_to_index=self.renew_class_to_index,
                      nhead=NHEAD, regions=regions, use_decouple=use_decouple)
        if fam == 'v2':
            self.model = ResNetTransformerV2(**kwargs)
        else:
            self.model = ResNetTransformer(**kwargs)
        self.model.unfreeze_all()   # best 保存的是最终全解冻状态的全部参数

        ck = torch.load(self.best_pth, map_location='cpu')
        ck_arch = ck.get('arch')
        cur_arch = type(self.model).__name__
        if ck_arch is not None and ck_arch != cur_arch:
            raise SystemExit(f'[arch] 断点架构 {ck_arch} ≠ 当前重建 {cur_arch}，拒绝加载 '
                             f'（MODEL_REGISTRY 与 PRESETS 可能不同步）')
        sd = ck.get('state_dict', ck.get('model_state_dict'))
        if sd is None:
            raise KeyError('断点里既没有 state_dict 也没有 model_state_dict')
        self.model.load_state_dict(sd)
        self.model.to(self.device)
        self.model.eval()
        if verbose:
            acc = ck.get('val_acc')
            ep = ck.get('epoch')
            extra = f"best_epoch={ep}, best_val_acc={acc:.2f}%)" if acc is not None else ")"
            print(f'[加载] {os.path.basename(self.best_pth)} 成功 '
                  f'(config={ck.get("config_id")}, seed={ck.get("seed")}, {extra}')
        self._class_labels = get_classnames()

    def warmup(self):
        """预热：用 dummy 图跑一次前向，触发 CUDA/cuDNN 首次初始化（CPU 模式则跳过 CUDA 同步）。"""
        dummy = torch.zeros(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            self.model(dummy)
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        print('[预热] 完成')

    def predict_image_path(self, image_path, topk=5):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        img = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.softmax(out, dim=1).cpu().numpy().flatten()
        top_idx = probs.argsort()[::-1][:topk]
        results = []
        for idx in top_idx:
            name = self.index_to_name.get(int(idx), f'类{idx}')
            results.append((self._clean_name(name), float(probs[idx])))
        return results

    def _clean_name(self, name):
        if name.startswith('私有类-'):
            return name[len('私有类-'):]
        return name


class ModelNotReadyError(RuntimeError):
    """该配置暂无 best checkpoint（实验未跑完/未启动）。"""



# ==================== GUI（PySide6 / Qt6，2026-09-08 会话9 从 tkinter 迁移） ====================
def _run_gui():
    import threading
    import queue as _queue
    import time as _time
    from PySide6.QtCore import Qt, QTimer, QRectF
    from PySide6.QtGui import (QImage, QPixmap, QFont, QIcon, QPainter, QColor)
    from PySide6.QtWidgets import (
        QApplication, QWidget, QLabel, QComboBox, QPushButton,
        QVBoxLayout, QHBoxLayout, QFrame, QPlainTextEdit, QFileDialog,
    )

    # —— Qt 样式表：浅色现代卡片风 ——
    QSS = """
* { font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif; }
QMainWindow, QWidget#root { background: #eef2f8; }
QFrame#card {
    background: #ffffff;
    border: 1px solid #e2e8f2;
    border-radius: 14px;
}
QLabel#secTitle { font-size: 13px; font-weight: 600; color: #2c3748; background: transparent; }
QLabel#cnName { color: #5b6b81; font-size: 12px; background: transparent; }
QLabel#status { color: #5b6b81; font-size: 12px; background: transparent; }
QLabel#imgInfo { color: #5b6b81; font-size: 12px; background: transparent; }
QLabel#preview {
    background: #f7f9fd;
    border: 2px dashed #c6d2e6;
    border-radius: 10px;
    color: #5b6b81;
    font-size: 13px;
}
QLabel#preview[loaded="true"] { border: 1px solid #b6c4da; }  /* 载图后：虚线→实线，暗示内容就位 */
QComboBox {
    background: #ffffff;
    color: #1c2534;                 /* 显式文字色：防系统深色主题白底白字复发 */
    border: 1px solid #d3dcea;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 13px;
    min-height: 20px;
}
QComboBox:hover { border-color: #8fb4f0; }
QComboBox:focus { border-color: #3b82f6; }
QComboBox::drop-down { border: none; width: 26px; }
QComboBox::down-arrow { image: none; border-left: 4px solid transparent;
    border-right: 4px solid transparent; border-top: 5px solid #5b6b81; margin-right: 8px; }
QComboBox QAbstractItemView {
    background: #ffffff;
    color: #1c2534;
    border: 1px solid #d3dcea;
    selection-background-color: #3b82f6;
    selection-color: #ffffff;
    outline: none;
    font-size: 13px;
}
QPushButton#upload {
    background: #2563eb;            /* 主色下压一档：白字对比度 4.5→5.2 */
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 10px 34px;
    font-size: 14px;
    font-weight: 600;
}
QPushButton#upload:hover { background: #1e4fd6; }
QPushButton#upload:pressed { background: #1d4fcc; }
QPushButton#upload:disabled { background: #ccd7ea; color: #6b7790; }
QPlainTextEdit#result {
    background: #ffffff;
    border: 1px solid #e2e8f2;
    border-radius: 10px;
    padding: 8px;
    color: #26303e;
    font-size: 13px;
    selection-background-color: #d7e4ff;
}

/* —— 自绘标题栏（无边框窗口）：浅色一体式，与内容卡片同色系 —— */
#titleBar {
    background: #fbfdff;
    border-bottom: 1px solid #e0e7f2;
}
#tbLogoBox {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #3b82f6, stop:0.5 #2b63e8, stop:1 #1f53d8);
    border: none;
    border-radius: 6px;
}
#tbLogoChar { color: #ffffff; font-size: 12px; font-weight: 700; }
#tbApp { color: #1c2534; font-size: 13.5px; font-weight: 700; }
#tbSub { color: #5b6b81; font-size: 12px; }
#tbModel {
    color: #2456c8;
    background: #eaf1ff;
    border: 1px solid #c9d9f7;
    border-radius: 13px;            /* 胶囊限高 26px 后圆角=半高 → 真药丸 */
    padding: 0 12px;
    font-size: 11px;
    font-weight: 600;
}
QPushButton#tbBtn, QPushButton#tbBtnClose {
    background: transparent;
    color: #5b6b81;
    border: none;
    font-size: 13px;
    font-weight: 600;
}
QPushButton#tbBtn:hover { background: #e9edf5; color: #1c2534; }
QPushButton#tbBtnClose:hover { background: #e81123; color: #ffffff; }
QPushButton#tbBtn:pressed { background: #d8deea; }
QPushButton#tbBtnClose:pressed { background: #c50f1e; }
"""


    # ---------- PIL ↔ Qt 转换 ----------
    def pil_to_qpixmap(pil_img):
        """PIL RGB 图像 → QPixmap（Qt 不持有原始 bytes，转完即安全）。"""
        img = pil_img.convert('RGB')
        w, h = img.size
        data = img.tobytes('raw', 'RGB')
        qimg = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def fit_pixmap(pix, box):
        """等比缩放进 box×box：大图缩小、小图保持原尺寸（不放大、不裁切）。"""
        if pix.width() <= box and pix.height() <= box:
            return pix
        return pix.scaled(box, box, Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)

    def cn_text(cfg):
        """配置 → 全称中文名（CN_NAMES 优先，退回注册表 desc）。"""
        cn = CN_NAMES.get(cfg)
        if cn is None:
            reg = MODEL_REGISTRY.get(cfg)
            cn = reg[4] if reg else cfg
        return cn

    class TitleBar(QWidget):
        """自绘标题栏（无边框窗口）：logo 块 + 应用名/副标题 + 当前模型胶囊 + 最小化/最大化/关闭按钮。
        浅色一体式风格（与内容卡片同色系 #fbfdff，深色文字保证对比度）。
        支持按住标题栏拖动窗口、双击最大化/还原。"""
        HEIGHT = 46

        def __init__(self, parent_win, app_title, app_sub):
            super().__init__(parent_win)
            self._win = parent_win
            self._drag_off = None
            self.setObjectName('titleBar')
            # 关键：自定义 QWidget 必须开启 StyledBackground，QSS 的 background 才会被绘制
            #（否则深蓝渐变不渲染，露出浅色底 + 白字 = "背景白字也白"）
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            self.setFixedHeight(self.HEIGHT)

            lay = QHBoxLayout(self)
            lay.setContentsMargins(12, 0, 0, 0)
            lay.setSpacing(8)

            # —— 左：渐变 logo 块 + 应用名 + 副标题 ——
            logo_box = QFrame()
            logo_box.setObjectName('tbLogoBox')
            logo_box.setFixedSize(22, 22)
            lb_lay = QVBoxLayout(logo_box)
            lb_lay.setContentsMargins(0, 0, 0, 0)
            logo_char = QLabel('分')
            logo_char.setObjectName('tbLogoChar')
            logo_char.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lb_lay.addWidget(logo_char)
            lay.addWidget(logo_box)

            self._title_lbl = QLabel(app_title)
            self._title_lbl.setObjectName('tbApp')
            self._sub_lbl = QLabel(app_sub)
            self._sub_lbl.setObjectName('tbSub')
            lay.addWidget(self._title_lbl)
            lay.addWidget(self._sub_lbl)
            lay.addStretch(1)

            # —— 中右：当前模型胶囊（随选择更新）——
            self.model_tag = QLabel('模型：—')
            self.model_tag.setObjectName('tbModel')
            self.model_tag.setFixedHeight(26)   # 限高成紧凑药丸（QLabel 默认会被布局撑满整条）
            lay.addWidget(self.model_tag)
            lay.addSpacing(6)

            # —— 右：窗口控制按钮 ——
            for sym, name, tip, fn in (
                ('─', 'tbBtn', '最小化', self._win.showMinimized),
                ('□', 'tbBtn', '最大化/还原', self._toggle_max),
                ('✕', 'tbBtnClose', '关闭', self._win.close),
            ):
                b = QPushButton(sym)
                b.setObjectName(name)
                b.setFixedSize(46, self.HEIGHT)
                b.setToolTip(tip)
                b.setCursor(Qt.CursorShape.PointingHandCursor)
                b.clicked.connect(fn)
                lay.addWidget(b)

        # —— 拖动窗口 / 双击最大化 ——
        def mousePressEvent(self, e):
            if e.button() == Qt.MouseButton.LeftButton:
                self._drag_off = e.globalPosition().toPoint() - self._win.frameGeometry().topLeft()
                e.accept()

        def mouseMoveEvent(self, e):
            if self._drag_off is not None and e.buttons() & Qt.MouseButton.LeftButton:
                self._win.move(e.globalPosition().toPoint() - self._drag_off)
                e.accept()

        def mouseReleaseEvent(self, e):
            self._drag_off = None

        def mouseDoubleClickEvent(self, e):
            if e.button() == Qt.MouseButton.LeftButton:
                self._toggle_max()

        def _toggle_max(self):
            self._win.showNormal() if self._win.isMaximized() else self._win.showMaximized()

        def set_model_tag(self, cfg):
            """标题栏上显示当前所选模型（短名），与主内容联动。"""
            self.model_tag.setText(f'模型：{cfg}')

    class App(QWidget):
        PREVIEW = 300  # 预览窗口固定边长

        def __init__(self):
            super().__init__()
            self.setObjectName('root')
            # 自定义 QWidget 子类需 WA_StyledBackground 才能渲染 QSS background（#root 底色）
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            # 无边框窗口：标题栏自绘（TitleBar）；setWindowTitle 仅用于任务栏/Alt-Tab 显示
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
            self.setWindowTitle('垃圾分类智能识别')
            self.resize(920, 660)
            self.setMinimumSize(780, 560)

            self._items = build_combo_items()
            self._load_gen = 0
            self._predict_gen = 0
            self.predictor = None
            self._current_image_path = None   # 已上传/正在显示的图片：切换模型后自动重识别
            self._ui_q = _queue.Queue()          # 工作线程 → 主线程消息

            self._build_ui()
            self._select_config(DEFAULT_CONFIG)

            # 周期设备检查（训练保护）：GPU 空闲后自动切回 GPU 重载
            threading.Thread(target=self._device_watch, daemon=True).start()

            # 主线程消息泵
            self._pump = QTimer(self)
            self._pump.timeout.connect(self._drain_ui)
            self._pump.start(60)

        # ---------- 界面搭建 ----------
        def _build_ui(self):
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)     # 无边框窗口：标题栏顶满宽度
            root.setSpacing(0)

            # —— 自绘标题栏（浅色一体式，取代系统标题栏）——
            self.title_bar = TitleBar(self, '垃圾分类智能识别', '多模型对照 · 单图预测 · 254 类')
            root.addWidget(self.title_bar)

            # —— 内容区（卡片带边距呼吸感）——
            body = QHBoxLayout()
            body.setContentsMargins(18, 14, 18, 14)
            body.setSpacing(14)
            root.addLayout(body, 1)

            # —— 左卡：图片预览 ——
            left_card = QFrame()
            left_card.setObjectName('card')
            left_card.setFixedWidth(348)
            lv = QVBoxLayout(left_card)
            lv.setContentsMargins(16, 14, 16, 14)
            lv.setSpacing(8)
            lv.addStretch(1)        # 顶部弹性：内容组在卡内垂直居中（消除底部悬空大白块）

            sec_l = QLabel('图片预览')
            sec_l.setObjectName('secTitle')
            lv.addWidget(sec_l, 0, Qt.AlignmentFlag.AlignHCenter)

            self.preview = QLabel('尚未选择图片')
            self.preview.setObjectName('preview')
            self.preview.setFixedSize(self.PREVIEW, self.PREVIEW)
            self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lv.addWidget(self.preview, 0, Qt.AlignmentFlag.AlignHCenter)

            self.img_info = QLabel('原图尺寸：—')
            self.img_info.setObjectName('imgInfo')
            self.img_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lv.addWidget(self.img_info)

            self.upload_btn = QPushButton('上传图片')
            self.upload_btn.setObjectName('upload')
            self.upload_btn.setEnabled(False)
            self.upload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.upload_btn.clicked.connect(self._on_upload)
            lv.addWidget(self.upload_btn, 0, Qt.AlignmentFlag.AlignHCenter)
            lv.addStretch(1)        # 底部弹性：与顶部 stretch 等权 → 内容垂直居中
            body.addWidget(left_card)

            # —— 右卡：模型选择 + 结果 ——
            right_card = QFrame()
            right_card.setObjectName('card')
            rv = QVBoxLayout(right_card)
            rv.setContentsMargins(16, 14, 16, 14)
            rv.setSpacing(8)

            sec_r = QLabel('选择模型（按最好准确率排序）')
            sec_r.setObjectName('secTitle')
            rv.addWidget(sec_r)

            self.combo = QComboBox()
            for cid, label in self._items:
                self.combo.addItem(label, cid)
            self.combo.setMinimumWidth(330)
            self.combo.currentIndexChanged.connect(self._on_combo_changed)
            rv.addWidget(self.combo)

            # 选中模型的全称中文名（选框下方说明行）
            self.cn_label = QLabel('')
            self.cn_label.setObjectName('cnName')
            self.cn_label.setWordWrap(True)
            rv.addWidget(self.cn_label)

            sec_r2 = QLabel('预测结果（前5）')
            sec_r2.setObjectName('secTitle')
            rv.addWidget(sec_r2)

            self.result_box = QPlainTextEdit()
            self.result_box.setObjectName('result')
            self.result_box.setReadOnly(True)
            self.result_box.setPlaceholderText('选择图片后，这里显示前 5 个预测类别')   # 空态占位
            rv.addWidget(self.result_box, 1)

            self.status = QLabel('模型加载中…')
            self.status.setObjectName('status')
            self.status.setWordWrap(True)
            rv.addWidget(self.status)
            body.addWidget(right_card, 1)

            # 默认选中项 = DEFAULT_CONFIG（找不到就选列表第一项）
            # 注意：setCurrentIndex 会触发 currentIndexChanged → _select_config，
            # 因此用 blockSignals 抑制，首次真正加载由 __init__ 末尾 _select_config(DEFAULT_CONFIG) 完成
            default_idx = 0
            for i in range(self.combo.count()):
                if self.combo.itemData(i) == DEFAULT_CONFIG:
                    default_idx = i
                    break
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(default_idx)
            self.combo.blockSignals(False)

        # ---------- 工作线程 → 主线程消息 ----------
        def _post(self, kind, *args):
            self._ui_q.put((kind, args))

        def _drain_ui(self):
            try:
                while True:
                    kind, args = self._ui_q.get_nowait()
                    getattr(self, '_ui_' + kind)(*args)
            except _queue.Empty:
                pass

        # —— 各 UI 处理器（主线程执行）——
        def _ui_loading(self, cfg):
            self.status.setText(f'加载 {cfg} …')
            self.upload_btn.setEnabled(False)

        def _ui_ready(self, cfg, gen, pred, note):
            if gen != self._load_gen:
                return  # 期间又切换了配置，丢弃过期加载
            self.predictor = pred
            self.upload_btn.setEnabled(True)
            if self._current_image_path:
                # 已有上传图片 → 新模型就绪后自动重识别
                self.status.setText(f'就绪：{cfg}（{os.path.basename(pred.best_pth)}）· {note} — '
                                    f'正在自动识别当前图片…')
                self._start_predict()
            else:
                self.status.setText(f'就绪：{cfg}（{os.path.basename(pred.best_pth)}）· {note} — 请上传图片')

        def _ui_loaderr(self, gen, text):
            if gen != self._load_gen:
                return
            self.status.setText(f'⚠ {text}')
            self.upload_btn.setEnabled(False)

        def _ui_results(self, gen, results):
            if gen != self._predict_gen:
                return  # 期间又发起新预测/切换了模型，丢弃过期结果
            lines = ['预测概率（前5）:', '']
            for name, prob in results:
                lines.append(f'{name}: {prob:.2%}')
            self.result_box.setPlainText('\n'.join(lines))
            self.status.setText('识别完成 — 可上传新图片或切换模型（会自动重新识别）')
            self.upload_btn.setEnabled(True)

        def _ui_prederr(self, gen, text):
            if gen != self._predict_gen:
                return
            self.status.setText(f'⚠ {text}')
            self.upload_btn.setEnabled(True)

        def _ui_switch_to_gpu(self, cfg):
            if self.predictor is None or cfg != self.predictor.config_id:
                return
            self.status.setText('检测到训练结束 / GPU 空闲 → 正在切回 GPU 模式…')
            self._select_config(cfg)

        # ---------- 配置切换 ----------
        def _on_combo_changed(self, _index):
            cid = self.combo.currentData()
            if cid:
                self._select_config(cid)

        def _select_config(self, cfg):
            self._load_gen += 1
            gen = self._load_gen
            self._predict_gen += 1          # 作废在途预测（若换模型时旧图预测还没回来）
            self.predictor = None
            self.result_box.clear()
            self.cn_label.setText(cn_text(cfg))   # 选框下方显示全称中文名
            self.title_bar.set_model_tag(cfg)     # 标题栏同步显示当前模型短名
            self._ui_loading(cfg)
            threading.Thread(target=self._load_worker, args=(cfg, gen), daemon=True).start()

        def _load_worker(self, cfg, gen):
            try:
                pred = Predictor(config_id=cfg, verbose=False)
                pred.warmup()
                self._post('ready', cfg, gen, pred, device_hint(pred.device_mode))
            except ModelNotReadyError as e:
                self._post('loaderr', gen, f'{e}')
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._post('loaderr', gen, f'模型加载失败: {e}')

        # ---------- 发起一次预测（上传 or 换模型后自动重识别共用） ----------
        def _start_predict(self):
            """用当前 predictor 对 _current_image_path 发起预测（线程内执行）。
            每次发起自增 _predict_gen，旧的在途预测结果回来时因代次不符被丢弃。"""
            if self.predictor is None or not self._current_image_path:
                return
            self._predict_gen += 1
            gen = self._predict_gen
            self.upload_btn.setEnabled(False)
            pred = self.predictor
            path = self._current_image_path
            threading.Thread(target=self._predict_worker, args=(pred, path, gen), daemon=True).start()

        def _predict_worker(self, pred, path, gen):
            try:
                results = pred.predict_image_path(path, topk=5)
                self._post('results', gen, results)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._post('prederr', gen, f'预测失败: {e}')

        # ---------- 设备守护：训练占显存 → CPU；GPU 空闲 → 切回 ----------
        def _device_watch(self):
            while True:
                _time.sleep(60)
                try:
                    if self.predictor is None:
                        continue
                    if self.predictor.device.type == 'cpu':
                        _dev, mode = pick_device()
                        if mode == 'gpu':
                            self._post('switch_to_gpu', self.predictor.config_id)
                except Exception:
                    pass

        # ---------- 上传 / 预测 ----------
        def _on_upload(self):
            path, _f = QFileDialog.getOpenFileName(
                self, '选择图片', '',
                '图片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)')
            if not path:
                return
            if self.predictor is None:
                self.status.setText('模型未就绪（当前配置可能暂无模型），请换一个配置')
                return
            try:
                img = Image.open(path)
                img.load()                      # 立即解码，损坏文件在此报错
            except Exception as e:
                self.status.setText(f'无法打开图片: {e}')
                return
            self._current_image_path = path     # 记住这张图：切换模型后自动重新识别
            self._show_preview(img)             # 固定窗口等比缩放居中 + 原图像素
            self.status.setText('预测中…')
            self._start_predict()

        # ---------- 预览（固定窗口；等比缩放居中，不破坏比例） ----------
        def _show_preview(self, pil_img):
            w, h = pil_img.size
            pix = fit_pixmap(pil_to_qpixmap(pil_img), self.PREVIEW - 18)
            self._keep_pix = pix                     # 防 GC 后图片消失
            self.preview.setText('')                 # 先清占位文字（QLabel 图文互斥）
            self.preview.setPixmap(pix)              # QLabel 比图大 → 自动居中
            self.img_info.setText(f'原图尺寸：{w} × {h} 像素')
            # 虚线→实线：提示"内容已就位"（配合 QSS [loaded="true"] 规则）
            if self.preview.property('loaded') is not True:
                self.preview.setProperty('loaded', True)
                self.preview.style().unpolish(self.preview)
                self.preview.style().polish(self.preview)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(QSS)
    try:
        app.setFont(QFont('Microsoft YaHei', 10))
    except Exception:
        pass
    win = App()
    try:
        # favicon 实际位于 src\favicon.ico（项目根兜底尝试）
        _ico = next((p for p in (os.path.join(_PROJECT_ROOT, 'src', 'favicon.ico'),
                                 os.path.join(_PROJECT_ROOT, 'favicon.ico'))
                     if os.path.exists(p)), None)
        if _ico:
            win.setWindowIcon(QIcon(_ico))
    except Exception:
        pass
    win.show()
    app.exec()


if __name__ == '__main__':
    # 命令行：predict_gui_snoprog.py <图片路径>            → 默认配置
    #         predict_gui_snoprog.py <配置ID> <图片路径>    → 指定配置
    args = [a for a in sys.argv[1:]]
    img_arg = None
    cfg_arg = DEFAULT_CONFIG
    if args:
        # 判定第一个参数是配置ID还是图片路径
        if args[0] in MODEL_REGISTRY:
            cfg_arg = args[0]
            img_arg = args[1] if len(args) > 1 else None
        else:
            img_arg = args[0]
            if len(args) > 1 and args[1] in MODEL_REGISTRY:
                cfg_arg = args[1]
    if img_arg:
        # 命令行预测前先探测设备（训练占满显存时自动 CPU，避免挤爆训练）
        _dev, _mode = pick_device()
        print(f'[设备] {device_hint(_mode)}')
        try:
            p = Predictor(config_id=cfg_arg, verbose=True, device=_dev)
            p.warmup()
            print('=' * 60)
            for name, prob in p.predict_image_path(img_arg, topk=5):
                print(f'{name:20s} {prob:.2%}')
        except ModelNotReadyError as e:
            print(f'⚠ {e}')
            sys.exit(2)
    else:
        # 无图片参数 → 命令行只做"探测哪些配置有模型"自检，然后进 GUI
        _dev, _mode = pick_device()
        print(f'[设备] {device_hint(_mode)}')
        print('— 各配置模型可用性探测（按最好准确率排序） —')
        for cid, label in build_combo_items():
            print(f'  {label}')
        _run_gui()
