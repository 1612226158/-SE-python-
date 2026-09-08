# -*- coding: utf-8 -*-
"""
predict_gui_snoprog.py —— 多模型通用单图预测 GUI（按 run_queue QUEUE 配置表驱动，2026-09-08 会话8 重写）

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

# Windows 控制台用 UTF-8 输出（避免中文乱码）
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk

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
    'G-Full':         ('v1', [1, 2], 3, True,  '渐进式·RLRP旧基线(留档)'),
    'G-NoSE':         ('v1', None,   3, True,  '渐进式·无SE·RLRP(留档)'),
    'S-NoProg':       ('v1', [1, 2], 3, True,  '★全解冻主模型候选(82.63%@74)'),
    'S-NoSE':         ('v1', None,   3, True,  '全解冻·无SE(SE消融)'),
    'G-Full-CAWR':    ('v1', [1, 2], 3, True,  '渐进式+CAWR(修复后重跑)'),
    # —— 进行中/待跑（QUEUE"待跑"区；没跑出 best 时 GUI 会提示"暂无模型"）——
    'G-Full-CAWR-Long': ('v1', [1, 2], 3, True, '渐进式160轮(83.57%@132)'),
    'V2-Full':        ('v2', [1, 2], 3, True,  '修正版49-token注意力(全解冻)'),
    'V2-NoTF':        ('v2', [1, 2], 0, True,  'V2架构去Transformer(纯注意力消融)'),
    # —— 曾出现/弃用（若想留测试入口可保留，没跑出 best 会提示暂无模型）——
    # 'G-NoSE-CAWR':  ('v1', None, 3, True, '渐进式无SE(弃用)'),
    # 'G-NoTF':       ('v1', [1, 2], 0, True, 'v1去Transformer(弃用)'),
    # 'G-SingleSE':   ('v1', [1], 3, True, '单尺度SE(暂缓)'),
    # 'G-PureBB':     ('v1', None, 0, False, '纯骨干(暂缓)'),
}
DEFAULT_CONFIG = 'S-NoProg'     # 下拉默认值（与旧版行为一致）
D_MODEL = 512
NHEAD = 8
SEEDS = (1, 2)                  # 主实验约定 seed1/2（G-Full 旧 seed0 也可被 glob 命中，不限此表）

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


# ==================== GUI ====================
def _run_gui():
    import tkinter as tk
    from tkinter import filedialog, ttk
    import threading

    class App:
        def __init__(self, root):
            self.root = root
            self.root.title('垃圾分类预测（多模型 · 254类 · 按 QUEUE 配置表）')
            self.root.geometry('800x620')
            left = ttk.Frame(root); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            right = ttk.Frame(root); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

            # —— 模型选择行 ——
            ttk.Label(right, text='选择模型（配置）', font=('Arial', 11, 'bold')).pack(pady=(4, 2))
            self.config_var = tk.StringVar(value=DEFAULT_CONFIG)
            combo = ttk.Combobox(right, textvariable=self.config_var, state='readonly', width=38)
            combo['values'] = [f'{k}  —  {v[4]}' for k, v in MODEL_REGISTRY.items()]
            combo.bind('<<ComboboxSelected>>', self.on_config_change)
            combo.pack(pady=4)

            self.image_label = ttk.Label(left); self.image_label.pack(pady=10)
            self.upload_btn = ttk.Button(left, text='上传图片', command=self.upload, state='disabled')
            self.upload_btn.pack(pady=10)
            ttk.Label(right, text='预测结果（前5）', font=('Arial', 14, 'bold')).pack()
            self.result_text = tk.Text(right, height=10, width=46)
            self.result_text.pack(pady=10)
            self.status_var = tk.StringVar(value='模型加载中…')
            ttk.Label(right, textvariable=self.status_var).pack(pady=4)
            self.predictor = None
            self._load_gen = 0
            self._select_config(DEFAULT_CONFIG)
            # —— 周期设备检查：训练结束（GPU 空闲）后自动把 CPU 模式切回 GPU ——
            threading.Thread(target=self._periodic_device_check, daemon=True).start()

        # ---- 周期设备检查（训练保护 + 自动恢复 GPU）----
        def _periodic_device_check(self):
            import time as _time
            while True:
                _time.sleep(60)
                try:
                    if self.predictor is None:
                        continue
                    if self.predictor.device.type == 'cpu':
                        _dev, _mode = pick_device()
                        if _mode == 'gpu':
                            cfg = self.predictor.config_id
                            self.root.after(0, lambda c=cfg: self.status_var.set(
                                '检测到训练结束 / GPU 空闲 → 正在切回 GPU 模式…'))
                            self._select_config(cfg)   # 用 GPU 重载当前配置
                except Exception:
                    pass

        # ---- 配置切换 ----
        def on_config_change(self, _event=None):
            raw = self.config_var.get()
            cfg = raw.split('  —  ')[0].strip()
            self._select_config(cfg)

        def _select_config(self, cfg):
            self._load_gen += 1
            gen = self._load_gen
            self.predictor = None
            self.result_text.delete(1.0, tk.END)
            self.status_var.set(f'加载 {cfg} …')
            self.upload_btn.configure(state='disabled')
            threading.Thread(target=self._load_and_warmup, args=(cfg, gen), daemon=True).start()

        def _load_and_warmup(self, cfg, gen):
            try:
                pred = Predictor(config_id=cfg)
                pred.warmup()
                if gen != self._load_gen:
                    return  # 期间用户又切换了配置，丢弃过期结果
                self.predictor = pred
                dev_note = device_hint(pred.device_mode)
                self.root.after(0, lambda: self.status_var.set(
                    f'就绪：{cfg}（{os.path.basename(pred.best_pth)}）· {dev_note} — 请上传图片'))
                self.root.after(0, lambda: self.upload_btn.configure(state='normal'))
            except ModelNotReadyError as e:
                if gen != self._load_gen:
                    return
                self.root.after(0, lambda: self.status_var.set(f'⚠ {e}'))
                self.root.after(0, lambda: self.upload_btn.configure(state='disabled'))
            except Exception as e:
                import traceback
                traceback.print_exc()
                if gen != self._load_gen:
                    return
                self.root.after(0, lambda: self.status_var.set(f'模型加载失败: {e}'))

        # ---- 上传预测 ----
        def upload(self):
            path = filedialog.askopenfilename(filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff')])
            if not path:
                return
            if self.predictor is None:
                self.status_var.set('模型未就绪（当前配置可能暂无模型），请换一个配置')
                return
            img = Image.open(path)
            img.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            self.status_var.set('预测中…')
            self.upload_btn.configure(state='disabled')
            threading.Thread(target=self._predict_async, args=(path,), daemon=True).start()

        def _predict_async(self, path):
            try:
                results = self.predictor.predict_image_path(path, topk=5)
                self.root.after(0, lambda: self._show_results(results))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.status_var.set(f'预测失败: {e}'))

        def _show_results(self, results):
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, '预测概率（前5）:\n\n')
            for name, prob in results:
                self.result_text.insert(tk.END, f'{name}: {prob:.2%}\n')
            self.status_var.set('就绪：请上传图片')
            self.upload_btn.configure(state='normal')

    root = tk.Tk()
    try:
        root.iconbitmap(os.path.join(_PROJECT_ROOT, 'favicon.ico'))
    except Exception:
        pass
    App(root)
    root.mainloop()


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
        print('— 各配置模型可用性探测 —')
        for cid in MODEL_REGISTRY:
            meta = pick_best(cid, verbose=False)
            if meta:
                acc = f"{meta['val_acc']:.2f}%@ep{meta['epoch']}" if meta['val_acc'] is not None else '?'
                print(f"  ✅ {cid:22s} {os.path.basename(meta['path'])}  (val {acc})")
            else:
                print(f"  ⬜ {cid:22s} 暂无模型（未跑完/未启动）")
        _run_gui()
