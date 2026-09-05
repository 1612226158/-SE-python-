# -*- coding: utf-8 -*-
"""
predict_gui_snoprog.py —— S-NoProg（全解冻主模型候选，best 82.63%@74）单图预测 GUI

【来源】src/eval2.py 的 src_v2 适配版：
  - 原版按"旧格式断点"加载（checkpoint['model'] 整对象 + model_state_dict）；
  - src_v2 的 best.pth 是"新格式精简 state_dict"（仅 epoch/config_id/seed/arch/state_dict/val_acc，
    无整模型、无 renew 映射）→ 本版改为：用 src_v2 的 merged_dict(254类) 重建 ResNetTransformer 后
    load_state_dict(best.pth['state_dict'])。
【用法】
  D:\\Python\\Python3.10.7\\python.exe E:\\DataSet\\垃圾分类图片-2\\src_v2\\predict_gui_snoprog.py
  或 PyCharm 直接 Run（工作目录无关，路径已绝对化）。
  若机器无显示器/GUI 失败，可改用同目录 evaluate_snoprog.py 的命令行模式。

【运行前提】
  1. 训练/排队器未占用 GPU 时不冲突（eval 只占 ~5GB，可共存但建议串行）；
  2. 首次运行会读 train/ 目录算 mean/std（有缓存则秒回）。
"""
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
from calculate import calculate_mean_and_std, generate_mappings, get_classnames
from src_v2 import merged_dict

# ==================== 可配置项 ====================
# S-NoProg（全解冻）best 模型：checkpoint_S-NoProg_seed1_best.pth（best val 82.63%@epoch74）
BEST_PTH = os.path.join(_HERE, 'runs', 'checkpoint_S-NoProg_seed1_best.pth')
TRANSFORMER_LAYERS = 3
D_MODEL = 512
NHEAD = 8
REGIONS = [1, 2]        # S-NoProg = 多区域 SE [1,2]
USE_DECOUPLE = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SNoProgPredictor:
    """加载 S-NoProg best 模型 + 254 类映射，提供单图预测。"""

    def __init__(self, best_pth=BEST_PTH, verbose=True):
        if not os.path.exists(best_pth):
            raise FileNotFoundError(f'未找到 best 模型: {best_pth}\n'
                                    f'请先跑完 S-NoProg seed1（runs 下应生成 checkpoint_S-NoProg_seed1_best.pth）')
        # —— mean/std（复用训练集自定义统计；有缓存则秒回）——
        file = os.path.join(_PROJECT_ROOT, 'train')
        self.mean, self.std, _ = calculate_mean_and_std(file, retrieve_file=True)

        # —— 254 类映射（与训练时同一 merged_dict / classname.txt / 排序逻辑）——
        self.id_to_main_class, self.renew_class_to_index = generate_mappings(merged_dict=merged_dict)
        self.index_to_name = {int(v): k for k, v in self.renew_class_to_index.items()}
        num_classes = len(self.renew_class_to_index)
        if verbose:
            print(f'[映射] 类别数 = {num_classes}（应为 254）')
            print(f'[映射] 前 5 类: {sorted(self.index_to_name.items())[:5]}')

        # —— 重建模型（结构须与训练时完全一致，否则 load_state_dict 会报 size mismatch）——
        self.model = ResNetTransformer(
            transformer_layers=TRANSFORMER_LAYERS,
            d_model=D_MODEL,
            id_to_main_class=self.id_to_main_class,
            renew_class_to_index=self.renew_class_to_index,
            nhead=NHEAD,
            regions=REGIONS,
            use_decouple=USE_DECOUPLE,
        )
        # 载入权重前先全解冻（best 保存的是最终全解冻状态的全部参数）
        self.model.unfreeze_all()

        ck = torch.load(best_pth, map_location='cpu')
        ck_arch = ck.get('arch')
        cur_arch = type(self.model).__name__
        if ck_arch is not None and ck_arch != cur_arch:
            raise SystemExit(f'[arch] 断点架构 {ck_arch} ≠ 当前模型 {cur_arch}，拒绝加载')
        sd = ck.get('state_dict', ck.get('model_state_dict'))
        if sd is None:
            raise KeyError('断点里既没有 state_dict 也没有 model_state_dict')
        self.model.load_state_dict(sd)
        self.model.to(device)
        self.model.eval()
        if verbose:
            print(f'[加载] {os.path.basename(best_pth)} 成功 '
                  f'(config={ck.get("config_id")}, seed={ck.get("seed")}, best_epoch={ck.get("epoch")}, '
                  f'best_val_acc={ck.get("val_acc"):.2f}%)' if ck.get('val_acc') is not None
                  else f'[加载] {os.path.basename(best_pth)} 成功')
        self._class_labels = get_classnames()  # 数字文件夹序号 → 原始类名（用于显示"私有类-XXX"的 XXX）

    def _clean_name(self, name):
        """把 '私有类-xxx' / '公共类-父类名' 转成更友好的显示名。"""
        if name.startswith('私有类-'):
            return name[len('私有类-'):]
        return name

    def warmup(self):
        """预热：用 dummy 图跑一次前向，触发 CUDA/cuDNN 首次初始化（kernel 编译、显存分配）。
        应放在 GUI 启动、模型加载完成后立即执行，这样用户点"上传"时不再卡第一张。"""
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            self.model(dummy)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        print('[预热] CUDA/cuDNN 初始化完成，后续预测不再卡顿')

    def predict_image_path(self, image_path, topk=5):
        """对单张图片预测，返回 [(类名, 概率), ...] 按概率降序。"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        img = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.softmax(out, dim=1).cpu().numpy().flatten()
        top_idx = probs.argsort()[::-1][:topk]
        results = []
        for idx in top_idx:
            name = self.index_to_name.get(int(idx), f'类{idx}')
            results.append((self._clean_name(name), float(probs[idx])))
        return results


# ==================== GUI（沿用 eval2 风格，加载逻辑换 SNoProgPredictor） ====================
def _run_gui():
    import tkinter as tk
    from tkinter import filedialog, ttk
    import threading

    class App:
        def __init__(self, root):
            self.root = root
            self.root.title('垃圾分类预测（S-NoProg 全解冻模型 · best 82.63% @254类）')
            self.root.geometry('800x600')
            left = ttk.Frame(root); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            right = ttk.Frame(root); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            self.image_label = ttk.Label(left); self.image_label.pack(pady=10)
            self.upload_btn = ttk.Button(left, text='上传图片', command=self.upload)
            self.upload_btn.pack(pady=10)
            ttk.Label(right, text='预测结果（前5）', font=('Arial', 14, 'bold')).pack()
            self.result_text = tk.Text(right, height=12, width=42)
            self.result_text.pack(pady=10)
            self.status_var = tk.StringVar(value='模型加载中…')
            ttk.Label(right, textvariable=self.status_var).pack(pady=4)
            # —— 后台线程加载模型 + 预热，避免启动界面假死 ——
            threading.Thread(target=self._load_and_warmup, daemon=True).start()

        def _load_and_warmup(self):
            try:
                self.predictor = SNoProgPredictor()
                self.predictor.warmup()
                self.root.after(0, lambda: self.status_var.set('就绪：请上传图片'))
                self.root.after(0, lambda: self.upload_btn.configure(state='normal'))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.status_var.set(f'模型加载失败: {e}'))

        def upload(self):
            path = filedialog.askopenfilename(filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff')])
            if not path:
                return
            if not hasattr(self, 'predictor'):
                self.status_var.set('模型还在加载中，请稍候…')
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
                # 线程里不能直接改 tk 控件，回主线程更新
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
    # 支持命令行快速测试：python predict_gui_snoprog.py <图片路径>
    if len(sys.argv) > 1:
        p = SNoProgPredictor(verbose=True)
        p.warmup()
        print('=' * 60)
        for name, prob in p.predict_image_path(sys.argv[1], topk=5):
            print(f'{name:20s} {prob:.2%}')
    else:
        _run_gui()
