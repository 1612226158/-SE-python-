# -*- coding: utf-8 -*-
"""
eval_confusion_matrix.py —— 混淆矩阵 + 逐类错误分析（254 合并类口径，2026-09-12 建）

【用途】回答三个审稿人一定会问的问题：
  1) 模型到底错在哪：254 类全量 val 推理 → 混淆矩阵（真值×预测）；
  2) 哪些类最差：逐类 precision/recall/F1/support（per_class_metrics_<model>.csv）；
  3) 最典型的混淆对：top-N "真值 → 误判"对（top_confusions_<model>.csv + 图），
     并据此评估"易混淆类合并策略"到底救了多少（合并组内互混 vs 组间互混）。

【链接】模型名用 src_v2\model_registry.py 的注册 key（如 S-NoProG / V2-Full）；
  checkpoint 与整体成绩由 chart_data 按名解析（best_ckpt_path / results_best）；
  支持 v1 头（ResNetTransformer）与 v2 头（ResNetTransformerV2）自动分派。

【设备】默认 CPU（训练占着 GPU 时也不抢；val 14,603 张，CPU 全量约 20-40 分钟）。

用法（分阶段，避免重复推理）：
  python eval_confusion_matrix.py smoke  --model S-NoProG             # 5 个文件夹快速自检
  python eval_confusion_matrix.py infer  --model S-NoProG             # 全 val 推理 → confusion_*.npz
  python eval_confusion_matrix.py report --model S-NoProG             # 打印总成绩+最差类+top混淆对
  python eval_confusion_matrix.py plot   --model S-NoProG             # 出图（top混淆对 + 逐类召回）
  python eval_confusion_matrix.py all    --model S-NoProG             # infer + report + plot
输出（本目录）：confusion_<model>.npz / per_class_metrics_<model>.csv /
  top_confusions_<model>.csv / confusion_top_pairs_<model>.pdf / per_class_recall_<model>.pdf
"""
import os
import sys
import json
import csv
import time
import argparse

# —— 路径注入（等效 PyCharm Sources Root）——
_HERE = os.path.dirname(os.path.abspath(__file__))     # src_v2\ReportChart
_SRC = os.path.dirname(_HERE)                          # src_v2
_ROOT = os.path.dirname(_SRC)                          # 项目根
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from src_v2 import merged_dict
from calculate import generate_mappings, calculate_mean_and_std
from ResNet import ResNetTransformer
from ResNetTransformer import ResNetTransformerV2
import chart_data as cd
try:
    import model_registry as mr
except Exception:               # 注册表缺失时降级（按 v1 头处理）
    mr = None

# ============ 配置 ============
VAL_ROOT = os.path.join(_ROOT, 'val')
TRAIN_ROOT = os.path.join(_ROOT, 'train')
D_MODEL, NHEAD = 512, 8
# 与训练 SafeImageFolder._scan_disk 一致的扩展名（不含 avif）
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif')
TOP_N_PAIRS = 20        # 报告/条形图展示的混淆对数量
HEATMAP_CLASSES = 25    # 热力图子矩阵规模（按累计混淆次数取最相关类）
MIN_SUPPORT_WORST = 30  # 统计"最差类"时的最小验证样本数


def _model_meta(model):
    """从注册表取 (arch, regions, transformer_layers, use_decouple)，取不到则退回 v1 默认。"""
    if mr is not None:
        return mr.arch_of(model), mr.regions_of(model), mr.tf_of(model), mr.dec_of(model)
    return 'v1', [1, 2], 3, True


def _out(name, model, ext):
    return os.path.join(_HERE, f'{name}_{model}.{ext}')


# ============ 模型与数据 ============
def build_and_load(model_name, verbose=True):
    """按注册表信息构建对应头结构，加载 best checkpoint（CPU）。返回 (model, ckpt, renew, idx2name, path)。"""
    arch, regions, tf_layers, use_decouple = _model_meta(model_name)
    p = cd.best_ckpt_path(model_name)
    if p is None:
        raise SystemExit(f'[cm] 找不到 {model_name} 的 best checkpoint（runs/ 下）')
    ck = torch.load(p, map_location='cpu')
    ck_arch = str(ck.get('arch', ''))
    if ck_arch and (('V2' in ck_arch) != (arch == 'v2')):
        raise SystemExit(f'[cm] 架构不一致：注册表={arch}（{arch}头），checkpoint arch={ck_arch}')
    id2main, renew = generate_mappings(merged_dict=merged_dict)
    cls = ResNetTransformerV2 if arch == 'v2' else ResNetTransformer
    net = cls(transformer_layers=tf_layers, d_model=D_MODEL, id_to_main_class=id2main,
              renew_class_to_index=renew, nhead=NHEAD, regions=regions, use_decouple=use_decouple)
    net.unfreeze_all()
    sd = ck.get('state_dict') or ck.get('model_state_dict')
    net.load_state_dict(sd)
    net.eval()
    net = net.to('cpu')
    idx2name = {int(v): k for k, v in renew.items()}
    if verbose:
        print(f'[cm] 加载 {os.path.basename(p)}：{model_name}（{arch} 头, tf={tf_layers}, regions={regions}） '
              f'epoch={ck.get("epoch")} val_acc={ck.get("val_acc")}')
    return net, ck, renew, idx2name, p


def build_truth_map():
    """返回 (fid->class_name, class_name->index, 类别数)。"""
    id2main, renew = generate_mappings(merged_dict=merged_dict)
    fid2cls = {int(k): v for k, v in id2main.items()}
    return fid2cls, renew, len(renew)


def _transform(mean, std):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def val_folders():
    """val/ 下数字文件夹（按 id 升序）。"""
    out = []
    for name in os.listdir(VAL_ROOT):
        fp = os.path.join(VAL_ROOT, name)
        if name.isdigit() and os.path.isdir(fp):
            out.append(int(name))
    return sorted(out)


# ============ 推理 ============
def stage_infer(model_name, limit=None, threads=8, verbose=True):
    torch.set_num_threads(threads)
    net, ck, renew, idx2name, ckpt_path = build_and_load(model_name, verbose=verbose)
    fid2cls, _renew, n_cls = build_truth_map()
    mean, std, _skip = calculate_mean_and_std(TRAIN_ROOT, retrieve_file=True)
    tf = _transform(mean.tolist() if hasattr(mean, 'tolist') else mean,
                    std.tolist() if hasattr(std, 'tolist') else std)

    fids = [f for f in val_folders() if f in fid2cls]
    if limit:
        fids = fids[:limit]
    total_imgs = sum(len([n for n in os.listdir(os.path.join(VAL_ROOT, str(f)))
                          if n.lower().endswith(VALID_EXTS)]) for f in fids)
    print(f'[cm] val 文件夹 {len(fids)} 个，待推理图像 {total_imgs} 张（CPU, threads={threads}）', flush=True)

    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    done, t0 = 0, time.time()
    with torch.no_grad():
        for k, fid in enumerate(fids, 1):
            folder = os.path.join(VAL_ROOT, str(fid))
            tcls = fid2cls[fid]
            ti = int(renew[tcls])
            names = sorted(n for n in os.listdir(folder) if n.lower().endswith(VALID_EXTS))
            for name in names:
                try:
                    img = Image.open(os.path.join(folder, name)).convert('RGB')
                except Exception:
                    continue
                out = net(tf(img).unsqueeze(0))
                pi = int(out.argmax(dim=1).item())
                cm[ti, pi] += 1
                done += 1
            if verbose and (k % 20 == 0 or k == len(fids)):
                acc = np.trace(cm) / max(1, cm.sum())
                el = time.time() - t0
                print(f'  [{k}/{len(fids)}] 图 {done}/{total_imgs} 累计acc={acc*100:.2f}% '
                      f'用时{el/60:.1f}min 预计剩余{el/max(1,done)*(total_imgs-done)/60:.1f}min', flush=True)
    names = [idx2name[i] for i in range(n_cls)]
    np.savez_compressed(_out('confusion', model_name, 'npz'),
                        cm=cm, names=np.array(names, dtype=object),
                        model=model_name, ckpt=os.path.basename(ckpt_path),
                        limit=(-1 if not limit else limit))
    print(f'[cm] 已保存 {_out("confusion", model_name, "npz")}')
    return cm, names


def load_cm(model_name):
    p = _out('confusion', model_name, 'npz')
    if not os.path.exists(p):
        raise SystemExit(f'[cm] 缺少 {p}，请先跑 infer 阶段')
    z = np.load(p, allow_pickle=True)
    return z['cm'].astype(np.int64), [str(x) for x in z['names']]


# ============ 指标与报告 ============
def per_class_metrics(cm, names):
    rows = []
    for i, nm in enumerate(names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        sup = int(cm[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append(dict(cls=nm, support=sup, tp=tp, fp=fp, fn=fn,
                         precision=prec, recall=rec, f1=f1))
    return rows


def top_confusions(cm, names, n=TOP_N_PAIRS):
    pairs = []
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and cm[i, j] > 0:
                pairs.append((names[i], names[j], int(cm[i, j]), int(cm[i, :].sum())))
    pairs.sort(key=lambda t: -t[2])
    return pairs[:n]


def stage_report(model_name, top_n=TOP_N_PAIRS, worst_k=15):
    cm, names = load_cm(model_name)
    rows = per_class_metrics(cm, names)
    total = int(cm.sum())
    acc = np.trace(cm) / max(1, total) * 100
    macro_f1 = float(np.mean([r['f1'] for r in rows])) * 100
    sup_sum = sum(r['support'] for r in rows) or 1
    weighted_f1 = sum(r['f1'] * r['support'] for r in rows) / sup_sum * 100
    print('=' * 78)
    print(f'模型 {model_name}｜验证样本 {total} 张｜类别 {len(names)}')
    print(f'整体准确率 {acc:.2f}%｜宏平均 F1 {macro_f1:.2f}%｜加权 F1 {weighted_f1:.2f}%')
    print('=' * 78)

    pairs = top_confusions(cm, names, top_n)
    print(f'—— top{top_n} 混淆对（真值 → 误判：次数 / 占该类错误比例 / 占该类样本比例）——')
    for t, p, c, tsup in pairs:
        err = tsup - int(cm[names.index(t), names.index(t)])
        print(f'  {t:　<22s} → {p:　<22s} {c:>5d}  {c/max(1,err)*100:5.1f}%  {c/max(1,tsup)*100:5.2f}%')

    cand = [r for r in rows if r['support'] >= MIN_SUPPORT_WORST]
    cand.sort(key=lambda r: (r['recall'], -r['support']))
    print(f'—— 最差 {worst_k} 类（验证样本 ≥{MIN_SUPPORT_WORST}，按召回率升序）——')
    for r in cand[:worst_k]:
        print(f'  {r["cls"]:　<22s} 召回 {r["recall"]*100:5.1f}%  精确 {r["precision"]*100:5.1f}%  '
              f'F1 {r["f1"]*100:5.1f}%  support={r["support"]}')

    # 落盘
    with open(_out('per_class_metrics', model_name, 'csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['cls', 'support', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])
        w.writeheader()
        for r in sorted(rows, key=lambda r: r['recall']):
            w.writerow(r)
    with open(_out('top_confusions', model_name, 'csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['true_class', 'pred_class', 'count', 'true_support', 'share_of_errors', 'share_of_support'])
        for t, p, c, tsup in top_confusions(cm, names, 200):
            err = tsup - int(cm[names.index(t), names.index(t)])
            w.writerow([t, p, c, tsup, round(c / max(1, err), 4), round(c / max(1, tsup), 4)])
    print(f'[cm] 已保存 per_class_metrics_{model_name}.csv 与 top_confusions_{model_name}.csv')
    return cm, names, rows


# ============ 绘图 ============
def stage_plot(model_name, top_n=TOP_N_PAIRS):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12,
                         'legend.fontsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9})

    cm, names = load_cm(model_name)
    pairs = top_confusions(cm, names, top_n)
    total = int(cm.sum())
    acc = np.trace(cm) / max(1, total) * 100

    # —— 图 1：top 混淆对条形 + 相关类子矩阵热力图 ——
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2), gridspec_kw={'width_ratios': [1.25, 1.0]})
    labels = [f'{t} → {p}' for t, p, _, _ in pairs][::-1]
    vals = [c for _, _, c, _ in pairs][::-1]
    ax1.barh(range(len(vals)), vals, color='#c44e52', alpha=0.85)
    ax1.set_yticks(range(len(vals)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel('误判次数（254 类验证集）')
    ax1.set_title(f'(a) 最典型混淆对 top{top_n}（{model_name}，整体 {acc:.2f}%）', fontsize=12)
    for yi, v in enumerate(vals):
        ax1.text(v + max(vals) * 0.01, yi, str(v), va='center', fontsize=8)

    idxs = []
    for t, p, _, _ in pairs:
        for nm in (t, p):
            i = names.index(nm)
            if i not in idxs:
                idxs.append(i)
    idxs = idxs[:HEATMAP_CLASSES]
    sub = cm[np.ix_(idxs, idxs)].astype(float)
    row = sub.sum(axis=1, keepdims=True)
    subn = np.divide(sub, row, out=np.zeros_like(sub), where=row > 0) * 100
    im = ax2.imshow(subn, cmap='YlOrRd', vmin=0, vmax=100)
    ax2.set_xticks(range(len(idxs)))
    ax2.set_yticks(range(len(idxs)))
    ax2.set_xticklabels([names[i] for i in idxs], rotation=90, fontsize=7)
    ax2.set_yticklabels([names[i] for i in idxs], fontsize=7)
    ax2.set_xlabel('预测类')
    ax2.set_ylabel('真值类')
    ax2.set_title('(b) 高混淆类子矩阵（行归一化 %，对角线=召回）', fontsize=12)
    # colorbar 用独立 axes，避免挤压缩短 ax2、导致左右面板标题错位
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax2).append_axes('right', size='4%', pad=0.15)
    fig.colorbar(im, cax=cax, label='行内占比 (%)')
    fig.tight_layout()
    out1 = _out('confusion_top_pairs', model_name, 'pdf')
    fig.savefig(out1, dpi=300)
    fig.savefig(out1.replace('.pdf', '_preview.png'), dpi=110)
    plt.close(fig)
    print(f'[cm] 已生成 {out1}')

    # —— 图 2：逐类召回分布（左）+ 最差 10 类条形（右）——
    # 右图单独列最差类，避免长类名在曲线左端低召回区挤成一团（2026-09-12 视觉自检后重构）
    rows = per_class_metrics(cm, names)
    rows_sorted = sorted(rows, key=lambda r: r['recall'])
    cand = [r for r in rows_sorted if r['support'] >= MIN_SUPPORT_WORST]
    worst = (cand or rows_sorted)[:10]
    pos = {r['cls']: i for i, r in enumerate(rows_sorted)}

    fig2, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 5.6), gridspec_kw={'width_ratios': [1.5, 1.0]})
    xs = np.arange(len(rows_sorted))
    rec = np.array([r['recall'] * 100 for r in rows_sorted])
    axL.plot(xs, rec, color='#4c72b0', lw=1.3, label='逐类召回率')
    axL.axhline(np.mean(rec), color='#dd8452', ls='--', lw=1.4, label=f'宏平均召回 {np.mean(rec):.1f}%')
    axL.scatter([pos[r['cls']] for r in worst], [r['recall'] * 100 for r in worst],
                color='#c44e52', zorder=5, s=30, label='最差 10 类（右图详列）')
    axL.set_xlabel('类别（按召回率升序，共 %d 类）' % len(rows_sorted))
    axL.set_ylabel('召回率 (%)')
    axL.set_ylim(-3, 103)
    axL.set_title(f'(c) 逐类召回分布（{model_name}）', fontsize=12)
    axL.legend(loc='lower right')

    nm = [r['cls'] for r in worst][::-1]
    vl = [r['recall'] * 100 for r in worst][::-1]
    sp = [r['support'] for r in worst][::-1]
    axR.barh(range(len(vl)), vl, color='#c44e52', alpha=0.85)
    axR.set_yticks(range(len(vl)))
    axR.set_yticklabels([f'{a}（n={b}）' for a, b in zip(nm, sp)], fontsize=8)
    axR.set_xlim(0, 100)
    axR.set_xlabel('召回率 (%)')
    axR.set_title('(d) 最差 10 类（验证样本 ≥%d）' % MIN_SUPPORT_WORST, fontsize=12)
    for yi, v in enumerate(vl):
        axR.text(v + 1.2, yi, f'{v:.1f}%', va='center', fontsize=8)
    fig2.tight_layout()
    out2 = _out('per_class_recall', model_name, 'pdf')
    fig2.savefig(out2, dpi=300)
    fig2.savefig(out2.replace('.pdf', '_preview.png'), dpi=110)
    plt.close(fig2)
    print(f'[cm] 已生成 {out2}')


# ============ CLI ============
def main():
    ap = argparse.ArgumentParser(description='混淆矩阵 + 逐类错误分析（254 合并类口径）')
    ap.add_argument('stage', choices=['smoke', 'infer', 'report', 'plot', 'all'])
    ap.add_argument('--model', default='S-NoProG', help='注册表模型 key（如 S-NoProG / V2-Full / G-Full-CAWR）')
    ap.add_argument('--limit', type=int, default=None, help='只推理前 N 个 val 文件夹（冒烟用）')
    ap.add_argument('--threads', type=int, default=8, help='CPU 线程数（默认 8，避免抢占训练）')
    a = ap.parse_args()

    if a.stage == 'smoke':
        cm, names = stage_infer(a.model, limit=a.limit or 5, threads=a.threads)
        acc = np.trace(cm) / max(1, cm.sum()) * 100
        print(f'[cm] 冒烟完成：{cm.sum()} 张，acc={acc:.2f}%（仅前 {a.limit or 5} 个文件夹，非正式成绩）')
    elif a.stage == 'infer':
        stage_infer(a.model, limit=a.limit, threads=a.threads)
    elif a.stage == 'report':
        stage_report(a.model)
    elif a.stage == 'plot':
        stage_plot(a.model)
    else:
        stage_infer(a.model, limit=a.limit, threads=a.threads)
        stage_report(a.model)
        stage_plot(a.model)


if __name__ == '__main__':
    main()
