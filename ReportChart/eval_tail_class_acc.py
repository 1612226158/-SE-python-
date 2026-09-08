# -*- coding: utf-8 -*-
"""
eval_tail_class_acc.py —— 长尾小样本类对比：S-NoProG(全解冻) vs G-Full-CAWR(渐进式)
研究问题：渐进式解冻是否在"训练样本最少的类"上更有优势？
口径：254 个合并类（merged_dict 合并后），每类样本数 = 其成员文件夹(id)的图片数之和。
切片：训练样本升序取最少 50 类 / 最少 100 类（最少50 ⊂ 最少100，嵌套）。

用法（分阶段，避免推理长任务阻塞）：
  D:\\Python\\Python3.10.7\\python.exe eval_tail_class_acc.py count   # 统计类分布+生成 tail_classes.json
  D:\\Python\\Python3.10.7\\python.exe eval_tail_class_acc.py infer   # CPU 推理两模型(仅tail100覆盖的val图) → per_class_acc.csv
  D:\\Python\\Python3.10.7\\python.exe eval_tail_class_acc.py plot    # 读 csv 出图
输出：本目录 tail_class_acc.pdf + per_class_acc.csv（SCI 风格，与既有论文图一致）

【语义】
- 聚合 acc（大数口径，更稳）= Σ正确 / Σ样本（该切片内全部 val 图）
- 宏平均 acc = 切片内每类 acc 的简单平均（每类 val 仅~10张，噪声大，仅供对照）
- 整体 254 类对照直接用 checkpoint 自带 val_acc：S-NoProG 82.63 / G-Full-CAWR 80.96
"""
import os
import sys
import json
import csv

# —— 路径注入（等效 PyCharm Sources Root）——
_HERE = os.path.dirname(os.path.abspath(__file__))          # src_v2\ReportChart
_SRC = os.path.dirname(_HERE)                                # src_v2
_ROOT = os.path.dirname(_SRC)                                # 项目根
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from src_v2 import merged_dict
from calculate import generate_mappings, get_classnames, calculate_mean_and_std
from ResNet import ResNetTransformer

# ============ 配置 ============
RUNS = os.path.join(_SRC, 'runs')
VAL_ROOT = os.path.join(_ROOT, 'val')
TRAIN_ROOT = os.path.join(_ROOT, 'train')
# 对比的模型（输入模型英文名即可，checkpoint 路径与整体 acc 由 chart_data 动态解析）
COMPARE_MODELS = ['S-NoProG', 'G-Full-CAWR']
from chart_data import best_ckpt_path, results_best, model_best  # noqa: E402

def _build_ckpt_and_overall():
    """动态生成 {模型名: best checkpoint 路径} 与 {模型名: 整体254类 val_acc}。
    整体 acc 论文口径优先 results JSON（82.63/80.96），无 results 退回 metrics best。"""
    ckpt, overall = {}, {}
    for name in COMPARE_MODELS:
        p = best_ckpt_path(name)
        if p is None:
            raise FileNotFoundError(f'[tail] 找不到 {name} 的 best checkpoint（runs/ 下）')
        ckpt[name] = p
        va, _, _ = results_best(name)
        overall[name] = va if va is not None else model_best(name)[0]
    return ckpt, overall

CKPT, OVERALL_ACC = _build_ckpt_and_overall()
TAIL_JSON = os.path.join(_HERE, 'tail_classes.json')
PER_CLASS_CSV = os.path.join(_HERE, 'per_class_acc.csv')

# 模型结构：两模型完全一致（与 predict_gui_snoprog.py / PRESETS 核对）
TRANSFORMER_LAYERS = 3
D_MODEL = 512
NHEAD = 8
REGIONS = [1, 2]
USE_DECOUPLE = True

# 与训练 SafeImageFolder._scan_disk 完全一致的扩展名（不含 avif）
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif')


# ============ 数据统计 ============
def count_images(root):
    """数字文件夹 -> 图片数（扩展名过滤，与训练缓存扫描一致）。"""
    cnt = {}
    for folder in os.listdir(root):
        fp = os.path.join(root, folder)
        if not folder.isdigit() or not os.path.isdir(fp):
            continue
        n = 0
        for name in os.listdir(fp):
            if name.lower().endswith(VALID_EXTS):
                n += 1
        cnt[int(folder)] = n
    return cnt


def build_class_table():
    """合并类口径：class_name -> dict(ids, train_n, val_n)。"""
    id_to_main_class, renew_class_to_index = generate_mappings(merged_dict=merged_dict)
    # id(str) -> class_name
    id2cls = {int(k): v for k, v in id_to_main_class.items()}
    train_cnt = count_images(TRAIN_ROOT)
    val_cnt = count_images(VAL_ROOT)

    # 按类聚合
    class_table = {}
    for fid in sorted(id2cls):
        cls = id2cls[fid]
        row = class_table.setdefault(cls, {'ids': [], 'train_n': 0, 'val_n': 0})
        row['ids'].append(fid)
        row['train_n'] += train_cnt.get(fid, 0)
        row['val_n'] += val_cnt.get(fid, 0)
    n_classes = len(class_table)
    assert n_classes == 254, f'合并类数应为 254，实际 {n_classes}'
    return class_table, id_to_main_class, renew_class_to_index


def stage_count():
    class_table, id2cls, renew = build_class_table()
    rows = sorted(class_table.items(), key=lambda kv: (kv[1]['train_n'], kv[0]))
    print(f'合并类总数: {len(rows)}')
    print(f'{"名次":>4} {"类名":<32} {"train样本":>8} {"val样本":>7} {"成员ids":>12}')
    for rank, (cls, r) in enumerate(rows, 1):
        if rank <= 30 or rank in (50, 100) or rank > 244:
            print(f'{rank:>4} {cls:<32} {r["train_n"]:>8} {r["val_n"]:>7} {str(r["ids"]):>12}')
    # 汇总切片
    def slice_stat(k):
        sub = rows[:k]
        tn = sum(r['train_n'] for _, r in sub)
        vn = sum(r['val_n'] for _, r in sub)
        return tn, vn
    for k in (50, 100):
        tn, vn = slice_stat(k)
        print(f'--- 最少 {k} 类: train 合计 {tn}, val 合计 {vn}（推理张数上限）---')
    tail50 = [c for c, _ in rows[:50]]
    tail100 = [c for c, _ in rows[:100]]
    with open(TAIL_JSON, 'w', encoding='utf-8') as f:
        json.dump({'tail50': tail50, 'tail100': tail100}, f, ensure_ascii=False, indent=1)
    print(f'[count] tail 名单已存 {TAIL_JSON}')
    # 校准总量
    print(f'[count] 校准: train 总数={sum(count_images(TRAIN_ROOT).values())} '
          f'(权威132293), val 总数={sum(count_images(VAL_ROOT).values())} (权威14603)')


# ============ 推理 ============
def _transform(mean, std):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _load_model(best_pth):
    ck = torch.load(best_pth, map_location='cpu')
    assert ck.get('arch') == 'ResNetTransformer', f"arch 不符: {ck.get('arch')}"
    id2main, renew = generate_mappings(merged_dict=merged_dict)
    model = ResNetTransformer(
        transformer_layers=TRANSFORMER_LAYERS,
        d_model=D_MODEL,
        id_to_main_class=id2main,
        renew_class_to_index=renew,
        nhead=NHEAD,
        regions=REGIONS,
        use_decouple=USE_DECOUPLE,
    )
    model.unfreeze_all()
    sd = ck.get('state_dict') or ck.get('model_state_dict')
    model.load_state_dict(sd)
    model.eval()
    model = model.to('cpu')
    # 类索引反查: model 输出 idx -> class_name
    index_to_name = {int(v): k for k, v in renew.items()}
    print(f'[infer] 加载 {os.path.basename(best_pth)}: config={ck.get("config_id")} '
          f'seed={ck.get("seed")} best_epoch={ck.get("epoch")} val_acc={ck.get("val_acc"):.2f}')
    return model, index_to_name


def stage_smoke():
    """冒烟：加载两模型，对 tail 里第一个文件夹的前 3 张图打印预测 vs 真值，验证映射。"""
    class_table, id2cls, renew = build_class_table()
    with open(TAIL_JSON, 'r', encoding='utf-8') as f:
        tails = json.load(f)
    fid0 = class_table[tails['tail50'][0]]['ids'][0]
    mean, std, _skip = calculate_mean_and_std(os.path.join(_ROOT, 'train'), retrieve_file=True)
    print(f'[smoke] 测试文件夹 id={fid0} 真值类={id2cls[str(fid0)]}')
    for tag, pth in CKPT.items():
        model, index_to_name = _load_model(pth)
        folder = os.path.join(VAL_ROOT, str(fid0))
        names = sorted(n for n in os.listdir(folder) if n.lower().endswith(VALID_EXTS))[:3]
        with torch.no_grad():
            for name in names:
                img = Image.open(os.path.join(folder, name)).convert('RGB')
                t = _transform(mean, std)(img).unsqueeze(0)
                out = model(t)
                probs = torch.softmax(out, dim=1)[0]
                pred_idx = int(out.argmax(dim=1).item())
                print(f'  [{tag}] {name}: pred={index_to_name.get(pred_idx)} '
                      f'(p={probs[pred_idx].item():.3f}) '
                      f'{"✓" if index_to_name.get(pred_idx) == id2cls[str(fid0)] else "✗"}')
    print('[smoke] 完成')


def stage_infer():
    torch.set_num_threads(16)
    class_table, id2cls, renew = build_class_table()
    with open(TAIL_JSON, 'r', encoding='utf-8') as f:
        tails = json.load(f)
    tail100 = set(tails['tail100'])
    # tail100 类覆盖的 val 文件夹 id
    member_ids = sorted(fid for cls in tail100 for fid in class_table[cls]['ids'])
    print(f'[infer] tail100 覆盖 {len(member_ids)} 个 val 文件夹')

    file = os.path.join(_ROOT, 'train')
    mean, std, _skip = calculate_mean_and_std(file, retrieve_file=True)
    print(f'[infer] mean={mean.tolist()} std={std.tolist()}')

    results = {}   # class_name -> {tag: [correct, total]}
    for tag, pth in CKPT.items():
        model, index_to_name = _load_model(pth)
        correct, total = 0, 0
        with torch.no_grad():
            for fid in member_ids:
                folder = os.path.join(VAL_ROOT, str(fid))
                true_cls = id2cls[str(fid)]
                names = sorted(n for n in os.listdir(folder)
                               if n.lower().endswith(VALID_EXTS))
                for name in names:
                    img = Image.open(os.path.join(folder, name)).convert('RGB')
                    t = _transform(mean, std)(img).unsqueeze(0)
                    out = model(t)
                    pred_idx = int(out.argmax(dim=1).item())
                    pred_cls = index_to_name.get(pred_idx, '?')
                    ok = 1 if pred_cls == true_cls else 0
                    r = results.setdefault(true_cls, {})
                    cell = r.setdefault(tag, [0, 0])
                    cell[0] += ok
                    cell[1] += 1
                    correct += ok
                    total += 1
                if fid % 20 == 0:
                    print(f'  [{tag}] 文件夹 {fid}/{member_ids[-1]} 累计 {total} 张 '
                          f'acc={correct/total*100:.2f}%')
        print(f'[{tag}] tail100 覆盖全部完成: {total} 张, 原始聚合 acc={correct/total*100:.2f}%')
        torch.cuda.empty_cache()

    # 存每类结果
    with open(PER_CLASS_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class_name', 'train_n', 'val_n', 'member_ids',
                    'S_correct', 'S_total', 'G_correct', 'G_total'])
        for cls in sorted(class_table):
            r = results.get(cls, {})
            ct = class_table[cls]
            s = r.get('S-NoProG', [0, 0])
            g = r.get('G-Full-CAWR', [0, 0])
            w.writerow([cls, ct['train_n'], ct['val_n'], str(ct['ids']),
                        s[0], s[1], g[0], g[1]])
    print(f'[infer] 每类结果已存 {PER_CLASS_CSV}')
    print('[infer] 注意: 非 tail100 的类 S_total=0（本脚本只推最少100类）')


def summarize_slice(rows_by_name, names):
    """rows_by_name: class_name -> dict(S_correct,S_total,G_correct,G_total)（csv 读入为 str）"""
    agg = {'S': [0, 0], 'G': [0, 0]}
    macro = {'S': [], 'G': []}
    for c in names:
        r = rows_by_name.get(c)
        if r is None:
            continue
        if int(r['S_total']):
            agg['S'][0] += int(r['S_correct']); agg['S'][1] += int(r['S_total'])
            macro['S'].append(int(r['S_correct']) / int(r['S_total']))
        if int(r['G_total']):
            agg['G'][0] += int(r['G_correct']); agg['G'][1] += int(r['G_total'])
            macro['G'].append(int(r['G_correct']) / int(r['G_total']))
    return agg, macro


def check_text_overlap(fig):
    """程序化检测图内文本包围盒两两重叠（无需视觉模型）。打印可疑对。"""
    from matplotlib.text import Text
    fig.canvas.draw()
    texts = [t for t in fig.findobj(Text) if t.get_text().strip() and t.get_visible()]
    boxes = {}
    for t in texts:
        bb = t.get_window_extent()
        boxes[t] = (bb.x0, bb.y0, bb.x1, bb.y1)
    n_overlap = 0
    items = list(boxes.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            t1, b1 = items[i]
            t2, b2 = items[j]
            # 同一父坐标轴内的文本才可能互相遮
            if t1.axes is not t2.axes:
                continue
            ix = min(b1[2], b2[2]) - max(b1[0], b2[0])
            iy = min(b1[3], b2[3]) - max(b1[1], b2[1])
            if ix > 2 and iy > 2:  # 像素级容差
                n_overlap += 1
                if n_overlap <= 12:
                    print(f'  [重叠?] {t1.get_text()[:18]!r} <-> {t2.get_text()[:18]!r} '
                          f'(ix={ix:.0f}px, iy={iy:.0f}px)')
    print(f'[check] 文本两两重叠对数(同轴内, 容差2px): {n_overlap}')
    return n_overlap


def stage_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    class_table, _i, _r = build_class_table()
    with open(TAIL_JSON, 'r', encoding='utf-8') as f:
        tails = json.load(f)
    with open(PER_CLASS_CSV, 'r', encoding='utf-8') as f:
        rows = {r['class_name']: r for r in csv.DictReader(f)}

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12,
                         'legend.fontsize': 10, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10})
    C_FULL = '#4c72b0'   # S-NoProG 全解冻（蓝）
    C_PROG = '#dd8452'   # G-Full-CAWR 渐进式（橙）

    def acc_agg(nm):
        agg, _ = summarize_slice(rows, nm)
        return (agg['S'][0] / agg['S'][1] * 100, agg['G'][0] / agg['G'][1] * 100)

    a50, b50 = acc_agg(tails['tail50'])
    a100, b100 = acc_agg(tails['tail100'])
    # 每类宏平均
    _, macro50 = summarize_slice(rows, tails['tail50'])
    _, macro100 = summarize_slice(rows, tails['tail100'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                   gridspec_kw={'width_ratios': [1.15, 1.4]})

    # ---- 左面板：三组聚合柱状 ----
    labels = ['最少50类\n(聚合)', '最少100类\n(聚合)', '全部254类\n(整体参考)']
    full = OVERALL_ACC['S-NoProG']
    prog = OVERALL_ACC['G-Full-CAWR']
    s_vals = [a50, a100, full]
    g_vals = [b50, b100, prog]

    x = np.arange(len(labels))
    w = 0.36
    b1 = ax1.bar(x - w / 2, s_vals, w, label='S-NoProG（全解冻）', color=C_FULL)
    b2 = ax1.bar(x + w / 2, g_vals, w, label='G-Full-CAWR（渐进式）', color=C_PROG)
    for xi, sv, gv in zip(x, s_vals, g_vals):
        ax1.text(xi - w / 2, sv + 1.2, f'{sv:.1f}', ha='center', fontsize=9,
                 color=C_FULL)
        ax1.text(xi + w / 2, gv + 1.2, f'{gv:.1f}', ha='center', fontsize=9,
                 color='#b85a1a')
        d = sv - gv
        ax1.text(xi, max(sv, gv) + 5.0, f'Δ{d:+.1f}pp', ha='center',
                 fontsize=10, fontweight='bold', color='#333333')
    ax1.set_ylabel('Top-1 准确率 (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 100)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_title('(a) 聚合准确率：小样本切片', fontsize=12)

    # ---- 右面板：前缀趋势（前 k 个最少类的聚合 acc，k=10..100 + 全体）----
    all_sorted = sorted(class_table.items(), key=lambda kv: (kv[1]['train_n'], kv[0]))
    ks = list(range(10, 101, 10))
    xs = []
    s_curve, g_curve = [], []
    for k in ks:
        names = [c for c, _ in all_sorted[:k]]
        sa, ga = acc_agg(names)
        max_train = max(class_table[c]['train_n'] for c in names)
        xs.append(k)
        s_curve.append(sa)
        g_curve.append(ga)
    # 全体 254 类的整体参考（checkpoint val_acc，非本次逐类推理）
    xs_all = xs + [254]
    s_all = s_curve + [full]
    g_all = g_curve + [prog]

    ax2.plot(xs_all, s_all, 'o-', color=C_FULL, label='S-NoProG（全解冻）',
             ms=5, lw=1.6)
    ax2.plot(xs_all, g_all, 's--', color=C_PROG, label='G-Full-CAWR（渐进式）',
             ms=5, lw=1.6)
    for xi, sv, gv in zip(xs_all, s_all, g_all):
        d = sv - gv
        ax2.annotate(f'{d:+.1f}', (xi, min(sv, gv) - 3.2), ha='center',
                     fontsize=8, color='#555555')
    ax2.set_xlabel('参与统计的类数（按训练样本数从少到多取前 k 类）')
    ax2.set_ylabel('聚合 Top-1 准确率 (%)')
    ax2.set_xticks(xs_all)
    ax2.set_xticklabels([str(k) for k in xs] + ['全体254'])
    ax2.set_ylim(0, 100)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_title('(b) 前缀趋势：差距随类样本量变化', fontsize=12)
    ax2.grid(True, alpha=0.4)

    fig.suptitle('长尾小样本类：渐进式解冻 vs 全解冻（254类口径，逐类推理）', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(_HERE, 'tail_class_acc.pdf')
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace('.pdf', '_preview.png'), dpi=110)
    check_text_overlap(fig)
    print(f'[plot] 已输出 {out}')

    # 控制台汇总
    print('=' * 70)
    print(f'{"切片":<16}{"S-NoProG":>12}{"G-Full-CAWR":>14}{"差值(全解冻-渐进)":>18}')
    for nm, (sa, ga) in [('最少50类', (a50, b50)), ('最少100类', (a100, b100)),
                         ('全部254类', (full, prog))]:
        print(f'{nm:<16}{sa:>11.2f}%{ga:>13.2f}%{sa - ga:>+17.2f}pp')
    print('-' * 70)
    print('每类宏平均（对照，单类 val 仅~10张噪声大）:')
    for nm, m in [('最少50类', macro50), ('最少100类', macro100)]:
        if m['S'] and m['G']:
            ms = sum(m['S']) / len(m['S']) * 100
            mg = sum(m['G']) / len(m['G']) * 100
            print(f'  {nm:<12} S={ms:.2f}%  G={mg:.2f}%  Δ={ms - mg:+.2f}pp')
    # 汇总每切片 val 样本数
    for nm, names in [('最少50类', tails['tail50']), ('最少100类', tails['tail100'])]:
        vn = sum(class_table[c]['val_n'] for c in names)
        tn = sum(class_table[c]['train_n'] for c in names)
        print(f'  {nm}: train={tn}, val={vn}')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'count'
    if stage == 'count':
        stage_count()
    elif stage == 'smoke':
        stage_smoke()
    elif stage == 'infer':
        stage_infer()
    elif stage == 'plot':
        stage_plot()
    else:
        raise SystemExit(f'未知阶段: {stage}（可选 count/infer/plot）')
