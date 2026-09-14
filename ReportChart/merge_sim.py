# -*- coding: utf-8 -*-
"""merge_sim.py —— 标签体系口径评估（离线合并模拟）

【目的】不改一张图、不训练，只用已算好的 254×254 混淆矩阵做纯算术重算，量化"标签口径"对成绩的
影响，并给候选合并排序。方案与全部限定条件见 `毕业设计\\离线合并模拟方案_v4.md` 与 `…_v5.md`。

【口径定义（务必按此解读，勿混用）】
  · 错误集中度  = 有向混淆格 n(A→B)               —— 论文 13.40/17.82/26.21%
  · 准确率(C-tol)= 逐格赦免：只把 top-K 无向混淆对两侧的混淆格计入正确，不做传递闭包
  · 准确率(C-part)= 等价类划分：按混淆对连边取连通分量后合并行列重算（做传递闭包）
  · 并列规则     = (-w, 类索引升序)；C-tol 列对并列稳健，C-part 列与类数列依赖该规则
  · 三口径不得互相替代：`top_confusions_*.csv` 仅有向 200 行（去重后 165 无向对），K>165 不足以
    表达"top-K 无向对"，故本脚本一律从 CM 全枚举取对。

【硬约束】
  1. 只读 train/、val/、src/；本脚本不写入任何数据集或旧代码目录；
  2. 核心计算只用 numpy + 标准库；`selfcheck` 需要 fid↔类名时 try-import `src_v2`
     （会连带 import torch，并在 src_v2/runs 上执行一次 makedirs —— 属既有行为），
     import 失败则跳过 #4/#5/#6 并在输出首行标 DEGRADED，**不得**静默视为通过；
  3. 产物一律写 ReportChart/ 顶层（子目录不在 .gitignore 覆盖范围，会被静默入库）；
  4. 不自行调用 `calculate_mean_and_std(retrieve_file=True)`（若 src/calculate.config 缺失，它会
     回退全量扫 train/ 并**写回 src/**，违反只读纪律）；如将来需要，必须先做三步护栏。

用法：
  python merge_sim.py selfcheck                 # 九条自证（退出码非 0 即失败）
  python merge_sim.py report   --spec merge_specs/v2_candidate_A.json
  python merge_sim.py ranking  --top 30
  python merge_sim.py direction --top 20
  python merge_sim.py all      --spec merge_specs/v2_candidate_A.json
"""
import os
import re
import sys
import csv
import json
import math
import hashlib
import zipfile
import pickle
import argparse
import collections
import itertools
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # src_v2/ReportChart
SRCV2 = os.path.dirname(HERE)                              # src_v2
ROOT = os.path.dirname(SRCV2)                              # 项目根
for _p in (SRCV2, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODEL = 'S-NoProG'
NPZ_DEFAULT = os.path.join(HERE, 'confusion_S-NoProG.npz')
SPEC_DIR = os.path.join(HERE, 'merge_specs')
VAL_ROOT = os.path.join(ROOT, 'val')
RUNS_DIR = os.path.join(SRCV2, 'runs')
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif')   # 与 SafeImageFolder._scan_disk 一致
TIE_RULE = '(-w, class index asc)'
# 期望值（来自论文表 tab:ambiguity-bound，用于自证 #9）
EXPECT_TOL = {5: 84.6059, 10: 85.4208, 20: 86.4617, 50: 88.1874}
EXPECT_PART = {5: 84.6059, 10: 85.5372, 20: 86.5713, 50: 88.9612}
EXPECT_COMPONENTS = {5: 249, 10: 244, 20: 236, 50: 211}
EXPECT_PARENT_ACC = 95.9734
# spec A（6.5.6 盲标"重叠型" + 语义闸门）
SPEC_A_PAIRS = [('私有类-可回收物-锅', '私有类-可回收物-电饭煲'),
                ('公共类-可回收物-鞋', '私有类-可回收物-拖鞋'),
                ('私有类-可回收物-插头电线', '私有类-可回收物-充电线')]


# ------------------------------------------------------------------ 基础工具
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sha256_json(obj):
    """冻结的口径：json.dumps(sort_keys=True, ensure_ascii=False, separators=(',',':')) 后 SHA256。"""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def load_cm(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    cm = z['cm'].astype(np.int64)
    names = [str(x) for x in z['names']]
    meta = {k: (z[k].item() if hasattr(z[k], 'item') else z[k])
            for k in ('model', 'ckpt', 'limit') if k in z}
    return cm, names, meta


def micro_acc(cm):
    return float(np.trace(cm)) / float(cm.sum())


def per_class_f1(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def macro_f1(cm):
    return float(per_class_f1(cm).mean())


def weighted_f1(cm):
    f1 = per_class_f1(cm)
    sup = cm.sum(axis=1).astype(float)
    return float((f1 * sup).sum() / sup.sum())


def ranked_pairs(cm):
    """无向类对按 (-w, 类索引升序) 排序 —— 并列规则写死在这里。"""
    n = len(cm)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            w = int(cm[i, j]) + int(cm[j, i])
            if w > 0:
                out.append((w, i, j))
    out.sort(key=lambda t: (-t[0], t[1], t[2]))
    return out


def components(n, edges):
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    grp = collections.defaultdict(list)
    for i in range(n):
        grp[find(i)].append(i)
    return list(grp.values())


def merge_cm(cm, groups):
    """按组聚合（向量化：np.add.at，避免 O(G²) 的 np.ix_ 循环——并列枚举会调用上千次）。"""
    n = len(cm)
    gid = np.empty(n, dtype=np.int64)
    for gi, g in enumerate(groups):
        gid[g] = gi
    out = np.zeros((len(groups), len(groups)), dtype=np.int64)
    np.add.at(out, (gid[:, None], gid[None, :]), cm)
    return out


def acc_tol(cm, pairs, K):
    """C-tol：逐格赦免（不做传递闭包）。"""
    N = cm.sum()
    return (float(np.trace(cm)) + float(sum(w for w, _, _ in pairs[:K]))) / float(N)


def groups_from_pairs(n, pairs, K):
    return components(n, [(i, j) for _, i, j in pairs[:K]])


def acc_part(cm, groups):
    return micro_acc(merge_cm(cm, groups))


def tie_envelope(cm, pairs, K):
    """并列包络：K 名处计数并列时，枚举全部取法下 C-part 的极值与分量数极值。"""
    if K <= 0 or K > len(pairs):
        return None
    w_cut = pairs[K - 1][0]
    above = [p for p in pairs if p[0] > w_cut]
    cand = [p for p in pairs if p[0] == w_cut]
    need = K - len(above)
    if need <= 0 or need >= len(cand):
        g = groups_from_pairs(len(cm), pairs, K)
        a = acc_part(cm, g)
        return dict(n_solutions=1, acc_min=a, acc_max=a, comp_min=len(g), comp_max=len(g),
                    classes_min=sum(1 for x in g if len(x) > 1), classes_max=sum(1 for x in g if len(x) > 1))
    combos = itertools.combinations(range(len(cand)), need)
    total = math.comb(len(cand), need)
    cap = 200                                  # 枚举上限：超过则确定性抽样，并如实标注 n_sampled
    accs, comps, cover = [], [], []
    for k, idx in enumerate(combos):
        if k >= cap:
            break
        sub = above + [cand[t] for t in idx]
        g = groups_from_pairs(len(cm), sub, len(sub))
        accs.append(acc_part(cm, g))
        comps.append(len(g))
        cover.append(sum(1 for x in g if len(x) > 1))
    return dict(n_solutions=total, n_sampled=len(accs),
                acc_min=min(accs), acc_max=max(accs), comp_min=min(comps), comp_max=max(comps),
                classes_min=min(cover), classes_max=max(cover))


# ------------------------------------------------------------------ 溯源：从 _last.pth 读类别顺序（不加载张量）
class _Stub(object):
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'collections':
            return getattr(collections, name, _Stub)
        if module == 'builtins':
            return getattr(__import__('builtins'), name, _Stub)
        return _Stub

    def persistent_load(self, pid):
        return None


def read_ckpt_meta(path):
    """从 checkpoint 的 data.pkl 读顶层标量元数据（zip+pickle + 桩，**不加载任何张量**）。

    返回 dict：含 val_acc / epoch / config_id / seed 等标量，以及 renew_class_to_index（若有）。
    best checkpoint 只存 epoch/config_id/seed/arch/state_dict/val_acc；断点额外存 renew_class_to_index。
    """
    with zipfile.ZipFile(path) as z:
        raw = z.read('archive/data.pkl')
    obj = _SafeUnpickler(__import__('io').BytesIO(raw)).load()
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[k] = v
            elif isinstance(k, str) and 'renew_class_to_index' in k and isinstance(v, dict):
                out['renew_class_to_index'] = v
    if 'renew_class_to_index' in obj:
        out['renew_class_to_index'] = obj['renew_class_to_index']
    return out


def read_renew_from_last_ckpt(path):
    """从 checkpoint 读 renew_class_to_index（兼容旧调用）。"""
    return read_ckpt_meta(path).get('renew_class_to_index')


# ------------------------------------------------------------------ src_v2 映射（try-import）
def load_mapping():
    """返回 (fid2cls, cls2fid, renew, merged_dict) 或 None（DEGRADED）。"""
    try:
        from src_v2 import merged_dict as md
        from calculate import generate_mappings
        id2main, renew = generate_mappings(merged_dict=md)
        fid2cls = {int(k): v for k, v in id2main.items()}
        return fid2cls, id2main, renew, md
    except Exception as e:
        sys.stderr.write(f'[merge_sim] DEGRADED：无法 import src_v2 映射（{e.__class__.__name__}: {e}）\n')
        return None


def disk_counts(fid2cls):
    """每个合并类的盘上图像数（非递归 + 扩展名白名单，与训练一致）。"""
    out = collections.Counter()
    for fid, cls in fid2cls.items():
        d = os.path.join(VAL_ROOT, str(fid))
        if not os.path.isdir(d):
            continue
        c = 0
        for nm in os.listdir(d):
            if nm.lower().endswith(VALID_EXTS):
                c += 1
        out[cls] += c
    return out


# ------------------------------------------------------------------ selfcheck
class Checker(object):
    def __init__(self, degraded=False):
        self.rows = []
        self.failed = 0
        self.degraded = degraded

    def check(self, name, cond, detail=''):
        ok = bool(cond)
        self.failed += (not ok)
        self.rows.append((name, ok, detail))
        flag = 'PASS' if ok else ('SKIP' if detail == 'SKIP' else 'FAIL')
        print(f'  [{flag}] {name}' + (f'　{detail}' if detail else ''))
        return ok


def stage_selfcheck(a):
    print('=' * 84)
    print('merge_sim.py selfcheck —— 九条自证（口径：见文件头）')
    cm, names, meta = load_cm(a.npz)
    n = len(names)
    N = int(cm.sum())
    mapping = load_mapping()
    ck = Checker(degraded=mapping is None)
    print(f'npz={os.path.basename(a.npz)}  n={n}  N={N}  trace={int(np.trace(cm))}  '
          f'model={meta.get("model")}  ckpt={meta.get("ckpt")}  limit={meta.get("limit")}')
    if mapping is None:
        print('  >>> DEGRADED：需要 fid↔类名的自证（#4/#5/#6）将被跳过，不得视为通过 <<<')
    fid2cls, id2main, renew, md = mapping if mapping else ({}, {}, {}, {})

    # #1 单位映射精确复现
    acc0, m0, w0 = micro_acc(cm), macro_f1(cm), weighted_f1(cm)
    ck.check('#1 单位映射 acc = 12066/14603 = 82.6269%', abs(acc0 * 100 - 82.6269) < 5e-3,
             f'实测 {acc0 * 100:.4f}%')
    ck.check('#1b 宏 F1 80.3544% / 加权 F1 82.6331%',
             abs(m0 * 100 - 80.3544) < 5e-3 and abs(w0 * 100 - 82.6331) < 5e-3,
             f'实测 {m0 * 100:.4f}% / {w0 * 100:.4f}%')

    # #2 守恒与组数
    ck.check('#2 Σ CM 守恒', int(merge_cm(cm, [[i] for i in range(n)]).sum()) == N, f'{N}')
    if md:
        n_merges = sum(len(g) - 1 for g in md.values())          # 7 组 / 18 成员 → 11 次归并
        n_single = len(id2main) - n_merges                       # 265 − 11 = 254
        ck.check('#2b 合并前 265 类 − Σ(|g|−1) == 254 个合并类',
                 (n_single == n and len(id2main) == 265 and n_merges == 11),
                 f'实测 265 − {n_merges} = {n_single}（组数 {len(md)}，成员 {sum(len(g) for g in md.values())}）')
    else:
        ck.check('#2b 合并前 265 类 − Σ(|g|−1) == 254 个合并类', False, 'SKIP')

    # #3 闭式校验（spec A）
    idx = {nm: i for i, nm in enumerate(names)}
    pairs_a = [(int(cm[idx[p[0]], idx[p[1]]]) + int(cm[idx[p[1]], idx[p[0]]]), idx[p[0]], idx[p[1]])
               for p in SPEC_A_PAIRS]
    in_err = sum(int(cm[i, j]) for _, i, j in pairs_a) + sum(int(cm[j, i]) for _, i, j in pairs_a)
    # 注意：pairs_a 的 w 是双向合计，组内非对角格 = 两向之和
    closed = in_err / N
    gA = components(n, [(i, j) for _, i, j in pairs_a])
    got = micro_acc(merge_cm(cm, gA)) - acc0
    ck.check('#3 闭式 Δacc == 实现值（spec A）', abs(closed - got) < 1e-12,
             f'闭式 {closed * 100:.4f}pp / 实现 {got * 100:.4f}pp（{in_err}/{N}）')

    # #4 names ↔ id2main
    if mapping:
        want = sorted(set(id2main.values()))
        ck.check('#4 names == sorted(set(id2main.values()))', names == want, f'n={len(want)}')
        ck.check('#4b 265 fid → 254 名满射（Σ|π⁻¹| = 265）',
                 len(id2main) == 265 and len(set(id2main.values())) == n,
                 f'fid 数 {len(id2main)}，去重类名 {len(set(id2main.values()))}，类数 {n}')

    # #5 每类 (support, tp) 比对盘上计数
    if mapping:
        dc = disk_counts(fid2cls)
        sup = collections.Counter()
        tp = collections.Counter()
        for i, nm in enumerate(names):
            sup[nm] = int(cm[i].sum())
            tp[nm] = int(cm[i, i])
        bad = [(nm, sup[nm], dc.get(nm, 0)) for nm in names if sup[nm] != dc.get(nm, 0)]
        ck.check('#5 每类 support == 盘上计数', not bad, f'不符 {len(bad)} 类；合计 {sum(sup.values())}')
        key_sup = collections.Counter(sup.values())
        blind_sup = sum(1 for nm in names if key_sup[sup[nm]] > 1)
        key_st = collections.Counter((sup[nm], tp[nm]) for nm in names)
        blind_st = sum(1 for nm in names if key_st[(sup[nm], tp[nm])] > 1)
        ck.check('#5b 盲区统计（仅 support / 加 tp）', True,
                 f'{blind_sup}/{n} = {blind_sup / n * 100:.1f}% → {blind_st}/{n} = {blind_st / n * 100:.1f}%')

    # #6 Σ CM == 盘上 val 有效图像数
    if mapping:
        tot_disk = sum(disk_counts(fid2cls).values())
        ck.check('#6 Σ CM == 盘上 val 有效图像数', tot_disk == N, f'盘上 {tot_disk} / CM {N}')

    # #7 主对齐检查：与 ckpt val_acc 差 ≤1 张（用 zip+pickle 读元数据，不加载张量）
    val_acc = None
    best = os.path.join(RUNS_DIR, str(meta.get('ckpt') or 'checkpoint_S-NoProg_seed1_best.pth'))
    if os.path.exists(best):
        try:
            val_acc = read_ckpt_meta(best).get('val_acc')
        except Exception as e:
            sys.stderr.write(f'[merge_sim] #7 读取 ckpt 元数据失败：{e}\n')
    if isinstance(val_acc, (int, float)):
        va = float(val_acc)
        frac = va / 100.0 if va > 1.5 else va          # ckpt 里存的是"百分数"（如 82.6337），也兼容 0-1 分数
        cnt = int(round(frac * N))
        delta = abs(cnt - int(np.trace(cm)))
        ck.check('#7 复算 acc 与 ckpt val_acc 差 ≤1 张', delta <= 1,
                 f'ckpt {frac * 100:.4f}%（{cnt} 张）vs CM {int(np.trace(cm))} 张，差 {delta}')
    else:
        ck.check('#7 复算 acc 与 ckpt val_acc 差 ≤1 张', False, 'SKIP（未读到 val_acc）')

    # #8 类别顺序闭环（_last.pth 的 renew_class_to_index）
    last = os.path.join(RUNS_DIR, 'checkpoint_S-NoProg_seed1_last.pth')
    if os.path.exists(last):
        rn = read_renew_from_last_ckpt(last)
        ok = isinstance(rn, dict) and len(rn) == n and all(
            str(rn.get(nm, '')) == str(names.index(nm)) for nm in names)
        ck.check('#8 类别顺序闭环（_last.pth renew == npz names 顺序）', ok,
                 f'renew 条目 {len(rn) if isinstance(rn, dict) else "None"}')
    else:
        ck.check('#8 类别顺序闭环', False, f'SKIP（无 {os.path.basename(last)}）')

    # #9 两口径全表复现（含并列包络）
    pairs = ranked_pairs(cm)
    ok9 = True
    detail = []
    for K in (5, 10, 20, 50):
        t = acc_tol(cm, pairs, K) * 100
        g = groups_from_pairs(n, pairs, K)
        p = acc_part(cm, g) * 100
        env = tie_envelope(cm, pairs, K)
        ok9 &= abs(t - EXPECT_TOL[K]) < 5e-3 and abs(p - EXPECT_PART[K]) < 5e-3 \
            and len(g) == EXPECT_COMPONENTS[K]
        detail.append(f'K={K}: tol {t:.2f} / part {p:.2f} / 组 {len(g)}'
                      + (f' [part 包络 {env["acc_min"] * 100:.2f}~{env["acc_max"] * 100:.2f}%, '
                         f'组 {env["comp_min"]}~{env["comp_max"]}]' if env and env['n_solutions'] > 1 else ''))
    ck.check('#9 两口径全表与论文表一致 + 并列包络', ok9, '；'.join(detail))
    # 4 父类
    par = collections.defaultdict(list)
    for i, nm in enumerate(names):
        parts = nm.split('-')
        par[parts[1] if len(parts) > 2 else '?'].append(i)
    g4 = list(par.values())
    acc4 = acc_part(cm, g4) * 100
    ck.check('#9b 仅 4 父类口径 95.97%', abs(acc4 - EXPECT_PARENT_ACC) < 5e-3,
             f'实测 {acc4:.4f}%（宏 F1 {macro_f1(merge_cm(cm, g4)) * 100:.2f}%）')

    print('-' * 84)
    print(f'结果：通过 {len(ck.rows) - ck.failed} / {len(ck.rows)}'
          + ('（DEGRADED：有跳过项）' if ck.degraded else ''))
    return 0 if ck.failed == 0 else 1


# ------------------------------------------------------------------ spec / 报告
def load_spec(path):
    with open(path, encoding='utf-8') as f:
        spec = json.load(f)
    return spec


def spec_groups(names, spec):
    idx = {nm: i for i, nm in enumerate(names)}
    groups = []
    for a, b in spec['pairs']:
        if a not in idx or b not in idx:
            raise SystemExit(f'[merge_sim] spec 里的类名不在 npz names 中：{a} / {b}')
        groups.append([idx[a], idx[b]])
    for i in range(len(names)):
        if not any(i in g for g in groups):
            groups.append([i])
    return groups


def stage_report(a):
    cm, names, meta = load_cm(a.npz)
    N = int(cm.sum())
    spec = load_spec(a.spec)
    groups = spec_groups(names, spec)
    cm2 = merge_cm(cm, groups)
    in_err = int(np.trace(cm2)) - int(np.trace(cm))
    delta = in_err / N
    print(f'[spec {spec.get("name")}] 组数 {len(names)} → {len(groups)}；'
          f'赦免 {in_err} 张（{delta * 100:+.4f}pp）；acc {micro_acc(cm) * 100:.4f}% → {micro_acc(cm2) * 100:.4f}%')
    print(f'  宏 F1 {macro_f1(cm) * 100:.4f}% → {macro_f1(cm2) * 100:.4f}%；'
          f'加权 F1 {weighted_f1(cm) * 100:.4f}% → {weighted_f1(cm2) * 100:.4f}%')
    # 区间
    if a.bootstrap:
        f1 = per_class_f1(cm)
        involved = [i for g in groups if len(g) > 1 for i in g]
        cells = [(i, j) for g in groups if len(g) > 1 for i in g for j in g if i != j]
        p = cm / N
        pc = float(sum(p[i, j] for i, j in cells))
        sd = math.sqrt(N * pc * (1 - pc)) / N
        rng = np.random.default_rng(a.seed)
        boots = []
        flat_cm = cm.ravel()
        for _ in range(a.bootstrap):
            m = rng.multinomial(N, p.ravel()).reshape(cm.shape)
            boots.append(sum(m[i, j] for i, j in cells) / N)
        boots = np.array(boots)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f'  图像级区间：解析 sd {sd * 100:.4f}pp ⇒ 95%CI [{max(0.0, (delta - 1.96 * sd)) * 100:.3f}, '
              f'{(delta + 1.96 * sd) * 100:.3f}]pp（主报，论文按两位小数写 [0.84, 1.16]）；'
              f'bootstrap({a.bootstrap}, seed={a.seed}) sd {boots.std(ddof=1) * 100:.4f}pp '
              f'95%CI [{lo * 100:.3f}, {hi * 100:.3f}]pp')
        cl = []
        for _ in range(a.bootstrap):
            sel = rng.choice(involved, size=len(involved), replace=True)
            s = 0
            for i, j in cells:
                s += min(int(np.sum(sel == i)), int(np.sum(sel == j))) * cm[i, j]
            cl.append(s / N)
        cl = np.array(cl)
        print(f'  类整群（仅 {len(involved)} 个涉及类，min(重数) 计权、分母固定 N）：sd {cl.std(ddof=1) * 100:.4f}pp、'
              f'零值比例 {np.mean(cl == 0) * 100:.1f}%、均值 {cl.mean() * 100:.4f}pp '
              f'⇒ sd 比 {cl.std(ddof=1) / sd:.2f}×（方差比 {(cl.std(ddof=1) / sd) ** 2:.1f}×）')
    # 写产物
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(HERE, f'merge_sim_{a.model}_{spec.get("name")}.csv')
    with open(out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow([f'# generated_at={stamp}', f'spec={os.path.basename(a.spec)}',
                    f'npz_sha256={sha256_file(a.npz)}'])
        w.writerow(['kind', 'name', 'support', 'recall', 'precision', 'f1', 'members', 'note'])
        w.writerow(['overall', 'micro_acc', f'{micro_acc(cm2) * 100:.4f}', '', '', '',
                    f'{len(groups)} groups', f'base {micro_acc(cm) * 100:.4f}%, delta {delta * 100:+.4f}pp'])
        w.writerow(['overall', 'macro_f1', f'{macro_f1(cm2) * 100:.4f}', '', '', '', '', ''])
        w.writerow(['overall', 'weighted_f1', f'{weighted_f1(cm2) * 100:.4f}', '', '', '', '', ''])
        f1s = per_class_f1(cm2)
        for gi, (g, f1v) in enumerate(zip(groups, f1s)):
            sup = int(cm2[gi].sum())
            rec = float(cm2[gi, gi]) / sup if sup else 0.0
            prec = float(cm2[gi, gi]) / float(cm2[:, gi].sum()) if cm2[:, gi].sum() else 0.0
            nm = '+'.join(names[i].split('-')[-1] for i in g[:4]) + ('+…' if len(g) > 4 else '')
            w.writerow(['group', nm, sup, f'{rec * 100:.2f}', f'{prec * 100:.2f}', f'{f1v * 100:.2f}',
                        len(g), 'merged' if len(g) > 1 else ''])
    print(f'  已写 {os.path.basename(out)}')
    return 0


def stage_stats(a):
    """M1b：把论文 6.5.5 用到的全部统计量纳入入库脚本复算（此前部分数字无脚本可算）。"""
    cm, names, meta = load_cm(a.npz)
    n = len(names)
    N = int(cm.sum())
    correct = int(np.trace(cm))
    err = N - correct
    off = cm.copy()
    np.fill_diagonal(off, 0)
    rows = []

    def add(k, v, note=''):
        rows.append((k, v, note))
        print(f'  {k:34s} {v}   {note}')

    add('N / correct / errors', f'{N} / {correct} / {err}')
    add('acc / macroF1 / weightedF1',
        f'{micro_acc(cm) * 100:.4f}% / {macro_f1(cm) * 100:.4f}% / {weighted_f1(cm) * 100:.4f}%')
    pr = per_class_f1(cm)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(axis=0), 1)
    rec = tp / np.maximum(cm.sum(axis=1), 1)
    add('macro P / R', f'{prec.mean() * 100:.4f}% / {rec.mean() * 100:.4f}%')

    # 有向混淆格占比
    directed = sorted((int(off[i, j]) for i in range(n) for j in range(n) if i != j and off[i, j] > 0),
                      reverse=True)
    for K in (10, 20, 50):
        add(f'有向混淆格 top{K} 占错误', f'{sum(directed[:K])}/{err} = {sum(directed[:K]) / err * 100:.2f}%')

    # 无向类对 / 双向
    pairs = ranked_pairs(cm)
    bi = [(w, i, j) for (w, i, j) in pairs if cm[i, j] > 0 and cm[j, i] > 0]
    err_all = sum(w for w, _, _ in bi)
    err_2min = sum(2 * int(min(cm[i, j], cm[j, i])) for _, i, j in bi)
    add('有误判的无向类对', f'{len(pairs)}（占全部类对 {n * (n - 1) // 2}）')
    add('双向误判类对', f'{len(bi)}（占全部类对 {len(bi) / (n * (n - 1) // 2) * 100:.2f}%；'
                    f'占有误判类对 {len(bi) / len(pairs) * 100:.2f}%）')
    add('双向类对所涉错误（两向全计）', f'{err_all}（占错误 {err_all / err * 100:.2f}%）')
    add('双向类所涉错误（2×min 保守）', f'{err_2min}（占错误 {err_2min / err * 100:.2f}%）')

    # 父类口径（块对角 / 行总 = 按样本加权的父类召回；论文 84.96/80.07/82.18/91.96）
    parent = []
    for nm in names:
        p = nm.split('-')
        parent.append(p[1] if len(p) > 2 else '?')
    pg = collections.defaultdict(list)
    for i, pp in enumerate(parent):
        pg[pp].append(i)
    cross = sum(int(off[i, j]) for i in range(n) for j in range(n)
                if i != j and parent[i] != parent[j])
    add('同父类内 / 跨父类错误', f'{(err - cross) / err * 100:.1f}% / {cross / err * 100:.1f}%')
    expect_parent = {'有害垃圾': 91.96, '其他垃圾': 84.96, '可回收物': 82.18, '厨余垃圾': 80.07}
    for p, idx in sorted(pg.items()):
        rowtot = int(cm[idx].sum())                     # 该父类的真实样本数（分母！不等于块和）
        correct_in = int(np.diag(cm)[idx].sum())        # 细类层面判对的数量
        acc = correct_in / rowtot * 100
        blk = int(cm[np.ix_(idx, idx)].sum())           # 真值与预测同属该父类的样本数（另一口径）
        note = ''
        if p in expect_parent:
            note = 'ok' if abs(acc - expect_parent[p]) < 0.05 else f'⚠ 与论文 {expect_parent[p]}% 不符'
        add(f'父类准确率 {p}', f'{acc:.2f}%（n={rowtot}）',
            f'{note}；条件口径（真值预测同父类）{correct_in / blk * 100:.2f}%（n={blk}）')
    g4 = list(pg.values())
    add('仅 4 父类口径 acc / macroF1',
        f'{acc_part(cm, g4) * 100:.4f}% / {macro_f1(merge_cm(cm, g4)) * 100:.2f}%')

    # 7 组合并类覆盖的错误（并集口径：真值或预测任一侧属于这 7 个合并类）
    mapping = load_mapping()
    if mapping:
        fid2cls, id2main, renew, md = mapping
        fids = {f for grp in md.values() for f, _ in grp}
        mnames = {id2main[f] for f in fids}
        M = {i for i, nm in enumerate(names) if nm in mnames}
        A = sum(int(off[i, j]) for i in M for j in range(n))      # 真值 ∈ M
        B = sum(int(off[i, j]) for j in M for i in range(n))      # 预测 ∈ M
        C = sum(int(off[i, j]) for i in M for j in M)             # 两者都在 M
        union = A + B - C
        note = 'ok' if union == 442 else '⚠ 与论文 442 不符'
        add('涉及 7 组合并类的错误（并集）',
            f'{union}（占错误 {union / err * 100:.2f}%）；覆盖 {len(M)} 类；'
            f'真值侧 {A}、预测侧 {B}、内部 {C}（论文 442 / 17.4%）', note)
    else:
        add('涉及 7 组合并类的错误（并集）', 'SKIP（DEGRADED）')

    # 两口径全表 + 并列包络
    for K in (5, 10, 20, 50, 100):
        t = acc_tol(cm, pairs, K) * 100
        g = groups_from_pairs(n, pairs, K)
        p = acc_part(cm, g) * 100
        env = tie_envelope(cm, pairs, K)
        extra = ''
        if env and env['n_solutions'] > 1:
            extra = (f'｜并列解 {env["n_solutions"]}'
                     + (f'（抽样 {env["n_sampled"]}）' if env['n_sampled'] < env['n_solutions'] else '')
                     + f'：C-part {env["acc_min"] * 100:.2f}~{env["acc_max"] * 100:.2f}%、组 {env["comp_min"]}~{env["comp_max"]}')
        add(f'top{K} C-tol / C-part / 组数',
            f'{t:.2f}% / {p:.2f}% / {len(g)}（宏 F1 {macro_f1(merge_cm(cm, g)) * 100:.2f}%）', extra)

    out = os.path.join(HERE, f'merge_sim_{a.model}_stats.csv')
    with open(out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow([f'# generated_at={datetime.now().strftime("%Y%m%d_%H%M%S")}',
                    f'npz_sha256={sha256_file(a.npz)}', f'tie_rule={TIE_RULE}'])
        w.writerow(['item', 'value', 'note'])
        w.writerows(rows)
    print(f'  已写 {os.path.basename(out)}')
    return 0


def stage_ranking(a):
    cm, names, meta = load_cm(a.npz)
    N = int(cm.sum())
    base_m = macro_f1(cm)
    pairs = ranked_pairs(cm)
    print(f'[ranking] 候选 {len(pairs)} 个无向类对')
    print('  · Δacc 轨道：闭式 Δ = Σ组内非对角格/N，对互斥类对严格可加 ⇒ 按 w 降序即精确最优序列')
    print('  · ΔmacroF1 轨道：逐步重算的**贪心**启发式（不保证全局最优；宏 F1 变化含"FP 缓解"构成效应）')
    out = os.path.join(HERE, f'merge_sim_{a.model}_ranking.csv')
    with open(out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow([f'# tie_rule={TIE_RULE}', f'npz_sha256={sha256_file(a.npz)}'])
        w.writerow(['step', 'pair_a', 'pair_b', 'w', 'marginal_delta_acc_pp', 'cum_delta_acc_pp',
                    'cum_delta_macroF1_pp', 'greedy_marginal_macroF1_pp', 'closed_form_ok'])
        used_pairs = []
        cum_acc = 0.0
        cum_m = base_m
        for step, (wt, i, j) in enumerate(pairs[:a.top], start=1):
            marg_acc = wt / N * 100.0
            cum_acc += marg_acc
            used_pairs.append((wt, i, j))
            g = components(len(names), [(x, y) for _, x, y in used_pairs])
            m = macro_f1(merge_cm(cm, g))
            # 闭式校验：实现值增量 == 本步 w/N
            closed_ok = abs((cum_acc - sum(x[0] for x in used_pairs[:-1]) / N * 100) - marg_acc) < 1e-9
            w.writerow([step, names[i], names[j], wt, f'{marg_acc:.4f}', f'{cum_acc:.4f}',
                        f'{(m - base_m) * 100:.4f}', f'{(m - cum_m) * 100:.4f}',
                        'ok' if closed_ok else 'MISMATCH'])
            cum_m = m
    print(f'  已写 {os.path.basename(out)}（前 {a.top} 步）')
    return 0


def fisher_exact_two_sided(a_, b_, c_, d_):
    """2×2 表 [[a,b],[c,d]] 的双侧 Fisher 精确检验（超几何求和）。"""
    n = a_ + b_ + c_ + d_
    r1, c1 = a_ + b_, a_ + c_
    def prob(x):
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    p_obs = prob(a_)
    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs + 1e-12))


def stage_direction(a):
    cm, names, meta = load_cm(a.npz)
    pairs = ranked_pairs(cm)
    out = os.path.join(HERE, f'merge_sim_direction_{a.model}.csv')
    with open(out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pair_a', 'pair_b', 'ab', 'ba', 'support_a', 'support_b',
                    'p_ab', 'p_ba', 'rate_ratio', 'rr_lo', 'rr_hi', 'method', 'note'])
        for (wt, i, j) in pairs[:a.top]:
            ab, ba = int(cm[i, j]), int(cm[j, i])
            sa, sb = int(cm[i].sum()), int(cm[j].sum())
            p_ab, p_ba = ab / sa, ba / sb
            if ab == 0 or ba == 0:
                pv = fisher_exact_two_sided(ab, sa - ab, ba, sb - ba)
                w.writerow([names[i], names[j], ab, ba, sa, sb, f'{p_ab:.4f}', f'{p_ba:.4f}',
                            'inf' if ba == 0 and ab > 0 else ('0' if ab == 0 else 'n/a'),
                            '', '', 'fisher', f'one-way, p={pv:.3g}'])
                continue
            rr = p_ab / p_ba
            se = math.sqrt(1 / ab - 1 / sa + 1 / ba - 1 / sb)
            lo, hi = math.exp(math.log(rr) - 1.96 * se), math.exp(math.log(rr) + 1.96 * se)
            note = 'symmetric-ish' if 0.5 <= rr <= 2 else ('A→B dominant' if rr > 1 else 'B→A dominant')
            w.writerow([names[i], names[j], ab, ba, sa, sb, f'{p_ab:.4f}', f'{p_ba:.4f}',
                        f'{rr:.3f}', f'{lo:.3f}', f'{hi:.3f}', 'katz', note])
    print(f'  已写 {os.path.basename(out)}（前 {a.top} 对；率比 + Katz 区间，含 0 用 Fisher）')
    print('  提醒：方向不对称度**仅为描述性线索**，与人工盲标冲突时以盲标为准。')
    return 0


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description='标签体系口径评估（离线合并模拟）')
    ap.add_argument('stage', choices=['selfcheck', 'stats', 'report', 'ranking', 'direction', 'all'])
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--npz', default=None)
    ap.add_argument('--spec', default=None)
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.npz is None:
        a.npz = NPZ_DEFAULT if a.model == MODEL else os.path.join(HERE, f'confusion_{a.model}.npz')
    if a.spec is None:
        a.spec = os.path.join(SPEC_DIR, 'v2_candidate_A.json')
    if not os.path.exists(a.npz):
        raise SystemExit(f'[merge_sim] 找不到混淆矩阵：{a.npz}')
    if a.stage == 'selfcheck':
        return stage_selfcheck(a)
    if a.stage in ('stats', 'all'):
        stage_stats(a)
    if a.stage in ('report', 'all'):
        if not os.path.exists(a.spec):
            raise SystemExit(f'[merge_sim] 找不到 spec：{a.spec}')
        rc = stage_report(a)
    if a.stage in ('ranking', 'all'):
        rc = stage_ranking(a)
    if a.stage in ('direction', 'all'):
        rc = stage_direction(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
