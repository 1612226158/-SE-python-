# -*- coding: utf-8 -*-
"""
chart_data.py —— ReportChart 统一实验数据读取模块（2026-09-08 会话8 建立）

【目的】消除各绘图脚本里"手写归档 csv 文件名 + 逐档合并"的重复逻辑。
  之前画图前必须人工查"某模型的数据散落在哪几个 runs/metrics_record*.csv 里"，
  新实验跑完/归档后还得改脚本。本模块自动扫描 runs/ 下全部 metrics 文件
  （当前 metrics_record.csv + 所有 metrics_record_archive_*.csv），
  按 CONFIG_ID（模型英文名）过滤合并 → 画图脚本只需：
      from chart_data import load_model, model_best, list_models
      df = load_model('S-NoProG')     # 自动取该模型最完整的 seed，含全部轮次
      best = model_best('S-NoProG')   # -> (best_val, best_epoch)

【通用性】任何新实验跑完（runs/results_*.json + metrics 归档后）无需改本模块；
  若模型在跑（metrics_record.csv 实时追加），load_model 也能拿到"截至当前"的数据。

【口径说明】
  - metrics_record.csv 每行含: config_id,seed,epoch,train_acc,train_loss,val_acc,val_loss,
    val_macro_f1,val_weighted_f1,lr,gradient_norm,time_spend_s,state,save_time
  - Long 配置（G-Full-CAWR-Long=160 轮）续跑产生的 100~159 与源配置 0~99 分处不同 csv，
    本模块按 (config_id, seed) 合并后去重排序，天然支持。
  - G-Full 历史 seed0/1 并存：load_model 默认取"行数最完整"的 seed（可显式传 seed 覆盖）。
"""
import os
import sys
import glob
import pandas as pd

# ============ 路径（基于本文件定位，与 cwd 无关） ============
_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(_HERE), 'runs')

# ---------- 模型名 → 元数据（中文全称/短说明）唯一来源 = src_v2\model_registry.py ----------
# 新增/改模型名一律去 model_registry.py；本模块按 id 索引，注册表缺失时优雅降级为原名。
_SRC_V2 = os.path.dirname(_HERE)
if _SRC_V2 not in sys.path:
    sys.path.insert(0, _SRC_V2)
try:
    import model_registry as _mr
except Exception:          # 注册表被隔离/未生成时降级：不查元数据，仅按原名使用
    _mr = None


def _cid(name):
    """显示写法 → 规范 key（如 S-NoProG → S-NoProg）；注册表不可用时原样返回。"""
    return _mr.normalize(name) if _mr is not None else name


def model_cn(name):
    """模型中文全称（GUI 说明行 / 汇报 / 图注用）。"""
    return _mr.cn_of(name) if _mr is not None else name


def model_short(name):
    """模型短说明。"""
    return _mr.short_of(name) if _mr is not None else name


def is_known(name):
    """该模型 id 是否已在 model_registry.py 注册。"""
    return _mr.is_known(name) if _mr is not None else True

# ---------- 输出文件名推导（2026-09-08 会话8 加：消灭 out*.pdf 硬编码） ----------

# 论文已引用的历史对比标签（document_new.tex \includegraphics 依赖这些文件名，不能变）
# 键 = (模型A, 模型B)（顺序无关，内部排序后匹配）；值 = 论文文件名里的对比段
PAIR_TAGS = {
    ('G-Full-CAWR', 'S-NoProg'): 'progressive_vs_full',
    ('S-NoProg', 'S-NoSE'):      'snoprog_vs_snose',
    ('G-Full-CAWR-Long', 'S-NoProg'): 'long_vs_full',
}


def pair_tag(model_a, model_b):
    """由模型对推导文件名对比段：优先命中论文历史标签，否则自动拼 modelA_vs_modelB。
    输入接受显示写法（自动归一化到规范 key）。"""
    key = tuple(sorted([_cid(model_a), _cid(model_b)]))
    if key in PAIR_TAGS:
        return PAIR_TAGS[key]
    return f'{model_a}_vs_{model_b}'


def fig_out(fig_type, model_a, model_b, out_dir=None):
    """生成绘图输出路径：<out_dir>/<fig_type>_<对比段>.pdf
    fig_type ∈ {loss_curve, val_acc, time_acc, ...}（由调用脚本传图类名）；
    模型对只需在脚本顶部声明一次，文件名（含论文历史标签）自动匹配，无需手写。"""
    out_dir = out_dir or _HERE
    return os.path.join(out_dir, f'{fig_type}_{pair_tag(model_a, model_b)}.pdf')


# ---------- 基础扫描 ----------

def all_metrics_files():
    """返回全部 metrics 文件路径：当前 metrics_record.csv + 所有归档，按名称排序。"""
    cur = os.path.join(RUNS, 'metrics_record.csv')
    files = ([cur] if os.path.exists(cur) else [])
    files += sorted(glob.glob(os.path.join(RUNS, 'metrics_record_archive_*.csv')))
    return files


def _read_all(config_id=None):
    """扫全部 metrics 文件，过滤出 config_id（可选）。返回单 DataFrame（未排序、未去重）。"""
    frames = []
    for f in all_metrics_files():
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
        except Exception as e:
            print(f'[chart_data] 跳过无法读取的 {os.path.basename(f)}: {e}')
            continue
        if config_id is not None:
            df = df[df['config_id'] == config_id]
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------- 对外 API ----------

def available_models():
    """返回 {config_id: {seed: {'rows': n, 'max_epoch': e}}}：当前有哪些模型、各 seed 记录情况。"""
    df = _read_all()
    out = {}
    if df.empty:
        return out
    for (cid, seed), g in df.groupby(['config_id', 'seed']):
        out.setdefault(cid, {})[int(seed)] = {'rows': len(g), 'max_epoch': int(g['epoch'].max())}
    return out


def list_models(verbose=True):
    """打印当前全部可用模型（config_id + seed + 行数 + 最高轮次）。返回 available_models() dict。"""
    avail = available_models()
    if verbose:
        print(f'[chart_data] 共 {len(avail)} 个模型配置：')
        for cid in sorted(avail):
            for seed in sorted(avail[cid]):
                info = avail[cid][seed]
                print(f'  {cid:22s} seed{seed}  行数 {info["rows"]:>4}  最高轮 ep{info["max_epoch"]}')
    return avail


def load_model(config_id, seed=None, verbose=True):
    """取某模型（按 config_id / 模型英文名）的全部训练轮次记录。

    - seed=None：自动取该模型"行数最完整"的 seed（多 seed 并存如 G-Full 的 0/1 时避免混线）；
    - seed=指定：只取该 seed；
    - 返回按 epoch 升序、同 epoch 去重（保留最后一条）的 DataFrame；
    - 无数据 → 抛 FileNotFoundError（提示该模型还没跑 / config_id 拼错）。
    """
    config_id = _cid(config_id)   # 兼容 'S-NoProG' 等显示写法
    df = _read_all(config_id)
    if df.empty:
        raise FileNotFoundError(
            f'[chart_data] 未找到 config={config_id} 的任何记录。可用模型: '
            f'{", ".join(sorted(available_models())) or "(暂无)"}（检查 runs/ 下 metrics_record*.csv）')
    # seed 归一化
    if seed is not None:
        df = df[df['seed'].astype(int) == int(seed)]
        if df.empty:
            raise FileNotFoundError(f'[chart_data] {config_id} 没有 seed={seed} 的记录')
    else:
        # 自动选最完整 seed：行数最多；同数取 max_epoch 大者
        best_seed = None
        best_key = (-1, -1)
        for s, g in df.groupby('seed'):
            key = (len(g), int(g['epoch'].max()))
            if key > best_key:
                best_key, best_seed = key, s
        df = df[df['seed'] == best_seed]
        if verbose:
            print(f'[chart_data] {config_id}: 自动选用 seed={int(best_seed)} '
                  f'（共 {len(df)} 行，最高 ep{int(df["epoch"].max())}）')
    df = df.sort_values('epoch').reset_index(drop=True)
    df = df.drop_duplicates(subset='epoch', keep='last').reset_index(drop=True)
    # 数值列尽量转 float（seed 保留原样）
    for col in ('epoch', 'train_acc', 'train_loss', 'val_acc', 'val_loss',
                'val_macro_f1', 'val_weighted_f1', 'lr', 'time_spend_s'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def model_best(config_id, seed=None, verbose=False):
    """取某模型 best val acc 与对应 epoch：返回 (best_val, best_epoch)。
    基于 metrics 记录（与论文"最佳观测值"口径一致），不读 pth。"""
    df = load_model(config_id, seed=seed, verbose=verbose)
    if df.empty:
        return (None, None)
    ib = df['val_acc'].idxmax()
    return (float(df.loc[ib, 'val_acc']), int(df.loc[ib, 'epoch']))


def model_final(config_id, seed=None, verbose=False):
    """取某模型末轮 val acc 与 epoch：返回 (final_val, final_epoch)。"""
    df = load_model(config_id, seed=seed, verbose=verbose)
    if df.empty:
        return (None, None)
    last = df.iloc[-1]
    return (float(last['val_acc']), int(last['epoch']))


# ---------- 论文口径（results JSON / best checkpoint，与 metrics 略有出入） ----------

def results_best(config_id):
    """从 results_<config>_seed*.json 读该模型的论文口径最佳值（跑完才生成）。
    返回 (best_val_acc, best_val_epoch, seed) 取最高 seed；无 results → (None, None, None)。
    注：results 的 best 是训练结束时汇总的"正式成绩"（如 S-NoProg 82.63@74）；
    与 model_best()（metrics 口径，如 82.634）可能差 0.00x，引用论文数字请用本函数。"""
    config_id = _cid(config_id)
    best = None
    for p in sorted(glob.glob(os.path.join(RUNS, f'results_{config_id}_seed*.json'))):
        try:
            import json
            with open(p, encoding='utf-8') as f:
                r = json.load(f)
            seed = int(r.get('seed'))
            va, ep = r.get('best_val_acc'), r.get('best_val_epoch')
            if va is None:
                continue
            key = (float(va), int(ep), seed)
            if best is None or key[0] > best[0]:
                best = key
        except Exception:
            continue
    return best if best else (None, None, None)


def best_ckpt_path(config_id, verbose=False):
    """返回该模型应加载的 best checkpoint 绝对路径（与 results_best 的 seed 对齐；
    无 results 时退回 glob 到的最高 val_acc 那个 best.pth）。无 → None。
    论文图/推理脚本只需输入模型英文名即可拿到正确 checkpoint。"""
    config_id = _cid(config_id)
    _, _, seed = results_best(config_id)
    cands = sorted(glob.glob(os.path.join(RUNS, f'checkpoint_{config_id}_seed*_best.pth')))
    if not cands:
        return None
    if seed is not None:
        p = os.path.join(RUNS, f'checkpoint_{config_id}_seed{seed}_best.pth')
        if os.path.exists(p):
            return p
    # 退回：读各 best.pth 的 val_acc 取最高（需 torch；懒加载）
    import torch
    best_p, best_v = None, None
    for p in cands:
        try:
            ck = torch.load(p, map_location='cpu')
            v = ck.get('val_acc')
            if v is not None and (best_v is None or v > best_v):
                best_p, best_v = p, v
        except Exception:
            continue
    return best_p


if __name__ == '__main__':
    # 自检：打印可用模型 + 各模型 best（仅读 csv，无 matplotlib/torch 依赖）
    import sys
    print('[chart_data 自检] 可用模型清单：')
    avail = available_models()
    if not avail:
        print('  (runs/ 下暂无 metrics 记录)')
        sys.exit(1)
    for cid in sorted(avail):
        bv, be = model_best(cid, verbose=False)
        fv, fe = model_final(cid, verbose=False)
        print(f'  {cid:24s} best {bv:7.3f}%@ep{be:<4d} final {fv:7.3f}%@ep{fe}')
