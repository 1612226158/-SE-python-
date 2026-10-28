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

【命令行】本文件同时是只读结果查询入口，不生成或修改训练产物：
    python chart_data.py list
    python chart_data.py summary S-NoProg:1 S-NoSE:1
    python chart_data.py compare S-NoProg:1 S-NoSE:1
    python chart_data.py data S-NoProg:1
    python chart_data.py data S-NoProg:1 --json
    python chart_data.py compare S-NoProg:1 S-NoSE:1 --json

【口径说明】
  - metrics_record.csv 每行含: config_id,seed,epoch,sampler_seed,train_acc,train_loss,val_acc,val_loss,
    val_macro_f1,val_weighted_f1,lr,gradient_norm,time_spend_s,state,save_time
  - Long 配置（G-Full-CAWR-Long=160 轮）续跑产生的 100~159 与源配置 0~99 分处不同 csv，
    本模块按 (config_id, seed) 合并后去重排序，天然支持。
  - G-Full 历史 seed0/1 并存：load_model 默认取"行数最完整"的 seed（可显式传 seed 覆盖）。
"""
import argparse
import glob
import json
import math
import os
import sys
import pandas as pd

# ============ 路径（基于本文件定位，与 cwd 无关） ============
_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(_HERE), 'runs')

# ============ 合并类口径常量（2026-09-26 加：消灭各绘图/评估脚本的硬编码 254） ============
# **唯一真源 = src_v2/taxonomy.py**（merged_dict + RAW_CLASS_COUNT）。本模块**派生**而非复制：
#   口径 = 原始子类文件夹数 − Σ(|组|−1)。
#   2026-09-26 之前：7 组 18 子类 → 265 − 11 = 254
#   2026-09-26 起  ：并入「蔬菜+菜根菜叶→菜叶」「插头电线+充电线→插头充电线」
#                   （组名取「插头充电线」而非「充电器」，避免与既有类别「充电头」(id 68) 语义重叠）
#                   共 9 组 22 子类 → 265 − 13 = 252
# ⇒ 改标签体系请**只改 src_v2/taxonomy.py**，本模块的 N_CLS 会自动跟随。
#   交叉校验：eval_tail_class_acc.build_class_table() 有 assert n_classes == N_CLS；
#   merge_sim.py 自证 #2b 还会额外比对 chart_data.N_CLS ↔ merged_dict 实际类数 ↔ npz 类数。


def _derive_n_cls():
    """按**文件路径**加载 src_v2/taxonomy.py，返回 (原始子类数, 合并后类数)。

    刻意不用 `import src_v2.taxonomy` —— 那会执行 src_v2/__init__.py 从而拉起 torch，
    而本模块要求绘图脚本保持轻量（plot_*.py 不需要 torch）。taxonomy.py 本身只依赖
    hashlib/json/os，按路径加载是安全的。
    """
    import importlib.util as _ilu
    _taxonomy = os.path.join(os.path.dirname(_HERE), 'taxonomy.py')
    _spec = _ilu.spec_from_file_location('_taxonomy_for_chart', _taxonomy)
    if _spec is None or _spec.loader is None:
        raise ImportError(f'无法定位 {_taxonomy}')
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    raw = int(_mod.RAW_CLASS_COUNT)
    merged = raw - sum(len(g) - 1 for g in _mod.merged_dict.values())
    return raw, merged


try:
    N_CLS_RAW, N_CLS = _derive_n_cls()
except Exception as _e:      # taxonomy 缺失/改名时不静默：打印告警 + 回退字面常量
    sys.stderr.write(
        f'[chart_data] ⚠ 无法从 src_v2/taxonomy.py 派生合并类数（{_e.__class__.__name__}: {_e}），'
        f'回退字面常量 265/252 —— 若标签体系已改，此处会画出错误口径，请立即核对。\n')
    N_CLS_RAW = 265          # 数据集原始子类文件夹数（train/ 与 val/ 各 265 个数字文件夹）
    N_CLS = 252              # 合并后模型输出类数 = N_CLS_RAW − Σ(|组|−1)（9 组 22 子类 → 减 13）

# ============ 已知脏记录登记（2026-09-09 加，用户指出"只按文件新旧掩盖症状是严重bug"） ============
# 背景：调度器 bug（CAWR 误传 val_acc → state2/3 从不重启、lr 贴 1e-6 地板）期间的
#       G-Full-CAWR seed1 run（70.82 脏成绩）与修复后干净重跑**同名同 seed**并存于归档。
# 规则：以下 = {归档文件名: {(config_id, seed), ...}}，命中的行一律剔除（行级，不影响
#       同文件里其他配置的合法数据——该归档还存着 S-NoProg/S-NoSE 等正常 run 的记录）。
# 今后再出现"已作废重跑的 run"：要么归档时把 csv 放进 runs\DIRTY* 子目录（本模块不扫），
#       要么在此登记 (文件名, 配置, seed) —— 二者等价，都保证画图/统计永远不会吃到脏数据。
DIRTY_RECORDS = {
    # 调度器 bug 期 G-Full-CAWR seed1 脏 run（与 20260908 干净重跑同 tag；val 封顶 ~70.82）
    'metrics_record_archive_20260905_230315.csv': {('G-Full-CAWR', 1)},
}

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
    """扫全部 metrics 文件，过滤出 config_id（可选）+ 剔除 DIRTY_RECORDS 已知脏行。
    返回单 DataFrame（未排序、未去重）。附加列 _fidx = 来源文件序号（越新越大）——
    供 load_model 去重时"同一 epoch 确定性保留最新文件"（同 epoch 多份记录时的兜底规则；
    已知脏 run 已在上面按 (文件, 配置, seed) 行级剔除，不依赖"恰好最新"）。"""
    frames = []
    for fidx, f in enumerate(all_metrics_files()):
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
        except Exception as e:
            print(f'[chart_data] 跳过无法读取的 {os.path.basename(f)}: {e}')
            continue
        tag = os.path.basename(f)
        dirty = DIRTY_RECORDS.get(tag)
        if dirty:
            before = len(df)
            mask = df.apply(lambda r: (str(r['config_id']), int(r['seed'])) in dirty, axis=1)
            df = df[~mask]
            dropped = before - len(df)
            if dropped:
                print(f'[chart_data] 剔除脏记录 {dropped} 行（{tag}: 已知作废 run {sorted(dirty)}）')
        if config_id is not None:
            df = df[df['config_id'] == config_id]
        if len(df):
            df = df.copy()
            df['_fidx'] = fidx
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
    df = df.sort_values(['epoch', '_fidx'], kind='stable').reset_index(drop=True)
    df = df.drop_duplicates(subset='epoch', keep='last').reset_index(drop=True)   # 同 epoch 保留最新文件
    df = df.drop(columns=['_fidx'], errors='ignore')
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


# ---------- 统一状态汇总 / 命令行 ----------

def _number(row, key):
    """从 pandas 行中安全读取有限浮点数；缺列、空值或无穷值返回 None。"""
    if key not in row:
        return None
    try:
        value = float(row[key])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _metric_point(df, column, mode='max'):
    """返回某指标的极值点 {'value', 'epoch'}；整列无有效值时返回空值。"""
    if column not in df.columns:
        return {'value': None, 'epoch': None}
    values = pd.to_numeric(df[column], errors='coerce')
    valid = values.dropna()
    if valid.empty:
        return {'value': None, 'epoch': None}
    idx = valid.idxmin() if mode == 'min' else valid.idxmax()
    return {'value': float(values.loc[idx]), 'epoch': int(df.loc[idx, 'epoch'])}


def _result_for_run(config_id, seed):
    """读取指定 config/seed 的正式 results JSON；未收官或不可读时返回 None。"""
    path = os.path.join(RUNS, f'results_{_cid(config_id)}_seed{int(seed)}.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            result = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if result.get('config_id') != _cid(config_id) or int(result.get('seed', -1)) != int(seed):
        return None
    return result


def summarize_run(config_id, seed=None):
    """汇总一个实验的状态与关键指标，供命令行和其他脚本统一调用。

    数据仍由 load_model() 负责扫描、合并、去重和排除脏记录。本函数不直接打开
    metrics CSV，只在其规范化结果上计算完整性、best/final、F1、loss、泛化差距和耗时。
    正式 best 优先采用指定 seed 的 results JSON；未收官时退回当前 metrics 峰值。
    """
    df = load_model(config_id, seed=seed, verbose=False)
    config_id = _cid(config_id)
    selected_seed = int(df['seed'].iloc[0])
    result = _result_for_run(config_id, selected_seed)
    registered_model = bool(_mr is not None and _mr.is_known(config_id))
    target_epochs = int(_mr.target_of(config_id)) if registered_model else None

    epochs = sorted({int(e) for e in df['epoch'].dropna().tolist()})
    min_epoch = epochs[0]
    max_epoch = epochs[-1]
    missing_epochs = sorted(set(range(0, max_epoch + 1)) - set(epochs))

    metrics_best = _metric_point(df, 'val_acc', 'max')
    best_epoch = metrics_best['epoch']
    best_row = df[df['epoch'].astype(int) == best_epoch].iloc[-1]
    final_row = df.iloc[-1]

    official_best = {
        'value': float(result['best_val_acc']) if result and result.get('best_val_acc') is not None
                 else metrics_best['value'],
        'epoch': int(result['best_val_epoch']) if result and result.get('best_val_epoch') is not None
                 else metrics_best['epoch'],
        'source': 'results_json' if result else 'metrics',
    }
    final_train = _number(final_row, 'train_acc')
    final_val = _number(final_row, 'val_acc')
    best_train = _number(best_row, 'train_acc')
    best_val = _number(best_row, 'val_acc')
    time_values = pd.to_numeric(df.get('time_spend_s'), errors='coerce') if 'time_spend_s' in df else None

    def protocol_value(key, cast=None):
        value = result.get(key) if result else None
        if value is None and key in final_row:
            value = final_row[key]
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if value is None or cast is None:
            return value
        try:
            return cast(value)
        except (TypeError, ValueError):
            return None

    return {
        'config_id': config_id,
        'seed': selected_seed,
        'registered_model': registered_model,
        'status': 'finished' if result else 'incomplete',
        'results_json': bool(result),
        'epochs': {
            'rows': int(len(df)),
            'first': min_epoch,
            'last': max_epoch,
            'target': target_epochs,
            'progress_percent': (100.0 * len(epochs) / target_epochs) if target_epochs else None,
            'remaining': max(target_epochs - len(epochs), 0) if target_epochs else None,
            'missing': missing_epochs,
            'continuous_from_zero': not missing_epochs and min_epoch == 0,
        },
        'official_best_val_acc': official_best,
        'metrics_best_val_acc': metrics_best,
        'final': {
            'epoch': int(final_row['epoch']),
            'train_acc': final_train,
            'val_acc': final_val,
            'val_loss': _number(final_row, 'val_loss'),
            'val_macro_f1': _number(final_row, 'val_macro_f1'),
            'val_weighted_f1': _number(final_row, 'val_weighted_f1'),
            'generalization_gap_pp': (final_train - final_val)
                                     if final_train is not None and final_val is not None else None,
        },
        'at_metrics_best_val': {
            'epoch': best_epoch,
            'train_acc': best_train,
            'val_acc': best_val,
            'val_loss': _number(best_row, 'val_loss'),
            'val_macro_f1': _number(best_row, 'val_macro_f1'),
            'val_weighted_f1': _number(best_row, 'val_weighted_f1'),
            'generalization_gap_pp': (best_train - best_val)
                                     if best_train is not None and best_val is not None else None,
        },
        'best_macro_f1': _metric_point(df, 'val_macro_f1', 'max'),
        'best_weighted_f1': _metric_point(df, 'val_weighted_f1', 'max'),
        'minimum_val_loss': _metric_point(df, 'val_loss', 'min'),
        'cumulative_train_hours': float(time_values.sum() / 3600.0)
                                  if time_values is not None else None,
        'protocol': {
            'training_protocol_version': protocol_value('training_protocol_version', int),
            'sampling_protocol_version': protocol_value('sampling_protocol_version', int),
            'label_schema_id': protocol_value('label_schema_id', str),
            'label_schema_version': protocol_value('label_schema_version', int),
            'label_mapping_hash': protocol_value('label_mapping_hash', str),
            'num_classes': protocol_value('num_classes', int),
        },
    }


def compare_runs(left_config, left_seed, right_config, right_seed):
    """比较两个实验；差值统一定义为 right - left，避免口径歧义。"""
    left = summarize_run(left_config, left_seed)
    right = summarize_run(right_config, right_seed)

    def delta(path):
        lv, rv = left, right
        for key in path:
            lv = lv.get(key) if isinstance(lv, dict) else None
            rv = rv.get(key) if isinstance(rv, dict) else None
        if lv is None or rv is None:
            return None
        return float(rv) - float(lv)

    protocol_keys = ('training_protocol_version', 'sampling_protocol_version',
                     'label_schema_version', 'label_mapping_hash', 'num_classes')
    comparable_protocol = all(
        left['protocol'].get(key) is not None and
        left['protocol'].get(key) == right['protocol'].get(key)
        for key in protocol_keys
    )
    return {
        'left': left,
        'right': right,
        'delta_definition': 'right_minus_left',
        'protocol_match': comparable_protocol,
        'delta': {
            'official_best_val_acc_pp': delta(('official_best_val_acc', 'value')),
            'final_val_acc_pp': delta(('final', 'val_acc')),
            'best_macro_f1': delta(('best_macro_f1', 'value')),
            'best_weighted_f1': delta(('best_weighted_f1', 'value')),
            'minimum_val_loss': delta(('minimum_val_loss', 'value')),
            'final_generalization_gap_pp': delta(('final', 'generalization_gap_pp')),
            'cumulative_train_hours': delta(('cumulative_train_hours',)),
        },
    }


def _parse_run_spec(spec):
    """解析 MODEL[:SEED] 或 MODEL@SEED；省略 seed 时由 load_model 自动选最完整者。"""
    for separator in (':', '@'):
        if separator in spec:
            model, seed_text = spec.rsplit(separator, 1)
            if not model or not seed_text.isdigit():
                raise argparse.ArgumentTypeError(f'invalid run spec {spec!r}; expected MODEL:SEED')
            return model, int(seed_text)
    return spec, None


def _fmt(value, digits=4, suffix=''):
    if value is None:
        return 'n/a'
    return f'{float(value):.{digits}f}{suffix}'


def _print_run_summary(summary):
    e = summary['epochs']
    ob = summary['official_best_val_acc']
    final = summary['final']
    target = f"/{e['target']} ({e['progress_percent']:.1f}%)" if e['target'] else ''
    print(f"{summary['config_id']} seed{summary['seed']}  [{summary['status']}]  "
          f"epochs={e['rows']}{target} (ep{e['first']}..{e['last']}, missing={len(e['missing'])})")
    print(f"  best val      : {_fmt(ob['value'], 4, '%')}@ep{ob['epoch']} ({ob['source']})")
    print(f"  final         : val={_fmt(final['val_acc'], 4, '%')}  "
          f"train={_fmt(final['train_acc'], 4, '%')}  "
          f"gap={_fmt(final['generalization_gap_pp'], 4, 'pp')}  "
          f"loss={_fmt(final['val_loss'], 6)}")
    print(f"  best macro-F1 : {_fmt(summary['best_macro_f1']['value'], 6)}"
          f"@ep{summary['best_macro_f1']['epoch']}")
    print(f"  min val loss  : {_fmt(summary['minimum_val_loss']['value'], 6)}"
          f"@ep{summary['minimum_val_loss']['epoch']}")
    print(f"  train time    : {_fmt(summary['cumulative_train_hours'], 2, ' h')}")
    if e['missing']:
        preview = ','.join(str(x) for x in e['missing'][:12])
        tail = '...' if len(e['missing']) > 12 else ''
        print(f'  missing epochs: {preview}{tail}')


def _json_ready(value):
    """把 NaN/Inf 清为 null，保证 --json 输出是严格 JSON。"""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def cli(argv=None):
    """chart_data 命令行入口；成功返回0，数据或参数错误返回2。"""
    parser = argparse.ArgumentParser(description='Read-only summary of src_v2/runs experiments')
    sub = parser.add_subparsers(dest='command')

    p_list = sub.add_parser('list', help='list available config/seed runs and epoch ranges')
    p_list.add_argument('--json', action='store_true', help='emit machine-readable JSON')

    p_summary = sub.add_parser('summary', help='summarize one or more MODEL[:SEED] runs')
    p_summary.add_argument('runs', nargs='+', help='for example: S-NoProg:1 S-NoSE:1')
    p_summary.add_argument('--json', action='store_true', help='emit machine-readable JSON')

    p_compare = sub.add_parser('compare', help='compare two MODEL[:SEED] runs; delta=right-left')
    p_compare.add_argument('left', help='left run, for example S-NoProg:1')
    p_compare.add_argument('right', help='right run, for example S-NoSE:1')
    p_compare.add_argument('--json', action='store_true', help='emit machine-readable JSON')

    p_data = sub.add_parser('data', help='print every merged epoch row and every source column')
    p_data.add_argument('run', help='run to read, for example S-NoProg:1')
    p_data.add_argument('--json', action='store_true', help='emit a JSON array instead of CSV')

    args = parser.parse_args(argv)
    command = args.command or 'list'
    try:
        if command == 'list':
            data = available_models()
            if getattr(args, 'json', False):
                print(json.dumps(_json_ready(data), ensure_ascii=False, indent=2, allow_nan=False))
            else:
                print(f'[chart_data] models={len(data)}')
                for cid in sorted(data):
                    for seed in sorted(data[cid]):
                        info = data[cid][seed]
                        print(f'  {cid:22s} seed{seed}  rows={info["rows"]:>4}  '
                              f'max_epoch={info["max_epoch"]}')
            return 0

        if command == 'summary':
            data = [summarize_run(*_parse_run_spec(spec)) for spec in args.runs]
            if args.json:
                print(json.dumps(_json_ready(data), ensure_ascii=False, indent=2, allow_nan=False))
            else:
                for index, summary in enumerate(data):
                    if index:
                        print()
                    _print_run_summary(summary)
            return 0

        if command == 'data':
            model, seed = _parse_run_spec(args.run)
            df = load_model(model, seed=seed, verbose=False)
            if args.json:
                # pandas负责把NaN转换为null，并把numpy标量转换为标准JSON数值。
                print(df.to_json(orient='records', force_ascii=False, indent=2))
            else:
                # 默认CSV便于终端查看、管道处理或由调用方自行重定向保存。
                sys.stdout.write(df.to_csv(index=False, lineterminator='\n'))
            return 0

        left = _parse_run_spec(args.left)
        right = _parse_run_spec(args.right)
        data = compare_runs(left[0], left[1], right[0], right[1])
        if args.json:
            print(json.dumps(_json_ready(data), ensure_ascii=False, indent=2, allow_nan=False))
        else:
            _print_run_summary(data['left'])
            print()
            _print_run_summary(data['right'])
            d = data['delta']
            print(f"\nDelta ({data['right']['config_id']} - {data['left']['config_id']})")
            print(f"  official best val : {_fmt(d['official_best_val_acc_pp'], 4, 'pp')}")
            print(f"  final val         : {_fmt(d['final_val_acc_pp'], 4, 'pp')}")
            print(f"  best macro-F1     : {_fmt(d['best_macro_f1'], 6)}")
            print(f"  final gap         : {_fmt(d['final_generalization_gap_pp'], 4, 'pp')}")
            print(f"  train time        : {_fmt(d['cumulative_train_hours'], 2, ' h')}")
            print(f"  protocol match    : {data['protocol_match']}")
        return 0
    except (FileNotFoundError, ValueError, TypeError) as error:
        parser.error(str(error))
        return 2


# ---------- 论文口径（results JSON / best checkpoint，与 metrics 略有出入） ----------

def results_best(config_id, seed=None):
    """从 results_<config>_seed*.json 读该模型的论文口径最佳值（跑完才生成）。
    返回 (best_val_acc, best_val_epoch, seed)；指定seed只读该run，未指定取最高准确率。
    无 results → (None, None, None)。
    注：results 的 best 是训练结束时汇总的"正式成绩"（如 S-NoProg 82.63@74）；
    与 model_best()（metrics 口径，如 82.634）可能差 0.00x，引用论文数字请用本函数。"""
    config_id = _cid(config_id)
    if seed is not None:
        result = _result_for_run(config_id, seed)
        if result is None:
            return (None, None, None)
        return (float(result['best_val_acc']), int(result['best_val_epoch']), int(seed))
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


def best_ckpt_path(config_id, verbose=False, seed=None):
    """返回该模型应加载的 best checkpoint 绝对路径（与 results_best 的 seed 对齐；
    无 results 时退回 glob 到的最高 val_acc 那个 best.pth）。无 → None。
    显式seed时仅返回该run的best路径，缺失不回退其他seed；旧自动选择行为保留。"""
    config_id = _cid(config_id)
    if seed is not None:
        path = os.path.join(RUNS, f'checkpoint_{config_id}_seed{int(seed)}_best.pth')
        return path if os.path.isfile(path) else None
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


def load_run_frames(config_ids, seeds=(1, 2), require_complete=True):
    """公共多run入口；显式seed、正式结果/连续轮次及协议一致性检查。"""
    frames, summaries, protocols = {}, {}, []
    for config_id in config_ids:
        for seed in seeds:
            key = (_cid(config_id), int(seed))
            summary = summarize_run(*key)
            frame = load_model(*key, verbose=False)
            if require_complete:
                epochs = summary['epochs']
                if (not summary['results_json'] or not epochs['continuous_from_zero']
                        or epochs['rows'] != epochs['target']
                        or epochs['last'] != epochs['target'] - 1):
                    raise ValueError(f'Incomplete run: {key}')
                official = summary['official_best_val_acc']
                observed = summary['metrics_best_val_acc']
                if (round(observed['value'], 4) != round(official['value'], 4)
                        or observed['epoch'] != official['epoch']):
                    raise ValueError(f'Results/metrics mismatch: {key}')
            frames[key], summaries[key] = frame, summary
            protocols.append(summary['protocol'])
    if require_complete and any(p != protocols[0] for p in protocols):
        raise ValueError('Run protocols differ')
    return frames, summaries


def window_statistics(frame, start=0, stop=100):
    """半开epoch窗口，loss面积/下降斜率及真实记录用时；不平滑或排异常轮。"""
    import numpy as np
    selected = frame[(frame.epoch >= start) & (frame.epoch < stop)]
    if len(selected) != stop - start:
        raise ValueError(f'Missing epochs in [{start}, {stop})')
    x = selected.epoch.to_numpy(float)
    hours = selected.time_spend_s.to_numpy(float).cumsum() / 3600
    result = {'start': start, 'stop_exclusive': stop, 'n_epochs': len(selected),
              'hours': float(hours[-1])}
    for column in ('train_loss', 'val_loss'):
        y = selected[column].to_numpy(float)
        result[column] = {
            'first': float(y[0]), 'last': float(y[-1]), 'mean': float(y.mean()),
            'minimum': float(y.min()), 'endpoint_drop': float(y[0] - y[-1]),
            'slope_per_epoch': float(np.polyfit(x, y, 1)[0]),
            'slope_per_hour': float(np.polyfit(hours, y, 1)[0]),
            'normalized_auc_epoch': float(np.trapz(y, x) / (x[-1] - x[0])),
            'normalized_auc_time': float(np.trapz(np.r_[y[0], y], np.r_[0., hours]) / hours[-1]),
        }
    return result


if __name__ == '__main__':
    sys.exit(cli())
