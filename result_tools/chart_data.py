# -*- coding: utf-8 -*-
"""V3公共结果读取：仅扫描runs下正式run目录，排除_smoke与analyses。

按配置/seed读取本地绑定的MySQL指标与结果，历史运行读取CSV与results JSON。
checkpoint始终留在本地；可通过V3_RESULTS_ROOT指定其他运行目录的上级目录。
查询不构造模型、不读原图；分析输出仅在显式请求时写runs/analyses。
例：python chart_data.py summary FullUnfreeze:1
"""
import argparse
import json
import math
import os
import sys
import pandas as pd

# 路径与训练/队列统一；查询不构造模型或读取图片。
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_V3 = os.path.dirname(_HERE)
if _SRC_V3 not in sys.path:
    sys.path.insert(0, _SRC_V3)
from run_paths import (RUNS_ROOT, checkpoint_path, formal_run_directories, load_config,
                       normalization_signature, result_path, run_directory)
import taxonomy as _taxonomy
import model_registry as _mr
RUNS = os.environ.get('V3_RESULTS_ROOT') or str(RUNS_ROOT)

# 类别数按taxonomy的原始类数和合并规则派生。
N_CLS_RAW = int(_taxonomy.RAW_CLASS_COUNT)
N_CLS = N_CLS_RAW - sum(len(group) - 1 for group in _taxonomy.merged_dict.values())

# V3不继承其他实验目录的历史作废记录。
DIRTY_RECORDS = {}


# 模型名称及元数据统一读取model_registry。
def _cid(name):
    """通过注册表取得规范配置ID。"""
    return _mr.normalize(name)


def model_cn(name):
    """模型中文全称（GUI 说明行 / 汇报 / 图注用）。"""
    return _mr.cn_of(name)


def model_short(name):
    """模型短说明。"""
    return _mr.short_of(name)


def is_known(name):
    """该模型 id 是否已在 model_registry.py 注册。"""
    return _mr.is_known(name)

# ---------- 输出文件名推导（2026-09-08 会话8 加：消灭 out*.pdf 硬编码） ----------

# 当前两基线对比使用本目录的分析标签，不复用既有论文文件名。
PAIR_TAGS = {
    ('FullUnfreeze', 'ProgressiveUnfreeze'): 'unfreeze_comparison',
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
    out_dir = out_dir or os.path.join(RUNS, 'analyses')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f'{fig_type}_{pair_tag(model_a, model_b)}.pdf')


# ---------- 基础扫描 ----------

def all_metrics_files(*, runs_root=RUNS_ROOT):
    """只读正式run子目录的当前指标，排除归档、_smoke与analyses。"""
    files = []
    for directory in formal_run_directories(runs_root=runs_root):
        current = directory / 'metrics_record.csv'
        if current.is_file():
            files.append(str(current))
    return files


def _mysql_binding(directory, config_id, seed):
    path = directory / 'config.json'
    try:
        with path.open(encoding='utf-8') as f:
            snapshot = json.load(f)
    except FileNotFoundError:
        return None
    if not isinstance(snapshot, dict):
        raise ValueError(f'运行配置必须是 JSON 对象：{path}')
    if 'MYSQL_RUN_ID' not in snapshot and 'MYSQL_DATABASE' not in snapshot:
        return None
    run_id = snapshot.get('MYSQL_RUN_ID')
    database = snapshot.get('MYSQL_DATABASE')
    if (snapshot.get('CONFIG_ID') != config_id or snapshot.get('SEED') != seed
            or snapshot.get('SMOKE') or type(run_id) is not int or run_id <= 0
            or not isinstance(database, str) or not database):
        raise ValueError(f'MySQL 运行绑定不完整或身份不匹配：{path}')
    return run_id, database


def _read_mysql(config_id, seed, binding):
    from mysql_store import read_run_results

    run_id, database = binding
    data = read_run_results(run_id, database=database)
    run = data['run']
    if (run['config_id'], run['seed']) != (config_id, seed):
        raise ValueError(f'MySQL {database} run_id={run_id} 与目录的模型或 seed 不匹配')
    return data


def _read_all(config_id=None, seed=None):
    """读取正式run当前指标，按配置和作废登记过滤；_fidx用于稳定去重。"""
    frames = []
    mysql_runs = {}
    bound_dirs = set()
    for directory in formal_run_directories(runs_root=RUNS):
        cid, marker, seed_text = directory.name.rpartition('_seed')
        if not marker or not seed_text.isascii() or not seed_text.isdecimal():
            continue
        run_seed = int(seed_text)
        if seed_text != str(run_seed):
            continue
        if (config_id is not None and cid != config_id) or (seed is not None and run_seed != seed):
            continue
        binding = _mysql_binding(directory, cid, run_seed)
        if binding is None:
            continue
        data = _read_mysql(cid, run_seed, binding)
        mysql_runs[cid, run_seed] = data['run'], data['result']
        bound_dirs.add(str(directory))
        rows = data['metrics']
        if rows:
            df = pd.DataFrame(rows)
            # 混合历史指标时仍保留 BIGINT UNSIGNED 采样种子的整数精度。
            df['sampler_seed'] = pd.Series([row['sampler_seed'] for row in rows], dtype=object)
            df['save_time'] = [str(row['save_time']) if row['save_time'] is not None else None
                               for row in rows]
            frames.append(df.assign(config_id=cid, seed=run_seed, _fidx=0))

    for fidx, f in enumerate(all_metrics_files(runs_root=RUNS)):
        if os.path.dirname(f) in bound_dirs:
            continue
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
        if seed is not None:
            df = df[df['seed'].astype(int) == seed]
        if mysql_runs and len(df):
            bound = pd.MultiIndex.from_frame(df[['config_id', 'seed']]).isin(mysql_runs)
            df = df[~bound]
        if len(df):
            df = df.copy()
            df['_fidx'] = fidx
            frames.append(df)
    if not frames:
        return pd.DataFrame(), mysql_runs
    return pd.concat(frames, ignore_index=True), mysql_runs


# ---------- 对外 API ----------

def available_models():
    """返回 {config_id: {seed: {'rows': n, 'max_epoch': e}}}：当前有哪些模型、各 seed 记录情况。"""
    df, _ = _read_all()
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

    - seed=None：自动选择行数最多的 seed；行数相同时，优先选择最高 epoch 更大者；
    - seed=指定：只取该 seed；
    - 返回按 epoch 升序、同 epoch 去重（保留最后一条）的 DataFrame；
    - 无数据 → 抛 FileNotFoundError（提示该模型还没跑 / config_id 拼错）。
    """
    config_id = _cid(config_id)   # 当前只传递配置ID，不转换旧名称。
    seed = int(seed) if seed is not None else None
    df, mysql_runs = _read_all(config_id, seed)
    if df.empty:
        identity = config_id if seed is None else f'{config_id} seed={seed}'
        raise FileNotFoundError(f'[chart_data] {identity} 没有逐轮指标（运行目录：{RUNS}）')
    # seed 归一化
    if seed is None:
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
    df = df.drop_duplicates(subset='epoch', keep='last').reset_index(drop=True)   # 同epoch保留扫描顺序最后一条
    df = df.drop(columns=['_fidx'], errors='ignore')
    # 数值列尽量转 float（seed 保留原样）
    for col in ('epoch', 'train_acc', 'train_loss', 'val_acc', 'val_loss',
                'val_macro_f1', 'val_weighted_f1', 'val_weighted_precision',
                'val_weighted_recall', 'lr', 'time_spend_s'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    mysql_run = mysql_runs.get((config_id, int(df['seed'].iloc[0])))
    if mysql_run is not None:
        df.attrs['mysql_run'] = mysql_run
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
    """从load_model已规范化的数据取极值点；缺列或全NaN返回空值。"""
    if column not in df.columns:
        return {'value': None, 'epoch': None}
    values = df[column]
    valid = values.dropna()
    if valid.empty:
        return {'value': None, 'epoch': None}
    idx = valid.idxmin() if mode == 'min' else valid.idxmax()
    return {'value': float(values.loc[idx]), 'epoch': int(df.loc[idx, 'epoch'])}


def _result_for_run(config_id, seed):
    """读取指定 config/seed 的 MySQL 结果或历史 results JSON。"""
    seed = int(seed)
    directory = run_directory(config_id, seed, runs_root=RUNS)
    binding = _mysql_binding(directory, config_id, seed)
    if binding is not None:
        return _read_mysql(config_id, seed, binding)['result']
    return _legacy_result_for_run(config_id, seed)


def _legacy_result_for_run(config_id, seed):
    path = str(result_path(_cid(config_id), int(seed), runs_root=RUNS))
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            result = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(result, dict) or result.get('config_id') != _cid(config_id) or int(result.get('seed', -1)) != int(seed):
        raise ValueError('V3结果的配置ID/seed身份不匹配')
    preset = _mr.params_of(_cid(config_id))
    expected = {'backbone': preset['backbone'], 'unfreeze_epochs': preset['unfreeze_epochs'],
                'experiment_signature': _mr.experiment_signature_of(_cid(config_id)),
                'normalization_signature': normalization_signature(load_config())}
    source = _mr.long_source_of(_cid(config_id), int(seed))
    if source:
        expected['long_source'] = source
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f'V3结果与当前注册配置不一致: {key}')

    return result


def summarize_run(config_id, seed=None):
    """汇总一个实验的状态与关键指标，供命令行和其他脚本统一调用。

    数据仍由 load_model() 负责扫描、合并、去重和排除脏记录。本函数不直接打开
    metrics CSV，只在其规范化结果上计算完整性、best/final、F1、loss、泛化差距和耗时。
    正式 best 优先采用该运行的数据源所提供的结果；未收官时使用当前指标峰值。
    """
    df = load_model(config_id, seed=seed, verbose=False)
    return _summarize_frame(config_id, df)


def _summarize_frame(config_id, df):
    """汇总load_model已规范化的逐轮数据，不修改传入数据。"""
    config_id = _cid(config_id)
    selected_seed = int(df['seed'].iloc[0])
    mysql_run = df.attrs.get('mysql_run')
    if mysql_run is None:
        run, result = {}, _legacy_result_for_run(config_id, selected_seed)
    else:
        run, result = mysql_run
    registered_model = bool(_mr.is_known(config_id))
    target_epochs = run['target_epochs'] if run else (int(_mr.target_of(config_id)) if registered_model else None)
    source = _mr.long_source_of(config_id, selected_seed) if registered_model else None

    epochs = sorted({int(e) for e in df['epoch'].dropna().tolist()})
    min_epoch = epochs[0]
    max_epoch = epochs[-1]
    missing_epochs = sorted(set(range(0, max_epoch + 1)) - set(epochs))

    metrics_best = _metric_point(df, 'val_acc', 'max')
    best_epoch = metrics_best['epoch']
    best_row = df[df['epoch'].astype(int) == best_epoch].iloc[-1]
    final_row = df.iloc[-1]

    official_best = dict(metrics_best, source='metrics')
    if result and result.get('best_val_acc') is not None and result.get('best_val_epoch') is not None:
        official_best = {
            'value': float(result['best_val_acc']),
            'epoch': int(result['best_val_epoch']),
            'source': 'mysql' if run else 'results_json',
        }
    metadata = (dict(run['training_params'], backbone=run['backbone'], num_classes=run['num_classes'],
                     unfreeze_epochs=run['model_params'].get('unfreeze_epochs'))
                if run else result or {})
    final_train = _number(final_row, 'train_acc')
    final_val = _number(final_row, 'val_acc')
    best_train = _number(best_row, 'train_acc')
    best_val = _number(best_row, 'val_acc')
    time_values = df.get('time_spend_s')

    def protocol_value(key, cast=None):
        value = metadata.get(key)
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
        **({'long_source': source} if source else {}),
        'registered_model': registered_model,
        'status': 'finished' if result else 'incomplete',
        'results_json': mysql_run is None and bool(result),
        'backbone': metadata.get('backbone'),
        'unfreeze_epochs': metadata.get('unfreeze_epochs'),
        'experiment_signature': metadata.get('experiment_signature'),
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
    parser = argparse.ArgumentParser(description='Read-only summary of src_v3/runs experiments')
    sub = parser.add_subparsers(dest='command')

    p_list = sub.add_parser('list', help='list available config/seed runs and epoch ranges')
    p_list.add_argument('--json', action='store_true', help='emit machine-readable JSON')

    p_summary = sub.add_parser('summary', help='summarize one or more MODEL[:SEED] runs')
    p_summary.add_argument('runs', nargs='+', help='for example: FullUnfreeze:1 ProgressiveUnfreeze:1')
    p_summary.add_argument('--json', action='store_true', help='emit machine-readable JSON')

    p_compare = sub.add_parser('compare', help='compare two MODEL[:SEED] runs; delta=right-left')
    p_compare.add_argument('left', help='left run, for example FullUnfreeze:1')
    p_compare.add_argument('right', help='right run, for example ProgressiveUnfreeze:1')
    p_compare.add_argument('--json', action='store_true', help='emit machine-readable JSON')

    p_data = sub.add_parser('data', help='print every merged epoch row and every source column')
    p_data.add_argument('run', help='run to read, for example FullUnfreeze:1')
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
    """返回正式结果(best_acc, epoch, seed)，显式seed不回退其他运行。"""
    config_id = _cid(config_id)
    if seed is not None:
        seeds = [int(seed)]
    else:
        prefix = config_id + '_seed'
        seeds = []
        for directory in formal_run_directories(runs_root=RUNS):
            suffix = directory.name[len(prefix):]
            if directory.name.startswith(prefix) and suffix.isascii() and suffix.isdecimal():
                seeds.append(int(suffix))
    candidates = []
    for selected_seed in seeds:
        result = _result_for_run(config_id, selected_seed)
        if result and result.get('best_val_acc') is not None and result.get('best_val_epoch') is not None:
            candidates.append((float(result['best_val_acc']), int(result['best_val_epoch']), selected_seed))
    return max(candidates, key=lambda value: value[0]) if candidates else (None, None, None)


def best_ckpt_path(config_id, verbose=False, seed=None):
    """只定位本run的best文件，不隐式加载权重猜测seed。"""
    config_id = _cid(config_id)
    if seed is None:
        _, _, seed = results_best(config_id)
    if seed is None:
        return None
    path = checkpoint_path(config_id, int(seed), which='best', runs_root=RUNS)
    return str(path) if path.is_file() else None


def load_run_frames(config_ids, seeds=(1, 2), require_complete=True):
    """公共多run入口；显式seed、正式结果/连续轮次及协议一致性检查。"""
    frames, summaries, protocols = {}, {}, []
    for config_id in config_ids:
        for seed in seeds:
            key = (_cid(config_id), int(seed))
            frame = load_model(*key, verbose=False)
            summary = _summarize_frame(key[0], frame)
            if require_complete:
                epochs = summary['epochs']
                if (summary['status'] != 'finished' or not epochs['continuous_from_zero']
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
