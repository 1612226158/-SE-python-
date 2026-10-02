"""Reusable seed1 comparison; all epoch data use chart_data, never direct CSV reads."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CHARTS = HERE.parents[1]
sys.path.insert(0, str(CHARTS))
from chart_data import load_model, summarize_run, RUNS
import model_registry

MODELS = ['S-NoSE', 'S-StdSE', 'S-NoProg', 'V2-Full', 'V2-NoTF']
COLORS = dict(zip(MODELS, ['#c44e52', '#dd8452', '#4c72b0', '#8172b2', '#55a868']))
LABELS = {'S-NoSE': 'S-NoSE（v1 / 无SE）', 'S-StdSE': 'S-StdSE（v1 / [1]）',
          'S-NoProg': 'S-NoProg（v1 / [1,2]）', 'V2-Full': 'V2-Full（V2 / 3层）',
          'V2-NoTF': 'V2-NoTF（V2 / 0层）'}
PAIR = ['V2-Full', 'V2-NoTF']
RESTARTS = [5, 15, 35, 75]
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': 10,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'grid.color': '#d9d9d9', 'grid.linestyle': '--', 'grid.linewidth': .7,
                     'legend.fontsize': 9, 'axes.titlesize': 12})


def save(fig, name, caption):
    fig.text(.5, .012, caption, ha='center', fontsize=8.5, color='#555555')
    fig.tight_layout(rect=(0, .055, 1, .96))
    for suffix in ['png', 'pdf']:
        fig.savefig(HERE / f'{name}.{suffix}', dpi=230, bbox_inches='tight')
    plt.close(fig)
    print(name)


def lines(ax, frames, models, field, scale=1):
    for model in models:
        df = frames[model]
        ax.plot(df.epoch, df[field] * scale, color=COLORS[model], linewidth=1.65,
                label=LABELS[model], alpha=.95)
    for epoch in RESTARTS:
        ax.axvline(epoch, color='#aaaaaa', linewidth=.75, linestyle='--', alpha=.6)
    ax.set_xlabel('Epoch（0–99；竖虚线为CAWR重启）')
    ax.legend(loc='lower right', frameon=True)


def load():
    frames, summaries, results, fingerprints = {}, {}, {}, {}
    keys = ['training_protocol_version', 'sampling_protocol_version', 'label_schema_id',
            'label_schema_version', 'label_mapping_hash', 'num_classes']
    baseline = None
    for model in MODELS:
        df = load_model(model, seed=1, verbose=False).reset_index(drop=True)
        summary = summarize_run(model, seed=1)
        if summary['status'] != 'finished' or not summary['results_json']:
            raise ValueError(f'{model}: not finished')
        if df.epoch.tolist() != list(range(100)):
            raise ValueError(f'{model}: incomplete/duplicate epochs')
        for field in ['train_acc', 'val_acc', 'val_loss', 'val_macro_f1', 'time_spend_s']:
            if not np.isfinite(df[field].to_numpy(float)).all():
                raise ValueError(f'{model}: missing/nonfinite {field}')
        protocol = {key: summary['protocol'][key] for key in keys}
        if any(value is None for value in protocol.values()):
            raise ValueError(f'{model}: missing protocol')
        result_path = Path(RUNS) / f'results_{model}_seed1.json'
        result = json.loads(result_path.read_text(encoding='utf-8'))
        common = {**protocol, **{key: result[key] for key in
                                  ['batch_size', 'train_samples', 'val_samples', 'unfreeze_mode']}}
        if baseline is None:
            baseline = common
        elif common != baseline:
            raise ValueError(f'{model}: protocol/budget/data-count mismatch')
        frames[model], summaries[model], results[model] = df, summary, result
        fingerprints[model] = {'results_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
                               'registered_params': model_registry.params_of(model),
                               'observed_model_metadata': result.get('model_metadata')}
    full, no = [results[model]['model_metadata'].copy() for model in PAIR]
    if full.pop('transformer_layers') != 3 or no.pop('transformer_layers') != 0 or full != no:
        raise ValueError('Full/NoTF metadata differs beyond transformer_layers')
    return frames, summaries, results, fingerprints, baseline


def main():
    frames, summaries, results, fingerprints, protocol = load()
    best = {m: summaries[m]['official_best_val_acc']['value'] for m in MODELS}
    at_best_f1 = {m: float(frames[m].loc[
        frames[m].epoch == summaries[m]['official_best_val_acc']['epoch'],
        'val_macro_f1'].iloc[0]) * 100 for m in MODELS}
    valid_times, exclusions, timing = {}, [], {}
    for model in MODELS:
        df = frames[model]
        mask = np.ones(len(df), dtype=bool)
        if model == 'V2-NoTF':
            mask = df.epoch.to_numpy() != 63
            raw = float(df.loc[df.epoch == 63, 'time_spend_s'].iloc[0])
            if abs(raw - 5728.7926187515) > 1e-5:
                raise ValueError('UI-67 excluded value changed; recheck identity')
            exclusions.append({'model': model, 'seed': 1, 'epoch': 63,
                               'field': 'time_spend_s', 'raw_value': raw,
                               'reason': 'UI-20261001-67 efficiency-only exclusion'})
        times = df.loc[mask, 'time_spend_s'].to_numpy(float)
        if (times <= 0).any():
            raise ValueError('Invalid duration')
        valid_times[model] = times / 60
        timing[model] = {'valid_epochs': len(times), 'excluded_epochs': 100 - len(times),
                         'median_minutes': float(np.median(times)/60),
                         'mean_minutes': float(np.mean(times)/60),
                         'q25_minutes': float(np.percentile(times, 25)/60),
                         'q75_minutes': float(np.percentile(times, 75)/60),
                         'recorded_timing_definition': 'Existing time_spend_s; not complete wall-clock time'}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    for ax in axes:
        lines(ax, frames, PAIR, 'val_acc')
        ax.set_ylabel('验证准确率 (%)')
    axes[0].set_title('Full与NoTF：完整100轮')
    axes[0].set_ylim(40, 87)
    axes[1].set_xlim(60, 99)
    axes[1].set_ylim(79, 85.5)
    axes[1].set_title('后期局部：epoch60–99')
    for model in PAIR:
        epoch = summaries[model]['official_best_val_acc']['epoch']
        axes[1].scatter(epoch, best[model], s=40, color=COLORS[model], zorder=5)
    axes[1].text(.03, .97, '\n'.join(f'{m}: {best[m]:.4f}% @ ep74' for m in PAIR),
                 transform=axes[1].transAxes, va='top', fontsize=9,
                 bbox={'facecolor':'white', 'edgecolor':'#dddddd', 'alpha':.9})
    save(fig, '01_full_notf_accuracy', '252类 / 训练协议v2 / seed1；仅本次观测，不代表跨seed稳定性。')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    lines(axes[0], frames, PAIR, 'val_macro_f1', 100)
    axes[0].set_ylabel('验证Macro-F1 (%)')
    axes[0].set_title('类别平均表现')
    lines(axes[1], frames, PAIR, 'val_loss')
    axes[1].set_ylabel('验证Loss')
    axes[1].set_title('验证损失（越低越好）')
    axes[1].legend(loc='upper right')
    save(fig, '02_full_notf_f1_loss', '原始逐轮曲线；重启后的回落须结合学习率周期解释。')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    delta = frames['V2-NoTF'].val_acc - frames['V2-Full'].val_acc
    axes[0].plot(range(100), delta, color=COLORS['V2-NoTF'], alpha=.3, label='原始逐轮差')
    axes[0].plot(range(100), delta.rolling(5, center=True, min_periods=1).mean(),
                 color=COLORS['V2-NoTF'], linewidth=1.9, label='5轮中心滑动均值')
    axes[0].axhline(0, color='#666666', linewidth=.8)
    axes[0].set_ylabel('NoTF − Full 验证准确率 (pp)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_title('正值表示该轮NoTF较高')
    axes[0].legend(loc='lower right')
    for model in PAIR:
        gap = frames[model].train_acc - frames[model].val_acc
        axes[1].plot(range(100), gap, color=COLORS[model], label=model)
    axes[1].set_ylabel('训练Accuracy − 验证Accuracy (pp)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_title('训练与验证差距')
    axes[1].legend(loc='lower right')
    save(fig, '03_full_notf_difference_gap', '训练Accuracy来自WRS增强采样，验证为验证口径；差距不能直接当自然分布泛化误差。')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    for ax, values, title in [(axes[0], best, '正式best验证准确率'),
                              (axes[1], at_best_f1, '最佳准确率同轮的Macro-F1')]:
        bars = ax.bar(range(5), [values[m] for m in MODELS], color=[COLORS[m] for m in MODELS], width=.62)
        ax.set_xticks(range(5), MODELS, rotation=15)
        ax.set_ylim(0, 90)
        ax.set_ylabel('百分比 (%)')
        ax.set_title(title)
        for model, bar in zip(MODELS, bars):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'{values[model]:.3f}', ha='center', fontsize=9)
    save(fig, '04_five_models_best_metrics', '五组性能概览；S系列为v1头、V2系列为49-token头，跨组差值不是纯SE或纯Transformer效应。')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lines(axes[0], frames, MODELS, 'val_acc')
    axes[0].set_ylabel('验证准确率 (%)')
    axes[0].set_title('五组完整验证Accuracy')
    lines(axes[1], frames, MODELS, 'val_macro_f1', 100)
    axes[1].set_ylabel('验证Macro-F1 (%)')
    axes[1].set_title('五组完整验证Macro-F1')
    save(fig, '05_five_models_curves', '协议、seed、100轮预算及样本数已核对；不能从混合架构曲线单独归因模块收益。')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    bridge = ['S-NoProg', 'V2-Full', 'V2-NoTF']
    lines(axes[0], frames, bridge, 'val_acc')
    axes[0].set_title('固定多区域SE [1,2] 的路线比较')
    axes[0].set_ylabel('验证准确率 (%)')
    for model in PAIR:
        diff = frames[model].val_acc - frames['S-NoProg'].val_acc
        axes[1].plot(range(100), diff.rolling(5, center=True, min_periods=1).mean(),
                     color=COLORS[model], label=f'{model} − S-NoProg')
    axes[1].axhline(0, color='#666666', linewidth=.8)
    axes[1].set_ylabel('相对S-NoProg差值 (pp)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_title('5轮中心滑动均值（描述性）')
    axes[1].legend(loc='lower right')
    save(fig, '06_same_se_head_routes', 'v1→V2改变token、位置编码、投影/池化及正则路径；同SE不等于仅改变Transformer。')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    offsets = {'S-NoSE':(-50,-22), 'S-StdSE':(8,-5), 'S-NoProg':(-75,-20),
               'V2-Full':(-78,10), 'V2-NoTF':(8,8)}
    for model in MODELS:
        x, y = results[model]['params']/1e6, best[model]
        axes[0].scatter(x,y,color=COLORS[model],s=50)
        axes[0].annotate(model,(x,y),xytext=offsets[model],textcoords='offset points',fontsize=9)
    axes[0].set_xlim(30,44)
    axes[0].set_ylim(83,85)
    axes[0].set_xlabel('模型参数量 (百万)')
    axes[0].set_ylabel('正式best验证准确率 (%)')
    axes[0].set_title('参数量与性能（架构路线综合对比）')
    boxes = axes[1].boxplot([valid_times[m] for m in MODELS], patch_artist=True, widths=.55,
                           showfliers=True, medianprops={'color':'#333333'})
    for patch, model in zip(boxes['boxes'], MODELS):
        patch.set_facecolor(COLORS[model]); patch.set_alpha(.65)
    axes[1].set_xticks(range(1,6), [f'{m}\nn={timing[m]["valid_epochs"]}' for m in MODELS], rotation=15)
    axes[1].set_ylabel('已记录逐轮耗时 (分钟)')
    axes[1].set_title('逐轮耗时分布（NoTF已排除epoch63）')
    save(fig, '07_parameters_epoch_time', 'NoTF n=99，其余n=100；不补缺失耗时，不绘清洗后的累计时间或time-to-accuracy；单机历史轮速。')

    report = {'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'scope': {'seed':1, 'epochs':100, 'num_classes':252},
              'protocol_and_common_fields':protocol, 'summaries_raw':summaries,
              'result_identities':fingerprints, 'timing_exclusions':exclusions,
              'valid_epoch_timing':timing, 'params':{m:results[m]['params'] for m in MODELS},
              'accuracy_selected_epoch_macro_f1_pct':at_best_f1,
              'NoTF_vs_Full_epoch_counts': {
                  'NoTF_higher': int((delta > 0).sum()),
                  'equal': int((delta == 0).sum()),
                  'Full_higher': int((delta < 0).sum())},
              'pairwise_best_acc_pp': {'NoTF_minus_Full':best['V2-NoTF']-best['V2-Full'],
                  'NoTF_minus_S_NoProg':best['V2-NoTF']-best['S-NoProg'],
                  'Full_minus_S_NoProg':best['V2-Full']-best['S-NoProg']},
              'comparison_boundaries': ['Full/NoTF: matched V2 3/0-layer comparison',
                  'Three S variants: within-v1 SE comparison',
                  'V2 vs S-NoProg: whole-head route comparison with SE [1,2] fixed',
                  'All five: descriptive model-performance comparison, not module attribution'],
              'limitations': ['Only seed1; no statistical significance or cross-seed stability claim',
                  'S results lack saved model_metadata; registry describes S architecture, not recovered run metadata',
                  'Protocol/batch/sample-count matching does not prove all historical hyperparameters match',
                  'Raw cumulative hours in summaries_raw contain excluded epoch63; do not use as cleaned efficiency',
                  'Current images/data splitting hashes are not available in the old run artifacts']}
    (HERE/'comparison_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'pairwise':report['pairwise_best_acc_pp'], 'timing':timing},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
