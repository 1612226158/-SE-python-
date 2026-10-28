"""Render every numeric thesis figure from traceable existing experiment data.

No training, inference, cache rebuilds or writes to runs/. Old exports are kept
as historical artifacts; document_new.tex points to this consistent figure set.
"""
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
SRC = Path(__file__).resolve().parents[3]
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
from ReportChart.chart_data import load_model
from ReportChart.paper_style import COLORS, SIZE, canvas, configure, save
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
SNAPSHOT = SRC / 'ReportChart/v2/seed1_summary/seed1_results_summary.json'
REPORT = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
FRAMES = {name: load_model(name, 1, verbose=False) for name in REPORT['runs']}
for name, frame in FRAMES.items():
    if list(frame.epoch.astype(int)) != list(range(100)):
        raise ValueError(f'{name}: continuous seed1 epochs 0..99 required')
    frozen = REPORT['runs'][name]['metrics_all_epochs']
    for col in ['val_acc', 'val_loss', 'val_macro_f1', 'time_spend_s']:
        if not np.allclose(frame[col], [row[col] for row in frozen], rtol=1e-8, atol=1e-8):
            raise ValueError(f'{name}: changed frozen source {col}')
    frame['hours'] = frame.time_spend_s.cumsum() / 3600

configure()
SOURCES = [SNAPSHOT]
FIGURES = []


def export(fig, name):
    save(fig, OUT, name)
    FIGURES.append(name)


def curves(ax, names, field, ylabel, x='epoch', best=False):
    for name in names:
        d = FRAMES[name]
        y = d[field] * (100 if field.endswith('_f1') else 1)
        if best:
            y = y.cummax()
        ax.plot(d[x], y, label=name, color=COLORS[name],
                drawstyle='steps-post' if best else 'default')
    ax.set_xlabel('训练轮次' if x == 'epoch' else '累计训练与验证时间（小时）')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()


def loss_lr(names, filename):
    fig, axs = canvas(3, 1, sharex=True)
    for ax, field, title in zip(axs, ['train_loss', 'val_loss', 'lr'],
                                ['训练损失', '验证损失', '学习率']):
        curves(ax, names, field, title)
        ax.set_title(title)
        for epoch in [30, 60]:
            ax.axvline(epoch, color='#999999', ls=':', lw=.8)
        if field != 'lr':
            ax.set_xlabel('')
        if field == 'lr':
            ax.set_yscale('log')
    for ax in axs[1:]:
        ax.get_legend().remove()
    export(fig, filename)


def accuracy(names, filename):
    fig, ax = canvas()
    curves(ax, names, 'val_acc', '验证准确率（%）')
    ax.set_title('验证准确率')
    export(fig, filename)


def time_accuracy(names, filename):
    fig, axs = canvas(1, 2)
    curves(axs[0], names, 'val_acc', '验证准确率（%）', x='hours')
    curves(axs[1], names, 'val_acc', '最佳验证准确率（%）', x='hours', best=True)
    axs[0].set_title('逐轮验证准确率')
    axs[1].set_title('时间预算内最佳准确率')
    export(fig, filename)


loss_lr(['S-NoProg', 'G-Full-CAWR'], 'progress_loss')
accuracy(['S-NoProg', 'G-Full-CAWR'], 'progress_accuracy')
time_accuracy(['S-NoProg', 'G-Full-CAWR'], 'progress_time')
loss_lr(['V2-Full', 'G-V2'], 'gv2_loss')
time_accuracy(['V2-Full', 'G-V2'], 'gv2_time')

fig, axs = canvas(2, 2)
for ax, field, title in zip(axs.flat, ['train_loss', 'val_loss', 'gap', 'val_acc'],
                           ['训练损失', '验证损失', '训练与验证准确率差', '验证准确率']):
    for name in ['V2-Full', 'V2-NoTF']:
        d = FRAMES[name]
        y = d.train_acc - d.val_acc if field == 'gap' else d[field]
        ax.plot(d.epoch, y, label=name, color=COLORS[name])
    ax.set_title(title)
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('准确率差（百分点）' if field == 'gap' else
                  '准确率（%）' if field == 'val_acc' else '损失')
    for epoch in [5, 15, 35, 75]:
        ax.axvline(epoch, color='#999999', ls=':', lw=.8)
    ax.grid(True)
    ax.legend()
export(fig, 'tf_fitting')

fig, axs = canvas(1, 2)
thresholds = [80, 82, 83]
for ax, key, title, ylabel in zip(axs, ['epoch', 'hours'],
                                 ['首次达到目标的完成轮数', '首次达到目标的时间'],
                                 ['完成训练轮数', '累计训练与验证时间（小时）']):
    for i, name in enumerate(['V2-Full', 'V2-NoTF']):
        values = [REPORT['runs'][name]['thresholds'][str(v)]['first'][key] for v in thresholds]
        if key == 'epoch':
            values = [v + 1 for v in values]
        positions = np.arange(3) + (i - .5) * .34
        bars = ax.bar(positions, values, width=.34, color=COLORS[name], label=name)
        ax.bar_label(bars, fmt='%d' if key == 'epoch' else '%.1f', fontsize=9, padding=2)
    ax.set_xticks(np.arange(3), [str(v) for v in thresholds])
    ax.set_xlabel('目标准确率（%）')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(y=.18)
    ax.legend(loc='upper left')
export(fig, 'tf_threshold')

fig, axs = canvas(1, 2)
names = ['S-NoProg', 'V2-Full', 'V2-NoTF']
curves(axs[0], names, 'val_acc', '验证准确率（%）')
axs[0].set_title('验证准确率')
for name in ['V2-Full', 'V2-NoTF']:
    diff = (FRAMES[name].val_acc - FRAMES['S-NoProg'].val_acc).rolling(5, center=True).mean()
    axs[1].plot(FRAMES[name].epoch, diff, color=COLORS[name], label=f'{name} - S-NoProg')
axs[1].set(title='五轮滑动平均差值', xlabel='训练轮次', ylabel='准确率差（百分点）')
axs[1].axhline(0, color='#888888', lw=.8)
axs[1].grid(True)
axs[1].legend()
export(fig, 'head_routes')

names = ['S-NoSE', 'S-StdSE', 'S-NoProg']
fig, axs = canvas(2, 1, sharex=True)
for ax, field, title in zip(axs, ['train_loss', 'val_loss'], ['训练损失', '验证损失']):
    curves(ax, names, field, '损失')
    ax.set_title(title)
export(fig, 'se_loss')
accuracy(names, 'se_accuracy')
time_accuracy(names, 'se_time')
fig, axs = canvas(2, 1, sharex=True)
for ax, field, title in zip(axs, ['val_macro_f1', 'val_weighted_f1'], ['宏平均F1', '加权F1']):
    curves(ax, names, field, 'F1（%）')
    ax.set_title(title)
export(fig, 'se_f1')

fig, axs = canvas(1, 2)
d = FRAMES['V2-NoTF']
for ax, fields, ylabel, title in zip(axs, [('train_loss', 'val_loss'), ('train_acc', 'val_acc')],
                                    ['损失', '准确率（%）'], ['损失轨迹', '准确率轨迹']):
    for field, linestyle, label in zip(fields, ['--', '-'], ['训练', '验证']):
        ax.plot(d.epoch, d[field], color=COLORS['V2-NoTF'], ls=linestyle, label=label)
    for epoch in [5, 15, 35, 75]:
        ax.axvline(epoch, color='#999999', ls=':', lw=.8)
    ax.set(xlabel='训练轮次', ylabel=ylabel, title=title)
    ax.grid(True)
    ax.legend()
export(fig, 'training_example')

tail_csv = SRC / 'ReportChart/per_class_acc.csv'
tail_json = SRC / 'ReportChart/tail_classes.json'
historic_results = {n: SRC / f'runs/DIRTY_unscale_bug_20260926/results_{n}_seed1.json'
                    for n in ['S-NoProg', 'G-Full-CAWR']}
SOURCES.extend([tail_csv, tail_json, *historic_results.values()])
with tail_csv.open(encoding='utf-8-sig', newline='') as h:
    tail_rows = {row['class_name']: row for row in csv.DictReader(h)}
tails = json.loads(tail_json.read_text(encoding='utf-8'))


def tail_acc(names, letter):
    correct = sum(int(tail_rows[n][letter + '_correct']) for n in names)
    total = sum(int(tail_rows[n][letter + '_total']) for n in names)
    return 100 * correct / total


fig, axs = canvas(1, 2)
ordered = sorted(tail_rows, key=lambda n: (int(tail_rows[n]['train_n']), n))
for i, name in enumerate(['S-NoProg', 'G-Full-CAWR']):
    letter = ['S', 'G'][i]
    old = json.loads(historic_results[name].read_text(encoding='utf-8'))
    if old['num_classes'] != 254:
        raise ValueError('Historical tail results require 254 classes')
    values = [tail_acc(tails['tail50'], letter), tail_acc(tails['tail100'], letter), old['best_val_acc']]
    bars = axs[0].bar(np.arange(3) + (i - .5) * .34, values, .34, color=COLORS[name], label=name)
    axs[0].bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
    ks = list(range(10, 101, 10))
    axs[1].plot(ks, [tail_acc(ordered[:k], letter) for k in ks], color=COLORS[name], label=name)
axs[0].set_xticks(np.arange(3), ['最少50类', '最少100类', '全部254类'])
axs[0].set(title='类别子集准确率', ylabel='验证准确率（%）', ylim=(0, 100))
axs[1].set(title='长尾类别前缀准确率', xlabel='参与统计的类别数', ylabel='验证准确率（%）', ylim=(0, 100))
for ax in axs:
    ax.grid(True, axis='y')
    ax.legend(loc='lower right')
export(fig, 'tail_accuracy')


def short_name(name):
    return str(name).split('-', 2)[-1]


def confusion_figures(path, prefix, model):
    SOURCES.append(path)
    with np.load(path, allow_pickle=True) as z:
        cm, names = z['cm'], z['names']
    support = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), support, out=np.zeros(len(cm), dtype=float), where=support > 0)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    cells = sorted(((int(off[i, j]), i, j) for i, j in zip(*np.nonzero(off))), reverse=True)[:20]
    fig, axs = canvas(1, 2, gridspec_kw={'width_ratios': [1.1, 1]})
    labels = [f'{short_name(names[i])}→{short_name(names[j])}' for v, i, j in cells]
    bars = axs[0].barh(np.arange(len(cells)), [v for v, i, j in cells], color=COLORS[model])
    axs[0].set_yticks(np.arange(len(cells)), labels, fontsize=7)
    axs[0].invert_yaxis()
    axs[0].bar_label(bars, padding=2, fontsize=7)
    axs[0].set(title='高频有向混淆', xlabel='误判次数')
    axs[0].margins(x=.16)
    error_count = off.sum(axis=0) + off.sum(axis=1)
    idx = np.sort(np.argsort(error_count, kind='stable')[-20:])
    percent = cm[np.ix_(idx, idx)] / support[idx, None] * 100
    im = axs[1].imshow(percent, cmap='Blues', vmin=0, vmax=100, aspect='equal')
    axs[1].set_xticks(np.arange(len(idx)), [short_name(names[i]) for i in idx], rotation=90, fontsize=6.5)
    axs[1].set_yticks(np.arange(len(idx)), [short_name(names[i]) for i in idx], fontsize=6.5)
    axs[1].set(title='高混淆类别矩阵', xlabel='预测类别', ylabel='真实类别')
    fig.colorbar(im, ax=axs[1], fraction=.04, pad=.02, label='占真值类样本（%）')
    export(fig, prefix + '_confusion')
    fig, axs = canvas(1, 2)
    order = np.argsort(recall, kind='stable')
    axs[0].plot(np.arange(1, len(cm) + 1), recall[order] * 100, color=COLORS[model])
    axs[0].axhline(recall.mean() * 100, color='#888888', ls='--', label=f'宏平均{recall.mean()*100:.1f}%')
    axs[0].set(title='逐类召回率分布', xlabel='类别排序', ylabel='召回率（%）', ylim=(0, 105))
    axs[0].grid(True)
    axs[0].legend()
    worst = [i for i in order if support[i] >= 30][:10]
    bars = axs[1].barh(np.arange(len(worst)), recall[worst] * 100, color=COLORS[model])
    axs[1].set_yticks(np.arange(len(worst)), [f'{short_name(names[i])}（{support[i]}张）' for i in worst], fontsize=8)
    axs[1].bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
    axs[1].invert_yaxis()
    axs[1].set(title='低召回十类（样本≥30）', xlabel='召回率（%）', xlim=(0, 100))
    export(fig, prefix + '_recall')


confusion_figures(SRC / 'ReportChart/v2/seed1_summary/evaluation/confusion_V2-NoTF.npz', 'current', 'V2-NoTF')
confusion_figures(SRC / 'ReportChart/confusion_S-NoProG.npz', 'history', 'S-NoProg')

timing = {}
for name, d in FRAMES.items():
    minutes = d.time_spend_s.to_numpy() / 60
    matched = d[d.epoch != 63]
    timing[name] = {
        'all_epochs': list(range(100)), 'n': 100,
        'raw_total_hours': float(minutes.sum() / 60),
        'raw_mean_minutes': float(minutes.mean()), 'median_minutes': float(np.median(minutes)),
        'q1_minutes': float(np.quantile(minutes, .25)), 'q3_minutes': float(np.quantile(minutes, .75)),
        'common_epoch63_removed': {'n': 99, 'excluded_epochs': [63],
                                  'mean_minutes': float(matched.time_spend_s.mean() / 60),
                                  'sum_hours': float(matched.time_spend_s.sum() / 3600)},
    }
summary = {
    'generated': datetime.now().astimezone().isoformat(timespec='seconds'),
    'reader': 'ReportChart.chart_data.load_model(config_id, seed=1); checked against fixed 700-row snapshot',
    'figure_style': {'canvas_inches': list(SIZE), 'dpi': 300, 'font': 'Microsoft YaHei',
                     'title_language': 'Chinese', 'configuration_colors': COLORS,
                     'remarks_in_plot': False, 'dense_class_label_size': '6.5–8 pt'},
    'timing': timing,
    'timing_rule': 'All groups retain all 100 epochs for total, median and quartiles; identical optional sensitivity removes epoch63 from every group, n=99. Raw results remain intact.',
    'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in SOURCES],
    'figures': [{'name': n, 'pdf': str(OUT / (n + '.pdf')),
                 'sha256': hashlib.sha256((OUT / (n + '.pdf')).read_bytes()).hexdigest()} for n in FIGURES],
}
(OUT / 'paper_analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'figures': len(FIGURES), 'timing': timing}, ensure_ascii=False))
