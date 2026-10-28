"""Protocol-v2 seed1 progressive/full comparison; preserve all recorded durations."""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ReportChart.chart_data import load_model, summarize_run, all_metrics_files

OUT = Path(__file__).resolve().parent
NAMES = ['G-Full-CAWR', 'S-NoProg']
COLORS = ['#568b76', '#587aab']
data = {}
report = {'seed': 1, 'time_basis': 'sum of original time_spend_s, all epochs retained; excludes downtime and inter-epoch waits',
          'limitations': 'single seed, observed wall times, different optimizer/scheduler histories; not a controlled hardware benchmark', 'runs': {}}
for name in NAMES:
    d = load_model(name, seed=1, verbose=False)
    summary = summarize_run(name, 1)
    if summary['status'] != 'finished' or list(d.epoch.astype(int)) != list(range(100)):
        raise ValueError(f'{name}: requires finished continuous 0..99')
    d['hours'] = d.time_spend_s.cumsum() / 3600
    data[name] = d
    thresholds = {}
    for threshold in [75, 80, 81, 82, 82.5]:
        r = d[d.val_acc >= threshold].iloc[0]
        thresholds[str(threshold)] = {'epoch': int(r.epoch), 'hours': float(r.hours)}
    report['runs'][name] = {'summary': summary, 'best_hours': float(d.loc[d.val_acc.idxmax(), 'hours']), 'thresholds': thresholds}
if report['runs'][NAMES[0]]['summary']['protocol'] != report['runs'][NAMES[1]]['summary']['protocol']:
    raise ValueError('protocol mismatch')
files = [Path(p) for p in all_metrics_files()] + [ROOT / 'runs' / f'results_{n}_seed1.json' for n in NAMES]
report['sources'] = [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
for name, color in zip(NAMES, COLORS):
    d = data[name]
    axes[0].plot(d.epoch, d.train_loss, label=name, color=color)
    axes[1].plot(d.epoch, d.val_loss, label=name, color=color)
    axes[2].plot(d.epoch, d.lr, label=name, color=color)
for ax, label in zip(axes, ['Training loss', 'Validation loss', 'Learning rate (log)']):
    ax.set_ylabel(label)
    ax.grid(alpha=.2)
    for e in [30, 60]: ax.axvline(e, color='#aaaaaa', linestyle=':', linewidth=1)
axes[0].legend()
axes[0].set_title('Protocol v2, seed 1 | dotted lines: G unfreezing at epochs 30 / 60')
axes[2].set_yscale('log')
axes[2].set_xlabel('Epoch (zero-based)')
save(fig, '01_loss_lr')

fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
for name, color in zip(NAMES, COLORS):
    d = data[name]
    ax.plot(d.epoch, d.val_acc, color=color, label=name)
    r = d.loc[d.val_acc.idxmax()]
    ax.scatter(r.epoch, r.val_acc, color=color, s=25)
    ax.text(.03, .96 if name == NAMES[0] else .89,
            f'{name}: {r.val_acc:.4f}% @ {int(r.epoch)}',
            transform=ax.transAxes, va='top', fontsize=9, color=color)
for e in [30, 60]: ax.axvline(e, color='#aaaaaa', linestyle=':', linewidth=1)
ax.set(xlabel='Epoch (zero-based)', ylabel='Validation accuracy (%)', ylim=(20, 90), title='Protocol v2, seed 1 | validation accuracy')
ax.grid(alpha=.2)
ax.legend(loc='lower right')
save(fig, '02_val_accuracy')

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for name, color in zip(NAMES, COLORS):
    d = data[name]
    axes[0].plot(d.hours, d.val_acc, color=color, label=name)
    axes[1].plot(d.hours, d.val_acc.cummax(), color=color, label=name, drawstyle='steps-post')
for ax, title in zip(axes, ['Per-epoch validation accuracy', 'Best observed accuracy by time budget']):
    ax.set(xlabel='Cumulative recorded training + validation time (h)', ylabel='Validation accuracy (%)', title=title, ylim=(55, 86))
    ax.grid(alpha=.2)
axes[0].legend(loc='lower right')
fig.suptitle('Protocol v2, seed 1 | all durations retained; downtime / inter-epoch waits excluded')
save(fig, '03_time_accuracy')
(OUT / 'comparison_analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved 3 PNG/PDF pairs and comparison_analysis.json:', OUT)
