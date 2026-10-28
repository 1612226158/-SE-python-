"""Collect finished protocol-v2 seed1 runs through the public results reader."""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from ReportChart.chart_data import all_metrics_files, load_model, summarize_run

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
NAMES = ['S-NoProg', 'S-NoSE', 'S-StdSE', 'V2-Full', 'V2-NoTF', 'G-Full-CAWR', 'G-V2']
COLORS = {'S-NoProg': '#587aab', 'V2-Full': '#587aab',
          'G-Full-CAWR': '#568b76', 'G-V2': '#568b76'}


def collect():
    report = {
        'read_time': datetime.now().astimezone().isoformat(timespec='seconds'),
        'reader': 'ReportChart.chart_data.load_model/summarize_run; explicit seed1',
        'time_basis': 'Original time_spend_s: training plus validation; excludes downtime and inter-epoch waits. NoTF epoch63 is retained in raw cumulative time, excluded only from per-epoch duration distributions.',
        'limitations': 'One seed, selected on validation accuracy; no independent test or significance claim. Progressive strategies also change optimizer/scheduler histories and initial weight decay.',
        'protocol': None, 'runs': {}, 'seed2': {}, 'sources': [],
    }
    frames = {}
    for name in NAMES:
        d = load_model(name, 1, verbose=False)
        s = summarize_run(name, 1)
        if s['status'] != 'finished' or list(d.epoch.astype(int)) != list(range(100)):
            raise ValueError(f'{name}: finished continuous epochs 0..99 required')
        if s['official_best_val_acc']['epoch'] != s['metrics_best_val_acc']['epoch']:
            raise ValueError(f'{name}: official/metrics best epoch mismatch')
        if abs(s['official_best_val_acc']['value'] - s['metrics_best_val_acc']['value']) > .0001:
            raise ValueError(f'{name}: official/metrics best accuracy mismatch')
        if report['protocol'] is None:
            report['protocol'] = s['protocol']
        if report['protocol'] != s['protocol']:
            raise ValueError(f'{name}: protocol mismatch')
        rp = SRC / 'runs' / f'results_{name}_seed1.json'
        r = json.loads(rp.read_text(encoding='utf-8'))
        if not (SRC / 'runs' / f'checkpoint_{name}_seed1_best.pth').exists():
            raise FileNotFoundError(f'{name}: best checkpoint absent')
        d['cumulative_hours'] = d.time_spend_s.cumsum() / 3600
        b = d[d.epoch == s['official_best_val_acc']['epoch']].iloc[-1]
        thresholds = {}
        for value in [75, 80, 81, 82, 83, 84]:
            above = d.val_acc.ge(value)
            hit = d[above]
            starts = [i for i in range(96) if bool(above.iloc[i:i + 5].all())]
            thresholds[str(value)] = {
                'first': None if hit.empty else {'epoch': int(hit.iloc[0].epoch), 'hours': float(hit.iloc[0].cumulative_hours)},
                'five_consecutive': None if not starts else {'start_epoch': starts[0], 'confirm_epoch': starts[0] + 4, 'confirm_hours': float(d.iloc[starts[0] + 4].cumulative_hours)},
            }
        timing = d[d.epoch != 63] if name == 'V2-NoTF' else d
        stages = []
        for a, z in [(0, 29), (30, 59), (60, 99)]:
            v = d[d.epoch.between(a, z)]
            bstage = v.loc[v.val_acc.idxmax()]
            stages.append({'epochs': [a, z], 'sum_hours': float(v.time_spend_s.sum() / 3600),
                           'median_minutes': float(v.time_spend_s.median() / 60),
                           'best_acc': float(bstage.val_acc), 'best_epoch': int(bstage.epoch)})
        report['runs'][name] = {'summary': s, 'formal_results': r,
                               'best_epoch_row': json.loads(b.to_json()), 'best_hours': float(b.cumulative_hours),
                               'thresholds': thresholds, 'stages': stages,
                               'timing_distribution': {'valid_epochs': len(timing), 'excluded_epochs': [63] if name == 'V2-NoTF' else [], 'mean_minutes': float(timing.time_spend_s.mean() / 60)},
                               'metrics_all_epochs': json.loads(d.to_json(orient='records'))}
        frames[name] = d
        try:
            ds = load_model(name, 2, verbose=False)
            ss = summarize_run(name, 2)
            report['seed2'][name] = {'status': ss['status'], 'rows': len(ds), 'last_epoch': int(ds.epoch.max())}
        except FileNotFoundError:
            report['seed2'][name] = {'status': 'not_started', 'rows': 0, 'last_epoch': None}
        report['sources'].append({'path': str(rp), 'sha256': hashlib.sha256(rp.read_bytes()).hexdigest()})
    for f in all_metrics_files():
        p = Path(f)
        report['sources'].append({'path': str(p), 'sha256_at_capture': hashlib.sha256(p.read_bytes()).hexdigest()})

    def delta(g, f):
        gr, fr = report['runs'][g], report['runs'][f]
        gs, fs = gr['summary'], fr['summary']
        gb, fb = gs['at_metrics_best_val'], fs['at_metrics_best_val']
        return {'accuracy_delta_pp': gb['val_acc'] - fb['val_acc'],
                'macro_f1_delta_pp': 100 * (gb['val_macro_f1'] - fb['val_macro_f1']),
                'weighted_f1_delta_pp': 100 * (gb['val_weighted_f1'] - fb['val_weighted_f1']),
                'time_saving_percent': 100 * (1 - gs['cumulative_train_hours'] / fs['cumulative_train_hours']),
                'best_hours_g': gr['best_hours'], 'best_hours_f': fr['best_hours']}

    report['comparisons'] = {'G-V2_minus_V2-Full': delta('G-V2', 'V2-Full'),
                             'G-V2_minus_G-Full-CAWR': delta('G-V2', 'G-Full-CAWR'),
                             'G-Full-CAWR_minus_S-NoProg': delta('G-Full-CAWR', 'S-NoProg')}
    (OUT / 'seed1_results_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return frames, report


def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot(frames, report):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, key, title in zip(axs, ['val_acc', 'val_macro_f1'], ['Best validation accuracy', 'Macro-F1 at selected epoch']):
        vals = [report['runs'][n]['summary']['at_metrics_best_val'][key] * (100 if key != 'val_acc' else 1) for n in NAMES]
        ax.barh(NAMES, vals, color=['#788fa8' if n != 'V2-NoTF' else '#568b76' for n in NAMES])
        ax.invert_yaxis()
        ax.set(xlim=(0, 90), xlabel='Percent', title=title)
        for i, v in enumerate(vals):
            ax.text(v + .4, i, f'{v:.4f}', va='center', fontsize=8)
    fig.suptitle('All seven seed1 runs | protocol v2, 252 classes, 100 epochs each')
    save(fig, '01_seed1_metrics')

    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for n in ['V2-Full', 'G-V2']:
        d = frames[n]
        for ax, key in zip(axs, ['train_loss', 'val_loss', 'lr']):
            ax.plot(d.epoch, d[key], color=COLORS[n], label=n)
    for ax, label in zip(axs, ['Training loss', 'Validation loss', 'Learning rate (log)']):
        ax.set_ylabel(label); ax.grid(alpha=.2)
        for e in [30, 60]: ax.axvline(e, color='#aaaaaa', linestyle=':', linewidth=1)
    axs[0].legend(); axs[0].set_title('Same 49-token head | complete training-strategy comparison')
    axs[2].set_yscale('log'); axs[2].set_xlabel('Epoch (zero-based)')
    save(fig, '02_gv2_full_loss_lr')

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for n in ['V2-Full', 'G-V2']:
        d = frames[n]
        axs[0].plot(d.epoch, d.val_acc, color=COLORS[n], label=n)
        axs[1].plot(d.cumulative_hours, d.val_acc.cummax(), color=COLORS[n], label=n, drawstyle='steps-post')
    for e in [30, 60]: axs[0].axvline(e, color='#aaaaaa', linestyle=':', linewidth=1)
    axs[0].set(xlabel='Epoch (zero-based)', title='Validation accuracy by epoch')
    axs[1].set(xlabel='Cumulative recorded time (h)', title='Best observed accuracy by time budget')
    for ax in axs: ax.set_ylabel('Validation accuracy (%)'); ax.grid(alpha=.2); ax.legend(loc='lower right')
    fig.suptitle('V2-Full / G-V2 | seed1, all durations retained; not a controlled hardware benchmark')
    save(fig, '03_gv2_full_accuracy_time')

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, pair, title in zip(axs, [('S-NoProg', 'V2-Full'), ('G-Full-CAWR', 'G-V2')], ['Full unfreezing: old / V2 head', 'Progressive strategy: old / V2 head']):
        for i, n in enumerate(pair):
            d = frames[n]
            ax.plot(d.epoch, d.val_acc, color=['#8e8b8b', '#587aab'][i], label=n)
        ax.set(xlabel='Epoch (zero-based)', ylabel='Validation accuracy (%)', title=title)
        ax.grid(alpha=.2); ax.legend(loc='lower right')
    fig.suptitle('Same multi-region SE | complete head-route comparison, not pure attention ablation')
    save(fig, '04_head_strategy_routes')


if __name__ == '__main__':
    data, results = collect()
    plot(data, results)
    print(json.dumps({'read_time': results['read_time'], 'comparisons': results['comparisons'], 'seed2': results['seed2'],
                      'G-V2_stages': results['runs']['G-V2']['stages'], 'G-V2_thresholds': results['runs']['G-V2']['thresholds']}, ensure_ascii=False))
