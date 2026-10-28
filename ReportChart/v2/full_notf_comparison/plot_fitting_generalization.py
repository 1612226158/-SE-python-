"""Full/NoTF seed1 fitting diagnostics via the unified chart_data interface."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib.pyplot as plt
from plot_full_notf_comparison import load, save, lines, HERE, PAIR, COLORS


def main():
    frames, summaries, results, fingerprints, protocol = load()
    report = {'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'seed': 1, 'epoch_index': 'zero-based, 0–99', 'protocol': protocol,
              'sources': {m: fingerprints[m] for m in PAIR}, 'models': {},
              'limits': ['One seed and one validation split; no independent test/domain evaluation.',
                         'Train metrics use weighted sampling, augmentation and train mode; gaps are descriptive.',
                         'Loss uses label smoothing; its minimum need not match maximum accuracy.',
                         'Timing is recorded training-plus-validation epoch duration, not strict GPU benchmarking.',
                         'NoTF epoch63 excluded only from timing mean; no time interpolation.',
                         'Threshold times below use actual cumulative records before NoTF epoch63.']}
    for m in PAIR:
        d = frames[m]
        assert np.isfinite(d[['train_loss', 'val_loss', 'train_acc', 'val_acc']]).all().all()
        times = d.loc[d.epoch.ne(63)] if m == 'V2-NoTF' else d
        if m == 'V2-NoTF':
            assert abs(float(d.loc[d.epoch.eq(63), 'time_spend_s'].iloc[0]) - 5728.7926187515) < 1e-5
        out = {'average_epoch_minutes': float(times.time_spend_s.mean() / 60),
               'timing_n': len(times), 'accuracy_first_crossings': {},
               'accuracy_first_five_consecutive_epochs': {}, 'cycle_loss_minima': []}
        for threshold in [80, 82, 83]:
            ep = int(d.loc[d.val_acc.ge(threshold), 'epoch'].iloc[0])
            if m == 'V2-NoTF':
                assert ep < 63
            out['accuracy_first_crossings'][str(threshold)] = {
                'epoch': ep, 'completed_rounds': ep + 1,
                'recorded_hours': float(d.loc[d.epoch.le(ep), 'time_spend_s'].sum() / 3600)}
            mask = d.val_acc.ge(threshold).to_numpy()
            start = next((i for i in range(len(mask) - 4) if mask[i:i+5].all()), None)
            out['accuracy_first_five_consecutive_epochs'][str(threshold)] = start
        for a, b in [(0, 4), (5, 14), (15, 34), (35, 74), (75, 99)]:
            q = d.loc[d.epoch.between(a, b)]
            out['cycle_loss_minima'].append({'range': [a, b], 'val_loss': float(q.val_loss.min())})
        best_ep = summaries[m]['official_best_val_acc']['epoch']
        row = d.loc[d.epoch.eq(best_ep)].iloc[0]
        out['at_best_accuracy'] = {k: float(row[k]) for k in
                                  ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'val_macro_f1']}
        out['at_best_accuracy']['gap_pp'] = float(row.train_acc - row.val_acc)
        for name, a, b in [('pre_restart', 60, 74), ('last_cycle', 75, 99)]:
            q = d.loc[d.epoch.between(a, b)]
            out[name] = {'epochs': [a, b], 'mean_val_acc': float(q.val_acc.mean()),
                         'mean_val_loss': float(q.val_loss.mean()),
                         'mean_gap_pp': float((q.train_acc - q.val_acc).mean())}
        out['final'] = {k: float(d.iloc[-1][k]) for k in
                        ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'val_macro_f1']}
        report['models'][m] = out
    f, n = [frames[m] for m in PAIR]
    report['NoTF_lower_train_loss_epochs'] = int((n.train_loss < f.train_loss).sum())
    report['NoTF_lower_val_loss_epochs'] = int((n.val_loss < f.val_loss).sum())

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for ax, field, title in zip(axs.flat[:2], ['train_loss', 'val_loss'], ['训练 Loss', '验证 Loss']):
        lines(ax, frames, PAIR, field)
        ax.set_title(title)
        ax.set_ylabel('记录 Loss（含 label smoothing）')
        ax.legend(loc='upper right')
    gap_frames = {m: d.assign(gap=d.train_acc-d.val_acc) for m, d in frames.items()}
    lines(axs[1, 0], gap_frames, PAIR, 'gap')
    axs[1, 0].set_title('训练 Accuracy − 验证 Accuracy')
    axs[1, 0].set_ylabel('百分点；训练与验证分布/模式不同')
    lines(axs[1, 1], frames, PAIR, 'val_acc')
    axs[1, 1].set_title('验证 Accuracy（完整100轮）')
    axs[1, 1].set_ylabel('Accuracy (%)')
    save(fig, '08_fitting_generalization',
         'seed=1；竖虚线为CAWR重启；训练含采样/增强，gap仅作描述；验证集优势不能代替独立测试结论。')

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(3)
    for j, m in enumerate(PAIR):
        pts = report['models'][m]['accuracy_first_crossings']
        for ax, field, fmt in [(axs[0], 'completed_rounds', '{:.0f}'), (axs[1], 'recorded_hours', '{:.2f}')]:
            vals = [pts[str(t)][field] for t in [80, 82, 83]]
            bars = ax.bar(x+(j-.5)*.34, vals, width=.34, color=COLORS[m], label=m)
            ax.bar_label(bars, labels=[fmt.format(v) for v in vals], padding=3, fontsize=9)
            ax.set_xticks(x, ['80%', '82%', '83%'])
            ax.set_xlabel('首次达到的验证 Accuracy')
            ax.legend()
            ax.set_ylim(0, max(ax.get_ylim()[1], max(vals)*1.2))
    axs[0].set_ylabel('已完成轮数（epoch + 1）')
    axs[0].set_title('首次达到同准确率：训练轮数')
    axs[1].set_ylabel('累计记录耗时（小时）')
    axs[1].set_title('首次达到同准确率：实际记录耗时')
    save(fig, '09_accuracy_threshold_speed',
         '仅首次跨阈值，不代表永久保持；全部NoTF阈值在epoch63之前，无异常耗时或插值；历史耗时非硬件基准。')
    (HERE / 'fitting_generalization_analysis.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report['models'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
