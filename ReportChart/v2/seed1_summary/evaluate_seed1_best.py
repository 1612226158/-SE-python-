"""CPU validation of the selected seed1 winner using the existing evaluator."""
import gc
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC / 'ReportChart'))
sys.path.insert(0, str(SRC))
import torch
import numpy as np
import eval_confusion_matrix as evaluator
from ReportChart.chart_data import best_ckpt_path, summarize_run

MODEL = 'V2-NoTF'
OUT = Path(__file__).resolve().parent / 'evaluation'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main(report_only=False):
    summary = summarize_run(MODEL, 1)
    ckpt_path = Path(best_ckpt_path(MODEL))
    expected = SRC / 'runs' / f'checkpoint_{MODEL}_seed1_best.pth'
    if ckpt_path.resolve() != expected.resolve():
        raise ValueError('The evaluator must use the selected seed1 checkpoint')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    for key, value in [('config_id', MODEL), ('seed', 1), ('training_protocol_version', 2),
                       ('num_classes', 252), ('label_mapping_hash', summary['protocol']['label_mapping_hash'])]:
        if ckpt.get(key) != value:
            raise ValueError(f'Checkpoint identity mismatch: {key}={ckpt.get(key)}')
    if int(ckpt['epoch']) != int(summary['official_best_val_acc']['epoch']):
        raise ValueError('Checkpoint best epoch mismatch')
    source = {'checkpoint_path': str(ckpt_path), 'checkpoint_sha256': sha256(ckpt_path),
              'checkpoint_epoch': int(ckpt['epoch']), 'protocol': summary['protocol'],
              'model_metadata': ckpt.get('model_metadata'), 'evaluator_path': str(Path(evaluator.__file__)),
              'evaluator_sha256': sha256(Path(evaluator.__file__))}
    del ckpt
    gc.collect()
    OUT.mkdir(exist_ok=True)
    # Only redirect the evaluator's analysis outputs; data/model source paths stay unchanged.
    evaluator._HERE = str(OUT)
    if report_only:
        previous = json.loads((OUT / 'evaluation_summary.json').read_text(encoding='utf-8'))
        started = previous['started']
        source['inference_evaluator_sha256'] = previous.get('inference_evaluator_sha256', previous['evaluator_sha256'])
        source['inference_completed'] = previous.get('inference_completed', previous['completed'])
        cm, names = evaluator.load_cm(MODEL)
    else:
        torch.set_num_interop_threads(1)
        started = datetime.now().astimezone().isoformat(timespec='seconds')
        source['inference_evaluator_sha256'] = source['evaluator_sha256']
        cm, names = evaluator.stage_infer(MODEL, threads=4, dump=True)
    if cm.shape != (252, 252) or int(cm.sum()) != 14603:
        raise ValueError(f'Unexpected full-validation denominator: shape={cm.shape}, n={cm.sum()}')
    _, _, rows = evaluator.stage_report(MODEL)
    evaluator.stage_plot(MODEL)
    n = int(cm.sum())
    accuracy = float(np.trace(cm) / n * 100)
    macro = float(np.mean([r['f1'] for r in rows]) * 100)
    weighted = float(sum(r['f1'] * r['support'] for r in rows) / n * 100)
    ref = summary['at_metrics_best_val']
    source.update({'started': started, 'completed': datetime.now().astimezone().isoformat(timespec='seconds'),
                   'model': MODEL, 'seed': 1, 'device': 'CPU', 'threads': 4,
                   'validation_preprocess': 'Resize(256), CenterCrop(224), ToTensor, Normalize using the public training-mean/std cache',
                   'num_classes': len(names), 'num_samples': n, 'num_errors': n - int(np.trace(cm)),
                   'accuracy': accuracy, 'macro_f1_percent': macro, 'weighted_f1_percent': weighted,
                   'training_best_accuracy': ref['val_acc'], 'accuracy_difference_pp': accuracy - ref['val_acc'],
                   'precision_note': 'CPU FP32 inference; original training validation used CUDA autocast. Do not replace original training metrics with these numbers.',
                   'top_confusions': evaluator.top_confusions(cm, names, 20),
                   'worst_classes_support_at_least_30': sorted([r for r in rows if r['support'] >= 30], key=lambda r: (r['recall'], -r['support']))[:10]})
    if abs(source['accuracy_difference_pp']) > .1:
        raise ValueError(f'CPU evaluation differs materially from training: {source["accuracy_difference_pp"]} pp')
    off = cm.copy()
    np.fill_diagonal(off, 0)
    errors = int(off.sum())
    directed = sorted(off[off > 0].tolist(), reverse=True)
    parents = [name.split('-', 2)[1] for name in names]
    if len(set(parents)) != 4:
        raise ValueError('Expected four garbage-parent labels after public/private prefix')
    same_parent_errors = sum(int(off[i, j]) for i in range(len(names)) for j in range(len(names)) if parents[i] == parents[j])
    source['error_distribution'] = {
        'top_directed_counts': {str(k): int(sum(directed[:k])) for k in [10, 20, 50]},
        'top_directed_share_percent': {str(k): float(sum(directed[:k]) / errors * 100) for k in [10, 20, 50]},
        'same_parent_errors': same_parent_errors,
        'same_parent_share_percent': float(same_parent_errors / errors * 100),
        'cross_parent_errors': errors - same_parent_errors,
        'parent_labels': sorted(set(parents)),
        'note': 'Derived from full current-label CM; directed cells, not unordered pairs. No assertion of label errors or irreducible semantic ambiguity.',
    }
    source['heatmap_normalization'] = 'Selected class submatrix divided by full true-class support, not selected-submatrix row sum; diagonal equals full-class recall.'
    source['plot_evaluator_sha256'] = sha256(Path(evaluator.__file__))
    source['outputs'] = [{'path': str(p), 'sha256': sha256(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'evaluation_summary.json']
    (OUT / 'evaluation_summary.json').write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Full CPU evaluation complete:', json.dumps({k: source[k] for k in ['accuracy', 'macro_f1_percent', 'weighted_f1_percent', 'accuracy_difference_pp', 'num_errors']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-only', action='store_true', help='Reuse completed CM; regenerate reports/plots without inference')
    main(parser.parse_args().report_only)
