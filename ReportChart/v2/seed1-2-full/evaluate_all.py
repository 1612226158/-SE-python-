"""Full validation for 14 completed runs; explicit seed, shared public interfaces."""
import gc
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
REPORT = OUT.parents[1]
for path in (REPORT, REPORT.parent, REPORT.parent.parent):
    sys.path.insert(0, str(path))
import torch
from paper_style import COLORS
from chart_data import load_run_frames, best_ckpt_path, all_metrics_files
from report_inference import validation_manifest, preload_validation_rgb, predict_validation, sha256


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    frames, summaries = load_run_frames(list(COLORS))
    protected = [REPORT.parent / name for name in ('train.py','run_queue.py','ResNet.py','ResNetTransformer.py','SE_Block.py','taxonomy.py','experiment_config.json')]
    protected += [REPORT.parent.parent / '毕业设计/document_new.tex',
                  REPORT.parent.parent / '毕业设计/ModifyLog/v3实验预实施方案.md']
    protected += [Path(p) for p in all_metrics_files()]
    protected += list((REPORT.parent/'runs').glob('results_*.json'))
    # Weight content hashes are stored per-run by the public inference cache.
    recovery = {str(p):sha256(p) for p in protected}
    protection_file = REPORT.parent.parent/'毕业设计/Handover/check_artifacts/2026-10-26/双seed完整结果分析/source_protection.json'
    protection_file.parent.mkdir(parents=True, exist_ok=True)
    if protection_file.exists():
        if json.loads(protection_file.read_text(encoding='utf-8')) != recovery:
            raise ValueError('Protected source changed since initial snapshot')
    else:
        protection_file.write_text(json.dumps(recovery, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = validation_manifest()
    if manifest['num_samples'] != 14603:
        raise ValueError('Validation denominator mismatch')
    manifest_path = OUT/'validation_manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old['fingerprint'] != manifest['fingerprint']:
            raise ValueError('Validation data changed')
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    needed = any(not (OUT/'predictions'/f'predictions_{m}_seed{s}.npz').exists() for m,s in frames)
    rgb = preload_validation_rgb(manifest) if needed else None
    for model, seed in frames:
        predict_validation(model, seed, manifest, OUT/'predictions', rgb,
                           device='cuda', batch_size=32, amp=True)
    del rgb; gc.collect()
    if any(sha256(p) != value for p,value in recovery.items()):
        raise ValueError('Protected source changed')
    print('ALL_14_FULL_VALIDATIONS_COMPLETE; protected sources unchanged', flush=True)


if __name__ == '__main__':
    main()
