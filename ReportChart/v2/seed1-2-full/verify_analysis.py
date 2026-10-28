"""Read-only cross-checks of source identity, denominators, group ranking and exports."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;REPORT=OUT.parents[1]
sys.path.insert(0,str(REPORT))
import numpy as np
import pandas as pd
import pymupdf
from chart_data import load_run_frames,results_best,best_ckpt_path,window_statistics
from paper_style import COLORS
from report_inference import sha256, evaluator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protection-manifest',type=Path,help='Explicit source guard for an authorized later revision')
    args=parser.parse_args()
    protected_file=args.protection_manifest or REPORT.parent.parent/'毕业设计/Handover/check_artifacts/2026-10-26/双seed完整结果分析/source_protection.json'
    if args.protection_manifest is None and not protected_file.exists():protected_file=OUT/'source_protection.json'
    protected=json.loads(protected_file.read_text(encoding='utf-8'))
    if 'protected_files' in protected:protected=protected['protected_files']
    assert all(sha256(path)==value for path,value in protected.items()),'Protected source changed'
    frames,summaries=load_run_frames(list(COLORS))
    manifest=json.loads((OUT/'validation_manifest.json').read_text(encoding='utf-8'))
    manifest_without_hash={k:v for k,v in manifest.items() if k!='fingerprint'}
    expected_hash=hashlib.sha256(json.dumps(manifest_without_hash,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    assert expected_hash==manifest['fingerprint']
    n=manifest['num_classes'];expected_truth=np.array([r['true_index'] for r in manifest['records']])
    _,current_renew,current_n=evaluator.build_truth_map()
    expected_names=[name for name,index in sorted(current_renew.items(),key=lambda item:int(item[1]))]
    assert current_n==n and manifest['names']==expected_names,'Class names must use numeric output indices'
    support=np.bincount(expected_truth,minlength=n);assert len(expected_truth)==14603 and np.all(support>0)
    class_table=pd.read_csv(OUT/'per_class_metrics_14runs.csv')
    assert len(class_table)==14*n
    baseline=[];max_delta=0.;checks=0
    for (model,seed),frame in frames.items():
        assert results_best(model,seed)[2]==seed
        assert f'seed{seed}_best.pth' in best_ckpt_path(model,seed=seed)
        assert best_ckpt_path(model,seed=987654) is None
        assert results_best(model,seed=987654)==(None,None,None)
        assert window_statistics(frame,0,45)['n_epochs']==45 and window_statistics(frame,75,100)['n_epochs']==25
        p=OUT/'predictions'/f'predictions_{model}_seed{seed}.npz'
        with np.load(p,allow_pickle=False) as z:
            truth=z['truth'];prediction=z['predictions'];prob=z['probabilities'];cm=z['cm']
            identity=json.loads(str(z['identity']))
            assert np.array_equal(truth,expected_truth)
            assert prob.shape==(14603,n) and np.isfinite(prob).all()
            assert (prob>=0).all() and (prob<=1).all()
            np.testing.assert_allclose(prob.sum(1),1,atol=1e-5)
            assert np.array_equal(prediction,prob.argmax(1))
            expected_cm=np.bincount(truth*n+prediction,minlength=n*n).reshape(n,n)
            assert np.array_equal(cm,expected_cm) and np.array_equal(cm.sum(1),support)
            assert identity['manifest_fingerprint']==manifest['fingerprint']
            assert sha256(best_ckpt_path(model,seed=seed))==identity['checkpoint_sha256']
            csv=class_table[(class_table.model==model)&(class_table.seed==seed)].sort_values('class_index')
            np.testing.assert_allclose(csv.recall,np.diag(cm)/support,atol=1e-12)
            if model=='S-NoProg':baseline.append(np.diag(cm)/support)
            acc=np.trace(cm)/14603*100;max_delta=max(max_delta,abs(acc-summaries[model,seed]['official_best_val_acc']['value']))
        checks+=1
    assert max_delta<.1
    baseline=np.mean(baseline,axis=0);order=np.lexsort((np.arange(n),-support,-baseline))
    ranking=pd.read_csv(OUT/'fixed_baseline_class_ranking.csv')
    assert np.array_equal(ranking.class_index.to_numpy(),order)
    groups=json.loads((OUT/'prediction_analysis.json').read_text(encoding='utf-8'))['fixed_groups']
    assert all(len(g)==50 for g in groups.values())
    assert len(set(sum(groups.values(),[])))==150
    grad=json.loads((OUT/'06_gradcam/gradcam_manifest.json').read_text(encoding='utf-8'))
    assert len(grad['details'])==14*6 and len(grad['figures'])==24
    for item in grad['details']:
        assert item['true_class']==manifest['names'][item['true_index']]
        assert item['predicted_class_fp32']==manifest['names'][item['predicted_index_fp32']]
        assert item['source']==manifest['records'][item['validation_index']]
    with np.load(OUT/'06_gradcam/gradcam_arrays.npz',allow_pickle=False) as z:
        assert len(z.files)==14*6*2
        for key in z.files:
            image=z[key];assert image.shape==(224,224) and np.isfinite(image).all()
            assert image.min()>=0 and image.max()<=1.000001
    figures=json.loads((OUT/'figure_index.json').read_text(encoding='utf-8'))
    for figure in figures:
        for ext in ('.pdf','.png'):assert (OUT/(figure['path']+ext)).is_file()
        with pymupdf.open(OUT/(figure['path']+'.pdf')) as pdf:
            assert len(pdf)==1 and len(pdf[0].get_text().strip())>10
            assert '\ufffd' not in pdf[0].get_text()
    with pymupdf.open(OUT/'双seed完整分析图册.pdf') as pdf:
        assert len(pdf)==len(figures) and len(pdf.get_toc())>len(figures)
    print(json.dumps({'ok':True,'runs':checks,'training_rows':sum(len(f) for f in frames.values()),
        'prediction_rows':checks*14603,'classes':n,'class_metric_rows':len(class_table),
        'fixed_rank_groups':{k:len(v) for k,v in groups.items()},'gradcam_arrays':168,
        'gradcam_degenerate_truth_maps':sum(d['degenerate_truth_map'] for d in grad['details']),
        'gradcam_amp_vs_fp32_prediction_disagreements':sum(d['predicted_index_fp32']!=d['amp_prediction'] for d in grad['details']),
        'max_inference_vs_training_accuracy_delta_pp':max_delta,'figure_pairs':len(figures),
        'protected_files':len(protected),'protected_unchanged':True},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
