"""Actual target-logit Grad-CAM for all 14 best checkpoints on identical cases."""
import gc
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;REPORT=OUT.parents[1]
for path in (REPORT,REPORT.parent,REPORT.parent.parent):sys.path.insert(0,str(path))
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from paper_style import COLORS,configure,save
from report_inference import evaluator,sha256,grad_cam,normalized_batch,load_prediction_cache

MODELS=list(COLORS);BASELINE='S-NoProg'


def main(render_only=False):
    configure();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    target_dir=OUT/'06_gradcam';target_dir.mkdir(exist_ok=True)
    manifest=json.loads((OUT/'validation_manifest.json').read_text(encoding='utf-8'))
    names=manifest['names'];records=manifest['records'];pred={}
    for model in MODELS:
        for seed in (1,2):
            pred[model,seed],_=load_prediction_cache(model,seed,OUT/'predictions',manifest)
    truth=pred[BASELINE,1]['truth'];a=pred[BASELINE,1]['predictions']==truth;b=pred[BASELINE,2]['predictions']==truth
    rng=np.random.RandomState(20261026)
    permutation=rng.permutation(len(truth));chosen=[]
    rules=[('两seed均正确',a&b),('两seed均错误',~a&~b),
           ('基线seed不一致',a^b),
           ('NoTF seed1纠正基线',~a&(pred['V2-NoTF',1]['predictions']==truth)),
           ('Full seed2纠正基线',~b&(pred['V2-Full',2]['predictions']==truth)),
           ('NoTF seed1相对基线退化',a&(pred['V2-NoTF',1]['predictions']!=truth))]
    for reason,mask in rules:
        used_classes={int(truth[i]) for _,i in chosen}
        candidate=next((int(i) for i in permutation if mask[i] and i not in [j for _,j in chosen] and int(truth[i]) not in used_classes),None)
        if candidate is None:candidate=next(int(i) for i in permutation if mask[i] and i not in [j for _,j in chosen])
        chosen.append((reason,candidate))
    transforms=evaluator._transform(manifest['mean'],manifest['std']);crops=[]
    for _,index in chosen:
        with Image.open(Path(evaluator.VAL_ROOT)/records[index]['relative_path']) as image:
            rgb=transforms.transforms[1](transforms.transforms[0](image.convert('RGB')))
            crops.append(np.asarray(rgb))
    heatmaps={};details=[]
    if render_only:
        previous=json.loads((target_dir/'gradcam_manifest.json').read_text(encoding='utf-8'))
        if [(r['reason'],r['validation_index']) for r in previous['cases']]!=chosen:
            raise ValueError('Selected cases changed; refusing stale Grad-CAM arrays')
        details=previous['details']
        with np.load(target_dir/'gradcam_arrays.npz',allow_pickle=False) as z:
            heatmaps={(m,s,c,mode):z[f'{m}_seed{s}_case{c}_{mode}'].copy()
                      for m in MODELS for s in (1,2) for c in range(6) for mode in ('truth','predicted')}
    for model in ([] if render_only else MODELS):
        for seed in (1,2):
            net,ck,renew,idx2name,path=evaluator.build_and_load(model,seed=seed)
            checkpoint_sha=sha256(path)
            previous=json.loads((OUT/'predictions'/f'predictions_{model}_seed{seed}.json').read_text(encoding='utf-8'))
            if checkpoint_sha!=previous['checkpoint_sha256']:raise ValueError('Checkpoint changed since inference')
            del ck;gc.collect();net=net.cuda().eval()
            for case,(reason,index) in enumerate(chosen):
                image=normalized_batch(crops[case][None],manifest['mean'],manifest['std'],'cuda')
                true_index=int(truth[index])
                true_cam,probs=grad_cam(net,image,true_index)
                guess=int(probs.argmax())
                predicted_cam=true_cam if guess==true_index else grad_cam(net,image,guess)[0]
                heatmaps[model,seed,case,'truth']=true_cam
                heatmaps[model,seed,case,'predicted']=predicted_cam
                details.append(dict(model=model,seed=seed,case=case,validation_index=index,reason=reason,
                    source=records[index],true_index=true_index,true_class=names[true_index],
                    predicted_index_fp32=guess,predicted_class_fp32=names[guess],confidence_fp32=float(probs[guess]),
                    true_probability_fp32=float(probs[true_index]),
                    amp_prediction=int(pred[model,seed]['predictions'][index]),
                    gradient_precision='CUDA FP32, eval mode',layer='resnet_backbone[-1] (layer4 output)',
                    checkpoint_sha256=checkpoint_sha,
                    degenerate_truth_map=bool(np.ptp(true_cam)<1e-8),
                    degenerate_prediction_map=bool(np.ptp(predicted_cam)<1e-8)))
            del net,image;gc.collect();torch.cuda.empty_cache()
            print('GRADCAM_COMPLETE',model,seed,flush=True)
    figures=[]
    for case,(reason,index) in enumerate(chosen):
        for seed in (1,2):
            for mode in ('truth','predicted'):
                fig,axs=plt.subplots(2,4,figsize=(12,8.5))
                axs.flat[0].imshow(crops[case]);axs.flat[0].axis('off')
                axs.flat[0].set_title(f'输入图像\n真值：{names[int(truth[index])].split("-",2)[-1]}',fontsize=9)
                for ax,model in zip(list(axs.flat)[1:],MODELS):
                    row=next(d for d in details if d['model']==model and d['seed']==seed and d['case']==case)
                    cam=heatmaps[model,seed,case,mode]
                    ax.imshow(crops[case]);ax.imshow(cam,cmap='jet',vmin=0,vmax=1,alpha=.43)
                    guess=row['predicted_class_fp32'].split('-',2)[-1]
                    target=names[int(truth[index])].split('-',2)[-1] if mode=='truth' else guess
                    ax.set_title(f'{model}\n预测：{guess} ({row["confidence_fp32"]:.2f})\n梯度目标：{target}',fontsize=8,color=COLORS[model])
                    ax.axis('off')
                fig.suptitle(f'案例{case+1}：{reason}｜seed{seed}｜'+('真实类别Grad-CAM' if mode=='truth' else '预测类别Grad-CAM'),fontsize=10)
                name=f'case{case+1:02d}_seed{seed}_{mode}'
                save(fig,target_dir,name,tight_layout_kwargs={'h_pad':4.0});figures.append({'path':f'06_gradcam/{name}','description':f'{reason} seed{seed} {mode} Grad-CAM'})
    np.savez_compressed(target_dir/'gradcam_arrays.npz',**{f'{m}_seed{s}_case{c}_{mode}':v for (m,s,c,mode),v in heatmaps.items()})
    report={'selection_seed':20261026,'selection_rule':'First unused class in a fixed shuffled validation order satisfying each outcome rule; descriptive purposive cases, not representative accuracy estimates',
            'cases':[dict(case=i,reason=reason,validation_index=index,source=records[index]) for i,(reason,index) in enumerate(chosen)],
            'gradcam':'ReLU(sum_channel(mean_spatial(d target_logit/d activation) * activation)), bilinear to224, per-map minmax',
            'interpretation':'Attribution to the selected score; not segmentation, causal proof, or evidence of spatial attention weights. True/predicted targets differ on errors.',
            'details':details,'figures':figures}
    (target_dir/'gradcam_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'gradcam_figures.json').write_text(json.dumps(figures,ensure_ascii=False,indent=2),encoding='utf-8')
    # Error cases use the exact same, content-hashed manifest as the gradients.
    fig,axs=plt.subplots(2,3,figsize=(12,12))
    for ax,(reason,index),crop in zip(axs.flat,chosen,crops):
        ax.imshow(crop,extent=(0,1,.72,1.72));ax.set_ylim(0,1.8);ax.axis('off');t=names[int(truth[index])].split('-',2)[-1]
        text=[]
        for model in MODELS:
            pp=[names[int(pred[model,s]['predictions'][index])].split('-',2)[-1] for s in (1,2)]
            text.append(f'{model}: {pp[0]} / {pp[1]}')
        ax.set_title(reason+'｜真值：'+t,fontsize=9)
        ax.text(0,.68,'\n'.join(text),va='top',fontsize=8)
    fig.suptitle('相同验证图的预测对照｜每行：seed1 / seed2',fontsize=12)
    save(fig,OUT/'05_errors','same_image_prediction_cases')
    print('ALL_GRADCAM_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true')
    main(parser.parse_args().render_only)
