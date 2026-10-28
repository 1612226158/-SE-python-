"""Reproducible full seed1/2 report. Read metrics only through chart_data.

Usage: python -B render_analysis.py --stage metrics|predictions|all
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
REPORT = OUT.parents[1]
for directory in (REPORT, REPORT.parent, REPORT.parent.parent):
    sys.path.insert(0, str(directory))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chart_data import load_run_frames, window_statistics, all_metrics_files
from paper_style import COLORS, configure, save
from report_inference import sha256, calibration_statistics, evaluator, load_prediction_cache, paired_prediction_statistics

MODELS = list(COLORS)
BASELINE = 'S-NoProg'
GROUPS = {
    'se': ('SE结构', ['S-NoSE','S-StdSE','S-NoProg']),
    'heads': ('全解冻模型头', ['S-NoProg','V2-Full','V2-NoTF']),
    'progressive_global': ('原单token头训练策略', ['S-NoProg','G-Full-CAWR']),
    'progressive_spatial': ('空间头训练策略', ['V2-Full','G-V2']),
}
FIGURES = []


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                         default=lambda x: x.item() if isinstance(x,np.generic) else str(x)), encoding='utf-8')


def export(fig, directory, name, description):
    target = OUT/directory; target.mkdir(exist_ok=True)
    save(fig, target, name)
    FIGURES.append({'path':f'{directory}/{name}', 'description':description})
    print('FIGURE',directory,name,flush=True)


def axes_style(ax, xlabel='训练轮次', ylabel='验证损失'):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(ls='--', alpha=.22)


def two_seed_curve(ax, frames, model, column, start=0, stop=100, x_time=False, envelope=False):
    ys = []
    for seed in (1,2):
        d = frames[model,seed]
        sub = d[(d.epoch >= start)&(d.epoch < stop)]
        y = sub[column].to_numpy(float)
        if envelope: y = np.minimum.accumulate(y)
        if x_time:
            x = d.time_spend_s.cumsum().to_numpy(float)[start:stop]/3600
            ax.plot(x, y, color=COLORS[model], ls='-' if seed==1 else '--',
                    label=f'{model} s{seed}', lw=1.3)
        else:
            ys.append(y)
    if not x_time:
        x=np.arange(start,stop); ys=np.asarray(ys)
        ax.plot(x,ys.mean(0),color=COLORS[model],label=model)
        ax.fill_between(x,ys.min(0),ys.max(0),color=COLORS[model],alpha=.13)


def metrics_report():
    frames,summaries=load_run_frames(MODELS)
    records=[];window_records=[];loss_targets=[];late=[];restart=[]
    for (model,seed),d in frames.items():
        for a,b in [(0,45),(45,75),(75,100),(0,75),(0,100)]:
            w=window_statistics(d,a,b)
            for field in ['train_loss','val_loss']:
                window_records.append(dict(model=model,seed=seed,window=f'{a}-{b-1}',
                                           field=field,hours=w['hours'],**w[field]))
        hours=d.time_spend_s.cumsum()/3600
        for target in [2.0,1.8,1.7]:
            hit=d[d.val_loss<=target]
            e=int(hit.iloc[0].epoch) if len(hit) else None
            loss_targets.append(dict(model=model,seed=seed,target_loss=target,epoch=e,
                                     hours=float(hours.iloc[e]) if e is not None else None))
        prior=d[d.epoch<75];tail=d[d.epoch>=75]
        late.append(dict(model=model,seed=seed,pre75_best_loss=float(prior.val_loss.min()),
                         late_best_loss=float(tail.val_loss.min()),
                         additional_best_loss_drop=float(prior.val_loss.min()-tail.val_loss.min()),
                         within_late_drop=float(tail.iloc[0].val_loss-tail.iloc[-1].val_loss),
                         final5_mean=float(tail.tail(5).val_loss.mean()),
                         first5_mean=float(tail.head(5).val_loss.mean()),
                         new_loss_best=bool(tail.val_loss.min()<prior.val_loss.min())))
        for epoch in range(1,100):
            if d.iloc[epoch].lr > d.iloc[epoch-1].lr*2:
                threshold=float(d.iloc[epoch-1].val_loss)
                recovered=d[(d.epoch>=epoch)&(d.val_loss<=threshold)]
                restart.append(dict(model=model,seed=seed,epoch=epoch,
                                    val_loss_jump=float(d.iloc[epoch].val_loss-threshold),
                                    recovery_epoch=int(recovered.iloc[0].epoch) if len(recovered) else None))
        c=d.copy();c['cumulative_hours']=hours;records.append(c)
    pd.concat(records).to_csv(OUT/'training_metrics_1400rows.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(window_records).to_csv(OUT/'window_statistics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(loss_targets).to_csv(OUT/'loss_target_times.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(late).to_csv(OUT/'late_improvement.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(restart).to_csv(OUT/'lr_restart_response.csv',index=False,encoding='utf-8-sig')
    for tag,(title,models) in GROUPS.items():
        for phase,start,stop in [('full',0,100),('early',0,45),('late',75,100)]:
            fig,axs=plt.subplots(1,2,figsize=(10.4,4.5))
            for ax,field,label in zip(axs,['train_loss','val_loss'],['训练损失','验证损失']):
                for model in models:two_seed_curve(ax,frames,model,field,start,stop)
                ax.set_title(label);axes_style(ax,ylabel=label)
                if phase=='full':
                    for e in [30,45,60,75]:ax.axvline(e,color='#888888',ls=':',lw=.7)
            axs[0].legend(fontsize=8)
            fig.suptitle(f'{title}：训练与验证损失（epoch {start}—{stop-1}）')
            export(fig,'01_dynamics',f'{tag}_{phase}_loss',f'{title}训练与验证损失（epoch {start}—{stop-1}）')
        fig,axs=plt.subplots(1,2,figsize=(10.4,4.5),sharey=True)
        for ax,seed in zip(axs,(1,2)):
            for model in models:
                d=frames[model,seed];ax.plot(d.epoch,d.lr,color=COLORS[model],label=model)
            ax.set_yscale('log');axes_style(ax,ylabel='学习率（log）');ax.set_title(f'seed{seed}')
            for e in [5,15,30,35,60,75,95]:ax.axvline(e,color='#aaaaaa',ls=':',lw=.6)
        axs[0].legend(fontsize=8);fig.suptitle(f'{title}：学习率曲线')
        export(fig,'01_dynamics',f'{tag}_lr',f'{title}同组学习率两seed')
        fig,axs=plt.subplots(1,2,figsize=(10.4,4.5))
        for ax,field,label in zip(axs,['train_loss','val_loss'],['训练损失','验证损失']):
            for model in models:two_seed_curve(ax,frames,model,field,x_time=True)
            axes_style(ax,xlabel='累计训练耗时（小时）',ylabel=label)
            ax.set_title(label)
        axs[0].legend(fontsize=7);fig.suptitle(f'{title}：损失与训练耗时')
        export(fig,'02_efficiency',f'{tag}_loss_time',f'{title}损失与累计训练耗时')
    for phase,start,stop in [('full',0,100),('early',0,45),('late',75,100)]:
        fig,axs=plt.subplots(2,3,figsize=(12,6.8),sharex=True)
        for ax,model in zip(axs.flat,[m for m in MODELS if m!=BASELINE]):
            diff=np.asarray([frames[model,s].val_loss.to_numpy()[start:stop]-
                             frames[BASELINE,s].val_loss.to_numpy()[start:stop] for s in (1,2)])
            x=np.arange(start,stop);mean=diff.mean(0)
            ax.plot(x,mean,color=COLORS[model]);ax.fill_between(x,diff.min(0),diff.max(0),color=COLORS[model],alpha=.18)
            ax.fill_between(x,0,mean,where=mean<=0,color='#2A8C74',alpha=.15)
            ax.fill_between(x,0,mean,where=mean>0,color='#C75664',alpha=.15)
            ax.axhline(0,color='#555555',lw=.8);ax.set_title(model);axes_style(ax,ylabel='验证loss差')
        fig.suptitle(f'相对{BASELINE}的验证损失差（epoch {start}—{stop-1}）')
        export(fig,'01_dynamics',f'baseline_residual_{phase}','配对seed验证loss残差与正负填充')
    ws=pd.DataFrame(window_records)
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for ax,(window,field,metric,title) in zip(axs.flat,[
        ('0-44','val_loss','normalized_auc_epoch','前45轮验证损失面积均值'),
        ('0-44','train_loss','normalized_auc_epoch','前45轮训练损失面积均值'),
        ('0-44','val_loss','slope_per_hour','前45轮验证loss/小时斜率'),
        ('75-99','val_loss','normalized_auc_epoch','后25轮验证损失面积均值')]):
        subset=ws[(ws.window==window)&(ws.field==field)]
        values=np.array([[float(subset[(subset.model==m)&(subset.seed==s)][metric].iloc[0]) for s in (1,2)] for m in MODELS])
        x=np.arange(len(MODELS));ax.bar(x,values.mean(1),color=[COLORS[m] for m in MODELS],alpha=.7)
        for i in (0,1):ax.scatter(x+(-.08 if i==0 else .08),values[:,i],c='#333333',s=14,marker='o' if i==0 else 'x',zorder=3)
        ax.set_xticks(x,MODELS,rotation=25,ha='right',fontsize=8);ax.set_title(title);ax.grid(axis='y',ls='--')
    fig.suptitle('训练阶段的损失水平与下降速率')
    export(fig,'02_efficiency','phase_efficiency','固定窗口loss AUC与每小时下降斜率')
    fig,axs=plt.subplots(1,2,figsize=(11,4.7))
    late_df=pd.DataFrame(late)
    for ax,column,title in zip(axs,['within_late_drop','additional_best_loss_drop'],
                              ['epoch75到99的loss下降','后25轮相对前75轮的历史最低loss改善']):
        vals=np.asarray([[float(late_df[(late_df.model==m)&(late_df.seed==s)][column].iloc[0]) for s in (1,2)] for m in MODELS])
        for i in (0,1):ax.bar(np.arange(7)+(i-.5)*.32,vals[:,i],width=.32,color=[COLORS[m] for m in MODELS],alpha=.95 if i==0 else .5,label=f'seed{i+1}')
        ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(range(7),MODELS,rotation=25,ha='right',fontsize=8)
        ax.set_title(title);ax.set_ylabel('损失下降量');ax.grid(axis='y',ls='--');ax.legend()
    fig.suptitle('后期损失恢复与最低损失改善')
    export(fig,'02_efficiency','late_recovery_vs_new_best','区分75轮重启后恢复与真正新增的最低loss')
    fig,axs=plt.subplots(1,3,figsize=(12,4.7))
    for ax,target in zip(axs,[2.,1.8,1.7]):
        for i,m in enumerate(MODELS):
            rr=[r for r in loss_targets if r['model']==m and r['target_loss']==target]
            for j,r in enumerate(rr):
                if r['hours'] is not None:ax.scatter(r['hours'],i+(-.12 if j==0 else .12),c=COLORS[m],marker='o' if j==0 else 'x',s=40)
        ax.set_yticks(range(7),MODELS,fontsize=8);ax.set_xlabel('首次达到目标的累计小时');ax.set_title(f'验证loss ≤ {target}');ax.grid(ls='--')
        ax.set_xlim(left=0)
    fig.suptitle('达到共同验证损失目标的训练耗时')
    export(fig,'02_efficiency','shared_loss_targets','共同loss目标的首次到达时间')
    fig,axs=plt.subplots(2,4,figsize=(12,6.2),sharex=True)
    for ax,m in zip(axs.flat,MODELS):
        vals=np.asarray([(frames[m,s].val_loss-frames[m,s].train_loss).to_numpy() for s in (1,2)])
        ax.plot(range(100),vals.mean(0),c=COLORS[m]);ax.fill_between(range(100),vals.min(0),vals.max(0),color=COLORS[m],alpha=.15)
        ax.set_title(m);axes_style(ax,ylabel='验证loss − 训练loss')
    axs.flat[-1].axis('off');fig.suptitle('训练与验证损失差')
    export(fig,'01_dynamics','loss_gap','训练/验证loss差异和两seed范围')
    json_write(OUT/'metrics_analysis.json',{
        'baseline':BASELINE,'reader':'chart_data.load_run_frames/load_model/summarize_run/window_statistics',
        'bands':'min/max of two seeds, not confidence intervals','window_convention':'early [0,45), late [75,100)',
        'loss':'Original smoothed cross entropy (label_smoothing=0.1); no temporal smoothing',
        'time':'Sum raw time_spend_s, includes known slow epochs, excludes downtime and inter-epoch waits',
        'run_summaries':{f'{m}_seed{s}':v for (m,s),v in summaries.items()},
        'windows':window_records,'late_improvement':late,'loss_targets':loss_targets,'restarts':restart,
        'sources':[{'path':p,'sha256':sha256(p)} for p in all_metrics_files()],
        'figures':FIGURES})
    json_write(OUT/'metrics_figures.json',FIGURES)


def short_name(name):
    return name.split('-',2)[-1]


def prediction_report():
    manifest=json.loads((OUT/'validation_manifest.json').read_text(encoding='utf-8'))
    names=manifest['names']; n=len(names); pred={};cms={};class_rows=[];cal={}
    for m in MODELS:
        for s in (1,2):
            pred[m,s],identity=load_prediction_cache(m,s,OUT/'predictions',manifest)
            cms[m,s]=pred[m,s]['cm']
            for i,row in enumerate(evaluator.per_class_metrics(cms[m,s],names)):
                mask=pred[m,s]['truth']==i
                true_probability=pred[m,s]['probabilities'][mask,i]
                class_rows.append(dict(model=m,seed=s,class_index=i,
                    mean_true_probability=float(true_probability.mean()),
                    mean_nll=float(-np.log(np.maximum(true_probability,1e-30)).mean()),**row))
            cal[m,s]=calibration_statistics(pred[m,s]['truth'],pred[m,s]['probabilities'])
    support=cms[BASELINE,1].sum(1)
    if not all(np.array_equal(c.sum(1),support) for c in cms.values()):raise ValueError('Unequal class supports')
    recall={m:np.asarray([np.diag(cms[m,s])/support for s in (1,2)]) for m in MODELS}
    order=np.lexsort((np.arange(n),-support,-recall[BASELINE].mean(0)))
    rank={int(i):j+1 for j,i in enumerate(order)}
    panels={'top50':order[:50], 'middle50':order[(n-50)//2:(n-50)//2+50], 'bottom50':order[-50:]}
    pd.DataFrame(class_rows).to_csv(OUT/'per_class_metrics_14runs.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([dict(rank=rank[i],class_index=int(i),class_name=names[i],support=int(support[i]),
                       baseline_recall_mean=float(recall[BASELINE][:,i].mean())) for i in order]).to_csv(OUT/'fixed_baseline_class_ranking.csv',index=False,encoding='utf-8-sig')
    group_stats=[]
    for tag,indices in panels.items():
        data=np.array([recall[m].mean(0)[indices]*100 for m in MODELS]).T
        fig,ax=plt.subplots(figsize=(11.8,13.5))
        im=ax.imshow(data,aspect='auto',cmap='YlGnBu',vmin=0,vmax=100)
        ax.set_xticks(range(7),MODELS,rotation=20,ha='right')
        ax.set_yticks(range(len(indices)),[f'{rank[int(i)]:03d}  {short_name(names[i])}  (n={support[i]})' for i in indices],fontsize=8)
        group_name={'top50':'前50类','middle50':'中间50类','bottom50':'后50类'}[tag]
        ax.set_title(f'基线排序{group_name}的验证召回率（两seed平均）')
        fig.colorbar(im,ax=ax,label='验证召回率（%）',shrink=.6)
        export(fig,'03_classes',f'{tag}_fixed_class_recall',f'固定{tag}类同一集合，显示原排名和全量样本数')
        perclass=pd.DataFrame(class_rows)
        loss_data=np.array([[float(perclass[(perclass.model==m)&(perclass.class_index==int(i))].mean_nll.mean())
                            for m in MODELS] for i in indices])
        fig,ax=plt.subplots(figsize=(11.8,13.5))
        im=ax.imshow(loss_data,aspect='auto',cmap='YlOrRd',vmin=0,vmax=max(3.0,float(loss_data.max())))
        ax.set_xticks(range(7),MODELS,rotation=20,ha='right')
        ax.set_yticks(range(len(indices)),[f'{rank[int(i)]:03d}  {short_name(names[i])}  (n={support[i]})' for i in indices],fontsize=8)
        ax.set_title(f'基线排序{group_name}的负对数似然（两seed平均）')
        fig.colorbar(im,ax=ax,label='平均NLL',shrink=.6)
        export(fig,'03_classes',f'{tag}_fixed_class_nll',f'固定{tag}类逐图NLL，揭示正确率之外的真实类概率')
        for m in MODELS:
            for s in (1,2):
                c=cms[m,s];group_stats.append(dict(group=tag,model=m,seed=s,
                    n_classes=len(indices),n_images=int(support[indices].sum()),
                    macro_recall=float(recall[m][s-1,indices].mean()),
                    micro_recall=float(np.diag(c)[indices].sum()/support[indices].sum())))
    pd.DataFrame(group_stats).to_csv(OUT/'fixed50_group_statistics.csv',index=False,encoding='utf-8-sig')
    # Separate own-rank export answers "each model's best/worst" without changing comparison groups.
    own=[]
    for m in MODELS:
        own_order=np.lexsort((np.arange(n),-support,-recall[m].mean(0)))
        for tag,idx in [('own_top50',own_order[:50]),('own_bottom50',own_order[-50:])]:
            for i in idx:own.append(dict(model=m,group=tag,class_index=int(i),class_name=names[i],support=int(support[i]),recall_mean=float(recall[m][:,i].mean())))
    pd.DataFrame(own).to_csv(OUT/'each_model_own_top_bottom50.csv',index=False,encoding='utf-8-sig')
    fig,axs=plt.subplots(1,3,figsize=(12,4.7))
    for ax,tag in zip(axs,panels):
        vals=np.array([[r['macro_recall']*100 for r in group_stats if r['group']==tag and r['model']==m] for m in MODELS])
        ax.bar(range(7),vals.mean(1),color=[COLORS[m] for m in MODELS],alpha=.7)
        ax.scatter(range(7),vals[:,0],c='#333333',s=12);ax.scatter(range(7),vals[:,1],c='#333333',s=16,marker='x')
        ax.set_xticks(range(7),MODELS,rotation=40,ha='right',fontsize=7);ax.set_title(tag);ax.set_ylabel('类等权平均召回率（%）');ax.grid(axis='y',ls='--')
    fig.suptitle('基线排序前、中、后50类的召回率')
    export(fig,'03_classes','fixed50_group_comparison','固定三组50类比较，类等权宏召回率')
    fig,axs=plt.subplots(2,3,figsize=(12,7.5))
    for ax,m in zip(axs.flat,[m for m in MODELS if m!=BASELINE]):
        difference=(recall[m].mean(0)-recall[BASELINE].mean(0))*100
        idx=np.argsort(difference); chosen=np.r_[idx[:7],idx[-7:]]
        ax.barh(range(14),difference[chosen],color=[COLORS[m] if v>=0 else '#7B8794' for v in difference[chosen]])
        ax.set_yticks(range(14),[short_name(names[i]) for i in chosen],fontsize=7)
        ax.axvline(0,c='#555555',lw=.7);ax.set_title(m);ax.set_xlabel('相对基线召回率变化（百分点）');ax.grid(axis='x',ls='--')
    fig.suptitle('相对固定基线：改善最多/下降最多各7类（双seed平均）')
    export(fig,'03_classes','class_gain_loss_vs_baseline','相对基线的类别召回率变化')
    # Full denominators are retained even in the small submatrices.
    cm_segments={'top20':order[:20], 'middle20':order[(n-20)//2:(n-20)//2+20], 'bottom20':order[-20:]}
    for m in MODELS:
        mean_cm=(cms[m,1]+cms[m,2])/2
        normalized=mean_cm/support[:,None]
        fig,ax=plt.subplots(figsize=(8.4,7.2))
        im=ax.imshow(normalized[np.ix_(order,order)],cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
        ticks=[0,49,99,149,199,251];ax.set_xticks(ticks,[str(t+1) for t in ticks]);ax.set_yticks(ticks,[str(t+1) for t in ticks])
        ax.set_xlabel('预测类别：固定基线召回率降序排名');ax.set_ylabel('真实类别：同一固定排序')
        ax.set_title(f'{m}：252类混淆矩阵（两seed平均）')
        fig.colorbar(im,ax=ax,label='行归一化比例')
        export(fig,'04_confusion',f'{m}_full252_sorted','全252类按同一基线排名降序，完整分母')
        fig,axs=plt.subplots(1,3,figsize=(15,6.5))
        for ax,(tag,indices) in zip(axs,cm_segments.items()):
            sub=normalized[np.ix_(indices,indices)]
            im=ax.imshow(sub,cmap='Blues',vmin=0,vmax=1)
            labels=[f'{rank[int(i)]} {short_name(names[i])}' for i in indices]
            ax.set_xticks(range(20),labels,rotation=90,fontsize=6.5);ax.set_yticks(range(20),labels,fontsize=6.5)
            ax.set_title(tag);ax.set_xlabel('预测类别');ax.set_ylabel('真实类别')
            for j in range(20):ax.text(j,j,f'{sub[j,j]:.2f}',ha='center',va='center',fontsize=5.5,color='white' if sub[j,j]>.5 else 'black')
        fig.suptitle(f'{m}：基线排序前、中、后20类的混淆矩阵')
        export(fig,'04_confusion',f'{m}_segments20','基线排序前、中、后20类的局部混淆矩阵')
        # Per-seed full matrices retained numerically; paired visual comparison in one page.
        fig,axs=plt.subplots(1,2,figsize=(11,5))
        wrong_flow=(mean_cm+mean_cm.T).copy();np.fill_diagonal(wrong_flow,0)
        related=np.argsort(wrong_flow.sum(0))[-20:][::-1]
        for ax,s in zip(axs,(1,2)):
            cm=cms[m,s];norm=cm/support[:,None]
            im=ax.imshow(norm[np.ix_(related,related)],cmap='Blues',vmin=0,vmax=1)
            labs=[short_name(names[i]) for i in related]
            ax.set_xticks(range(20),labs,rotation=90,fontsize=6.5);ax.set_yticks(range(20),labs,fontsize=6.5)
            ax.set_title(f'seed{s}');ax.set_xlabel('预测');ax.set_ylabel('真实')
        fig.suptitle(f'{m}｜参与错误最多20类（该模型两seed错误流量排序；两图同序）')
        export(fig,'04_confusion',f'{m}_error20_seed_compare','各模型错误相关20类的双seed并列混淆矩阵')
    error_stats=[];pairs=[]
    fig,axs=plt.subplots(1,2,figsize=(11,4.8))
    for ax,parent_only in zip(axs,(False,True)):
        for i,m in enumerate(MODELS):
            counts=[]
            for s in (1,2):
                p=pred[m,s];truth=p['truth'];guess=p['predictions'];wrong=truth!=guess
                parents=np.array([x.split('-',2)[1] for x in names])
                same=(parents[truth]==parents[guess])&wrong
                counts.append([int(same.sum()),int((wrong&~same).sum())])
                if not parent_only:
                    error_stats.append(dict(model=m,seed=s,errors=int(wrong.sum()),same_parent=int(same.sum()),cross_parent=int((wrong&~same).sum())))
            mean=np.mean(counts,axis=0)
            vals=mean/mean.sum()*100 if parent_only else mean
            ax.bar(i,vals[0],color=COLORS[m],alpha=.85)
            ax.bar(i,vals[1],bottom=vals[0],color=COLORS[m],alpha=.3,hatch='//')
        ax.set_xticks(range(7),MODELS,rotation=30,ha='right',fontsize=8)
        ax.set_ylabel('错误占比（%）' if parent_only else '双seed平均错误数');ax.set_title('错误构成比例' if parent_only else '错误数量')
        ax.grid(axis='y',ls='--')
    fig.suptitle('错误组成：同垃圾大类（实色）与跨垃圾大类（斜线）')
    export(fig,'05_errors','parent_error_decomposition','错误在同垃圾大类/跨大类间的数量及比例')
    fig,axs=plt.subplots(2,4,figsize=(15,8))
    for ax,m in zip(axs.flat,MODELS):
        cm=(cms[m,1]+cms[m,2])/2;off=cm.copy();np.fill_diagonal(off,0)
        cells=np.argsort(off.ravel())[-12:][::-1];ij=[divmod(int(k),n) for k in cells]
        vals=[float(off[i,j]) for i,j in ij]
        ax.barh(range(len(ij)),vals,color=COLORS[m]);ax.invert_yaxis()
        ax.set_yticks(range(len(ij)),[f'{short_name(names[i])} → {short_name(names[j])}' for i,j in ij],fontsize=6.5)
        ax.set_title(m);ax.set_xlabel('双seed平均有向误判次数');ax.grid(axis='x',ls='--')
        for i,j in ij:pairs.append(dict(model=m,true_index=i,pred_index=j,true_class=names[i],pred_class=names[j],mean_count=float(off[i,j]),true_support=int(support[i])))
    axs.flat[-1].axis('off');fig.suptitle('各配置前12个有向混淆对（次数降序）')
    export(fig,'05_errors','top_directed_confusions','每模型真实类别→预测类别的错误流量')
    pd.DataFrame(error_stats).to_csv(OUT/'parent_error_statistics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pairs).to_csv(OUT/'top_directed_confusions.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(9,5.5))
    agreement=[]
    for i,m in enumerate(MODELS):
        truth=pred[m,1]['truth']
        stats=paired_prediction_statistics(truth,pred[m,1]['predictions'],pred[m,2]['predictions'])
        vals=np.array([stats['both_correct'],stats['one_correct'],stats['both_wrong']])/stats['n']*100
        left=0
        for j,(v,color) in enumerate(zip(vals,['#2A8C74','#D78A35','#C75664'])):
            ax.barh(i,v,left=left,color=color,label=['两seed都正确','一对一错','两seed都错误'][j] if i==0 else None)
            if v>3:ax.text(left+v/2,i,f'{v:.1f}%',ha='center',va='center',fontsize=8,color='white')
            left+=v
        agreement.append(dict(model=m,**{key:stats[key]for key in ('both_correct','one_correct','both_wrong','prediction_disagreement')}))
    ax.set_yticks(range(7),MODELS);ax.set_xlabel('完整验证集占比（%）');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=3,fontsize=8)
    ax.set_title('两seed对同一验证图像的预测一致性')
    export(fig,'05_errors','seed_prediction_consistency','两个seed的共同正确、单seed正确与共同错误')
    fig,axs=plt.subplots(1,2,figsize=(11,5))
    for m in MODELS:
        rr=[cal[m,s] for s in (1,2)]
        for s,r in zip((1,2),rr):
            rows=[x for x in r['bins'] if x['count']]
            axs[0].plot([x['confidence'] for x in rows],[x['accuracy'] for x in rows],c=COLORS[m],ls='-' if s==1 else '--',alpha=.8,label=m if s==1 else None)
        cover=np.linspace(.1,1,19);risk=[]
        for s in (1,2):
            p=pred[m,s];order_conf=np.argsort(-p['probabilities'].max(1),kind='stable');wrong=(p['predictions']!=p['truth'])[order_conf]
            risk.append([wrong[:max(1,int(len(wrong)*c))].mean() for c in cover])
        risk=np.asarray(risk);axs[1].plot(cover,risk.mean(0),c=COLORS[m],label=m)
        axs[1].fill_between(cover,risk.min(0),risk.max(0),color=COLORS[m],alpha=.1)
    axs[0].plot([0,1],[0,1],ls=':',color='#555555');axes_style(axs[0],xlabel='平均最大Softmax概率',ylabel='经验正确率')
    axs[0].set_title('可靠性曲线｜实线s1，虚线s2（10等宽箱）');axs[0].legend(fontsize=7)
    axes_style(axs[1],xlabel='按置信度保留比例',ylabel='保留样本错误率');axs[1].set_title('风险—覆盖率｜两seed均值与范围')
    export(fig,'07_calibration','reliability_risk_coverage','可靠性曲线与风险—覆盖率')
    calibration_rows=[dict(model=m,seed=s,**{k:v for k,v in cal[m,s].items() if k!='bins'}) for m in MODELS for s in (1,2)]
    pd.DataFrame(calibration_rows).to_csv(OUT/'calibration_statistics.csv',index=False,encoding='utf-8-sig')
    json_write(OUT/'prediction_analysis.json',{'baseline':BASELINE,'manifest_fingerprint':manifest['fingerprint'],
               'rank_definition':'Baseline two-seed mean recall descending, support descending, index ascending',
               'fixed_groups':{k:[int(i) for i in v] for k,v in panels.items()},
               'cm_normalization':'Full true-class support; submatrix rows may sum below 1',
               'per_class_metric':'Validation recall (class-conditional accuracy), not training fitting accuracy',
               'own_rank_warning':'Each model own top/bottom50 are different populations; use fixed groups for comparisons',
               'seed_agreement':agreement,'calibration':{f'{m}_seed{s}':v for (m,s),v in cal.items()},
               'figures':FIGURES})
    json_write(OUT/'prediction_figures.json',FIGURES)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['metrics','predictions','all'],default='all')
    stage=parser.parse_args().stage;configure()
    if stage in ('metrics','all'):metrics_report()
    if stage in ('predictions','all'):
        FIGURES.clear();prediction_report()


if __name__=='__main__':main()
