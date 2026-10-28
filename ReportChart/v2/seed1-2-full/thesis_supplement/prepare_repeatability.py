"""Add thesis statistics from completed runs; never open source image files."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
FULL=OUT.parent
sys.path[:0]=[str(OUT),str(FULL.parents[1])]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paper_style import COLORS,configure,save
from report_inference import load_prediction_cache,paired_prediction_statistics,sha256
from chart_data import load_run_frames
from prepare_supplement import table,figure

MODELS=list(COLORS)


def main():
    sys.stdout.reconfigure(encoding='utf-8');configure()
    manifest=json.loads((FULL/'validation_manifest.json').read_text(encoding='utf-8'))
    previous=json.loads((FULL/'prediction_analysis.json').read_text(encoding='utf-8'))
    diagnosis=json.loads((FULL/'bottom20_diagnosis/diagnostics.json').read_text(encoding='utf-8'))
    assert previous['manifest_fingerprint']==manifest['fingerprint']==diagnosis['manifest_fingerprint']
    truth=np.asarray([r['true_index']for r in manifest['records']])
    rows=[];table_rows=[];sources={};cms={}
    groups={'all':None,**previous['fixed_groups']}
    for model in MODELS:
        pair=[]
        for seed in (1,2):
            data,_=load_prediction_cache(model,seed,FULL/'predictions',manifest)
            assert np.array_equal(data['truth'],truth)
            pair.append(data['predictions']);cms[model,seed]=data['cm']
            source=FULL/'predictions'/f'predictions_{model}_seed{seed}.npz';sources[str(source)]=sha256(source)
        for group,indices in groups.items():
            mask=None if indices is None else np.isin(truth,indices)
            stat=paired_prediction_statistics(truth,*pair,mask)
            rows.append(dict(model=model,group=group,**stat))
        overall=next(r for r in rows if r['model']==model and r['group']=='all')
        original=next(r for r in previous['seed_agreement']if r['model']==model)
        assert all(overall[k]==v for k,v in original.items()if k!='model')
        table_rows.append([model,*[f'{overall[k]/overall["n"]*100:.2f}'for k in
            ('both_correct','seed1_only_correct','seed2_only_correct','both_wrong','prediction_disagreement')]])
    pd.DataFrame(rows).to_csv(OUT/'paired_prediction_statistics.csv',index=False,encoding='utf-8-sig')
    # Reuse the existing complete 252-class diagnosis, retaining original support.
    paired=pd.read_csv(FULL/'bottom20_diagnosis/paired_model_class_profiles.csv')
    class_rows=[];direction_rows=[];recalls=np.empty((20,7))
    for i,c in enumerate(diagnosis['classes']):
        index=c['class_index'];p=c['pooled'];target=p['top_targets'][0]
        direction_rows.append([str(c['rank']),c['class_name'].split('-',2)[-1],str(c['support']),
            target['class_name'].split('-',2)[-1],str(target['count']),f'{target["share_of_errors"]*100:.1f}'])
        aggregate=sum(cms[m,s][index]for m in MODELS for s in (1,2))
        assert aggregate.sum()==c['support']*14 and aggregate[index]==p['correct']
        assert aggregate[target['class_index']]==target['count']
        for j,model in enumerate(MODELS):
            source=paired[(paired.model==model)&(paired.class_index==index)].iloc[0]
            recall=(cms[model,1][index,index]+cms[model,2][index,index])/(2*c['support'])
            assert abs(recall-source.recall)<1e-12
            recalls[i,j]=recall*100
            class_rows.append(dict(rank=c['rank'],class_index=index,class_name=c['class_name'],model=model,
                support_per_seed=c['support'],correct_seed1=int(source.correct_seed1),correct_seed2=int(source.correct_seed2),
                mean_recall_percent=recall*100,main_wrong_targets=source.top_targets))
    pd.DataFrame(class_rows).to_csv(OUT/'fixed_bottom20_model_statistics.csv',index=False,encoding='utf-8-sig')
    fig,axs=plt.subplots(1,2,figsize=(10,4.6),gridspec_kw={'width_ratios':[1.05,1]})
    for i,model in enumerate(MODELS):
        r=next(r for r in rows if r['model']==model and r['group']=='all');left=0
        for key,color,label in [('both_correct','#2A8C74','均正确'),('one_correct','#D78A35','一对一错'),('both_wrong','#C75664','均错误')]:
            value=r[key]/r['n']*100
            axs[0].barh(i,value,left=left,color=color,label=label if i==0 else None)
            axs[0].text(left+value/2,i,f'{value:.1f}',ha='center',va='center',fontsize=7,color='white')
            left+=value
    axs[0].set_yticks(range(7),MODELS);axs[0].invert_yaxis();axs[0].set_xlabel('验证图像占比（%）')
    axs[0].set_xlim(0,100);axs[0].set_title('完整验证集的正确性组成')
    axs[0].legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,fontsize=8)
    matrix=np.array([[next(r for r in rows if r['model']==m and r['group']==g)['one_correct']/next(r for r in rows if r['model']==m and r['group']==g)['n']*100
                      for g in groups]for m in MODELS])
    image=axs[1].imshow(matrix,cmap='YlOrRd',vmin=0,vmax=16,aspect='auto')
    axs[1].set_yticks(range(7),MODELS);axs[1].set_xticks(range(4),['全部','前50类','中间50类','后50类'])
    axs[1].set_title('固定类别组的一对一错比例')
    for i in range(7):
        for j in range(4):axs[1].text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white'if matrix[i,j]>9 else 'black')
    fig.colorbar(image,ax=axs[1],label='组内图像占比（%）',fraction=.045,pad=.03)
    save(fig,OUT,'seed_stability_groups',tight_layout_kwargs={'w_pad':2.5,'rect':(0,.045,1,1)})
    fig,ax=plt.subplots(figsize=(7.6,6.5))
    image=ax.imshow(recalls,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto')
    ax.set_yticks(range(20),[f'{c["rank"]} {c["class_name"].split("-",2)[-1]}（{c["support"]}）'for c in diagnosis['classes']],fontsize=9)
    ax.set_xticks(range(7),MODELS,rotation=25,ha='right',fontsize=8)
    for i in range(20):
        for j in range(7):ax.text(j,i,f'{recalls[i,j]:.1f}',ha='center',va='center',fontsize=7,color='white'if recalls[i,j]>55 else 'black')
    fig.colorbar(image,ax=ax,label='两seed平均召回率（%）',fraction=.035,pad=.02)
    save(fig,OUT,'fixed_bottom20_models')
    overall=[r for r in rows if r['group']=='all']
    flip=[r['one_correct']/r['n']*100for r in overall]
    changed=[r['prediction_disagreement']/r['n']*100for r in overall]
    full=next(r for r in overall if r['model']=='V2-NoTF')
    bottom=[r['one_correct']/r['n']*100for r in rows if r['group']=='bottom50']
    top=[r['one_correct']/r['n']*100for r in rows if r['group']=='top50']
    consistency=table('两个随机种子在相同验证图像上的预测组成（百分比）','tab:supp-seed-stability',
        ['配置','均正确','仅s1正确','仅s2正确','均错误','预测类别不同'],table_rows,'lccccc')
    direction=table('固定后20类的主要误判去向（七配置双seed汇总）','tab:supp-bottom20-directions',
        ['基线排名','真实类别','图像数','主要错误去向','次数','占本类错误(\\%)'],direction_rows,'clclcc')
    fragment=r'''\Needspace{12\baselineskip}
\subsubsection{同一样本的重复运行一致性}
在相同验证图像上，将两次运行的结果分为均正确、仅seed1正确、仅seed2正确和均错误四种互斥情形。表\ref{tab:supp-seed-stability}中各比例的分母均为14,603张图像，四种正确性情形覆盖全部样本。仅一次运行正确的两部分相加，得到“一对一错”比例；预测类别不同还包括两次均错误但指向不同标签的情形。此处s1、s2分别表示seed1、seed2。

@@CONSISTENCY@@

七种配置的一对一错比例为@@FLIP@@，预测类别不同的比例为@@CHANGED@@。V2-NoTF有@@NOTF_COUNT@@张图像在两次运行之间改变了正确性，占@@NOTF_FLIP@@；同时有@@NOTF_WRONG@@张图像在两次运行中均被误判。总体正确率只反映正确图像数，两个方向的变化可以相互抵消；逐图配对则进一步呈现预测对象的变化。

图\ref{fig:supp-seed-stability}对固定类别组分别统计一对一错比例，分母为组内图像数。前50类为@@TOP@@，后50类为@@BOTTOM@@。正确性变化集中于困难类别，整体指标的接近与困难样本上的预测变化同时存在。

@@STABILITY_FIG@@

\subsubsection{困难类别的完整误判去向}
沿用S-NoProg的固定类别排序，选择排名233--252的20类，共455张验证图像。图\ref{fig:supp-bottom20-models}在七种配置中比较相同类别的两seed平均召回率。表\ref{tab:supp-bottom20-directions}进一步追踪这些图像在完整252类输出中的去向，合并七配置、两seed的6,370次预测，错误占比以该真实类别在14次运行中的总错误数为分母。同一图像在不同模型下的预测分别计数，原始验证图像数保持为455张。

@@BOTTOM_FIG@@

@@DIRECTION@@

火腿的139次错误中，88次预测为肠、45次预测为肉类，两者合占95.7\%；锅的678次错误中，505次预测为电饭煲，占74.5\%；奶茶杯的122次错误中，87次预测为一次性杯子，占71.3\%。这些类别的错误主要集中于少数相邻标签。模具、餐垫、巧克力、灯罩和铝制用品的首要去向所占比例较低，呈现多个错误方向。局部混淆矩阵之外的错误由完整输出统计进一步补充。

火腿每个seed包含10张验证图像，单次运行的召回率以10个百分点为最小非零步长，两seed平均则以5个百分点为步长。除G-Full-CAWR的seed2正确识别1张外，其余13次运行均为0张。因此图中的0.0对应实际零正确计数。G-V2对灯罩和过滤网的两seed平均召回率分别为44.0\%和57.7\%，S-NoProg为26.0\%和38.5\%，显示不同结构的收益随类别而变化。

对已有误例的原图与输入裁剪作对照，可以观察到器物与材料、容器与内容物、整机与部件等标签层级交叉。例如模具类的部分画面呈现造型蛋糕或饼干成品，锅类包含完整电饭煲；这些可见对象与相邻标签共享外观。铝制用品中还包含一张$1\times1$纯白输入，该图未提供物体纹理与轮廓。上述样本保留在原验证统计中，用于解释误判来源。

结合损失曲线、类别组和逐图结果，V2-NoTF在当前训练设置下具有更低的前期损失和较快的共同损失目标到达过程；G系列在后期继续取得最低损失改善，G-V2的固定后50类平均召回率更高；SE三组的前期损失曲线较接近。模型的损失拟合、困难类识别与逐图稳定性分别刻画不同表现，形成了随结构和训练策略变化的收益分布。
'''
    replacements={'CONSISTENCY':consistency,'DIRECTION':direction,
        'FLIP':f'{min(flip):.2f}\\%至{max(flip):.2f}\\%',
        'CHANGED':f'{min(changed):.2f}\\%至{max(changed):.2f}\\%',
        'NOTF_COUNT':str(full['one_correct']),'NOTF_FLIP':f'{full["one_correct"]/full["n"]*100:.2f}\\%',
        'NOTF_WRONG':str(full['both_wrong']),'TOP':f'{min(top):.2f}\\%至{max(top):.2f}\\%',
        'BOTTOM':f'{min(bottom):.2f}\\%至{max(bottom):.2f}\\%',
        'STABILITY_FIG':figure('seed_stability_groups','双seed的正确性组成与固定类别组的一对一错比例。','fig:supp-seed-stability'),
        'BOTTOM_FIG':figure('fixed_bottom20_models','固定基线排名233--252的逐类召回率。括号内为每个seed的验证图像数。','fig:supp-bottom20-models')}
    for key,value in replacements.items():fragment=fragment.replace('@@'+key+'@@',value)
    assert '@@'not in fragment
    (OUT/'repeatability_content.tex').write_text(fragment,encoding='utf-8')
    frames,_=load_run_frames(MODELS)
    fitting={f'seed{s}':{'notf_lower_train_loss_epochs':int((frames['V2-NoTF',s].train_loss<frames['V2-Full',s].train_loss).sum()),
                        'notf_lower_val_loss_epochs':int((frames['V2-NoTF',s].val_loss<frames['V2-Full',s].val_loss).sum())}for s in(1,2)}
    data={'manifest_fingerprint':manifest['fingerprint'],'sample_count':len(truth),'paired_statistics':rows,
          'consistency_table':table_rows,'direction_table':direction_rows,'fixed_bottom20':class_rows,
          'fitting_comparison':fitting,'prediction_sources':sources,'new_training':False,'new_inference':False,
          'source_images_read':False,'filtered_dataset_used':False}
    (OUT/'repeatability_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ADDITIONS_READY',len(rows),'paired rows',len(class_rows),'class-model rows',fitting,flush=True)


if __name__=='__main__':main()
