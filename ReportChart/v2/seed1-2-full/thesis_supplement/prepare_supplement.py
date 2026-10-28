"""Prepare additive thesis experiments from existing public APIs and frozen caches."""
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;FULL=OUT.parent;REPORT=FULL.parents[1]
for directory in (FULL,REPORT,REPORT.parent,REPORT.parent.parent):sys.path.insert(0,str(directory))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paper_style import configure,save,COLORS
from chart_data import load_run_frames,window_statistics
from report_inference import load_prediction_cache,calibration_statistics,sha256,evaluator
from render_analysis import two_seed_curve

MODELS=list(COLORS);BASE='S-NoProg'


def pm(values,decimals=4,scale=1):
    vals=np.asarray(values)*scale
    return f'${vals.mean():.{decimals}f}\\pm{vals.std(ddof=1):.{decimals}f}$'


def table(caption,label,headers,rows,columns):
    return '\n'.join([r'\begin{table}[H]',r'\centering\small',r'\caption{'+caption+'}',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}'+columns+'@{}}',r'\toprule',
        ' & '.join(headers)+r' \\',r'\midrule',*[' & '.join(row)+r' \\' for row in rows],
        r'\bottomrule',r'\end{tabular*}',r'\label{'+label+'}',r'\end{table}'])


def figure(name,caption,label,tall=False):
    sizing=r'width=\textwidth,height=0.82\textheight,keepaspectratio' if tall else r'width=\textwidth'
    return '\n'.join([r'\begin{figure}[H]',r'\centering',r'\includegraphics['+sizing+r']{../src_v2/ReportChart/v2/seed1-2-full/thesis_supplement/'+name+'.pdf}',
                      r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{figure}'])


def main():
    configure();plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8})
    frames,summaries=load_run_frames(MODELS)
    manifest=json.loads((FULL/'validation_manifest.json').read_text(encoding='utf-8'))
    names=manifest['names'];runs={};cal={};class_metrics={};errors={}
    for m in MODELS:
        for s in (1,2):
            data,identity=load_prediction_cache(m,s,FULL/'predictions',manifest);runs[m,s]=data
            cal[m,s]=calibration_statistics(data['truth'],data['probabilities'])
            class_metrics[m,s]=evaluator.per_class_metrics(data['cm'],names)
            parents=np.array([name.split('-',2)[1]for name in names])
            wrong=data['truth']!=data['predictions'];same=(parents[data['truth']]==parents[data['predictions']])&wrong
            errors[m,s]={'total':int(wrong.sum()),'same':int(same.sum()),'cross':int((wrong&~same).sum())}
    windows={(m,s,a,b):window_statistics(frames[m,s],a,b) for m in MODELS for s in (1,2) for a,b in [(0,45),(75,100),(0,75)]}
    late={(m,s):float(windows[m,s,0,75]['val_loss']['minimum']-windows[m,s,75,100]['val_loss']['minimum']) for m in MODELS for s in (1,2)}
    phase_rows=[];target_rows=[];class_rows=[];error_rows=[];cal_rows=[]
    support=runs[BASE,1]['cm'].sum(1)
    recall={m:np.array([np.diag(runs[m,s]['cm'])/support for s in (1,2)]) for m in MODELS}
    order=np.lexsort((np.arange(len(names)),-support,-recall[BASE].mean(0)))
    groups={'top':order[:50],'middle':order[101:151],'bottom':order[-50:]}
    group_info={key:{'n_images':int(support[idx].sum()),'min_support':int(support[idx].min()),'max_support':int(support[idx].max())} for key,idx in groups.items()}
    group_numbers={}
    for m in MODELS:
        phase_rows.append([m,pm([windows[m,s,0,45]['train_loss']['normalized_auc_epoch']for s in(1,2)]),
            pm([windows[m,s,0,45]['val_loss']['normalized_auc_epoch']for s in(1,2)]),f'{late[m,1]:+.4f}',f'{late[m,2]:+.4f}'])
        values=[]
        for target in(2.,1.8,1.7):
            for s in(1,2):
                d=frames[m,s];hit=d[d.val_loss<=target]
                values.append(f'{d.time_spend_s.cumsum().iloc[int(hit.iloc[0].epoch)]/3600:.2f}' if len(hit) else '--')
        target_rows.append([m,*values])
        grouped={key:[float(recall[m][s-1,idx].mean())for s in(1,2)]for key,idx in groups.items()}
        nll=[]
        for s in(1,2):
            p=runs[m,s];cls_nll=[float(-np.log(np.maximum(p['probabilities'][p['truth']==i,i],1e-30)).mean())for i in groups['bottom']]
            nll.append(float(np.mean(cls_nll)))
        group_numbers[m]={'recall':grouped,'bottom_nll':nll}
        class_rows.append([m,*[pm(grouped[key],2,100)for key in('top','middle','bottom')],pm(nll,4)])
        error_rows.append([m,pm([errors[m,s]['total']for s in(1,2)],1),
            pm([errors[m,s]['same']/errors[m,s]['total']for s in(1,2)],2,100),
            pm([errors[m,s]['cross']/errors[m,s]['total']for s in(1,2)],2,100)])
        cal_rows.append([m,pm([cal[m,s]['ece']for s in(1,2)],2,100),
            pm([cal[m,s]['nll']for s in(1,2)],5),pm([cal[m,s]['brier']for s in(1,2)],4)])
    # Compact plots use the same shared styles and the existing curve helper.
    fig,axs=plt.subplots(1,2,figsize=(7.6,4.1))
    for ax,field,title in zip(axs,['train_loss','val_loss'],['训练损失','验证损失']):
        for m in MODELS:two_seed_curve(ax,frames,m,field,0,45)
        ax.set_title(title);ax.set_xlabel('训练轮次');ax.set_ylabel(title);ax.grid(ls='--')
    axs[0].legend(fontsize=6.5,ncol=2)
    save(fig,OUT,'early_loss')
    fig,axs=plt.subplots(2,2,figsize=(7.6,6.3))
    for column,(models,title)in enumerate([([BASE,'G-Full-CAWR'],'全局单token头'),(['V2-Full','G-V2'],'空间模型头')]):
        for m in models:
            two_seed_curve(axs[0,column],frames,m,'val_loss',75,100)
            two_seed_curve(axs[1,column],frames,m,'lr',0,100)
        axs[0,column].set_title(title);axs[0,column].set_ylabel('验证损失');axs[1,column].set_ylabel('学习率')
        axs[1,column].set_yscale('log');axs[0,column].legend(fontsize=7)
        for ax in axs[:,column]:ax.set_xlabel('训练轮次');ax.grid(ls='--')
    save(fig,OUT,'late_loss_lr',tight_layout_kwargs={'h_pad':2})
    fig,axs=plt.subplots(2,3,figsize=(7.6,5.1),sharex=True)
    for ax,m in zip(axs.flat,[m for m in MODELS if m!=BASE]):
        delta=np.array([(frames[m,s].val_loss-frames[BASE,s].val_loss).to_numpy()for s in(1,2)])
        mean=delta.mean(0);ax.plot(range(100),mean,c=COLORS[m]);ax.fill_between(range(100),delta.min(0),delta.max(0),color=COLORS[m],alpha=.17)
        ax.axhline(0,c='#666666',lw=.7);ax.set_title(m,fontsize=8);ax.set_xlabel('训练轮次');ax.set_ylabel('验证损失差');ax.grid(ls='--')
    save(fig,OUT,'baseline_loss_difference')
    fig,ax=plt.subplots(figsize=(7.6,10.2))
    idx=groups['bottom'];data=np.array([recall[m].mean(0)[idx]*100for m in MODELS]).T
    im=ax.imshow(data,aspect='auto',cmap='YlGnBu',vmin=0,vmax=100)
    ax.set_xticks(range(7),MODELS,rotation=30,ha='right',fontsize=8)
    ax.set_yticks(range(50),[f'{203+j} {names[int(i)].split("-",2)[-1]}（{support[i]}）'for j,i in enumerate(idx)],fontsize=8.5)
    fig.colorbar(im,ax=ax,label='召回率（%）',fraction=.035,pad=.02)
    save(fig,OUT,'bottom50_recall')
    fig,axs=plt.subplots(1,2,figsize=(7.6,4.7));idx=order[-16:]
    for ax,m in zip(axs,[BASE,'G-V2']):
        cm=(runs[m,1]['cm']+runs[m,2]['cm'])/2;norm=cm/support[:,None];sub=norm[np.ix_(idx,idx)]
        im=ax.imshow(sub,cmap='Blues',vmin=0,vmax=1)
        labs=[name.split('-',2)[-1]for name in np.array(names)[idx]]
        ax.set_xticks(range(16),labs,rotation=90,fontsize=7.5);ax.set_yticks(range(16),labs,fontsize=7.5)
        ax.set_title(m);ax.set_xlabel('预测类别');ax.set_ylabel('真实类别')
        for i in range(16):ax.text(i,i,f'{sub[i,i]:.2f}',ha='center',va='center',fontsize=6,color='white'if sub[i,i]>.5 else 'black')
    save(fig,OUT,'difficult_class_confusion')
    fig,axs=plt.subplots(1,2,figsize=(7.6,4.1))
    for m in MODELS:
        for s in(1,2):
            bins=[b for b in cal[m,s]['bins']if b['count']]
            axs[0].plot([b['confidence']for b in bins],[b['accuracy']for b in bins],c=COLORS[m],ls='-'if s==1 else '--',lw=1,label=m if s==1 else None)
        coverage=np.linspace(.1,1,19);risk=[]
        for s in(1,2):
            p=runs[m,s];rank=np.argsort(-p['probabilities'].max(1),kind='stable');wrong=(p['predictions']!=p['truth'])[rank]
            risk.append([wrong[:max(1,int(len(wrong)*v))].mean()for v in coverage])
        risk=np.array(risk);axs[1].plot(coverage,risk.mean(0),c=COLORS[m],lw=1);axs[1].fill_between(coverage,risk.min(0),risk.max(0),color=COLORS[m],alpha=.1)
    axs[0].plot([0,1],[0,1],ls=':',c='#666666');axs[0].set_xlabel('平均最大Softmax分数');axs[0].set_ylabel('经验正确率');axs[0].set_title('可靠性曲线');axs[0].legend(fontsize=6.2,ncol=2)
    axs[1].set_xlabel('样本覆盖率');axs[1].set_ylabel('错误率');axs[1].set_title('风险—覆盖率')
    for ax in axs:ax.grid(ls='--')
    save(fig,OUT,'reliability_risk')
    grad=json.loads((FULL/'06_gradcam/gradcam_manifest.json').read_text(encoding='utf-8'))
    transform=evaluator._transform(manifest['mean'],manifest['std'])
    from PIL import Image
    with np.load(FULL/'06_gradcam/gradcam_arrays.npz',allow_pickle=False)as z:
        for case,filename in[(1,'gradcam_paperbox'),(3,'gradcam_printer')]:
            rec=grad['cases'][case]['source']
            with Image.open(Path(evaluator.VAL_ROOT)/rec['relative_path'])as image:crop=np.array(transform.transforms[1](transform.transforms[0](image.convert('RGB'))))
            fig,axs=plt.subplots(2,4,figsize=(7.6,4.3))
            for row,s in enumerate((1,2)):
                axs[row,0].imshow(crop);axs[row,0].set_title(f'输入图像（seed{s}）',fontsize=8);axs[row,0].axis('off')
                for ax,m in zip(axs[row,1:],[BASE,'V2-NoTF','G-V2']):
                    detail=next(d for d in grad['details']if d['case']==case and d['model']==m and d['seed']==s)
                    ax.imshow(crop);ax.imshow(z[f'{m}_seed{s}_case{case}_predicted'],cmap='jet',vmin=0,vmax=1,alpha=.43);ax.axis('off')
                    ax.set_title(f'{m}\n{detail["predicted_class_fp32"].split("-",2)[-1]}（{detail["confidence_fp32"]:.2f}）',fontsize=7.5)
            save(fig,OUT,filename,tight_layout_kwargs={'h_pad':1.8})
    blocks={
        'phase_table':table('前45轮损失面积与后25轮最低损失改善','tab:supp-loss-window',['配置','训练损失面积','验证损失面积','后期改善s1','后期改善s2'],phase_rows,'lcccc'),
        'target_table':table('共同验证损失阈值的首次到达耗时（小时）','tab:supp-loss-target',['配置','2.0/s1','2.0/s2','1.8/s1','1.8/s2','1.7/s1','1.7/s2'],target_rows,'lcccccc'),
        'class_table':table('固定基线类别组的召回率与后50类负对数似然','tab:supp-class-groups',['配置',r'前50类(\%)',r'中间50类(\%)',r'后50类(\%)','后50类NLL'],class_rows,'lcccc'),
        'error_table':table('重复运行的错误数量与垃圾大类组成','tab:supp-error-groups',['配置','错误图像数',r'同大类占比(\%)',r'跨大类占比(\%)'],error_rows,'lccc'),
        'cal_table':table('最佳验证模型的输出分数评价','tab:supp-calibration',['配置',r'ECE(\%)','NLL','Brier'],cal_rows,'lccc'),
        'early_fig':figure('early_loss','七种配置前45轮的训练与验证损失。曲线为两seed均值，阴影为两次运行的取值范围。','fig:supp-early-loss'),
        'late_fig':figure('late_loss_lr','两种模型头下的后期验证损失与学习率。上排为后25轮验证损失，下排为完整训练的学习率。','fig:supp-late-lr'),
        'residual_fig':figure('baseline_loss_difference','各配置与S-NoProg配对seed的验证损失差。','fig:supp-loss-difference'),
        'class_fig':figure('bottom50_recall','基线排序后50类的两seed平均召回率。类名左侧为固定排名，括号内为验证样本数。','fig:supp-bottom50',True),
        'cm_fig':figure('difficult_class_confusion','S-NoProg与G-V2在基线排序后16类上的局部混淆矩阵。','fig:supp-difficult-cm'),
        'cal_fig':figure('reliability_risk','输出分数的可靠性曲线与风险--覆盖率。可靠性图中实线与虚线分别对应seed1和seed2。','fig:supp-reliability'),
        'box_fig':figure('gradcam_paperbox','纸箱图像的预测类别Grad-CAM。第一、二行分别对应seed1和seed2，模型名下给出预测类别与最大Softmax分数。','fig:supp-gradcam-box'),
        'printer_fig':figure('gradcam_printer','打印机图像的预测类别Grad-CAM。第一、二行分别对应seed1和seed2。','fig:supp-gradcam-printer'),
    }
    template=r'''
\subsection{重复运行的训练过程与预测分析}\label{sec:repeated-process}

\subsubsection{比较设置与损失度量}
本节在表\ref{tab:formal254}所列七种配置上比较seed1与seed2的重复运行。每次训练100轮，沿用252类标签、固定训练与验证划分、数据增强和优化设置。以S-NoProg作为共同基线，训练过程按相同轮次配对，类别评价使用各次运行按验证准确率选出的模型。

训练过程划分为前45轮（epoch0--44）与后25轮（epoch75--99）。对窗口内的损失按轮次作梯形积分，再除以轮次跨度，得到损失面积均值；该量概括整个窗口的损失水平。共同损失目标的到达耗时取首次满足指定验证损失阈值时的累计训练与验证耗时。后期最低损失改善定义为前75轮最低验证损失减去后25轮最低验证损失，正值对应后期取得了更低的损失。表格中的均值与标准差以两次运行分别计算，曲线阴影给出两次运行的取值范围。

逐类与输出分数评价在14,603张验证图像上进行，使用Resize(256)、CenterCrop(224)和训练集通道统计标准化，批量为32，前向计算采用CUDA混合精度。类别响应图采用FP32梯度。各模型的类别输出顺序与252类标签映射一致。

\subsubsection{前期拟合与后期损失变化}
@@phase_table@@

前45轮中，V2-NoTF的验证损失面积均值为1.7615，V2-Full为1.7962，S-NoProg为2.0014。图\ref{fig:supp-early-loss}中，两种空间头在训练开始阶段下降较快；G-Full-CAWR与G-V2在骨干冻结阶段具有较高的训练损失，解冻后进一步下降。无SE、标准SE和多区域SE的验证损失面积分别为1.9954、1.9937和2.0014，曲线总体接近。

@@early_fig@@

表\ref{tab:supp-loss-target}从共同损失目标比较拟合过程。V2-NoTF在两次运行中均较早达到验证损失2.0与1.8；从轮次曲线和累计耗时两个维度观察，可以同时看到损失下降过程与训练成本。表中横线对应100轮内未达到的阈值。

@@target_table@@

后25轮中，G-Full-CAWR的最低验证损失分别比前75轮降低0.0355和0.0168，G-V2分别降低0.0108和0.0183。其余五种配置在该窗口的最低损失高于前75轮最低值。图\ref{fig:supp-late-lr}显示，全程解冻配置在epoch75进入新的学习率周期，渐进式配置在epoch95附近重启。G系列在epoch75--94继续降低验证损失，重启后出现回升；前者表现为后期新增改善，后者体现了周期切换下的恢复过程。

@@late_fig@@

图\ref{fig:supp-loss-difference}按相同seed逐轮减去S-NoProg的验证损失。空间头配置的差值在大部分轮次低于零；G系列在前期差值较大，随后随解冻阶段推进而下降。该图把初期拟合、阶段切换和后期变化放到共同基线上比较。

@@residual_fig@@

\subsubsection{基线分组的类别表现与局部混淆}
按S-NoProg两次运行的平均逐类召回率对252类降序排列，同分时依次按验证样本数降序、类别索引升序排列。取排名1--50、102--151与203--252作为前、中、后50类，所有模型使用相同类别组。三组分别包含@@group_counts@@张验证图像。逐类召回率为该真实类别中正确分类的比例，组内结果对50个类别等权平均。

@@class_fig@@

@@class_table@@

前50类中，各配置的平均召回率集中在97\%附近；后50类的差异更明显，S-NoProg为53.89\%，G-V2为56.84\%，G-Full-CAWR为56.73\%。图\ref{fig:supp-bottom50}显示了相同50类在各模型下的逐类变化。不同类别的改善方向并不一致：整体组均值的变化由多个类别共同形成。

后50类同时计算逐图负对数似然NLL，即真实类别输出概率的负对数，再按类别等权平均。该指标补充了正确分类比例，反映模型分配给真实类别的分数。图\ref{fig:supp-difficult-cm}选择同一排序的后16类，纵轴为真实类别、横轴为预测类别，使用真实类的完整验证样本数归一化，对角线对应逐类召回率。局部矩阵使模具、餐垫、巧克力、灯罩和铝制用品等低召回类别的分类差异更加清晰。

@@cm_fig@@

\subsubsection{错误组成与输出分数}
将误判按真实与预测标签所属的四种垃圾大类分为同大类与跨大类两部分，结果见表\ref{tab:supp-error-groups}。同大类错误占各配置误判的主要部分，类别边界相近的物体构成了较集中的错误来源。

@@error_table@@

输出分数采用ECE、NLL与Brier评价。ECE将最大Softmax分数按10个等宽区间分组，计算各区间经验正确率与平均最大分数之差的绝对值，并按样本占比加权。NLL取真实类别概率的负对数；Brier对252维概率向量与独热标签之间的平方差求和，再对图像平均。上述分数使用各次运行的最佳验证模型计算。

@@cal_table@@

V2-NoTF的平均ECE为4.01\%，NLL为0.7905，Brier为0.2356；V2-Full分别为6.29\%、0.8505与0.2462。图\ref{fig:supp-reliability}左侧按分数组展示经验正确率，右侧按最大Softmax分数降序保留图像，并计算不同覆盖率下的错误率。两种视角分别呈现分数与正确率的对应关系以及误例在分数排序中的分布。

@@cal_fig@@

\subsubsection{252类模型的同图响应}
采用骨干最后一组卷积特征生成Grad-CAM\cite{ref22}，以预测类别logit对特征图的空间平均梯度作为通道权重，加权求和并取ReLU。响应图分别线性缩放至$[0,1]$，叠加于相同的$224\times224$输入裁剪。相同颜色对应各自响应图中的相对水平。

图\ref{fig:supp-gradcam-box}中，纸箱图像在所展示配置下均被预测为纸制品，各配置在盒体表面、边缘及周边背景上的响应位置有所不同。图\ref{fig:supp-gradcam-printer}中，打印机处于背景物体较多的画面，不同配置和seed对打印机与盒子等类别给出不同预测。相同输入下的类别分数与空间响应共同呈现了这些预测差异。

@@box_fig@@

@@printer_fig@@
'''
    for key,value in blocks.items():template=template.replace('@@'+key+'@@',value)
    template=template.replace('@@group_counts@@','、'.join(f'{group_info[k]["n_images"]:,}'for k in('top','middle','bottom')))
    assert '@@'not in template
    (OUT/'supplement_content.tex').write_text(template,encoding='utf-8')
    data={'baseline':BASE,'windows':{f'{m}_s{s}_{a}_{b}':v for(m,s,a,b),v in windows.items()},
          'late_improvement':{f'{m}_s{s}':v for(m,s),v in late.items()},'class_groups':group_info,
          'class_numbers':group_numbers,'calibration':{f'{m}_s{s}':v for(m,s),v in cal.items()},
          'phase_table':phase_rows,'target_table':target_rows,'class_table':class_rows,'error_table':error_rows,'calibration_table':cal_rows,
          'manifest_fingerprint':manifest['fingerprint'],'source_paths':{'metrics':str(FULL/'metrics_analysis.json'),'predictions':str(FULL/'predictions'),'gradcam':str(FULL/'06_gradcam/gradcam_arrays.npz')},
          'figures':[{'name':p.stem,'sha256':sha256(p)}for p in sorted(OUT.glob('*.pdf'))]}
    (OUT/'supplement_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('SUPPLEMENT_READY',len(data['figures']),'figures, 5 tables',flush=True)


if __name__=='__main__':main()
