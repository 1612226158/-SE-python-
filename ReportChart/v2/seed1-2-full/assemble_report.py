"""Build a bookmarked PDF atlas and a data-driven analysis guide after all stages."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;REPORT=OUT.parents[1]
sys.path.insert(0,str(REPORT))
import numpy as np
import pandas as pd
import pymupdf
from report_inference import sha256


def main():
    metric=json.loads((OUT/'metrics_analysis.json').read_text(encoding='utf-8'))
    prediction=json.loads((OUT/'prediction_analysis.json').read_text(encoding='utf-8'))
    grad=json.loads((OUT/'06_gradcam/gradcam_manifest.json').read_text(encoding='utf-8'))
    figures=[]
    for filename in ['metrics_figures.json','prediction_figures.json','gradcam_figures.json']:
        figures.extend(json.loads((OUT/filename).read_text(encoding='utf-8')))
    figures.append({'path':'05_errors/same_image_prediction_cases','description':'相同六张验证图的14模型预测案例'})
    figures.sort(key=lambda r:(r['path'].split('/')[0],r['path']))
    doc=pymupdf.open();toc=[];last_group=None;page=1
    for figure in figures:
        group=figure['path'].split('/')[0]
        if group!=last_group:
            toc.append([1,group,page]);last_group=group
        source=OUT/(figure['path']+'.pdf')
        with pymupdf.open(source) as pdf:
            if len(pdf)!=1:raise ValueError(f'Expected one-page figure {source}')
            doc.insert_pdf(pdf)
        toc.append([2,figure['description']+' | '+Path(figure['path']).name,page])
        figure['atlas_page']=page;page+=1
    doc.set_toc(toc)
    atlas=OUT/'双seed完整分析图册.pdf'
    doc.save(atlas,garbage=4,deflate=True);doc.close()
    (OUT/'figure_index.json').write_text(json.dumps(figures,ensure_ascii=False,indent=2),encoding='utf-8')
    ws=pd.read_csv(OUT/'window_statistics.csv')
    means=ws[(ws.window=='0-44')&(ws.field=='val_loss')].groupby('model').normalized_auc_epoch.mean().sort_values()
    late=pd.read_csv(OUT/'late_improvement.csv')
    calibration=pd.read_csv(OUT/'calibration_statistics.csv')
    group=pd.read_csv(OUT/'fixed50_group_statistics.csv')
    inference=[]
    for path in sorted((OUT/'predictions').glob('*.json')):
        value=json.loads(path.read_text(encoding='utf-8'))
        inference.append(dict(model=value['model'],seed=value['seed'],
            inference_accuracy_percent=value['accuracy_percent'],
            original_accuracy_percent=value['training_best_accuracy_percent'],
            difference_pp=value['accuracy_percent']-value['training_best_accuracy_percent'],
            inference_seconds=value['inference_seconds'],checkpoint_sha256=value['checkpoint_sha256']))
    pd.DataFrame(inference).to_csv(OUT/'inference_vs_training.csv',index=False,encoding='utf-8-sig')
    models=list(metric['run_summaries'].keys())
    guide=['# 双seed完整模型分析','',
        '入口：[完整图册](双seed完整分析图册.pdf)（带章节/单图书签）；每图均提供同名PNG和PDF。',
        '',f'本批共{len(figures)}组图，覆盖七种配置×seed1/2、1400轮训练记录和14×14,603张验证图预测。',
        '', '## 统一口径','',
        '- 固定基线：S-NoProg；每个run显式指定seed，训练数据只经chart_data公共方法读取。',
        '- 前期：epoch0—44（45轮）；后期：epoch75—99（25轮）。原始loss不平滑，已含训练时label smoothing=0.1。',
        '- 线为两seed均值，填充为min—max范围，不是置信区间；epoch间有关联，不当作独立重复实验。',
        '- 累计耗时曲线以实线/虚线区分seed1/seed2；共同loss目标散点以圆点/叉号区分seed1/seed2，空白表示100轮内未达到。可靠性图以实线/虚线区分seed1/seed2。',
        '- 类别拟合效果采用完整验证集逐类召回率，即真实类内正确比例；没有训练集逐类轨迹，不把它称为训练逐类准确率。',
        '- 类别排名按S-NoProg双seed平均召回率降序；同分按验证support降序、类别索引升序。前/中/后50类均固定这一排名，同类比较。各模型自己的前后50类另存CSV，不能当成相同类群比较。',
        '- 混淆矩阵：纵轴真实、横轴预测。完整252类固定基线排名，标签映射见CSV；前/中/后20类子矩阵均以完整真实类support归一化，不把流向其他类的错误隐藏为行和1。',
        '- 时间是原始time_spend_s累加，含已登记慢轮，不含停机、轮间等待；机器现场与运行日期不同，不作为严格硬件基准。',
        '- 逐类NLL为−log p(真实类)，不含标签平滑，与原训练loss定义不同。校准指标使用原始Softmax概率，未拟合校准器。',
        '- 全验证缓存采用CUDA AMP、batch32；与原训练评价的批量/执行设置可能产生少量预测差异。差值逐run写入inference_vs_training.csv，原正式results/metrics未改。热图梯度单独采用CUDA FP32，并同时登记其预测与AMP预测。',
        '', '## 从loss与效率看结果','',
        '### 前45轮验证loss面积（越低越好）','',
        '| 配置 | 两seed平均归一化AUC |','|---|---:|']
    for model,value in means.items():guide.append(f'| {model} | {value:.4f} |')
    guide += ['',f'前期最低为{means.index[0]}。AUC同时受初始loss和下降过程影响；请结合训练/验证双曲线、每小时斜率、共同loss目标到达时间判断，不能仅用某一次骤降认定更高效率。',
        '', '### 75轮后的恢复与新增提升','',
        '| 配置 | seed1相对前75轮最低loss的改善 | seed2改善 |','|---|---:|---:|']
    for model in means.index:
        subset=late[late.model==model].sort_values('seed')
        vals=subset.additional_best_loss_drop.tolist()
        guide.append(f'| {model} | {vals[0]:+.4f} | {vals[1]:+.4f} |')
    guide += ['', '正值表示后25轮创造了更低的验证loss，负值表示末段仍未恢复到此前最低loss。S/V2全解冻配置epoch75发生重启，而G系列在epoch95附近重启，二者的末段周期相位不同；同组lr图给出实际记录。',
        '', '## 固定基线后50类：双seed类等权召回率','',
        '| 配置 | 均值（%） | seed1 / seed2（%） |','|---|---:|---|']
    tail=group[group.group=='bottom50']
    for model in means.index:
        vals=tail[tail.model==model].sort_values('seed').macro_recall.to_numpy()*100
        guide.append(f'| {model} | {vals.mean():.2f} | {vals[0]:.2f} / {vals[1]:.2f} |')
    guide += ['', '前50/中50/后50类分别提供召回率与NLL热图；样本数直接写在类名旁，完整指标见per_class_metrics_14runs.csv。类别由基线验证结果选择，是本验证集上的描述性分组，不是另一个独立测试集。',
        '', '## 热力图与错误案例','',
        '- 六个案例来自固定随机顺序，再按共同正确、共同错误、seed不一致、模型纠正/退化六种规则选择，并尽量选择不同真实类；不是代表性样本或性能估计。全部图片路径、内容哈希和选择规则见06_gradcam/gradcam_manifest.json。',
        '- 每个案例同时展示两个seed、七种配置，分别绘制“同一真实类梯度目标”和“各模型自身预测类目标”。后者对应实际预测类别；前者便于解释错误时真实类证据。',
        '- Grad-CAM使用backbone layer4输出和真实目标logit梯度，不是套用模板的伪热图，也不是Transformer注意力权重。每张独立归一化[0,1]，跨模型不能直接比较颜色强度；归因图不等同于分割或因果证明。',
        '', '## 补充分析','',
        '错误同/跨垃圾大类分解、有向混淆对、相对基线改善/退化类别、同图seed一致性、可靠性图、风险—覆盖率、ECE/NLL/Brier均已生成。所有统计都是既有最佳checkpoint在同源验证集的分析，不增加训练实验或独立泛化证据。',
        '', '## 文件与复现','',
        '- `evaluate_all.py`：调用公共report_inference，一次准备验证裁剪，逐run缓存预测，可断点复用；不写runs。',
        '- `render_analysis.py --stage metrics`：训练动力学/效率；`--stage predictions`：类别/混淆矩阵/错误/校准。',
        '- `render_gradcam.py`：同图真实Grad-CAM与预测案例。',
        '- `assemble_report.py`：完整图册、索引与本说明；`verify_analysis.py`：独立一致性检查，输出终端。',
        '- 论文获授权补充后，核验使用`verify_analysis.py --protection-manifest 毕业设计/Handover/check_artifacts/2026-10-26/论文缺失实验补充/修订基线.json`，按本轮38个保护文件检查；原始分析的保护清单仍保留为历史回滚依据。',
        '- 公共接口：`chart_data.load_run_frames/window_statistics/results_best/best_ckpt_path`；`eval_confusion_matrix.build_and_load(seed=...)`；`report_inference`验证manifest/预测缓存/Grad-CAM/校准统计；样式全部复用`paper_style`。',
        '- 原始数据、模型实现、训练代码/配置/队列与runs保持不变。论文第6.7节使用`thesis_supplement/`中的8组精简图和5张数据表，补充重复运行分析；既有摘要、主结果表与结论保留。修订与来源核验见`毕业设计/Handover/check_artifacts/2026-10-26/论文缺失实验补充/`。',
        '', '## 图册逐图索引','', '| 页 | 图 | 内容 |','|---:|---|---|']
    for r in figures:guide.append(f'| {r["atlas_page"]} | [{Path(r["path"]).name}]({r["path"]}.png) | {r["description"]} |')
    (OUT/'README.md').write_text('\n'.join(guide)+'\n',encoding='utf-8')
    # Exact numerical provenance without an additional inference pass.
    output_files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ('output_manifest.json','progress.md','task_plan.md')]
    (OUT/'output_manifest.json').write_text(json.dumps({
        'figure_count':len(figures),'atlas_pages':len(figures),'atlas_sha256':sha256(atlas),
        'manifest_fingerprint':prediction['manifest_fingerprint'],
        'public_sources':{str(REPORT/name):sha256(REPORT/name) for name in ['chart_data.py','eval_confusion_matrix.py','report_inference.py','paper_style.py']},
        'files':[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(output_files)]
    },ensure_ascii=False,indent=2),encoding='utf-8')
    print('REPORT_COMPLETE',len(figures),'figures/pages',flush=True)


if __name__=='__main__':main()
