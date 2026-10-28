# S-NoProg / S-NoSE / S-StdSE 对比图

## 对比范围

- 类别体系：252 类，`waste265-merge9-v2`
- 实验：`S-NoProg seed1`、`S-NoSE seed1`、`S-StdSE seed1`
- 三组实验均已完成 100 轮，epoch 为 0--99，无缺失轮次。
- 三组的训练协议版本、采样协议版本、类别数和标签映射哈希一致，可作同协议对比。
- 数据统一通过上级目录的 `chart_data.py` 读取；绘图脚本不直接解析 `runs` 中的 CSV。

## 运行方式

在本目录执行：

```bash
python plot_se_threeway.py
```

脚本会重新生成 8 组 PNG/PDF 图和 `comparison_metrics.json`。运行前会检查三组实验是否完成、轮次是否连续以及协议是否一致；条件不满足时直接报错，不输出容易误解的对比图。

## 风格约定

延续 `ReportChart` 既有论文图风格：白底、浅灰虚线网格、低饱和配色和简洁图例。颜色固定为：

- `S-NoProg`：蓝色 `#4c72b0`
- `S-NoSE`：红色 `#c44e52`
- `S-StdSE`：橙色 `#dd8452`

## 核心结果

| 模型 | SE 设置 | 最佳验证准确率 | 最佳 Macro-F1 | 最佳 Weighted-F1 | 最短验证损失 | 总耗时 |
|---|---|---:|---:|---:|---:|---:|
| S-NoProg | 多区域 SE `[1,2]` | 83.503%（ep71） | **81.126%**（ep74） | 83.407%（ep72） | 1.7572（ep34） | 22.11 h |
| S-NoSE | 无 SE | 83.291%（ep74） | 80.723%（ep73） | 83.228%（ep74） | **1.7458**（ep33） | **21.90 h** |
| S-StdSE | 标准全局 SE `[1]` | **83.736%**（ep74） | 81.030%（ep74） | **83.635%**（ep74） | 1.7495（ep72） | 22.91 h |

相对 `S-NoSE`，`S-StdSE` 的最佳验证准确率提高 0.4451 个百分点，`S-NoProg` 提高 0.2123 个百分点；`S-StdSE` 又比 `S-NoProg` 高 0.2328 个百分点。

## 可以得出的结论

1. 标准 SE 在本次 seed1 中取得最高的 Top-1 和 Weighted-F1，且 100 轮中当轮验证准确率领先 43 次，是三者中最稳定占优的一组。
2. 多区域 SE 没有赢得最高 Top-1，但取得最高 Macro-F1，说明它对类别平均表现可能略有帮助；优势只有约 0.10 个百分点，不能夸大。
3. 无 SE 的最低验证损失最好、训练耗时最短。加入标准 SE 约多耗时 1.00 小时（约 4.6%），换来的最佳 Top-1 增益小于 0.5 个百分点。
4. 三条曲线整体高度重合，差异会随 CAWR 周期改变。第 75 轮重启后，三组都没有恢复第 35--74 轮的历史峰值，说明末段低指标主要是共同调度行为，不是某一种 SE 独有的异常。
5. 目前只有 seed1，不能据此宣称差异具有统计显著性或可稳定复现。论文中适合写成“本次同协议实验观察到的有限增益”，不宜写成“多区域 SE 显著优于标准 SE”。

## 图表目录

1. `01_best_metrics_comparison`：最佳 Top-1、Macro-F1、Weighted-F1。
2. `02_val_acc_comparison`：100 轮验证准确率与最佳轮次。
3. `03_loss_curves`：三组训练/验证损失。
4. `04_f1_comparison`：Macro-F1 与 Weighted-F1 全程曲线。
5. `05_time_acc_comparison`：累计耗时与验证准确率。
6. `06_se_gain_vs_nose`：两条 SE 路线相对无 SE 的 5 轮滑动差值。
7. `07_cawr_cycle_response`：各 CAWR 周期峰值和重启冲击。
8. `08_epoch_lead_heatmap`：逐轮领先关系与领先次数。

机器可复核的完整汇总、阈值到达时间和各周期统计见 `comparison_metrics.json`。

## Full / NoTF 与SE路线对比（2026-10-02，UI-86）

新增产物目录：[full_notf_comparison](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/full_notf_comparison/comparison_analysis.json)，包含7组PNG/PDF、机器汇总和[可重复绘图脚本](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/full_notf_comparison/plot_full_notf_comparison.py)。不新增Markdown；原SE三组图保持原样。

| 对比 | 可解释范围 | 限制 |
|---|---|---|
| V2-Full vs V2-NoTF | 同V2结构3/0层注意力栈；saved model_metadata除transformer_layers相同 | 仅seed1；协议/batch/样本数一致不等于全部历史超参数已留完整快照 |
| S-NoSE vs S-StdSE vs S-NoProg | v1架构内无SE/[1]/[1,2]对照 | 旧S结果无saved model_metadata，架构依据注册表和实现，不伪造恢复元数据 |
| V2-Full/NoTF vs S-NoProg | 固定SE[1,2]的整套头部路线比较 | token/池化/位置编码/正则等同时变化，不是纯SE或纯Transformer效应 |
| 五组总体 | 性能及参数/历史轮速的描述性对照 | 不从混合架构的排序归因单模块；G系列尚未收官不纳入 |

### 图表索引

| 图 | PNG |
|---|---|
| 01 完整与后期Accuracy | [01_full_notf_accuracy](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/full_notf_comparison/01_full_notf_accuracy.png) |
| 02 Full/NoTF Macro-F1与Loss | [02_full_notf_f1_loss](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/full_notf_comparison/02_full_notf_f1_loss.png) |
| 03 逐轮差值与训练/验证差距 | [03_full_notf_difference_gap](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/full_notf_comparison/03_full_notf_difference_gap.png) |
| 04 五组best Accuracy与同轮Macro-F1 | [04_five_models_best_metrics](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/full_notf_comparison/04_five_models_best_metrics.png) |
| 05 五组验证曲线 | [05_five_models_curves](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/full_notf_comparison/05_five_models_curves.png) |
| 06 同SE[1,2]的头部路线 | [06_same_se_head_routes](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/full_notf_comparison/06_same_se_head_routes.png) |
| 07 参数量与有效轮耗时分布 | [07_parameters_epoch_time](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/full_notf_comparison/07_parameters_epoch_time.png) |

每图同名PDF可用于导出。Accuracy best来自正式results；04的Macro-F1来自正式最佳Accuracy对应epoch，而非独立选最高F1轮。逐轮数据只经chart_data统一加载；脚本检查finished、0—99连续、协议/batch/样本数/解冻方式一致、Full/NoTF结构身份。

结果：NoTF最佳Accuracy84.6128%、Full84.0512%，差0.5616个百分点；NoTF/Full相对S-NoProg分别+1.1094/+0.5478个百分点。仅本次seed1。NoTF本次100个epoch验证Accuracy均高于Full；不据相邻epoch当独立重复样本做显著性结论。

效率口径：NoTF epoch63原值5728.7926187515秒仅从耗时分布/均值中排除，其他指标与原始文件保持；NoTF n=99，其余n=100，不估算补值、不绘清洗后累计时间或time-to-accuracy。有效平均已记录轮耗时Full15.8053分钟、NoTF13.3407分钟；历史运行轮速受现场影响，不当严格硬件基准。汇总summaries_raw内原累计小时仍含异常，明确不作清洗后效率。

验证：7组PNG已逐张目视检查，无标题/图注/标签重叠或截断；PNG/PDF与严格JSON成功生成。论文/预实施文件SHA256保持不变，不改训练代码、队列或产物。数据划分内容哈希未在旧run留存，当前匹配不冒充新manifest核验。

### 拟合速度与泛化补充（2026-10-02，UI-91）

新增[08训练/验证Loss与泛化差距](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/full_notf_comparison/08_fitting_generalization.png)和[09首次达到同准确率的轮数/耗时](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/full_notf_comparison/09_accuracy_threshold_speed.png)，均有同名PDF。复用脚本为full_notf_comparison/plot_fitting_generalization.py，完整机器汇总为fitting_generalization_analysis.json；逐轮数据仍使用chart_data，源指纹与限制随汇总保留。

NoTF训练Loss100/100轮较低，验证Loss88/100轮较低。两组在epoch5/15/35/75重启时均回升，不以锯齿认定发散或过拟合；NoTF验证Loss全程最低1.565922@33，Full1.587060@71，最低Loss不是最佳Accuracy选模点，也不等于已永久收敛。最终epoch99验证Loss为Full1.679194/NoTF1.684369，NoTF并非全指标每轮领先。

首次达到80/82/83%验证Accuracy的epoch为Full27/30/63、NoTF12/29/32，对应实际记录累计小时Full7.9209/8.7175/17.1014、NoTF2.7953/6.4341/7.0767。这里NoTF全部阈值早于异常epoch63，因此可直接使用原始前缀耗时；没有估算缺失值、生成清洗全程累计或比较跨异常的84%时间。首次跨阈值不代表持续保持：连续5轮不低于80/82/83%的起始epoch分别Full27/30/63、NoTF25/29/60；需观察至起始+4轮才确认。

两组best Accuracy同为epoch74：Full/NoTF训练Loss1.057423/1.034701，验证Loss1.588569/1.568517，Accuracy84.0512/84.6128%，Macro-F1 81.4468/82.2040%，训练减验证Accuracy差13.8639/13.5534pp。epoch75—99平均Accuracy79.8822/80.7511%、平均验证Loss1.709770/1.701202，NoTF后期整体仍较好。训练含加权采样/增强及train模式，gap不是同分布clean-train评估，不直接当过拟合强度。仅seed1/现有验证集证据，无多seed或独立测试/跨域泛化结论。当前实现loss含label_smoothing=0.1；不以loss约1认定未拟合。

补充验证：脚本执行成功，完整100轮/字段有限/结构身份/协议检查通过，两组新增PNG目視无截断/重叠，PDF/JSON成功。无临时核验文件或日志；不改训练/runs/论文/预实施，不代提交Git。

### 渐进式与全解冻时间--精度对照（2026-10-02，UI-113）

[损失与学习率](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/progressive_comparison/01_loss_lr.png)、[验证准确率](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/progressive_comparison/02_val_accuracy.png)、[累计耗时与准确率](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/progressive_comparison/03_time_accuracy.png)均有同名PDF。复现脚本progressive_comparison/plot_progressive_comparison.py；机器汇总comparison_analysis.json含统一读取summary、首次共同阈值时间、各自best时间与源SHA256。

G-Full-CAWR/S-NoProg seed1各100轮，累计16.225/22.107h、best82.7844/83.5034%，完整时间减少26.6%且best低0.719pp；首次80%为5.628/6.494h、82%13.633/13.684h。保留全部慢轮，不包括停机或轮间守护、不插值；单seed与不同运行时段，不当严格硬件基准。解冻策略还包含阶段重建AdamW/CAWR及首阶段weight_decay差异。loss在固定解冻前仍改善，阶段切换/重启共同作用，自适应仅待验证方向，未实施。


### 七组seed1收官汇总（2026-10-03，UI-118）

[统一采集与绘图脚本](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_summary/collect_seed1_results.py)通过chart_data显式读取seed1；[收官汇总JSON](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_summary/seed1_results_summary.json)保留七组0—99共700行、正式results、同best轮F1/loss、末轮、最小loss、累计耗时、首次/连续5轮阈值、分阶段耗时和源哈希。读取时间13:58:42；S-NoProg seed2只有epoch0，其余未开始，均未混入本汇总。

[01七组准确率与Macro-F1](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_summary/01_seed1_metrics.png)、[02 G-V2/Full loss与LR](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_summary/02_gv2_full_loss_lr.png)、[03 G-V2/Full准确率与累计时间](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_summary/03_gv2_full_accuracy_time.png)、[04两种策略下的头部路线](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_summary/04_head_strategy_routes.png)均有同名PDF，四张PNG已逐张目视。论文引用02/03，其余提供完整结果入口。

G-V2正式best83.7910%@93、Macro-F1 81.4551%、Weighted-F1 83.7948%、val loss1.5664、总15.2084h；对V2-Full best低0.2602pp、总记录用时少42.3%。首次80/82/83%累计4.5508/6.2007/13.1835h；连续5轮保持82%的确认13.4088h比Full9.7810h更慢，不概括为所有稳定目标更快。对G-Full-CAWR，G-V2 best高1.0066pp、Macro-F1高0.9154pp、记录用时少6.3%；是完整头部路线及训练策略比较，不是纯注意力/冻结单变量。仅单seed同源验证，无统计显著性或独立test结论。NoTF epoch63只排单轮时间分布，原累计不压缩或补值。

252类全量误差分析：复用[现有评估工具的CPU入口](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_summary/evaluate_seed1_best.py)，选V2-NoTF seed1最佳epoch74，threads4，产物位于seed1_summary/evaluation/，不覆盖旧254类混淆矩阵，也不占GPU。已完成14603张/252类，正确12357、错误2246；CPU Accuracy84.6196%、Macro-F1 82.2107%、Weighted-F1 84.5507%；原训练Accuracy84.6128%保留，差0.0068pp，不断言具体预测变化。

[评价汇总与输出SHA256](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_summary/evaluation/evaluation_summary.json)、[高频混淆/完整分母热图](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_summary/evaluation/confusion_top_pairs_V2-NoTF_preview.png)、[252类召回与最低10类](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_summary/evaluation/per_class_recall_V2-NoTF_preview.png)；两图均有同名PDF（无_preview后缀）。同目录保存confusion_V2-NoTF.npz、preds_V2-NoTF.csv、per_class_metrics_V2-NoTF.csv、top_confusions_V2-NoTF.csv。top200不是完整错误，完整统计用NPZ。

公共stage_plot原用所选子矩阵行和，现用完整真值类support，对角线为实际召回率，子集行和可小于100%；xlabel用实际类数。推理stage_infer AST不变；`python ReportChart/v2/seed1_summary/evaluate_seed1_best.py --report-only`可重生成报告、不再推理。旧254类图保持历史原物。执行推理/绘图源码SHA256及旧源码备份分别登记。

CPU推理14:08:27—14:27:14，报告14:27:58，threads4；未占GPU或干预seed2进程，但可能共享CPU/磁盘，不能假定训练计时无影响或自行排除。只有NoTF完成本轮逐图评价，其他六组/独立test/盲标尚未完成，不做模型间McNemar或逐图显著性结论。


### 当前论文统一图表（2026-10-03，UI-119—147，主稿v5.0.0）

论文数字图现全部引用[统一导出目录](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/paper_figures/)，前述各日期图组保留为历史导出，不能据其样式或旧耗时处理推断当前稿。复现入口[render_paper_figures.py](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/paper_figures/render_paper_figures.py)，共享[paper_style.py](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/paper_style.py)。通过chart_data.load_model显式读取七组seed1，与13:58:42固定700行快照核对；历史长尾/254类混淆及当前252类误差分别使用既有CSV/NPZ，不启动训练或推理。

18组PDF与PNG覆盖渐进/全解冻、空间头Full/NoTF/原头、SE三组、长尾、当前与历史混淆/召回、NoTF训练示例。统一7.6×5.2英寸（547.2×374.4pt）、微软雅黑、中文标题与坐标、固定配置颜色、标题11/坐标10/刻度及图例9pt，密集类别标签6.5—8pt。PNG由同PDF以300dpi直接渲染为2280×1560，避免渲染后端造成标签偏移；图内不写备注性结论或风险说明。表格统一三线表，图注只标对象与必要条件。

本稿耗时协议替代以前只排NoTF第63轮的分布统计：七组主统计均保留全部100轮，报告累计、中位数与四分位；可选敏感性对全部配置共同移除epoch63，均为99轮。不删原始metrics或精度、loss/F1/checkpoint，不估算正常时间，累计和共同精度曲线保持原值。Full/NoTF中位数15.8548/12.8780分钟，共同99轮均值15.8194/13.3407分钟。阈值柱图用完成轮数=epoch+1，其余轨迹横轴保留0—99的epoch编号。不同运行时段不是严格硬件基准。

[统一样式、计时和来源哈希](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/paper_figures/paper_analysis.json)；[29项落实与成稿核验](E:/DataSet/垃圾分类图片-2/毕业设计/Handover/check_artifacts/2026-10-03/论文29项修订/逐项处理与核验记录.json)。本轮无新增训练、推理、数据修改或runs附属文件。

### seed1箱线图与降维（2026-10-05，UI159/160）

[21页图册](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_boxplots_embeddings/seed1_comparison_atlas.pdf)汇集六张箱线图、一张训练状态PCA及七模型各一张图像特征PCA/t-SNE。单图PDF/300dpi PNG和可复用分析脚本均在同目录，沿用统一尺寸、中文标签与配置颜色。

| 图册页 | 内容 / 单图PNG |
|---|---|
| 1 | [全100轮验证准确率](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_boxplots_embeddings/01_accuracy_all.png) |
| 2 | [末25轮验证准确率](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_boxplots_embeddings/02_accuracy_last25.png) |
| 3 | [末25轮Macro-F1](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_boxplots_embeddings/03_macro_f1_last25.png) |
| 4 | [末25轮验证损失](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_boxplots_embeddings/04_loss_last25.png) |
| 5 | [全100轮单轮耗时](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_boxplots_embeddings/05_time_all.png) |
| 6 | [全组共同排除epoch63的99轮耗时](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_boxplots_embeddings/06_time_common99.png) |
| 7 | [训练状态PCA](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_boxplots_embeddings/07_training_state_pca.png) |
| 8/9 | S-NoProg 图像特征PCA/t-SNE |
| 10/11 | S-NoSE 图像特征PCA/t-SNE |
| 12/13 | S-StdSE 图像特征PCA/t-SNE |
| 14/15 | V2-Full 图像特征PCA/t-SNE |
| 16/17 | V2-NoTF 图像特征PCA/t-SNE |
| 18/19 | G-Full-CAWR 图像特征PCA/t-SNE |
| 20/21 | G-V2 图像特征PCA/t-SNE |

逐轮指标通过公共chart_data显式seed1读取，七组0—99共700行与收官快照匹配（数值表示舍入容差1e-9）。箱体Q1—Q3、中线中位数、须1.5 IQR，圆点为超出须的观测值，不能自动视为数据错误。末25轮固定epoch75—99；全100轮保留NoTF异常95.48分钟观测，共同99轮对所有配置统一排63，均不删除原始记录。连续epoch不是独立重复seed，箱线图不能证明跨seed稳健性或显著性。

训练状态PCA用train_acc/train_loss/val_acc/val_loss/val_macro_f1五变量，对合并700点统一标准化后拟合；X形为epoch99，线按轮次相连。均值、尺度、载荷、解释方差和坐标保存在metrics_analysis.json。

图像降维固定锅/电饭煲、盒子/纸箱、鞋/拖鞋、水杯/保温杯八类，每类随机种子20261005取30张验证图，共240张；原始路径/图像SHA256、公共映射及公共验证预处理均记录于image_sample_manifest.json。使用公共模型加载器，在本分析进程限定七份checkpoint_*_seed1_best.pth并验证内部配置/seed/最佳epoch/标签身份，不用自动seed选择。CPU FP32、两线程、batch8，不使用GPU；分类器parent_classifier输入512维特征，原始特征/真实类/预测与缓存身份保存在features_*.npz。

每模型独立对L2归一化特征做PCA；t-SNE先独立PCA到50维，再perplexity30、初始化PCA、learning_rate=auto、1000迭代、随机种子20261005。所有图类别颜色和样本相同，但不同模型的坐标轴、旋转及簇间全局距离不可直接比较；仅八类验证子集，不代表252类全体或独立测试泛化。image_feature_analysis.json含七份权重SHA256、推理时间、子集预测表现、投影坐标/参数与解释边界；子集准确率不能替换正式全验证集指标。

复现：系统Python加-B运行同目录plot_seed1_analysis.py；缓存样本/预处理/权重SHA相符时直接复用特征，避免重复CPU推理。本分析不修改论文、训练代码、原数据或runs产物，不进入新增v3实验；预实施不更新、不复审。

### 类别中心余弦相似度热力图（2026-10-05，UI161）

[八页图册](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_boxplots_embeddings/centroid_heatmaps/class_centroid_heatmaps_atlas.pdf)含七模型各一张8×8中心相似度及[四组类对跨模型对照](E:/DataSet/垃圾分类图片-2/src_v2/ReportChart/v2/seed1_boxplots_embeddings/centroid_heatmaps/centroid_pair_comparison.png)。单图同名PDF/PNG位于centroid_heatmaps子目录；类别顺序、色阶[-1,1]、RdBu_r、数字两位小数及共享论文样式一致。红色越深，中心方向越相近；对角线为1，不是识别率。

复用此前同8类各30张验证图的512维分类头前特征，按真实类计算：每图L2归一化→类内均值→类中心L2归一化→中心两两点积。七模型均在自己的特征空间中计算，不混合不同模型坐标，也不用PCA/t-SNE二维坐标。类中心只体现平均方向，不反映类内多峰/离散程度；相似度不是混淆矩阵或错误率，不单凭较低中心相似度判断更好的分类或泛化。锅/电饭煲中心相似度七组约0.94—0.97；仅此八类固定子集，不外推252类全体。

centroid_analysis.json保存完整8×8矩阵、均值中心/单位中心、类内平均方向长度、四组对照值、源NPZ/清单/元数据SHA及输出SHA；plot_centroid_heatmaps.py可直接用系统Python -B复现，无需再次模型推理。八图目视、对称性/对角线/维度/样本身份、九份PDF及PNG尺寸、缓存未改校验通过。论文/main/预实施未改，不进入新增v3实验，不向runs写文件。


## 2026-10-26 seed1/2完整分析

[结果说明与逐图索引](seed1-2-full/README.md) · [完整图册](seed1-2-full/双seed完整分析图册.pdf)。包括固定S-NoProg基线的loss动力学、前45轮/75轮后效率、两seed范围与残差填充、真实lr、固定前中后50类召回率/NLL、252类排序及分段混淆矩阵、错误分解、一致性、真实Grad-CAM、可靠性与风险覆盖率。分析缓存与复现脚本在同目录；训练数据由公共chart_data读取，预测/热图由公共评估器及report_inference生成，沿用paper_style样式。不改原始训练结果或论文。
