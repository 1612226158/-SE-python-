# 双seed完整模型分析

入口：[完整图册](双seed完整分析图册.pdf)（带章节/单图书签）；每图均提供同名PNG和PDF。

本批共85组图，覆盖七种配置×seed1/2、1400轮训练记录和14×14,603张验证图预测。

## 统一口径

- 固定基线：S-NoProg；每个run显式指定seed，训练数据只经chart_data公共方法读取。
- 前期：epoch0—44（45轮）；后期：epoch75—99（25轮）。原始loss不平滑，已含训练时label smoothing=0.1。
- 线为两seed均值，填充为min—max范围，不是置信区间；epoch间有关联，不当作独立重复实验。
- 累计耗时曲线以实线/虚线区分seed1/seed2；共同loss目标散点以圆点/叉号区分seed1/seed2，空白表示100轮内未达到。可靠性图以实线/虚线区分seed1/seed2。
- 类别拟合效果采用完整验证集逐类召回率，即真实类内正确比例；没有训练集逐类轨迹，不把它称为训练逐类准确率。
- 类别排名按S-NoProg双seed平均召回率降序；同分按验证support降序、类别索引升序。前/中/后50类均固定这一排名，同类比较。各模型自己的前后50类另存CSV，不能当成相同类群比较。
- 混淆矩阵：纵轴真实、横轴预测。完整252类固定基线排名，标签映射见CSV；前/中/后20类子矩阵均以完整真实类support归一化，不把流向其他类的错误隐藏为行和1。
- 时间是原始time_spend_s累加，含已登记慢轮，不含停机、轮间等待；机器现场与运行日期不同，不作为严格硬件基准。
- 逐类NLL为−log p(真实类)，不含标签平滑，与原训练loss定义不同。校准指标使用原始Softmax概率，未拟合校准器。
- 全验证缓存采用CUDA AMP、batch32；与原训练评价的批量/执行设置可能产生少量预测差异。差值逐run写入inference_vs_training.csv，原正式results/metrics未改。热图梯度单独采用CUDA FP32，并同时登记其预测与AMP预测。

## 从loss与效率看结果

### 前45轮验证loss面积（越低越好）

| 配置 | 两seed平均归一化AUC |
|---|---:|
| V2-NoTF | 1.7615 |
| V2-Full | 1.7962 |
| S-StdSE | 1.9937 |
| S-NoSE | 1.9954 |
| S-NoProg | 2.0014 |
| G-V2 | 2.0998 |
| G-Full-CAWR | 2.3701 |

前期最低为V2-NoTF。AUC同时受初始loss和下降过程影响；请结合训练/验证双曲线、每小时斜率、共同loss目标到达时间判断，不能仅用某一次骤降认定更高效率。

### 75轮后的恢复与新增提升

| 配置 | seed1相对前75轮最低loss的改善 | seed2改善 |
|---|---:|---:|
| V2-NoTF | -0.1102 | -0.0963 |
| V2-Full | -0.0921 | -0.0854 |
| S-StdSE | -0.1521 | -0.1712 |
| S-NoSE | -0.1536 | -0.1485 |
| S-NoProg | -0.1595 | -0.1577 |
| G-V2 | +0.0108 | +0.0183 |
| G-Full-CAWR | +0.0355 | +0.0168 |

正值表示后25轮创造了更低的验证loss，负值表示末段仍未恢复到此前最低loss。S/V2全解冻配置epoch75发生重启，而G系列在epoch95附近重启，二者的末段周期相位不同；同组lr图给出实际记录。

## 固定基线后50类：双seed类等权召回率

| 配置 | 均值（%） | seed1 / seed2（%） |
|---|---:|---|
| V2-NoTF | 54.65 | 55.73 / 53.57 |
| V2-Full | 54.41 | 54.87 / 53.95 |
| S-StdSE | 54.56 | 53.79 / 55.34 |
| S-NoSE | 55.44 | 55.83 / 55.05 |
| S-NoProg | 53.89 | 53.99 / 53.78 |
| G-V2 | 56.84 | 56.75 / 56.94 |
| G-Full-CAWR | 56.73 | 55.80 / 57.66 |

前50/中50/后50类分别提供召回率与NLL热图；样本数直接写在类名旁，完整指标见per_class_metrics_14runs.csv。类别由基线验证结果选择，是本验证集上的描述性分组，不是另一个独立测试集。

## 热力图与错误案例

- 六个案例来自固定随机顺序，再按共同正确、共同错误、seed不一致、模型纠正/退化六种规则选择，并尽量选择不同真实类；不是代表性样本或性能估计。全部图片路径、内容哈希和选择规则见06_gradcam/gradcam_manifest.json。
- 每个案例同时展示两个seed、七种配置，分别绘制“同一真实类梯度目标”和“各模型自身预测类目标”。后者对应实际预测类别；前者便于解释错误时真实类证据。
- Grad-CAM使用backbone layer4输出和真实目标logit梯度，不是套用模板的伪热图，也不是Transformer注意力权重。每张独立归一化[0,1]，跨模型不能直接比较颜色强度；归因图不等同于分割或因果证明。

## 补充分析

错误同/跨垃圾大类分解、有向混淆对、相对基线改善/退化类别、同图seed一致性、可靠性图、风险—覆盖率、ECE/NLL/Brier均已生成。所有统计都是既有最佳checkpoint在同源验证集的分析，不增加训练实验或独立泛化证据。

## 文件与复现

- `evaluate_all.py`：调用公共report_inference，一次准备验证裁剪，逐run缓存预测，可断点复用；不写runs。
- `render_analysis.py --stage metrics`：训练动力学/效率；`--stage predictions`：类别/混淆矩阵/错误/校准。
- `render_gradcam.py`：同图真实Grad-CAM与预测案例。
- `assemble_report.py`：完整图册、索引与本说明；`verify_analysis.py`：独立一致性检查，输出终端。
- 论文获授权补充后，核验使用`verify_analysis.py --protection-manifest 毕业设计/Handover/check_artifacts/2026-10-26/论文缺失实验补充/修订基线.json`，按本轮38个保护文件检查；原始分析的保护清单仍保留为历史回滚依据。
- 公共接口：`chart_data.load_run_frames/window_statistics/results_best/best_ckpt_path`；`eval_confusion_matrix.build_and_load(seed=...)`；`report_inference`验证manifest/预测缓存/Grad-CAM/校准统计；样式全部复用`paper_style`。
- 原始数据、模型实现、训练代码/配置/队列与runs保持不变。论文第6.7节使用`thesis_supplement/`中的8组精简图和5张数据表，补充重复运行分析；既有摘要、主结果表与结论保留。修订与来源核验见`毕业设计/Handover/check_artifacts/2026-10-26/论文缺失实验补充/`。

## 图册逐图索引

| 页 | 图 | 内容 |
|---:|---|---|
| 1 | [baseline_residual_early](01_dynamics/baseline_residual_early.png) | 配对seed验证loss残差与正负填充 |
| 2 | [baseline_residual_full](01_dynamics/baseline_residual_full.png) | 配对seed验证loss残差与正负填充 |
| 3 | [baseline_residual_late](01_dynamics/baseline_residual_late.png) | 配对seed验证loss残差与正负填充 |
| 4 | [heads_early_loss](01_dynamics/heads_early_loss.png) | 全解冻模型头训练与验证损失（epoch 0—44） |
| 5 | [heads_full_loss](01_dynamics/heads_full_loss.png) | 全解冻模型头训练与验证损失（epoch 0—99） |
| 6 | [heads_late_loss](01_dynamics/heads_late_loss.png) | 全解冻模型头训练与验证损失（epoch 75—99） |
| 7 | [heads_lr](01_dynamics/heads_lr.png) | 全解冻模型头同组学习率两seed |
| 8 | [loss_gap](01_dynamics/loss_gap.png) | 训练/验证loss差异和两seed范围 |
| 9 | [progressive_global_early_loss](01_dynamics/progressive_global_early_loss.png) | 原单token头训练策略训练与验证损失（epoch 0—44） |
| 10 | [progressive_global_full_loss](01_dynamics/progressive_global_full_loss.png) | 原单token头训练策略训练与验证损失（epoch 0—99） |
| 11 | [progressive_global_late_loss](01_dynamics/progressive_global_late_loss.png) | 原单token头训练策略训练与验证损失（epoch 75—99） |
| 12 | [progressive_global_lr](01_dynamics/progressive_global_lr.png) | 原单token头训练策略同组学习率两seed |
| 13 | [progressive_spatial_early_loss](01_dynamics/progressive_spatial_early_loss.png) | 空间头训练策略训练与验证损失（epoch 0—44） |
| 14 | [progressive_spatial_full_loss](01_dynamics/progressive_spatial_full_loss.png) | 空间头训练策略训练与验证损失（epoch 0—99） |
| 15 | [progressive_spatial_late_loss](01_dynamics/progressive_spatial_late_loss.png) | 空间头训练策略训练与验证损失（epoch 75—99） |
| 16 | [progressive_spatial_lr](01_dynamics/progressive_spatial_lr.png) | 空间头训练策略同组学习率两seed |
| 17 | [se_early_loss](01_dynamics/se_early_loss.png) | SE结构训练与验证损失（epoch 0—44） |
| 18 | [se_full_loss](01_dynamics/se_full_loss.png) | SE结构训练与验证损失（epoch 0—99） |
| 19 | [se_late_loss](01_dynamics/se_late_loss.png) | SE结构训练与验证损失（epoch 75—99） |
| 20 | [se_lr](01_dynamics/se_lr.png) | SE结构同组学习率两seed |
| 21 | [heads_loss_time](02_efficiency/heads_loss_time.png) | 全解冻模型头损失与累计训练耗时 |
| 22 | [late_recovery_vs_new_best](02_efficiency/late_recovery_vs_new_best.png) | 区分75轮重启后恢复与真正新增的最低loss |
| 23 | [phase_efficiency](02_efficiency/phase_efficiency.png) | 固定窗口loss AUC与每小时下降斜率 |
| 24 | [progressive_global_loss_time](02_efficiency/progressive_global_loss_time.png) | 原单token头训练策略损失与累计训练耗时 |
| 25 | [progressive_spatial_loss_time](02_efficiency/progressive_spatial_loss_time.png) | 空间头训练策略损失与累计训练耗时 |
| 26 | [se_loss_time](02_efficiency/se_loss_time.png) | SE结构损失与累计训练耗时 |
| 27 | [shared_loss_targets](02_efficiency/shared_loss_targets.png) | 共同loss目标的首次到达时间 |
| 28 | [bottom50_fixed_class_nll](03_classes/bottom50_fixed_class_nll.png) | 固定bottom50类逐图NLL，揭示正确率之外的真实类概率 |
| 29 | [bottom50_fixed_class_recall](03_classes/bottom50_fixed_class_recall.png) | 固定bottom50类同一集合，显示原排名和全量样本数 |
| 30 | [class_gain_loss_vs_baseline](03_classes/class_gain_loss_vs_baseline.png) | 相对基线的类别召回率变化 |
| 31 | [fixed50_group_comparison](03_classes/fixed50_group_comparison.png) | 固定三组50类比较，类等权宏召回率 |
| 32 | [middle50_fixed_class_nll](03_classes/middle50_fixed_class_nll.png) | 固定middle50类逐图NLL，揭示正确率之外的真实类概率 |
| 33 | [middle50_fixed_class_recall](03_classes/middle50_fixed_class_recall.png) | 固定middle50类同一集合，显示原排名和全量样本数 |
| 34 | [top50_fixed_class_nll](03_classes/top50_fixed_class_nll.png) | 固定top50类逐图NLL，揭示正确率之外的真实类概率 |
| 35 | [top50_fixed_class_recall](03_classes/top50_fixed_class_recall.png) | 固定top50类同一集合，显示原排名和全量样本数 |
| 36 | [G-Full-CAWR_error20_seed_compare](04_confusion/G-Full-CAWR_error20_seed_compare.png) | 各模型错误相关20类的双seed并列混淆矩阵 |
| 37 | [G-Full-CAWR_full252_sorted](04_confusion/G-Full-CAWR_full252_sorted.png) | 全252类按同一基线排名降序，完整分母 |
| 38 | [G-Full-CAWR_segments20](04_confusion/G-Full-CAWR_segments20.png) | 基线排序前、中、后20类的局部混淆矩阵 |
| 39 | [G-V2_error20_seed_compare](04_confusion/G-V2_error20_seed_compare.png) | 各模型错误相关20类的双seed并列混淆矩阵 |
| 40 | [G-V2_full252_sorted](04_confusion/G-V2_full252_sorted.png) | 全252类按同一基线排名降序，完整分母 |
| 41 | [G-V2_segments20](04_confusion/G-V2_segments20.png) | 基线排序前、中、后20类的局部混淆矩阵 |
| 42 | [S-NoProg_error20_seed_compare](04_confusion/S-NoProg_error20_seed_compare.png) | 各模型错误相关20类的双seed并列混淆矩阵 |
| 43 | [S-NoProg_full252_sorted](04_confusion/S-NoProg_full252_sorted.png) | 全252类按同一基线排名降序，完整分母 |
| 44 | [S-NoProg_segments20](04_confusion/S-NoProg_segments20.png) | 基线排序前、中、后20类的局部混淆矩阵 |
| 45 | [S-NoSE_error20_seed_compare](04_confusion/S-NoSE_error20_seed_compare.png) | 各模型错误相关20类的双seed并列混淆矩阵 |
| 46 | [S-NoSE_full252_sorted](04_confusion/S-NoSE_full252_sorted.png) | 全252类按同一基线排名降序，完整分母 |
| 47 | [S-NoSE_segments20](04_confusion/S-NoSE_segments20.png) | 基线排序前、中、后20类的局部混淆矩阵 |
| 48 | [S-StdSE_error20_seed_compare](04_confusion/S-StdSE_error20_seed_compare.png) | 各模型错误相关20类的双seed并列混淆矩阵 |
| 49 | [S-StdSE_full252_sorted](04_confusion/S-StdSE_full252_sorted.png) | 全252类按同一基线排名降序，完整分母 |
| 50 | [S-StdSE_segments20](04_confusion/S-StdSE_segments20.png) | 基线排序前、中、后20类的局部混淆矩阵 |
| 51 | [V2-Full_error20_seed_compare](04_confusion/V2-Full_error20_seed_compare.png) | 各模型错误相关20类的双seed并列混淆矩阵 |
| 52 | [V2-Full_full252_sorted](04_confusion/V2-Full_full252_sorted.png) | 全252类按同一基线排名降序，完整分母 |
| 53 | [V2-Full_segments20](04_confusion/V2-Full_segments20.png) | 基线排序前、中、后20类的局部混淆矩阵 |
| 54 | [V2-NoTF_error20_seed_compare](04_confusion/V2-NoTF_error20_seed_compare.png) | 各模型错误相关20类的双seed并列混淆矩阵 |
| 55 | [V2-NoTF_full252_sorted](04_confusion/V2-NoTF_full252_sorted.png) | 全252类按同一基线排名降序，完整分母 |
| 56 | [V2-NoTF_segments20](04_confusion/V2-NoTF_segments20.png) | 基线排序前、中、后20类的局部混淆矩阵 |
| 57 | [parent_error_decomposition](05_errors/parent_error_decomposition.png) | 错误在同垃圾大类/跨大类间的数量及比例 |
| 58 | [same_image_prediction_cases](05_errors/same_image_prediction_cases.png) | 相同六张验证图的14模型预测案例 |
| 59 | [seed_prediction_consistency](05_errors/seed_prediction_consistency.png) | 两个seed的共同正确、单seed正确与共同错误 |
| 60 | [top_directed_confusions](05_errors/top_directed_confusions.png) | 每模型真实类别→预测类别的错误流量 |
| 61 | [case01_seed1_predicted](06_gradcam/case01_seed1_predicted.png) | 两seed均正确 seed1 predicted Grad-CAM |
| 62 | [case01_seed1_truth](06_gradcam/case01_seed1_truth.png) | 两seed均正确 seed1 truth Grad-CAM |
| 63 | [case01_seed2_predicted](06_gradcam/case01_seed2_predicted.png) | 两seed均正确 seed2 predicted Grad-CAM |
| 64 | [case01_seed2_truth](06_gradcam/case01_seed2_truth.png) | 两seed均正确 seed2 truth Grad-CAM |
| 65 | [case02_seed1_predicted](06_gradcam/case02_seed1_predicted.png) | 两seed均错误 seed1 predicted Grad-CAM |
| 66 | [case02_seed1_truth](06_gradcam/case02_seed1_truth.png) | 两seed均错误 seed1 truth Grad-CAM |
| 67 | [case02_seed2_predicted](06_gradcam/case02_seed2_predicted.png) | 两seed均错误 seed2 predicted Grad-CAM |
| 68 | [case02_seed2_truth](06_gradcam/case02_seed2_truth.png) | 两seed均错误 seed2 truth Grad-CAM |
| 69 | [case03_seed1_predicted](06_gradcam/case03_seed1_predicted.png) | 基线seed不一致 seed1 predicted Grad-CAM |
| 70 | [case03_seed1_truth](06_gradcam/case03_seed1_truth.png) | 基线seed不一致 seed1 truth Grad-CAM |
| 71 | [case03_seed2_predicted](06_gradcam/case03_seed2_predicted.png) | 基线seed不一致 seed2 predicted Grad-CAM |
| 72 | [case03_seed2_truth](06_gradcam/case03_seed2_truth.png) | 基线seed不一致 seed2 truth Grad-CAM |
| 73 | [case04_seed1_predicted](06_gradcam/case04_seed1_predicted.png) | NoTF seed1纠正基线 seed1 predicted Grad-CAM |
| 74 | [case04_seed1_truth](06_gradcam/case04_seed1_truth.png) | NoTF seed1纠正基线 seed1 truth Grad-CAM |
| 75 | [case04_seed2_predicted](06_gradcam/case04_seed2_predicted.png) | NoTF seed1纠正基线 seed2 predicted Grad-CAM |
| 76 | [case04_seed2_truth](06_gradcam/case04_seed2_truth.png) | NoTF seed1纠正基线 seed2 truth Grad-CAM |
| 77 | [case05_seed1_predicted](06_gradcam/case05_seed1_predicted.png) | Full seed2纠正基线 seed1 predicted Grad-CAM |
| 78 | [case05_seed1_truth](06_gradcam/case05_seed1_truth.png) | Full seed2纠正基线 seed1 truth Grad-CAM |
| 79 | [case05_seed2_predicted](06_gradcam/case05_seed2_predicted.png) | Full seed2纠正基线 seed2 predicted Grad-CAM |
| 80 | [case05_seed2_truth](06_gradcam/case05_seed2_truth.png) | Full seed2纠正基线 seed2 truth Grad-CAM |
| 81 | [case06_seed1_predicted](06_gradcam/case06_seed1_predicted.png) | NoTF seed1相对基线退化 seed1 predicted Grad-CAM |
| 82 | [case06_seed1_truth](06_gradcam/case06_seed1_truth.png) | NoTF seed1相对基线退化 seed1 truth Grad-CAM |
| 83 | [case06_seed2_predicted](06_gradcam/case06_seed2_predicted.png) | NoTF seed1相对基线退化 seed2 predicted Grad-CAM |
| 84 | [case06_seed2_truth](06_gradcam/case06_seed2_truth.png) | NoTF seed1相对基线退化 seed2 truth Grad-CAM |
| 85 | [reliability_risk_coverage](07_calibration/reliability_risk_coverage.png) | 可靠性曲线与风险—覆盖率 |


## 论文一致性与困难类增补

第6.7.6/6.7.7的两表两图单独位于thesis_supplement，不覆盖原85页图册。

- [双seed正确性组成与固定类别组](thesis_supplement/seed_stability_groups.png)
- [固定后20类七配置召回](thesis_supplement/fixed_bottom20_models.png)
- [配对统计](thesis_supplement/paired_prediction_statistics.csv)、[逐类模型统计](thesis_supplement/fixed_bottom20_model_statistics.csv)
- `python -B thesis_supplement/prepare_repeatability.py`从公共预测/metrics接口和既有bottom20诊断重建增补图表；不打开原图、不训练或新推理。
- repeatability_content.tex为本次两分节片段，repeatability_data.json保存数字/来源身份；既有prepare_supplement.py继续负责前五分节，不自动改写主论文。
