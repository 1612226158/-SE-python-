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
