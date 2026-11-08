# v3公共结果读取工具

公共读取模块仅在本目录维护，来源为src_v2/ReportChart/chart_data.py，v2副本保持原样。来源和目标SHA见[复制清单](../docs/copy_manifest.json)。

UI246已接入`runs/<config_id>_seed<seed>/`正式目录，枚举排除`analyses/`和`_smoke/`。仅读取当前metrics_record.csv（归档保留，不自动合并）、results和best checkpoint路径；结果核对注册骨干、解冻时点、实验/归一化签名与标签协议。队列另检查100轮连续完整和best指标一致性。

结果查询默认只读，不构造模型、不读图片、不触发推理；图表输出经显式调用写runs/analyses。临时目录中实际指标写入→读取→队列完成判定及身份冲突拒绝已验证；未读取真实图片或训练产物。不得默默混读v2产物补齐缺失数据。

```powershell
python -B src_v3/result_tools/chart_data.py list
python -B src_v3/result_tools/chart_data.py summary FullUnfreeze:1
```

目前尚无v3正式结果。数据manifest与独立test身份属于后续推进任务，不冒充已完成。

## 单表指标协议（RF12-S）

每个run仅以`metrics_record.csv`保存逐轮指标；不再生成`metrics.csv`。现有规范表快照与归档仍可保留。MySQL主存/CSV降级/双失败暂停仍仅规划，本次没有数据库实现。

字段按写入顺序如下，原18列保持，最后追加两项：

| 字段 | 含义/单位 |
|---|---|
| config_id、seed、epoch | 实验身份、seed、0起算轮次 |
| sampler_seed | 当前绝对epoch的采样seed |
| label_schema_version、label_mapping_hash、num_classes | 标签协议、映射哈希、输出类别数 |
| train_acc、train_loss、val_acc、val_loss | 训练/验证准确率（0–100）与损失 |
| val_macro_f1、val_weighted_f1 | 验证Macro/Weighted F1（0–1） |
| lr、gradient_norm | 数值学习率与梯度范数，无字符串后缀 |
| time_spend_s、state、save_time | 本轮训练/验证耗时秒（指标及pth写入前计时）、解冻阶段、指标保存时间 |
| val_weighted_precision、val_weighted_recall | 验证加权Precision/Recall（0–1），未传时留空 |

写入仍调用`calculate.add_record_metrics_v2`；名称保留不表示写入v2目录。旧`add_record_metrics`已删除，直接调用者改用规范接口。P/R通过末尾可选参数传入，训练入口传当前validate结果，不重新计算。五项Child零占位不保留。

`load_model`等读取接口与返回DataFrame类型保持，正常新表为20列，新增两列按原规则转数值；按原字段名取值不受影响，依赖固定18列宽度的外部调用须调整。汇总展示内容未扩展。

Long完整继承来源规范表前缀（包含P/R），改为目标config_id后继续追加；来源文件不动，目标重启不重复复制。普通续跑仍要求CSV覆盖0到last.epoch，随后从last.epoch+1追加。CSV与last不一致仍停止并保留现场；本次不自动修表、增重试或改变恢复策略。RF12-H已取消旧表头自动迁移；训练入口每次启动/恢复仅校验一次固定20列、当前配置/seed、连续epoch和末行换行，旧格式拒绝追加并保留文件。writer不再逐轮读取表头或重写全表。


## 指标与断点顺序（RF12-H）

20列顺序统一定义在`run_paths.METRICS_COLUMNS`。正常新文件首次写入同时输出表头和第一行，各自换行；后续只追加数据。独立调用writer已有文件时，调用者须先确认固定格式，writer内部不重复校验。writer函数参数、返回epoch及读取API保持。

每轮先训练/验证并推进原CAWR，再写指标；成功后才保存best、阈值留样、每轮完整模型，最后保存标准last。原控制台提示文字和次数保持，保存提示顺序随此流程调整。`time_spend_s`采样在保存前，因此不再包含本轮pth写入时间；整体耗时另行计量，不将两种口径混用。

写入发生OSError或短写时，撤回本次追加后停止，不输出指标保存成功，也不保存本轮任何pth。排障后重新启动相同配置/seed并设RESUME=true，从上一次last还原权重、优化器、调度和AMP状态，重跑未保存的一轮；不从失败时内存状态直接接着跑。队列沿用原有的一次续跑重试，未新增重试机制。第一轮失败尚无last时不能凭空续训，须排障后按既有新跑/目录规则重训。

若撤回也失败，或进程在CSV成功但pth尚未保存时被强制结束，CSV与last可能不一致；恢复入口停止并保留现场，不自动截断历史/跳行/补造指标。本次不是跨文件事务，也不承诺断电持久性；MySQL双失败暂停仍未实现。

模型A中断后训练B，再切回A时，仍按配置ID+seed选各自run目录，并在新训练进程重新校验；没有跨模型的“已校验”缓存。Long的来源完成检查使用同一20列格式，首次继承先复制指标再复制pth，最后写初始化配置；中途失败的不完整目标仍拒绝自动续跑。Long自己再次恢复只追加目标文件、不重复制来源；来源不改。Long快照也先复制指标，再复制last/best。
