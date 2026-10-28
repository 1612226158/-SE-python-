# v3公共结果读取工具

公共读取模块仅在本目录维护，来源为src_v2/ReportChart/chart_data.py，v2副本保持原样。来源和目标SHA见[复制清单](../docs/copy_manifest.json)。

UI246已接入`runs/<config_id>_seed<seed>/`正式目录，枚举排除`analyses/`和`_smoke/`。读取metrics_record.csv及其归档、results和best checkpoint路径；结果核对注册骨干、解冻时点、实验/归一化签名与标签协议。队列另检查100轮连续完整和best指标一致性。

结果查询默认只读，不构造模型、不读图片、不触发推理；图表输出经显式调用写runs/analyses。临时目录中实际指标写入→读取→队列完成判定及身份冲突拒绝已验证；未读取真实图片或训练产物。不得默默混读v2产物补齐缺失数据。

```powershell
python -B src_v3/result_tools/chart_data.py list
python -B src_v3/result_tools/chart_data.py summary FullUnfreeze:1
```

目前尚无v3正式结果。数据manifest与独立test身份属于后续推进任务，不冒充已完成。
