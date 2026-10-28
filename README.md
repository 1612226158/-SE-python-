# src_v3：实验运行与推进计划

当前已注册6项NoTF实验：全解冻、30/60、15/40、45/75渐进解冻，以及ResNet-18/34全解冻。均首轮seed1、100轮，顺序为基线→时点→骨干。配置ID省略系列与共同NoTF前缀，例如`FullUnfreeze`、`ProgressiveUnfreeze`；NoTF仍在模型参数中记录。具体ID见[实验命名规范](docs/实验命名规范.md)。

UI246已移除7处初始化占位拦截，修复直接运行的导入路径、冻结归一化读取、每run输出与公共结果读取。入口和临时指标/结果联通检查通过；尚未运行真实数据smoke或正式训练，不能据此宣称GPU训练已验收。

## 使用

沿用v2现有全局Python，不新建venv。项目根目录执行：

```powershell
python -B src_v3/train.py --check
python -B src_v3/run_queue.py --check
python -B src_v3/run_queue.py --list
```

`--check`只读核对配置、路径是否存在、依赖模块是否可定位及任务列表，不读图片、不创建训练产物、不下载权重。它不替代真实数据和GPU检查。

直接运行`python -B src_v3/train.py`执行活动JSON选定的实验；当前JSON为3轮SMOKE。直接运行`python -B src_v3/run_queue.py`会按该文件显式`QUEUE`列表的顺序执行，当前列出六项seed1、各100轮，并按每项状态决定跳过或续跑。增删任务、修改seed或顺序直接编辑列表中的`(配置名, seed)`；`--check`和`--list`也读取该列表，新增注册模型不会自动加入队列。

## 配置与结果

- [model_registry.py](model_registry.py)是骨干、时点、预算和共同训练策略的唯一来源；内部`v2`构造键不是旧实验名。
- [experiment_config.json](experiment_config.json)是单次训练的活动配置，含冻结mean/std。历史统计跳过数590不用于删除或过滤训练图片；运行中禁止自动重算归一化。
- [run_paths.py](run_paths.py)统一路径、运行快照和归一化签名；[launch_checks.py](launch_checks.py)提供帮助与只读检查。
- 正式结果写`runs/<config_id>_seed<seed>/`，含config.json、metrics.csv、metrics_record.csv、results_<run_id>.json及best/last checkpoint。非续跑时拒绝覆盖已有非空目录。
- 短运行写`runs/_smoke/<run_id>_<时间戳>/`；分析写`runs/analyses/`。公共读取和队列正式完成判断排除这两类目录。
- 公共结果读取只改[result_tools/](result_tools/README.md)；v2代码、结果和归一化文件保持原样。

## 后续研究

[详细推进计划](docs/v3实验推进计划.md)继续列出数据清单、独立评价、P05调度和P07等参数SE等研究步骤。这些后续项目未实现不再通过无条件raise封锁当前六项入口。正式训练前仍需真实子集smoke与GPU/AMP/数据加载检查；manifest及独立test未完成时，结果只用于现有验证集上的工程筛选。

[复制清单](docs/copy_manifest.json)记录源与目标SHA；[环境说明](docs/environment.md)记录全局环境约定。`docs/reference/`仅作历史来源。原`v3实验预实施方案.md`不因本轮运行入口修复自动更新。

`docs/`及智能体规则、交接与记忆文件仅在本地保留，已加入Git忽略规则；上述docs链接供本地阅读，代码仓库不包含这些文件。

## Git提交位置

本目录是独立Git工作区，origin为`https://github.com/1612226158/-SE-python-.git`，使用`v3`分支；项目父目录不是Git仓库。v3分支继承现有main历史，后续v3代码在本分支维护，src_v2保留原工作区。

提交前在本目录查看`git status`和`git diff`。忽略规则排除整个runs和docs目录、智能体规则/交接日志/记忆、权重、缓存、日志、压缩包、本地环境、私有.env及分析产物；源码和活动JSON仍纳入版本管理。精简副本缺少的旧GUI/图表文件会显示为相对main的删除差异。
