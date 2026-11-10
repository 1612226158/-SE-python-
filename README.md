# src_v3 垃圾分类实验

本分支维护 252 类垃圾图像分类实验。当前注册 14 项 NoTF 配置，包括原有 10 项和 4 项固定融合 HierSE。训练标签仍由 265 个原始数字类别目录映射到 252 个输出类。模型、训练预算和实验签名见 [model_registry.py](model_registry.py)。

## 当前正式队列

| 配置 ID | 骨干 | SE | seed | 轮数 |
|---|---|---|---:|---:|
| ResNet18-SE1-FullUnfreeze | ResNet18 | 全局 SE | 1 | 100 |
| ShuffleNetV2x1.0-SE1-FullUnfreeze | ShuffleNetV2 1.0× | 全局 SE | 1 | 100 |

两项基线均已完成真实数据的三轮冒烟。正式任务由 [run_queue.py](run_queue.py) 中的 `QUEUE` 逐项指定。增删任务、调整顺序或修改 seed 时，直接编辑 `(配置 ID, seed)`。单独运行 [train.py](train.py) 会读取 [experiment_config.json](experiment_config.json) 或 `V3_CONFIG_FILE` 指定的配置。

## HierSE 多层区域实验

四项配置暂用 ShuffleNetV2 1.0× 预训练骨干、NoTF 和全解冻。共用骨干字段位于 `HIERARCHICAL_SE_PLAN`。以下四项均已接入单模型入口，正式队列仍为上表两项基线。

| 配置 ID | 区域层级 | 父级门控参与子区修正 | 层间融合比例 |
|---|---|---|---|
| HierSE-Parallel-L2 | 整图 → 左右两区 | 否 | 0.6 / 0.4 |
| HierSE-Nested-L2 | 整图 → 左右两区 | 是 | 0.6 / 0.4 |
| HierSE-Parallel-L3 | 整图 → 左右两区 → 四区 | 否 | 0.6 / 0.2 / 0.2 |
| HierSE-Nested-L3 | 整图 → 左右两区 → 四区 | 是 | 0.6 / 0.2 / 0.2 |

`level_grids` 定义分区，`parent_modulation` 控制父级门控是否参与局部修正，`level_weights` 定义各层分类概率在最终输出中的比例。各区共用分类头，先在同层平均区域概率，再做层间融合。实现见 [SE_Block.py](SE_Block.py) 和 [ResNetTransformer.py](ResNetTransformer.py)。动态融合目前是研究候选。

**2026-11-10 冒烟验收**：四项各完成一个真实数据 epoch 和对应验证，退出码均为 0。训练损失、验证损失及轮末梯度范数为有限值。另用 CUDA 混合精度小批量核验了分区、融合及骨干与 SE 各分支的反向梯度。运行期间验证集有效样本数有变化，本批结果用于流程验收。

## 运行方式

使用已有的全局 Python、PyTorch 和 CUDA 环境。项目布局中，`train/`、`val/` 和 `classname.txt` 位于 `src_v3/` 的上一级。以下命令在 `src_v3/` 内执行。

~~~powershell
python -B train.py --check
python -B run_queue.py --check
python -B run_queue.py --list
~~~

`--check` 预览模型、预算、数据路径和依赖定位结果。准备好正式队列后，用 `python -B run_queue.py` 启动。

~~~powershell
# 冒烟预览。每项默认 3 轮
python -B run_queue.py --smoke --check

# 预览并运行每项 1 轮
python -B run_queue.py --smoke --smoke-epochs 1 --check
python -B run_queue.py --smoke --smoke-epochs 1
~~~

`--smoke-epochs N` 是可选的正整数。队列命令只对当前 `QUEUE` 中的任务生效。要单独冒烟某个 HierSE 配置，使用该配置 ID 和 `SMOKE=true` 的运行配置，再执行 `python -B train.py --smoke-epochs 1`。可先加 `--check` 核对配置绑定与轮数。

## 结果与版本

- 正式结果位于 `runs/<config_id>_seed<seed>/`，包括运行配置、`metrics_record.csv`、结果 JSON 和 best/last 断点。续跑会核对断点与指标轮次。
- 冒烟写入 `runs/_smoke/` 下独立的带时间戳目录。正式结果读取和队列完成判断会排除冒烟和分析目录。公共读取入口见 [result_tools](result_tools/README.md)。
- 归一化参数取自运行配置。训练、验证以及后续结构对照应记录实际数据版本与样本数。
- MySQL 入口目前用于测试库登记和首轮指标验收。常规训练沿用本地 CSV 和断点流程。
- GitHub 上的 v3 源码维护在 `v3` 分支。运行产物和本地详细实验文档按仓库忽略规则保留在本机。
