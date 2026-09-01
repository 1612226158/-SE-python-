# src_v2 运行说明（2026-08-31 · 消融实验版）

> 本目录是 `src\` 的**改造副本**：原目录一字未动，跑坏了随时回退。
> 所有输出落在 `src_v2\runs\`；数据集只读读取 `train\ / val\`；
> **v2 增量缓存**：每次启动校验一遍磁盘并自动更新（新增/删除/变动文件），原子写入不损坏；原缓存仅作首次种子。

## 零、目录结构与外部依赖（重要：本仓库只上传了 src_v2）

> 本仓库**只包含 `src_v2/` 代码**，不含数据集、类名表、权重。运行需把它放进一个**项目根目录**，与以下外部文件/目录并列：

```
<项目根目录>              # = PROJECT_ROOT = src_v2 的上一级（原为 E:\DataSet\垃圾分类图片-2）
├── src_v2/              # ← 本仓库内容放这里
│   ├── train.py / ResNet.py / SE_Block.py / calculate.py / calculate_update.py / __init__.py
│   ├── experiment_config.json   # 改实验配置只动它
│   └── runs/                    # 训练产物（自动生成，不入库）
├── src/                 # 原代码（只读依赖）
│   ├── images_cache_train.json  # 原训练缓存（v2 首次启动当"种子"，只读）
│   ├── images_cache_val.json
│   └── calculate.config         # mean/std 配置（只读）
├── train/               # 训练集：265 个数字子文件夹 0~264，每类一个
├── val/                 # 验证集：265 个数字子文件夹
└── classname.txt        # 类名表（项目根这一份是代码实际读取的）
```

**关键点**：

1. **路径定位**：所有路径由 `__file__` 推导（`PROJECT_ROOT = src_v2 的上一级`），所以 **src_v2 必须与 train/、val/、src/、classname.txt 同级放置**；工作目录（cwd）随便设都不影响；
2. **classname.txt**：265 行纯文本，**第 N 行 = 文件夹 N 的类别名**（0-based 对齐，例：第 25 行 `厨余垃圾-果壳` 对应文件夹 24）；新增/改名类别时需同步改此文件与 train/、val/ 的文件夹（train/ 里也有一份残留 copy，代码不读它，可忽略）；
3. **数据集存储**：`train/` 与 `val/` 各含 265 个数字子文件夹（0~264），每类一个文件夹、图片直接散放其中；加载扩展名 = `jpg/jpeg/png/bmp/tif/tiff/jfif`（**avif 不被加载**）；
4. **缓存**：`src_v2` 启动时以 `src/` 的原缓存为种子，生成自己的 `src_v2/images_cache_*_v2.json`（增量自动更新，不入库）；
5. **合并类**：`src_v2/__init__.py` 的 `merged_dict` 定义哪些类合并（当前 7 组 → 输出 254 类）；改它**无需重建缓存**（合并发生在标签生成层）。

## 一、怎么跑（PyCharm）

1. 用 PyCharm 打开本文件同目录的 `experiment_config.json`，**只改这个 JSON**（不用再改 train.py）：
   - `CONFIG_ID`：`G-Full` / `G-NoTF` / `G-NoSE` / `G-SingleSE` / `G-PureBB` / `S-NoProg`
   - `SEED`：`0` / `1` / `2`
   - `SMOKE`：先 `true` 冒烟（3 轮）→ 通过后改 `false` 挂正式 100 轮
   - `RESUME`：`false` = 全新训练；`true` = 从上次断点继续（**中途暂停后继续就改这个**）
2. 右键运行 `train.py`（工作目录无关，路径已全部绝对化）。

## 二、冒烟通过的标准（SMOKE=true 跑完检查）

- 控制台出现 3 个 epoch 的 Train/Validation Accuracy；
- `runs\metrics_record.csv` 生成且有 3 行（含 config_id/seed 列）；
- `runs\checkpoint_<配置>_seed<种子>_best.pth` 存在（约 170MB）；
- `runs\logging_<配置>_seed<种子>.log` 有输出；
- `runs\metrics.csv`（旧 18 列格式）同步生成。

## 三、输出文件（全部在 src_v2\runs\）

| 文件 | 内容 |
|---|---|
| `metrics.csv` | 旧 18 列格式（与历史口径一致，便于和旧 CSV 对齐看） |
| `metrics_record.csv` | 新格式：config_id/seed/train_acc/val_acc/val_macro_f1/val_weighted_f1/lr/gradient_norm 等纯数值列 |
| `checkpoint_<配置>_seed<种子>_best.pth` | **实时 best（state_dict），论文数字以它为准** |
| `checkpoint_<配置>_seed<种子>_last.pth` | **每轮断点（state_dict+optimizer+scheduler，约317MB）**：中途暂停后 `RESUME=true` 就从它续跑 |
| `results_<配置>_seed<种子>.json` | **本运行结果汇总**（params/样本数/best 轮次与 acc/显存峰值/时长等，训练结束生成） |
| `last_config.json` | 当前运行配置快照；每次正式全新训练会随产物一起归档 `last_config_archive_*` |
| `garbage_classification_model.pth` | 训练结束的完整 checkpoint（含 optimizer/scheduler 状态，用于续跑） |
| `logging_*.log` | 运行日志 |

## 四、本副本相对原代码的改动（共 7 类）

1. **修 CAWR**：`T_0=5`（epoch 单位，每 5/10/20...轮重启；你已亲手改过，保留）；
2. **模块开关**：`regions`（[1,2]/[1]/None）、`transformer_layers`（3/0）、`use_decouple`（True/False），由顶部 `PRESETS` 一键切换；
3. **解冻固定触发**：state1→2 第 30 轮、→3 第 60 轮（替换 CSV 自适应判据，保证消融可比）；
4. **固定种子**：`seed_all` + DataLoader/Sampler 传 generator；
5. **best.pth 实时保存**：只在 val_acc 刷新时覆盖（对应"best 不保证全局最优，但报的就是存的"）；
6. **metrics 双写**：原 `add_record_metrics` 照旧写 metrics.csv；新增 `add_record_metrics_v2` 写 metrics_record.csv；
7. **续跑三件套**：checkpoint 存 `scheduler_state_dict`；续跑按 state 建对应类型调度器并 load；`RESUME=True` 一键续跑（无交互）。

## 五、预设与消融矩阵（每配置跑 SEED 0/1/2）

| CONFIG_ID | SE | Transformer | 解耦层 | 解冻 |
|---|---|---|---|---|
| G-Full | [1,2] | 3 | ✓ | 渐进(30/60) |
| G-NoTF | [1,2] | **0** | ✓ | 渐进 |
| G-NoSE | **无** | 3 | ✓ | 渐进 |
| G-SingleSE | **[1]** | 3 | ✓ | 渐进 |
| G-PureBB | 无 | 0 | 无 | 渐进 |
| S-NoProg | [1,2] | 3 | ✓ | **全程全解冻** |

## 六、自动归档（防误操作覆盖，绝不删除）

- **正式全新训练开始前**（`RESUME=False` 且 `SMOKE=False`）：自动把上次的 `garbage_classification_model.pth`、`metrics.csv`、`metrics_record.csv` 带时间戳改名为 `*_archive_YYYYMMDD_HHMMSS.*`，然后才写全新文件；
- **冒烟（SMOKE=True）不触发归档**（防止把正式结果冲进档案）；**续跑（RESUME=True）不触发**（接着写）；
- 归档的 pth 内已存 `config_id`/`seed`，随时可溯源；
- 所以误操作重跑：上一次的权重和两条指标文件都在 archive 里，丢不了。

## 七、注意事项

- **消除 PyCharm 黄字（一次设置）**：右键 `src_v2` 目录 → `Mark Directory as → Sources Root`；再右键项目根 `垃圾分类图片-2` → 同样标记为 `Sources Root`——`from ResNet import` / `from src_v2 import` 即可全部解析（代码已改为显式赋值，无 globals 注入）；
- **暂停续跑**：训练中途停掉（PyCharm 停止）→ 把 `experiment_config.json` 里 `RESUME` 改 `true` → 再 Run，从 `checkpoint_*_last.pth` 无缝继续（权重/优化器/调度器状态全恢复，无 LR 跳变）；
- **每轮断点保存会多花 30-40 秒**（约 317MB 写入），换来的是随时可暂停；
- **v2 缓存**：首次启动 train 会生成 `src_v2\images_cache_train_v2.json`（以原缓存为种子，约 1-3 分钟）；此后每次启动仅数秒~1 分钟校验；日志打印 `[缓存v2] +新增X -删除Y 变动Z`；
- **不要**升级 torch/torchvision（1.11/0.12）；
- 每 50 轮有 600 秒散热休眠（沿用原设计）；每轮末尾 sleep 5 秒；
- 续跑：把 `RESUME=True`，优先从 `runs\checkpoint_<配置>_seed<种子>_last.pth`（每轮断点）继续；找不到时才回退用 `garbage_classification_model.pth`；
- 换配置/种子前，`runs\metrics_record.csv` 无需改名（config_id 列可区分）；
- 测试集（10%）与 test 评估尚未接入，属下一阶段（G1）。
