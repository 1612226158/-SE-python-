# human_loop —— 人工干预/人工核验专区（2026-09-12 建）

> 这里放**需要人来判断**的事务：标注一致性检验（Cohen's κ）、疑似脏标注类的人工核验、类别合并清单的人工评审等。
> 纪律：**只读 train/val 数据集**；产物只写本目录 `exports/`（不入 git）；不参与训练、不改数据集。

## 一、为什么需要它
混淆矩阵分析（论文 6.5.5 节）显示：残余错误中约三成集中在一小批类对上（如 锅/电饭煲、盒子/纸箱、插头电线/充电线）。
要把它写成"**任务内在歧义**（inherent label ambiguity）"而不是模型缺陷，必须有**人的证据**：
两人独立盲标的 **Cohen's κ** 低 → 说明该类对的标签边界人工都难以稳定复现 → 模型错误接近该标签体系下的可达上限。

## 二、文件与用法

| 文件 | 作用 |
|---|---|
| `hl_common.py` | 共用工具：254 合并类映射、短名解析、图片清单、**Cohen's κ 计算**、κ 分级 |
| `make_kappa_pack.py` | 生成**盲标抽样包**：易混淆类对平衡抽样 → 打乱文件名 → 两张空白标注表 + 答案键 |
| `kappa_report.py` | 读标注表 → 逐类对 **p_o / p_e / κ / 分级** + 两人**人类准确率** +（可选）**模型同批准确率** → CSV + Markdown |

### 1) 生成抽样包
```bash
D:\Python\Python3.10.7\python.exe src_v2\human_loop\make_kappa_pack.py
# 默认：内置 8 组易混淆对 × 每类 15 张 = 240 张（val 集）
# 自定义：--pairs "锅:电饭煲,盒子:纸箱"  --n-per-class 20  --seed 20260912
```
产物 `exports/<时间戳>_kappa_pack/`：
```
images/P01/P01_001.jpg ...    图片副本（文件名不含类别线索，保证盲标）
pairs.csv                     类对清单
sheetA.csv / sheetB.csv       两张空白标注表（option_1/2 为随机顺序的两个候选类）
_key.csv                      ⚠ 答案键（标注者不得打开）
README_ANNOTATOR.txt          标注者须知
```

### 2) 人工标注（关键规矩）
- **两位标注者各填一张表**（`sheetA.csv` / `sheetB.csv`），互相独立、不得讨论；
- 每张图**二选一**：`choice` 填 1 或 2（实在无法判断填 0 并在 notes 说明）；
- 不要打开 `_key.csv`；不要看模型预测；
- 建议每 20-30 分钟休息一次，避免疲劳造成系统性偏差。

### 3) 出报告
```bash
D:\Python\Python3.10.7\python.exe src_v2\human_loop\kappa_report.py --pack exports/<包名>
# 附模型对照（在同一批图片上跑模型，CPU，秒级~分钟级）：
D:\Python\Python3.10.7\python.exe src_v2\human_loop\kappa_report.py --pack exports/<包名> --with-model S-NoProG
# κ 计算自检（不读任何数据）：
D:\Python\Python3.10.7\python.exe src_v2\human_loop\kappa_report.py --selftest
```
输出：控制台汇总表 + `<pack>/kappa_report_<时间戳>.csv` + `.md`（Markdown 表可直接摘进论文）。

### 4) 工具链自检（改完代码先跑这个）
```bash
D:\Python\Python3.10.7\python.exe src_v2\human_loop\selftest_human_loop.py
# 额外验证"模型对照"分支（会加载 torch，较慢）：
D:\Python\Python3.10.7\python.exe src_v2\human_loop\selftest_human_loop.py --with-model S-NoProG
# 保留临时目录以便人工检查：
D:\Python\Python3.10.7\python.exe src_v2\human_loop\selftest_human_loop.py --keep
```
- 覆盖 48 项断言：κ 数学（含与独立 2×2 计数实现对照）、κ 分级阈值、类名/类对解析（唯一/未知/歧义/全角冒号）、抽样包产物完备性、**盲标性**（文件名不含类别线索）、**平衡抽样**、**同种子可复现**、报告生成的 κ=1/负值/弃权三种夹具、模型列填充；
- 零依赖（不需要 pytest），**失败时退出码非 0**；
- 只读 val 数据集，产物写系统临时目录并在结束时删除；
- ⚠ 测试里的"标注"是**程序生成的测试夹具**，不是真实人工标注，**不得用于论文**。

## 三、κ 怎么读（Landis & Koch）
| κ | 一致性 | 论文含义 |
|---|---|---|
| <0.21 | 极弱 | 标签边界几乎不可复现 → 强支持"固有歧义" |
| 0.21–0.40 | 一般 | 边界不可靠 → 支持"固有歧义" |
| 0.41–0.60 | 中等 | 边界模糊，归因需谨慎（可写"部分歧义"） |
| 0.61–0.80 | 较好 | 边界基本清晰 → 不宜整体归因于歧义 |
| 0.81–1.00 | 几乎完全一致 | 标签清晰，模型错误应视为可改进项 |

⚠ 注意事项（写进论文时必须交代）：① **平衡抽样**（每类等量，避免患病率悖论压低 κ）；② **选项数固定为 2**（不能有的图 2 选 1、有的 3 选 1）；③ 必须**盲标**；④ 报告 `p_o`、`p_e`、样本量、标注者人数。

## 四、后续可加的工具（占位，按需开发）
- `audit_labels.py`：对"铝制用品"等**疑似脏标注**类做抽样人工核验（含缩略图拼图，便于快速过目）；
- `review_merge_list.py`：类别合并清单 v2（蔬菜/菜根菜叶 等 7 组候选）的两轮人工评审表；
- `model_vs_human.py`：把"人类上限 vs 模型"结果汇总成论文图（柱状对比）。
