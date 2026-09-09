# ReportChart 目录说明（2026-09-08 会话8 建立）

> 本目录是论文图表脚本目录（SCI 风格）。本文档说明各文件用途；**新会话画图前先读本文件 + chart_data.py 的用法**，不要重新发明数据读取。
> 注意：本目录的 **PDF/PNG/CSV/JSON 产物不入 git**（.gitignore），只入库 .py 脚本；但产物是本机论文编译依赖（document_new.tex 从 `images/` 引用，部分图需手动复制过去）。

## 一、chart_data.py —— 数据读取模块（画任何图的第一步）

**作用**：统一按"模型英文名（CONFIG_ID）"取实验数据，消灭各脚本手写"归档 csv 文件名"的重复与错误。

**为什么需要它**：runs/ 下的训练记录会随每次正式训练被**自动归档**（`metrics_record.csv` 归档成 `metrics_record_archive_<时间戳>.csv`），且一个模型的数据可能散落在**多个归档文件**里（如中断续跑、多次重跑）。手工找数据极易错。

**核心 API**：
```python
from chart_data import (load_model, model_best, model_final,
                        results_best, best_ckpt_path, list_models,
                        pair_tag, fig_out)

load_model('S-NoProg')    # → DataFrame：该模型全部轮次曲线数据
                          #   自动扫 runs/ 当前 + 所有归档 csv，按 config_id 过滤合并
                          #   多 seed 自动选最完整的；按 epoch 去重排序
                          #   支持 Long 配置跨档续跑（100~159 自动拼上 0~99）
                          #   也接受显示写法 'S-NoProG'（注册表自动归一化到规范 key）
model_best('S-NoProg')    # → (82.634, 74)   metrics 口径 best（画曲线用）
results_best('S-NoProg')  # → (82.6337, 74, 1) 论文口径 best（读 results JSON）
                          #   ⚠ 论文表格/正文引用数字请用这个（results 是训练结束的正式成绩）
best_ckpt_path('S-NoProg')# → best checkpoint 绝对路径（推理/加载模型用）
list_models()             # → 打印当前有哪些模型可用（跑过的就有）

# —— 输出文件名推导（不再手写 out1/out2/out3）——
fig_out('val_acc', 'G-Full-CAWR', 'S-NoProg')
#   → <本目录>/val_acc_progressive_vs_full.pdf   （命中论文历史标签，文件名不可变）
#   新组合自动命名：fig_out('val_acc','V2-Full','S-NoProg')
#   → val_acc_V2-Full_vs_S-NoProg.pdf
```

**模型英文名（key）唯一真源 = `src_v2\model_registry.py`（2026-09-09 拆出；train.py PRESETS / 预测 GUI / 本目录脚本共用）**：
`G-Full`(渐进RLRP旧) / `G-NoSE` / `S-NoProg`(全解冻；显示写法常作 S-NoProG，自动归一化) / `S-NoSE`(全解冻无SE) / `G-Full-CAWR`(渐进CAWR) / `G-Full-CAWR-Long`(160轮) / `V2-Full`(修正版Transformer) / `V2-NoTF`(V2去Transformer) / `G-V2`(渐进+V2修正头，2×2第四格)。

## 二、三个绘图脚本各画什么

### 1. `plot_progressive_vs_full.py` —— 解冻策略对比（渐进式 vs 全解冻）
- **对比**：`G-Full-CAWR`（渐进式解冻） vs `S-NoProg`（全程全解冻），各 100 轮
- **输出 3 张图**（论文 6.5 节 消融拓展 已引用）：
  - `loss_curve_progressive_vs_full.pdf` —— 训练/验证损失曲线（左右两子图各一个模型）
  - `val_acc_progressive_vs_full.pdf` —— 验证集准确率曲线对比 + 各自 best 峰标注
  - `time_acc_progressive_vs_full.pdf` —— 耗时-准确率曲线（收敛效率视角）
- **论文对应**：fig:loss-curve-progressive-vs-full / fig:val-acc-progressive-vs-full / fig:time-acc-progressive-vs-full（渐进式 vs 全解冻，长尾分析前的整体对比）

### 2. `plot_snoprog_vs_snose.py` —— SE 消融对比（含 SE vs 去 SE）
- **对比**：`S-NoProg`（含多区域SE） vs `S-NoSE`（去多区域SE），各 100 轮
- **输出 3 张图**（论文 SE 消融节已引用）：
  - `loss_curve_snoprog_vs_snose.pdf` —— 训练/验证损失曲线
  - `val_acc_snoprog_vs_snose.pdf` —— 验证集准确率对比（差异仅 0.10pp）
  - `time_acc_snoprog_vs_snose.pdf` —— 耗时-准确率曲线
- **论文对应**：fig:loss-curve-snoprog-vs-snose / fig:val-acc-snoprog-vs-snose / fig:time-acc-snoprog-vs-snose（SE 消融：证明强基线下 SE 增益≈0）

### 3. `eval_tail_class_acc.py` —— 长尾小样本类分析（渐进式 vs 全解冻在尾部类上）
- **不是普通曲线图**：对 **254 个合并类按训练样本量升序切片**（最少 50 类 / 最少 100 类），逐类推理两模型（CPU），统计尾部类上的准确率对比
- **四阶段运行**（用法见脚本 docstring）：
  - `count` —— 统计类分布 → 生成 `tail_classes.json`
  - `infer` —— CPU 逐类推理（S-NoProG + G-Full-CAWR，仅 tail100 覆盖的 val 图）→ `per_class_acc.csv`
  - `plot` —— 读 csv 出图
  - `smoke` —— 冒烟验证
- **输出**：`tail_class_acc.pdf`（双面板：左=50/100/全体聚合柱状 + 右=前缀趋势）+ `per_class_acc.csv`
- **论文对应**：fig:tail-class-acc（长尾反转：最少50类渐进式 +3.64pp、最少100类 +2.15pp、全体 254 全解冻仍 +1.67pp → 互补）
- ⚠ infer 阶段需要 GPU/CPU 推理较久（~1817 张 × 2 模型），平时只重跑 `plot` 即可（读已有 csv）。

## 三、怎么画一张新对比图（模板步骤）

```python
# 1. 复制 plot_progressive_vs_full.py 或 plot_snoprog_vs_snose.py
# 2. 只改顶部 MODEL_A / MODEL_B 两行（模型英文名 + 中文图名 + 颜色）
# 3. 数据与输出文件名自动处理（load_model / fig_out），无需改 main()
# 4. 跑完核对控制台汇总数字（应与你预期的 best/final 一致）
```
示例：想画 V2-Full vs S-NoProg → MODEL_A=('V2-Full', 'V2修正Transformer', 颜色)，MODEL_B=('S-NoProg', 'S-NoProG(全解冻)', 颜色)；输出自动为 `*_V2-Full_vs_S-NoProg.pdf`。若该对比要进论文且定了图名，先在 `chart_data.PAIR_TAGS` 里登记一个标签（保持文件名稳定），再在 document_new.tex 引用。

## 四、口径与纪律（改脚本必读）

1. **论文表格/正文数字用 `results_best`（results JSON 口径）**；`load_model`/`model_best` 是 metrics 口径，两者在个别模型上有小差（如 G-Full-CAWR：metrics 80.89@94 vs results 80.96@92），别混用；
2. **论文历史文件名不可变**（document_new.tex 的 \includegraphics 依赖）：`progressive_vs_full` 与 `snoprog_vs_snose` 两组标签已锁在 chart_data.PAIR_TAGS；
3. 只读纪律：脚本只读 runs/ 与 train/val（统计用），产物只写本目录；不碰训练、不改数据集；
4. 中文字体：SimHei（脚本已设），若缺字体图内中文会变方块。

## 五、模型链接架构（2026-09-09 会话9 起）

- **新增/改模型 = 两步**：① `src_v2\model_registry.py` 的 `MODELS` 加/改条目（`params`=训练参数 / `cn`=中文全称 / `short`=短说明 / `target_epochs`=完整轮数）；② 要训练就在 `run_queue.py` QUEUE 加 `(id, seed)`。train.py（PRESETS 别名）、预测 GUI（下拉/中文名/轮数）、本目录全部脚本自动跟随，**不再四处手抄模型名**。
- **本目录脚本的链接约定**：数据一律 `chart_data.load_model(注册key)`；checkpoint/results 用 `best_ckpt_path/results_best`；中文全称 `chart_data.model_cn(id)`；注册校验 `chart_data.is_known(id)`。三个画图脚本（plot_progressive_vs_full / plot_snoprog_vs_snose / eval_tail_class_acc）顶部均有 id 自检：用了未注册的模型名会直接报错。
- **key 大小写规范**：代码里一律规范写法（如 `S-NoProg`）；论文/口头的 "S-NoProG" 只是显示写法，由注册表 ALIASES 归一化——别在代码里手写两种写法。
- 图内中文名/主色是"单图显示样式"（在各脚本顶部 MODEL_A/B 或 _META 里声明，可按图意改）；模型本身的 id/中文全称/参数在注册表，二者分工互不干扰。
