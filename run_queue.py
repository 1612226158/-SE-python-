# -*- coding: utf-8 -*-
"""
run_queue.py —— G 系列实验自动排队器

用法（确认 PyCharm 没有手动训练在跑，二者互斥）：
    D:\\Python\\Python3.10.7\\python.exe E:\\DataSet\\垃圾分类图片-2\\src_v2\\run_queue.py

它会按 QUEUE 顺序依次处理每个 (CONFIG_ID, SEED)：
  1. 已完成（有 results_<tag>.json，或 metrics_record*.csv 里存在"最后轮"行：
     普通配置=99 行，Long 配置=目标轮数-1 行）→ 跳过
  2. 正在训练（checkpoint_<tag>_last.pth 最近 15 分钟还在写）→ 等待 5 分钟重查
  3. Long 配置且自己无断点 → 从源配置断点复制续跑（ensure_long_resume，省重跑前 100 轮）
  4. 有断点但已停 → 自动 RESUME=true 续跑；无断点 → RESUME=false 全新跑
  5. 运行 train.py（阻塞等待自然结束）；失败自动用 RESUME=true 重试一次，再失败则停队
进度/报错写入 src_v2/runs/queue_log.txt；任务切换前经 temp_guard 温度守护。
队列"已完成"项仅作留档（会被自动跳过），实际待跑顺序见 QUEUE 的"待跑"分组。
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import temp_guard  # 温度守护：任务切换前检测 CPU/GPU 温度，过高自动休息 10 分钟

SRC_V2 = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(SRC_V2, "runs")
CONFIG = os.path.join(SRC_V2, "experiment_config.json")
TRAIN = os.path.join(SRC_V2, "train.py")
LOG = os.path.join(RUNS, "queue_log.txt")
PYTHON = sys.executable  # 用启动本脚本的同一个 Python（保证 torch 1.11 环境）

# —— 执行队列总览（2026-09-09 更新）——
# 模型"参数/中文名/家族/目标轮数"唯一真源 = src_v2\model_registry.py（train.py PRESETS / GUI / ReportChart 共用）；
# 本文件只维护 QUEUE 顺序与 Long 机制——"选队列/换顺序"改这里，"改模型定义/新增模型"改 model_registry.py。
# 命名速查（与 train.py PRESETS 顶部族谱一致）：
#   G- = 渐进式解冻（state1→2 @30轮、state2→3 @60轮）；S- = 全解冻（unfreeze='none'，标准微调）
#   -CAWR = CAWR 调度（T_0=5, T_mult=2, eta_min=1e-6）；无后缀 = RLRP 旧时代（已过时仅留档）
#   V2- = 修正版 Transformer 头（ResNetTransformerV2：7×7=49 patch token + pos_embed + patch-emb dropout 0.1，
#        替代 v1 头 avgpool 1×1 单 token + dropout 0.5 + 退化 seq_len=1 注意力的组合）
#   SE 消融 = S-NoSE（全解冻无SE）；G-NoSE / G-NoSE-CAWR（渐进式系）已弃用，仅留档勿再排
# 已完成基线（results JSON 权威，254 类/CAWR 口径）：
#   S-NoProg 82.63@74 / S-NoSE 82.53@73（SE 增益≈0.10pp）/ G-Full-CAWR 80.96@92（调度器修复后干净重跑）
#   G-Full-CAWR-Long 83.565@132（160 轮预算）/ V2-Full 83.353@74（vs S-NoProG +0.72pp，n=1 初步）
# ⚠️ 调度器 bug 史（2026-09-07）：CAWR 曾被误传 val_acc（原 step(val_accuracy[0])）→ state2/3 从不重启、
#    lr 贴 1e-6 地板 → 旧 G-Full-CAWR 70.82 是脏数据（已存档 runs\DIRTY_scheduler_bug_20260907\）。
# 🧪 V2-NoTF 提前到 seed2 复现之前（2026-09-09 用户重排）：它是 V2 归因的关键对照
#    （V2-Full − V2-NoTF = 纯注意力栈增益；S-NoProg 停跑~10轮留断点，RESUME 补跑）。
#    早期观测：V2-NoTF 拟合速度 ≈ V2-Full >> v1 头(S-NoProg) → 初期提速主要来自 V2 侧头部修正
#    （49-token 化 + patch-emb dropout 0.5→0.1），不是注意力栈本身；注意力净增益待 V2-NoTF 收官对比。
# 🆕 G-V2（2026-09-09 立项）：渐进式解冻(30/60) + V2 修正头 = 2×2 网格第四格（头 v1/V2 × 解冻 全/渐进），
#    排在 V2-NoTF 之后（100轮 n=1）：回答"V2 修正头下渐进式的小样本优势与预算收益是否保持"。
QUEUE = [
    # —— 已完成，仅留档（done() 会自动跳过）——
    ("G-Full", 1),                             # ✅ 完成（RLRP 旧基线，56.93/56.82 过时勿用）
    ("G-NoSE", 1), ("G-NoSE", 2),             # ✅ 完成（RLRP 旧基线，过时勿用）
    ("S-NoProg", 1),                          # ✅ 完成（82.63%@74，全解冻主基线）
    ("S-NoSE", 1),                            # ✅ 完成（82.53%@73，SE 消融）
    ("G-Full-CAWR", 1),                       # ✅ 完成（修复后干净重跑 80.96@92；旧脏 70.82 已存档）
    ("G-Full-CAWR-Long", 1),                  # ✅ 完成（83.565@132，160 轮预算断点续跑）
    ("V2-Full", 1),                           # ✅ 完成（83.353@74，全解冻+V2 修正头；vs S-NoProg +0.72pp n=1）
    # —— 待跑（按优先级排序；2026-09-09 起 V2-NoTF 优先于 seed2 复现）——
    ("V2-NoTF", 1),                           # 🧪【跑中 09-09】V2 架构去掉 Transformer 栈（Identity 直通）：
                                              #    V2-Full − V2-NoTF = 纯注意力增益；同 dropout/pos_embed 设置
    ("G-V2", 1),                              # 🧪【新 09-09 立项】渐进式解冻(30/60)+V2修正头 = 2×2网格第四格：
                                              #    对照 G-Full-CAWR(v1头渐进) / V2-Full(V2头全解冻)，100轮 n=1 验证趋势
    ("S-NoProg", 2),                          # 📊 全解冻主结果 seed2 → n=2（09-09 曾跑 ~10 轮被停，RESUME 补跑）
    ("G-Full-CAWR", 2),                       # 📊 渐进式对照 seed2 → n=2（长尾反转复现用）
    ("S-NoSE", 2),                            # 📊 SE 消融 seed2
    ("V2-Full", 2), ("V2-NoTF", 2),           # 🧪 V2 种子2（趋势验证通过后才值得跑）
    # 可选/暂缓：

    # ("G-NoSE-CAWR", 1), ("G-NoSE-CAWR", 2), # 渐进式无SE（留档，不再排）
    # ("G-NoTF", 1), ("G-NoTF", 2),          # v1 头 Transformer 消融（v1 头已整体退场，不再排）
    # ("G-SingleSE", 1), ("G-SingleSE", 2),
    # ("G-PureBB", 1), ("G-PureBB", 2),
]

# Long 配置：目标轮数（默认 100）。这类配置跑到 SNAPSHOT_EPOCH 轮时，train.py 会自动把该轮 checkpoint/metrics 复制存档。
LONG_CONFIGS = {
    "G-Full-CAWR-Long": 160,
}
SNAPSHOT_EPOCH = 100  # Long 配置跑到该轮时，自动存档 100 轮快照

# Long 配置 → 其续跑的"源配置"：从源配置的 100 轮断点续跑，省去重跑前 100 轮
LONG_BASE = {
    "G-Full-CAWR-Long": "G-Full-CAWR",
}

FRESH_SECONDS = 15 * 60  # checkpoint 最近 15 分钟有写 = 训练进行中


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done(config_id, seed):
    tag = f"{config_id}_seed{seed}"
    # 强制重跑集合（2026-09-07）：仅当"还没有 results JSON"时无视归档 metrics 判定已完成。
    # 目的：调度器修复后重跑 G-Full-CAWR seed1（旧 70.82 脏数据已存档），但跑完后 results 生成即恢复正常，
    #      避免 run_once 完成后 done() 仍返回 False 导致被误判"失败"停队。
    RERUN = {("G-Full-CAWR", 1)}
    if (config_id, seed) in RERUN:
        results_p = os.path.join(RUNS, f"results_{tag}.json")
        if not os.path.exists(results_p):
            return False  # 重跑中且尚未出 results → 视为未完成（跳过归档 metrics 的"已完成"误判）
        # results 已生成（本次重跑完成）→ 继续走正常判断，视为完成
    # 主标记：results JSON（训练结束生成，RUN_TAG 命名，不归档）
    if os.path.exists(os.path.join(RUNS, f"results_{tag}.json")):
        return True
    # 兜底：扫描当前 + 所有归档的 metrics_record*.csv，找该配置该种子的"最后轮"行
    # （普通配置=99；Long 配置=目标轮数-1，避免 Long 跑到 99 就被误判"已完成"）
    target_epoch = LONG_CONFIGS.get(config_id, 100) - 1
    prefix = f"{config_id},{seed},{target_epoch},"
    for mr in glob.glob(os.path.join(RUNS, "metrics_record*.csv")):
        try:
            with open(mr, encoding="utf-8") as f:
                for line in f:
                    if line.startswith(prefix):
                        return True
        except OSError:
            continue
    return False


def in_progress(config_id, seed):
    p = os.path.join(RUNS, f"checkpoint_{config_id}_seed{seed}_last.pth")
    if not os.path.exists(p):
        return False
    return (time.time() - os.path.getmtime(p)) < FRESH_SECONDS


def write_config(config_id, seed, resume):
    max_epochs = LONG_CONFIGS.get(config_id, 100)
    cfg = {
        "CONFIG_ID": config_id,
        "SEED": seed,
        "SMOKE": False,
        "MAX_EPOCHS": max_epochs,
        "RESUME": resume,
        "AUTO": True,
        "UNFREEZE1_EPOCH": 30,
        "UNFREEZE2_EPOCH": 60,
    }
    # Long 配置：跑到 SNAPSHOT_EPOCH 轮时，train.py 会自动把该轮 checkpoint/metrics 复制存档
    if config_id in LONG_CONFIGS:
        cfg["SNAPSHOT_EPOCH"] = SNAPSHOT_EPOCH
        log(f"📌 {config_id} 为 Long 配置：MAX_EPOCHS={max_epochs}，跑到第 {SNAPSHOT_EPOCH} 轮自动存档 100 轮快照")
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def run_once(config_id, seed, resume):
    write_config(config_id, seed, resume)
    log(f"▶ 启动 {config_id} seed{seed}（RESUME={resume}）")
    r = subprocess.run([PYTHON, TRAIN], cwd=SRC_V2)
    return r.returncode


def ensure_long_resume(config_id, seed):
    """Long 配置：若自己没有断点、但源配置有断点，则复制过来续跑（省去重跑前 100 轮）。"""
    base = LONG_BASE.get(config_id)
    if base is None:
        return
    long_pth = os.path.join(RUNS, f"checkpoint_{config_id}_seed{seed}_last.pth")
    if os.path.exists(long_pth):
        return  # 自己已有断点，不覆盖
    src_pth = os.path.join(RUNS, f"checkpoint_{base}_seed{seed}_last.pth")
    if os.path.exists(src_pth):
        shutil.copy(src_pth, long_pth)
        log(f"📌 {config_id} seed{seed} 从 {base} seed{seed} 的断点续跑（已复制 checkpoint）")


def process(config_id, seed):
    if done(config_id, seed):
        log(f"跳过已完成 {config_id} seed{seed}")
        return "done"
    if in_progress(config_id, seed):
        log(f"⏳ {config_id} seed{seed} 正在训练，等待 5 分钟")
        return "running"

    ensure_long_resume(config_id, seed)  # Long 配置：从源配置断点续跑
    resume = os.path.exists(os.path.join(RUNS, f"checkpoint_{config_id}_seed{seed}_last.pth"))
    rc = run_once(config_id, seed, resume)
    if rc == 0 and done(config_id, seed):
        log(f"✔ 完成 {config_id} seed{seed}")
        return "done"

    log(f"⚠ {config_id} seed{seed} 异常（退出码 {rc}），改用 RESUME=true 重试一次")
    rc2 = run_once(config_id, seed, True)
    if rc2 == 0 and done(config_id, seed):
        log(f"✔ RESUME 完成 {config_id} seed{seed}")
        return "done"

    log(f"✖ {config_id} seed{seed} 二次失败，停止队列，请人工检查后重启本脚本")
    return "failed"


def main():
    log("=" * 60)
    log("run_queue 启动；队列：")
    for c, s in QUEUE:
        log(f"    {c} seed{s}")
    log("=" * 60)

    i = 0
    while i < len(QUEUE):
        config_id, seed = QUEUE[i]
        st = process(config_id, seed)
        if st == "done":
            i += 1
            # 一个任务跑完、下一个开始前：温度过高则休息 10 分钟
            if i < len(QUEUE):
                _nxt_c, _nxt_s = QUEUE[i]
                temp_guard.check_temps_and_rest(
                    tag=f" 切换前(下一项 {_nxt_c} seed{_nxt_s})",
                    log_info=log, log_warn=log,
                )
        elif st == "running":
            time.sleep(300)  # 5 分钟后再查
        else:  # failed
            return 1
    log("🎉 队列全部完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
