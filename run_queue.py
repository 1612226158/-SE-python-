# -*- coding: utf-8 -*-
"""
run_queue.py —— G 系列实验自动排队器

用法（在 G-Full seed1 手动跑完之后）：
    D:\\Python\\Python3.10.7\\python.exe E:\\DataSet\\垃圾分类图片-2\\src_v2\\run_queue.py

它会按 QUEUE 顺序依次：
  1. 检查该 (CONFIG_ID, SEED) 是否已完成（有 results_*.json，或 metrics_record.csv 里有 epoch=99 行）→ 跳过
  2. 检查是否正在训练（checkpoint_last.pth 最近 15 分钟还在写）→ 等待 5 分钟重查
  3. 有断点但已停 → 自动 RESUME=true 续跑；无断点 → RESUME=false 全新跑
  4. 运行 train.py（阻塞等待自然结束）；失败自动用 RESUME=true 重试一次，再失败则停队
进度/报错写入 src_v2/runs/queue_log.txt。
"""
import glob
import json
import os
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

# G 系列执行队列（每配置 2 种子；G-Full seed2 暂缓、S-NoProg seed2 暂缓——用户 2026-09-03 定 S-NoProg 先跑 seed1 再说）
QUEUE = [
    ("G-Full", 1),
    ("G-NoSE", 1), ("G-NoSE", 2),
    ("S-NoProg", 1),
    ("G-NoTF", 1), ("G-NoTF", 2),
    ("G-SingleSE", 1), ("G-SingleSE", 2),
    ("G-PureBB", 1), ("G-PureBB", 2),
]

FRESH_SECONDS = 15 * 60  # checkpoint 最近 15 分钟有写 = 训练进行中


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done(config_id, seed):
    tag = f"{config_id}_seed{seed}"
    # 主标记：results JSON（训练结束生成，RUN_TAG 命名，不归档）
    if os.path.exists(os.path.join(RUNS, f"results_{tag}.json")):
        return True
    # 兜底：扫描当前 + 所有归档的 metrics_record*.csv，找该配置该种子的 epoch=99 行
    prefix = f"{config_id},{seed},99,"
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
    cfg = {
        "CONFIG_ID": config_id,
        "SEED": seed,
        "SMOKE": False,
        "MAX_EPOCHS": 100,
        "RESUME": resume,
        "AUTO": True,
        "UNFREEZE1_EPOCH": 30,
        "UNFREEZE2_EPOCH": 60,
    }
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def run_once(config_id, seed, resume):
    write_config(config_id, seed, resume)
    log(f"▶ 启动 {config_id} seed{seed}（RESUME={resume}）")
    r = subprocess.run([PYTHON, TRAIN], cwd=SRC_V2)
    return r.returncode


def process(config_id, seed):
    if done(config_id, seed):
        log(f"跳过已完成 {config_id} seed{seed}")
        return "done"
    if in_progress(config_id, seed):
        log(f"⏳ {config_id} seed{seed} 正在训练，等待 5 分钟")
        return "running"

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
