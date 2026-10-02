# -*- coding: utf-8 -*-
"""三组SE实验的统一对比图。

数据只通过上级 ``chart_data`` 模块读取；本脚本不直接打开 runs 下的CSV，
因此沿用统一的归档扫描、脏记录排除、seed筛选、epoch去重和协议汇总规则。

运行：
    python plot_se_threeway.py

输出均写入本脚本所在的 ``ReportChart/v2`` 目录。
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
REPORT_CHART = HERE.parent
if str(REPORT_CHART) not in sys.path:
    sys.path.insert(0, str(REPORT_CHART))

from chart_data import N_CLS, is_known, load_model, summarize_run  # noqa: E402


MODELS = [
    ("S-NoProg", "S-NoProg（多区域SE [1,2]）", "#4c72b0"),
    ("S-NoSE", "S-NoSE（无SE）", "#c44e52"),
    ("S-StdSE", "S-StdSE（标准SE [1]）", "#dd8452"),
]
SEED = 1
TARGET_EPOCHS = 100
RESTARTS = [5, 15, 35, 75]
CYCLES = [(0, 4), (5, 14), (15, 34), (35, 74), (75, 99)]
THRESHOLDS = [80.0, 82.0, 83.0]


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "grid.color": "#d9d9d9",
    "grid.linestyle": "--",
    "grid.linewidth": .7,
})


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _save(fig, name):
    path = HERE / name
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] {path.name} / {pdf_path.name}")
    return path


def _rolling(series, window=5):
    return series.rolling(window=window, min_periods=1, center=True).mean()


def _first_hit(df, threshold):
    hit = df[df["val_acc"] >= threshold]
    if hit.empty:
        return None
    idx = int(hit.index[0])
    return {
        "epoch": int(df.loc[idx, "epoch"]),
        "hours": float(df.loc[:idx, "time_spend_s"].sum() / 3600.0),
    }


def _sustained_hit(df, threshold, span=3):
    values = df["val_acc"].to_numpy(float)
    for start in range(0, len(values) - span + 1):
        if np.all(values[start:start + span] >= threshold):
            return int(df.iloc[start]["epoch"])
    return None


def _cycle_rows(df):
    rows = []
    for index, (start, end) in enumerate(CYCLES):
        part = df[(df["epoch"] >= start) & (df["epoch"] <= end)]
        peak_idx = part["val_acc"].idxmax()
        item = {
            "cycle": index + 1,
            "start": start,
            "end": end,
            "peak_val_acc": float(df.loc[peak_idx, "val_acc"]),
            "peak_epoch": int(df.loc[peak_idx, "epoch"]),
            "mean_val_acc": float(part["val_acc"].mean()),
            "std_val_acc": float(part["val_acc"].std(ddof=0)),
        }
        if start > 0:
            before = float(df.loc[df["epoch"] == start - 1, "val_acc"].iloc[0])
            after = float(df.loc[df["epoch"] == start, "val_acc"].iloc[0])
            previous_peak = max(row["peak_val_acc"] for row in rows)
            recovery = part[part["val_acc"] >= previous_peak]
            item.update({
                "restart_drop_pp": after - before,
                "previous_global_peak": previous_peak,
                "recovery_epoch": None if recovery.empty else int(recovery.iloc[0]["epoch"]),
                "recovery_epochs_after_restart": None if recovery.empty else int(recovery.iloc[0]["epoch"] - start),
            })
        rows.append(item)
    return rows


def load_runs():
    runs, summaries = {}, {}
    protocol = None
    protocol_keys = (
        "training_protocol_version", "sampling_protocol_version",
        "label_schema_id", "label_schema_version", "label_mapping_hash", "num_classes",
    )
    for model, _label, _color in MODELS:
        if not is_known(model):
            raise RuntimeError(f"模型未注册：{model}")
        df = load_model(model, seed=SEED, verbose=False).reset_index(drop=True)
        summary = summarize_run(model, seed=SEED)
        epochs = df["epoch"].astype(int).tolist()
        if summary["status"] != "finished" or not summary["results_json"]:
            raise RuntimeError(f"{model} 尚未正式收官")
        if epochs != list(range(TARGET_EPOCHS)):
            raise RuntimeError(f"{model} epoch不连续：{epochs[:3]}...{epochs[-3:]}")
        current = {key: summary["protocol"].get(key) for key in protocol_keys}
        if protocol is None:
            protocol = current
        elif current != protocol:
            raise RuntimeError(f"协议不一致：{model}={current}，基准={protocol}")
        runs[model] = df
        summaries[model] = summary
    return runs, summaries, protocol


def build_analysis(runs, summaries, protocol):
    derived = {}
    for model, _label, _color in MODELS:
        df = runs[model]
        val = df["val_acc"].to_numpy(float)
        top10 = np.sort(val)[-10:]
        derived[model] = {
            "mean_val_acc_100_epochs": float(np.mean(val)),
            "top10_epoch_mean_val_acc": float(np.mean(top10)),
            "late_pre_restart_65_74_mean": float(df[df["epoch"].between(65, 74)]["val_acc"].mean()),
            "late_pre_restart_65_74_std": float(df[df["epoch"].between(65, 74)]["val_acc"].std(ddof=0)),
            "final_window_90_99_mean": float(df[df["epoch"].between(90, 99)]["val_acc"].mean()),
            "final_window_90_99_std": float(df[df["epoch"].between(90, 99)]["val_acc"].std(ddof=0)),
            "thresholds": {
                str(int(t)): {
                    "first_hit": _first_hit(df, t),
                    "first_3_consecutive_epoch": _sustained_hit(df, t, span=3),
                }
                for t in THRESHOLDS
            },
            "cycles": _cycle_rows(df),
        }

    best = {m: summaries[m]["official_best_val_acc"]["value"] for m, _, _ in MODELS}
    pairwise = {
        "StdSE_minus_NoSE_best_acc_pp": float(best["S-StdSE"] - best["S-NoSE"]),
        "MultiRegion_minus_NoSE_best_acc_pp": float(best["S-NoProg"] - best["S-NoSE"]),
        "StdSE_minus_MultiRegion_best_acc_pp": float(best["S-StdSE"] - best["S-NoProg"]),
    }
    return {
        "scope": {"seed": SEED, "epochs": TARGET_EPOCHS, "num_classes": N_CLS},
        "protocol": protocol,
        "summaries": summaries,
        "derived": derived,
        "pairwise": pairwise,
        "limitations": [
            "仅seed1，不能据此声称统计显著或跨随机种子稳定。",
            "best、F1、loss和末轮值的最优排序不同，不能用单一指标替代完整判断。",
        ],
    }


def plot_overview(analysis):
    labels = [label for _, label, _ in MODELS]
    colors = [color for _, _, color in MODELS]
    x = np.arange(len(MODELS))
    best = [analysis["summaries"][m]["official_best_val_acc"]["value"] for m, _, _ in MODELS]
    macro = [100 * analysis["summaries"][m]["best_macro_f1"]["value"] for m, _, _ in MODELS]
    weighted = [100 * analysis["summaries"][m]["best_weighted_f1"]["value"] for m, _, _ in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    short_labels = [label.split("（", 1)[0] for label in labels]
    bars = axes[0].bar(x, best, color=colors, width=.58)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("验证准确率 (%)")
    axes[0].set_title("(a) 最佳验证准确率")
    for bar, value in zip(bars, best):
        axes[0].text(bar.get_x() + bar.get_width()/2, value + 1.0, f"{value:.3f}%", ha="center", fontsize=9)

    width = .34
    bars1 = axes[1].bar(x - width/2, macro, width, label="Macro-F1", color=colors)
    bars2 = axes[1].bar(x + width/2, weighted, width, label="Weighted-F1", color=colors, alpha=.58)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("F1 (%)")
    axes[1].set_title("(b) 各自最佳F1")
    axes[1].legend(
        handles=[
            Patch(facecolor="#777777", label="Macro-F1"),
            Patch(facecolor="#bbbbbb", label="Weighted-F1"),
        ],
        loc="lower right",
        frameon=True,
    )
    for bars in (bars1, bars2):
        for bar in bars:
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + .8,
                         f"{bar.get_height():.2f}", ha="center", fontsize=8)
    for ax in axes:
        ax.set_xticks(x, short_labels)
        ax.grid(True, alpha=.55, linestyle="--")
    fig.suptitle(f"SE配置的最佳指标对比（{N_CLS}类，seed1）", fontsize=13, y=1.01)
    fig.tight_layout()
    return _save(fig, "01_best_metrics_comparison.png")


def plot_accuracy(runs):
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    text_positions = {
        "S-NoProg": (37, 85.6),
        "S-NoSE": (69, 80.6),
        "S-StdSE": (64, 88.1),
    }
    for model, label, color in MODELS:
        df = runs[model]
        ax.plot(df["epoch"], df["val_acc"], color=color, linewidth=1.8, alpha=.92, label=label)
        ib = df["val_acc"].idxmax()
        x, y = int(df.loc[ib, "epoch"]), float(df.loc[ib, "val_acc"])
        ax.scatter(x, y, color=color, s=50, zorder=5)
        ax.annotate(f"{label.split('（',1)[0]}：{y:.3f}%@ep{x}", xy=(x, y),
                    xytext=text_positions[model], color=color, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=.95),
                    arrowprops=dict(arrowstyle="-", color=color, lw=.9))
    ax.set_xlabel("训练轮次（Epoch）")
    ax.set_ylabel("验证集准确率 (%)")
    ax.set_ylim(45, 90)
    ax.set_title(f"验证集准确率对比（{N_CLS}类，CAWR调度，100轮）")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=.6, linestyle="--")
    fig.tight_layout()
    return _save(fig, "02_val_acc_comparison.png")


def plot_loss(runs):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), sharex=True, sharey=True)
    for ax, (model, label, _color) in zip(axes, MODELS):
        df = runs[model]
        ax.plot(df["epoch"], df["train_loss"], label="训练损失", color="#1f77b4", linewidth=1.5)
        ax.plot(df["epoch"], df["val_loss"], label="验证损失", color="#ff7f0e", linewidth=1.5)
        ax.set_title(label)
        ax.set_xlabel("训练轮次（Epoch）")
        ax.grid(True, alpha=.6, linestyle="--")
        ax.legend(loc="upper right", frameon=True)
    axes[0].set_ylabel("损失值（Loss）")
    fig.suptitle("三组SE实验的训练与验证损失曲线", fontsize=13, y=1.02)
    fig.tight_layout()
    return _save(fig, "03_loss_curves.png")


def plot_metrics(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    specs = [
        ("val_macro_f1", "Macro-F1", 100.0, "Macro-F1 (%)"),
        ("val_weighted_f1", "Weighted-F1", 100.0, "Weighted-F1 (%)"),
    ]
    for ax, (column, title, scale, ylabel) in zip(axes, specs):
        for model, label, color in MODELS:
            df = runs[model]
            ax.plot(df["epoch"], df[column] * scale, color=color, linewidth=1.7, label=label)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("训练轮次（Epoch）")
        ax.grid(True, alpha=.6, linestyle="--")
    axes[0].legend(loc="lower right", frameon=True)
    fig.suptitle("验证集F1曲线对比", fontsize=13, y=1.01)
    fig.tight_layout()
    return _save(fig, "04_f1_comparison.png")


def plot_efficiency(runs):
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for model, label, color in MODELS:
        df = runs[model]
        hours = df["time_spend_s"].cumsum() / 3600.0
        ax.plot(hours, df["val_acc"], color=color, linewidth=1.8, alpha=.92, label=label)
    ax.set_xlabel("累计训练耗时（小时）")
    ax.set_ylabel("验证集准确率 (%)")
    ax.set_title("耗时-准确率曲线（收敛效率对比）")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=.6, linestyle="--")
    fig.tight_layout()
    return _save(fig, "05_time_acc_comparison.png")


def plot_delta(runs):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    baseline = _rolling(runs["S-NoSE"]["val_acc"])
    for model, label, color in MODELS:
        if model == "S-NoSE":
            continue
        delta = _rolling(runs[model]["val_acc"]) - baseline
        ax.plot(runs[model]["epoch"], delta, color=color, linewidth=1.8, label=f"{label} − 无SE")
    for restart in RESTARTS:
        ax.axvline(restart, color="#aaaaaa", linestyle="--", linewidth=.8, alpha=.75)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xlabel("训练轮次（Epoch；竖虚线为CAWR重启点）")
    ax.set_ylabel("相对无SE差值 (pp)")
    ax.set_title("SE路线相对无SE的验证准确率差异（5轮滑动均值）")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=.6, linestyle="--")
    fig.tight_layout()
    return _save(fig, "06_se_gain_vs_nose.png")


def plot_cycles(analysis):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    cycle_labels = [f"{a}–{b}" for a, b in CYCLES]
    for model, label, color in MODELS:
        rows = analysis["derived"][model]["cycles"]
        axes[0].plot(cycle_labels, [r["peak_val_acc"] for r in rows], marker="o",
                     linewidth=1.8, color=color, label=label)
    axes[0].set_ylabel("周期内峰值验证准确率 (%)")
    axes[0].set_xlabel("Epoch区间")
    axes[0].set_title("各CAWR周期的最好结果")
    axes[0].legend(loc="lower right", frameon=True)
    axes[0].grid(True, alpha=.6, linestyle="--")

    width = .23
    x = np.arange(len(RESTARTS))
    for i, (model, label, color) in enumerate(MODELS):
        rows = analysis["derived"][model]["cycles"][1:]
        vals = [r["restart_drop_pp"] for r in rows]
        axes[1].bar(x + (i - 1) * width, vals, width, color=color, label=label)
    axes[1].axhline(0, color="#111827", linewidth=1)
    axes[1].set_xticks(x, [f"ep{e}" for e in RESTARTS])
    axes[1].set_ylabel("重启点相对前一轮变化 (pp)")
    axes[1].set_title("学习率重启的即时冲击")
    axes[1].set_xlabel("重启点")
    axes[1].legend(loc="lower left", frameon=True)
    axes[1].grid(True, axis="y", alpha=.6, linestyle="--")
    fig.suptitle("CAWR周期与重启响应", fontsize=13, y=1.01)
    fig.tight_layout()
    return _save(fig, "07_cawr_cycle_response.png")


def plot_epoch_lead(runs):
    matrix = np.vstack([runs[m]["val_acc"].to_numpy(float) for m, _, _ in MODELS])
    best = matrix.max(axis=0, keepdims=True)
    deficit = matrix - best
    leaders = np.isclose(matrix, best, atol=1e-12)
    counts = leaders.sum(axis=1)
    labels = [label for _, label, _ in MODELS]
    colors = [color for _, _, color in MODELS]

    fig, axes = plt.subplots(2, 1, figsize=(14, 5.8), gridspec_kw={"height_ratios": [1.4, 1]})
    vmax = 0.0
    vmin = min(-4.0, float(np.nanmin(deficit)))
    im = axes[0].imshow(deficit, aspect="auto", cmap="YlOrRd_r", vmin=vmin, vmax=vmax,
                        extent=(-.5, 99.5, 2.5, -.5))
    axes[0].set_yticks(range(3), labels)
    axes[0].set_xlabel("Epoch")
    axes[0].set_title("每轮相对当轮最好模型的差距（0表示当轮领先）")
    for restart in RESTARTS:
        axes[0].axvline(restart, color="white", linestyle="--", linewidth=.9, alpha=.9)
    cbar = fig.colorbar(im, ax=axes[0], fraction=.025, pad=.02)
    cbar.set_label("差距 (pp)")

    bars = axes[1].bar(labels, counts, color=colors, width=.58)
    axes[1].set_ylabel("领先轮数")
    axes[1].set_ylim(0, max(counts) * 1.18)
    axes[1].set_title("100轮中验证准确率当轮领先次数（并列均计入）")
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, count + 1, str(int(count)), ha="center")
    fig.suptitle("逐轮领先关系", fontsize=13, y=1.01)
    fig.tight_layout()
    return _save(fig, "08_epoch_lead_heatmap.png")


def main():
    os.chdir(HERE)
    runs, summaries, protocol = load_runs()
    analysis = build_analysis(runs, summaries, protocol)
    outputs = [
        plot_overview(analysis),
        plot_accuracy(runs),
        plot_loss(runs),
        plot_metrics(runs),
        plot_efficiency(runs),
        plot_delta(runs),
        plot_cycles(analysis),
        plot_epoch_lead(runs),
    ]
    json_path = HERE / "comparison_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"[data] {json_path.name}")
    print("[done]", ", ".join(path.name for path in outputs))


if __name__ == "__main__":
    main()
