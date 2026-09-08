# -*- coding: utf-8 -*-
"""
plot_snoprog_vs_snose.py —— 全解冻含SE vs 全解冻无SE 三图对比（loss / val-acc / 耗时-acc）
论文图：SE 消融对比（S-NoProG 含SE vs S-NoSE 去SE，100 轮）

【参数化（2026-09-08 会话8）】顶部 MODEL_PAIR 声明对比的两个模型：
  - 数据：chart_data.load_model(模型名) 自动取（无需手写归档 csv）；
  - 输出文件名：chart_data.fig_out() 自动推导（命中论文历史标签 snoprog_vs_snose，
    换模型对自动变名）。

用法：
  D:\\Python\\Python3.10.7\\python.exe E:\\DataSet\\垃圾分类图片-2\\src_v2\\ReportChart\\plot_snoprog_vs_snose.py
输出：本脚本同目录 loss_curve/val_acc/time_acc_<对比段>.pdf（dpi=300，SCI 风格）
风格与 statistics\\ 既有脚本一致：seaborn-v0_8-whitegrid + SimHei。
"""
import os
import matplotlib.pyplot as plt
from chart_data import load_model, fig_out

_HERE = os.path.dirname(os.path.abspath(__file__))

# ============ 绘图风格（与既有论文图一致） ============
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# ============ 对比模型配置（画别的对比：改这一行即可） ============
# 每个元素: (模型英文名, 图内中文名, 主色)
MODEL_A = ('S-NoProg', 'S-NoProG（含多区域SE）', '#4c72b0')   # 蓝
MODEL_B = ('S-NoSE',   'S-NoSE（去多区域SE）',   '#c44e52')   # 红（去 SE 用红区分）
MAX_EPOCH = 99             # 截取公共轮数上限（100 轮协议）
C_TRAIN = '#1f77b4'
C_VAL = '#ff7f0e'


def main():
    name_a, cn_a, color_a = MODEL_A
    name_b, cn_b, color_b = MODEL_B
    df_a = load_model(name_a)
    df_b = load_model(name_b)
    print(f'{cn_a} 行数: {len(df_a)}, {cn_b} 行数: {len(df_b)}')

    df_a = df_a[df_a['epoch'] <= MAX_EPOCH].reset_index(drop=True)
    df_b = df_b[df_b['epoch'] <= MAX_EPOCH].reset_index(drop=True)
    n = min(len(df_a), len(df_b))
    df_a, df_b = df_a.iloc[:n], df_b.iloc[:n]
    ep = df_a['epoch']

    # ============================================================
    # 图 1：训练/验证损失曲线对比（2 子图并排：左=MODEL_A 右=MODEL_B）
    # ============================================================
    fig1, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, df, cn in [(axes[0], df_a, cn_a), (axes[1], df_b, cn_b)]:
        ax.plot(df['epoch'], df['train_loss'], label='训练损失', color=C_TRAIN, linewidth=1.5, alpha=0.9)
        ax.plot(df['epoch'], df['val_loss'], label='验证损失', color=C_VAL, linewidth=1.5, alpha=0.9)
        ax.set_title(cn, fontsize=12, fontweight='bold')
        ax.set_xlabel('训练轮次 (Epoch)')
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.legend(frameon=True, loc='best')
    axes[0].set_ylabel('损失值 (Loss)')
    fig1.suptitle(f'{cn_a} 与 {cn_b} 训练过程损失曲线对比', fontsize=13, y=1.02)
    fig1.tight_layout()
    out1 = fig_out('loss_curve', name_a, name_b)
    fig1.savefig(out1, bbox_inches='tight', dpi=300)
    plt.close(fig1)
    print(f'已生成: {out1}')

    # ============================================================
    # 图 2：验证集准确率曲线对比（同图双线 + best 峰标注）
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(7.5, 5))
    ax2.plot(ep, df_a['val_acc'], label=cn_a, color=color_a, linewidth=1.8, alpha=0.9)
    ax2.plot(ep, df_b['val_acc'], label=cn_b, color=color_b, linewidth=1.8, alpha=0.9)
    ba, bb = df_a['val_acc'].idxmax(), df_b['val_acc'].idxmax()
    ax2.plot(ep[ba], df_a['val_acc'][ba], 'o', color=color_a, markersize=7, zorder=5)
    ax2.plot(ep[bb], df_b['val_acc'][bb], 's', color=color_b, markersize=7, zorder=5)
    # best 标注（视觉诊断修正：y 向错开防叠压 + 白底框防穿字；两点几乎重合时 A 标上方、B 标下方）
    ax2.annotate(f"{cn_a}: {df_a['val_acc'][ba]:.2f}%@ep{int(ep[ba])}",
                 xy=(ep[ba], df_a['val_acc'][ba]), xytext=(ep[ba] + 2, df_a['val_acc'][ba] + 1.4),
                 arrowprops=dict(arrowstyle='-', color=color_a, lw=1.0),
                 color=color_a, fontsize=10, fontweight='bold', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=color_a, alpha=0.9))
    ax2.annotate(f"{cn_b}: {df_b['val_acc'][bb]:.2f}%@ep{int(ep[bb])}",
                 xy=(ep[bb], df_b['val_acc'][bb]), xytext=(ep[bb] + 2, df_b['val_acc'][bb] - 3.2),
                 arrowprops=dict(arrowstyle='-', color=color_b, lw=1.0),
                 color=color_b, fontsize=10, fontweight='bold', va='top',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=color_b, alpha=0.9))
    ax2.set_xlabel('训练轮次 (Epoch)')
    ax2.set_ylabel('验证集准确率 (%)')
    ax2.set_ylim(20, 88)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.legend(frameon=True, loc='lower right')
    ax2.set_title(f'验证集准确率对比（254类，CAWR调度，100轮）', fontsize=13)
    fig2.tight_layout()
    out2 = fig_out('val_acc', name_a, name_b)
    fig2.savefig(out2, bbox_inches='tight', dpi=300)
    plt.close(fig2)
    print(f'已生成: {out2}')

    # ============================================================
    # 图 3：耗时-准确率（收敛速度视角）
    # ============================================================
    cum_a = df_a['time_spend_s'].cumsum() / 60.0
    cum_b = df_b['time_spend_s'].cumsum() / 60.0
    fig3, ax3 = plt.subplots(figsize=(7.5, 5))
    ax3.plot(cum_a, df_a['val_acc'], label=cn_a, color=color_a, linewidth=1.8, alpha=0.9)
    ax3.plot(cum_b, df_b['val_acc'], label=cn_b, color=color_b, linewidth=1.8, alpha=0.9)
    ax3.set_xlabel('累计训练耗时 (分钟)')
    ax3.set_ylabel('验证集准确率 (%)')
    ax3.grid(True, alpha=0.4, linestyle='--')
    ax3.legend(frameon=True, loc='best')
    ax3.set_title('耗时-准确率曲线（收敛效率对比）', fontsize=13)
    fig3.tight_layout()
    out3 = fig_out('time_acc', name_a, name_b)
    fig3.savefig(out3, bbox_inches='tight', dpi=300)
    plt.close(fig3)
    print(f'已生成: {out3}')

    # ============ 控制台汇总 ============
    print(f'\n========== 汇总（0~{MAX_EPOCH}轮） ==========')
    for cn, df in [(cn_a, df_a), (cn_b, df_b)]:
        ib = df['val_acc'].idxmax()
        print(f"{cn}: best {df['val_acc'][ib]:.2f}%@ep{int(df['epoch'][ib])}, "
              f"final {df['val_acc'].iloc[-1]:.2f}%")


if __name__ == '__main__':
    main()
