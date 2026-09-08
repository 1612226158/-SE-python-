# -*- coding: utf-8 -*-
"""
plot_snoprog_vs_snose.py —— S-NoProG(全解冻+SE) vs S-NoSE(全解冻无SE) 对比图
论文图：SE 消融对比（loss 曲线 / 验证集准确率 / 耗时-准确率）

数据源：自动扫描 runs/ 下全部 metrics_record*.csv（当前+归档），按模型英文名取数据
  S-NoProG / S-NoSE —— 见 chart_data.py（数据读取模块化，无需手写归档文件名）

用法：
  D:\\Python\\Python3.10.7\\python.exe E:\\DataSet\\垃圾分类图片-2\\src_v2\\ReportChart\\plot_snoprog_vs_snose.py
输出：本脚本同目录下 *_snoprog_vs_snose.pdf（dpi=300，SCI 风格）

风格与 statistics\\ 下既有脚本保持一致：seaborn-v0_8-whitegrid + SimHei + 蓝/橙配色。
"""
import os
import matplotlib.pyplot as plt
from chart_data import load_model

# ============ 路径与本模块无关（数据读取交给 chart_data） ============
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

# 配色（延续既有脚本：训练蓝/验证橙，柱状图 4 色板）
C_TRAIN = '#1f77b4'
C_VAL = '#ff7f0e'
C_NOPROG = '#4c72b0'   # S-NoProG 主色
C_NOSE = '#c44e52'     # S-NoSE 主色（去 SE 用红区分）

# ============ 数据加载（模块化：chart_data 自动扫全部归档） ============
def main():
    # —— 按模型英文名取数据：S-NoProG / S-NoSE（自动选最完整 seed，无需手写归档名）——
    df_noprog = load_model('S-NoProg')
    df_nose = load_model('S-NoSE')

    print(f'S-NoProG 行数: {len(df_noprog)}, S-NoSE 行数: {len(df_nose)}')

    # 截取公共轮数（两模型都是 0~99）
    n = min(len(df_noprog), len(df_nose))
    df_p, df_s = df_noprog.iloc[:n], df_nose.iloc[:n]
    ep = df_p['epoch']

    # ============================================================
    # 图 1：训练/验证损失曲线对比（2 子图并排：左=S-NoProG 右=S-NoSE）
    # ============================================================
    fig1, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, df, name in [(axes[0], df_p, 'S-NoProG（全解冻+SE）'),
                         (axes[1], df_s, 'S-NoSE（全解冻，去SE）')]:
        ax.plot(df['epoch'], df['train_loss'], label='训练损失', color=C_TRAIN, linewidth=1.5, alpha=0.9)
        ax.plot(df['epoch'], df['val_loss'], label='验证损失', color=C_VAL, linewidth=1.5, alpha=0.9)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('训练轮次 (Epoch)')
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.legend(frameon=True, loc='best')
    axes[0].set_ylabel('损失值 (Loss)')
    fig1.suptitle('S-NoProG 与 S-NoSE 训练过程损失曲线对比', fontsize=13, y=1.02)
    fig1.tight_layout()
    out1 = os.path.join(_HERE, 'loss_curve_snoprog_vs_snose.pdf')
    fig1.savefig(out1, bbox_inches='tight', dpi=300)
    plt.close(fig1)
    print(f'已生成: {out1}')

    # ============================================================
    # 图 2：验证集准确率曲线对比（同图双线）
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(7.5, 5))
    ax2.plot(ep, df_p['val_acc'], label='S-NoProG（含多区域SE）', color=C_NOPROG, linewidth=1.8, alpha=0.9)
    ax2.plot(ep, df_s['val_acc'], label='S-NoSE（去多区域SE）', color=C_NOSE, linewidth=1.8, alpha=0.9)
    # 标注各自 best（两点几乎重合(82.63@74 vs 82.53@73)，必须错开：S-NoProG 标上方、S-NoSE 标下方）
    # 视觉诊断(2026-09-07)修正：①红标注下移避开曲线通道；②两标记 y 向错开防叠压；③标注加白底框防穿字；④ylim 放宽留呼吸
    bp, bs = df_p['val_acc'].idxmax(), df_s['val_acc'].idxmax()
    # best 点标记：y 向错开 0.35（视觉上分离，不改变数据含义）
    ax2.plot(ep[bp], df_p['val_acc'][bp] + 0.35, 'o', color=C_NOPROG, markersize=7, zorder=5)
    ax2.plot(ep[bs], df_s['val_acc'][bs] - 0.35, 's', color=C_NOSE, markersize=7, zorder=5)
    ax2.annotate(f"S-NoProG: {df_p['val_acc'][bp]:.2f}%@ep{int(ep[bp])}",
                 xy=(ep[bp], df_p['val_acc'][bp]), xytext=(ep[bp] + 2, df_p['val_acc'][bp] + 1.4),
                 arrowprops=dict(arrowstyle='-', color=C_NOPROG, lw=1.0),
                 color=C_NOPROG, fontsize=10, fontweight='bold', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=C_NOPROG, alpha=0.9))
    ax2.annotate(f"S-NoSE: {df_s['val_acc'][bs]:.2f}%@ep{int(ep[bs])}",
                 xy=(ep[bs], df_s['val_acc'][bs]), xytext=(ep[bs] + 2, df_s['val_acc'][bs] - 3.2),
                 arrowprops=dict(arrowstyle='-', color=C_NOSE, lw=1.0),
                 color=C_NOSE, fontsize=10, fontweight='bold', va='top',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=C_NOSE, alpha=0.9))
    ax2.set_xlabel('训练轮次 (Epoch)')
    ax2.set_ylabel('验证集准确率 (%)')
    ax2.set_ylim(20, 88)  # 放宽上下界：上界容纳蓝标注白框，下界去掉 10-20 大段空白
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.legend(frameon=True, loc='lower right')
    ax2.set_title('验证集准确率对比（254类，CAWR调度，100轮）', fontsize=13)
    fig2.tight_layout()
    out2 = os.path.join(_HERE, 'val_acc_snoprog_vs_snose.pdf')
    fig2.savefig(out2, bbox_inches='tight', dpi=300)
    plt.close(fig2)
    print(f'已生成: {out2}')

    # ============================================================
    # 图 3：耗时-准确率（收敛速度视角：横轴累计耗时/分钟）
    # ============================================================
    # 每轮 time_spend_s 是"本轮训练耗时"，累计得截至该轮的总训练时长
    cum_p = df_p['time_spend_s'].cumsum() / 60.0
    cum_s = df_s['time_spend_s'].cumsum() / 60.0
    fig3, ax3 = plt.subplots(figsize=(7.5, 5))
    ax3.plot(cum_p, df_p['val_acc'], label='S-NoProG（含多区域SE）', color=C_NOPROG, linewidth=1.8, alpha=0.9)
    ax3.plot(cum_s, df_s['val_acc'], label='S-NoSE（去多区域SE）', color=C_NOSE, linewidth=1.8, alpha=0.9)
    ax3.set_xlabel('累计训练耗时 (分钟)')
    ax3.set_ylabel('验证集准确率 (%)')
    ax3.grid(True, alpha=0.4, linestyle='--')
    ax3.legend(frameon=True, loc='best')
    ax3.set_title('耗时-准确率曲线（收敛效率对比）', fontsize=13)
    fig3.tight_layout()
    out3 = os.path.join(_HERE, 'time_acc_snoprog_vs_snose.pdf')
    fig3.savefig(out3, bbox_inches='tight', dpi=300)
    plt.close(fig3)
    print(f'已生成: {out3}')

    # ============================================================
    # 控制台汇总（供论文文字引用）
    # ============================================================
    print('\n========== 汇总 ==========')
    for name, df in [('S-NoProG', df_p), ('S-NoSE', df_s)]:
        ib = df['val_acc'].idxmax()
        print(f"{name}: best val {df['val_acc'][ib]:.2f}%@ep{int(df['epoch'][ib])}, "
              f"final(ep{int(df['epoch'].iloc[-1])}) {df['val_acc'].iloc[-1]:.2f}%")

if __name__ == '__main__':
    main()
