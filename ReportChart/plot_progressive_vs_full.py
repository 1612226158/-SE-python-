# -*- coding: utf-8 -*-
"""
plot_progressive_vs_full.py —— 渐进式解冻(G-Full-CAWR修复后) vs 全解冻(S-NoProG) 对比图（仅 100 轮）
论文图：解冻策略对比（loss 曲线 / 验证集准确率 / 耗时-准确率）

数据源：自动扫描 runs/ 下全部 metrics_record*.csv（当前+归档），按模型英文名取数据
  G-Full-CAWR / S-NoProG —— 见 chart_data.py（数据读取模块化，无需手写归档文件名）

用法：
  D:\\Python\\Python3.10.7\\python.exe E:\\DataSet\\垃圾分类图片-2\\src_v2\\ReportChart\\plot_progressive_vs_full.py
输出：本脚本同目录 *_progressive_vs_full.pdf（dpi=300，SCI 风格）
风格与 statistics\\ 既有脚本一致：seaborn-v0_8-whitegrid + SimHei。
"""
import os
import matplotlib.pyplot as plt
from chart_data import load_model

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

C_TRAIN = '#1f77b4'
C_VAL = '#ff7f0e'
C_PROG = '#4c72b0'   # 渐进式主色（蓝）
C_FULL = '#8172b3'   # 全解冻主色（紫，与渐进式区分）

# ============ 数据加载（模块化：chart_data 自动扫全部归档） ============
def main():
    # —— 按模型英文名取数据：G-Full-CAWR(渐进式) / S-NoProG(全解冻) ——
    df_prog = load_model('G-Full-CAWR')
    df_full = load_model('S-NoProg')

    print(f'渐进式 G-Full-CAWR 行数: {len(df_prog)}, 全解冻 S-NoProG 行数: {len(df_full)}')

    # 只取 0~99 轮
    df_p = df_prog[df_prog['epoch'] <= 99].reset_index(drop=True)
    df_f = df_full[df_full['epoch'] <= 99].reset_index(drop=True)
    n = min(len(df_p), len(df_f))
    df_p, df_f = df_p.iloc[:n], df_f.iloc[:n]
    ep = df_p['epoch']

    # ============================================================
    # 图 1：训练/验证损失曲线（2 子图并排：左=渐进式 右=全解冻）
    # ============================================================
    fig1, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, df, name in [(axes[0], df_p, '渐进式解冻 (G-Full-CAWR)'),
                         (axes[1], df_f, '全解冻 (S-NoProG)')]:
        ax.plot(df['epoch'], df['train_loss'], label='训练损失', color=C_TRAIN, linewidth=1.5, alpha=0.9)
        ax.plot(df['epoch'], df['val_loss'], label='验证损失', color=C_VAL, linewidth=1.5, alpha=0.9)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('训练轮次 (Epoch)')
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.legend(frameon=True, loc='best')
    axes[0].set_ylabel('损失值 (Loss)')
    fig1.suptitle('渐进式解冻与全解冻训练过程损失曲线对比（各100轮）', fontsize=13, y=1.02)
    fig1.tight_layout()
    out1 = os.path.join(_HERE, 'loss_curve_progressive_vs_full.pdf')
    fig1.savefig(out1, bbox_inches='tight', dpi=300)
    plt.close(fig1)
    print(f'已生成: {out1}')

    # ============================================================
    # 图 2：验证集准确率曲线对比（同图双线）
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(7.5, 5))
    ax2.plot(ep, df_p['val_acc'], label='渐进式解冻 (G-Full-CAWR)', color=C_PROG, linewidth=1.8, alpha=0.9)
    ax2.plot(ep, df_f['val_acc'], label='全解冻 (S-NoProG)', color=C_FULL, linewidth=1.8, alpha=0.9)
    # 标注各自 best（视觉诊断两轮修正后定稿：峰点就近小标注+白底，不用长 leader 线——ep92-99 两曲线交叉无空白区，
    # 长引线必然与曲线重叠。蓝(渐进式)峰 ep92 在曲线簇中，标注放峰点左上方小偏移+白底；紫(全解冻)峰 ep74 上方有空，标峰点上方）
    bp, bf = df_p['val_acc'].idxmax(), df_f['val_acc'].idxmax()
    ax2.plot(ep[bp], df_p['val_acc'][bp], 'o', color=C_PROG, markersize=6, zorder=6)
    ax2.plot(ep[bf], df_f['val_acc'][bf], 's', color=C_FULL, markersize=6, zorder=6)
    # 紫色(全解冻)峰在上，标注放其正上方
    ax2.annotate(f"全解冻: {df_f['val_acc'][bf]:.2f}%@ep{int(ep[bf])}",
                 xy=(ep[bf], df_f['val_acc'][bf]), xytext=(ep[bf] - 2, df_f['val_acc'][bf] + 2.6),
                 color=C_FULL, fontsize=10, fontweight='bold', va='bottom', ha='center',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=C_FULL, alpha=0.95))
    # 蓝色(渐进式)峰 ep92 两侧都是曲线，标注放峰点上方居中（收进轴内，防右缘贴边）
    ax2.annotate(f"渐进式: {df_p['val_acc'][bp]:.2f}%@ep{int(ep[bp])}",
                 xy=(ep[bp], df_p['val_acc'][bp]), xytext=(ep[bp] - 1, df_p['val_acc'][bp] + 2.8),
                 color=C_PROG, fontsize=10, fontweight='bold', va='bottom', ha='center',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=C_PROG, alpha=0.95))
    ax2.set_xlabel('训练轮次 (Epoch)')
    ax2.set_ylabel('验证集准确率 (%)')
    ax2.set_ylim(10, 90)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.legend(frameon=True, loc='lower right')
    ax2.set_title('验证集准确率对比（254类，CAWR调度，各100轮）', fontsize=13, pad=14)
    fig2.tight_layout()
    out2 = os.path.join(_HERE, 'val_acc_progressive_vs_full.pdf')
    fig2.savefig(out2, bbox_inches='tight', dpi=300)
    plt.close(fig2)
    print(f'已生成: {out2}')

    # ============================================================
    # 图 3：耗时-准确率（收敛效率视角）
    # ============================================================
    cum_p = df_p['time_spend_s'].cumsum() / 60.0
    cum_f = df_f['time_spend_s'].cumsum() / 60.0
    fig3, ax3 = plt.subplots(figsize=(7.5, 5))
    ax3.plot(cum_p, df_p['val_acc'], label='渐进式解冻 (G-Full-CAWR)', color=C_PROG, linewidth=1.8, alpha=0.9)
    ax3.plot(cum_f, df_f['val_acc'], label='全解冻 (S-NoProG)', color=C_FULL, linewidth=1.8, alpha=0.9)
    ax3.set_xlabel('累计训练耗时 (分钟)')
    ax3.set_ylabel('验证集准确率 (%)')
    ax3.grid(True, alpha=0.4, linestyle='--')
    ax3.legend(frameon=True, loc='lower right')
    ax3.set_title('耗时-准确率曲线（收敛效率对比，各100轮）', fontsize=13)
    fig3.tight_layout()
    out3 = os.path.join(_HERE, 'time_acc_progressive_vs_full.pdf')
    fig3.savefig(out3, bbox_inches='tight', dpi=300)
    plt.close(fig3)
    print(f'已生成: {out3}')

    # ============ 控制台汇总 ============
    print('\n========== 汇总（0~99轮） ==========')
    for name, df in [('渐进式 G-Full-CAWR', df_p), ('全解冻 S-NoProG', df_f)]:
        ib = df['val_acc'].idxmax()
        print(f"{name}: best {df['val_acc'][ib]:.2f}%@ep{int(df['epoch'][ib])}, final {df['val_acc'].iloc[-1]:.2f}%")

if __name__ == '__main__':
    main()
