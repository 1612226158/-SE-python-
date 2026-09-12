# -*- coding: utf-8 -*-
"""
plot_progressive_vs_full.py —— 渐进式解冻 vs 全解冻 三图对比（loss / val-acc / 耗时-acc）
论文图：解冻策略对比（G-Full-CAWR 渐进式 vs S-NoProG 全解冻，仅 100 轮）

【参数化（2026-09-08 会话8）】顶部 MODEL_PAIR 声明对比的两个模型：
  - 数据：chart_data.load_model(模型名) 自动取（无需手写归档 csv）；
  - 输出文件名：chart_data.fig_out() 自动推导（命中论文历史标签 progressive_vs_full，
    换模型对自动变名，无需手写 out*.pdf）。

用法：
  D:\\Python\\Python3.10.7\\python.exe E:\\DataSet\\垃圾分类图片-2\\src_v2\\ReportChart\\plot_progressive_vs_full.py
输出：本脚本同目录 loss_curve/val_acc/time_acc_<对比段>.pdf（dpi=300，SCI 风格）
风格与 statistics\\ 既有脚本一致：seaborn-v0_8-whitegrid + SimHei。
"""
import os
import matplotlib.pyplot as plt
from chart_data import load_model, fig_out, is_known, model_cn

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
# 模型英文名必须是 src_v2\model_registry.py 注册过的 id（数据由 chart_data 自动按名读取）；
# 图内中文名/主色是"本图的显示样式"，可按图意改，不会影响任何其他脚本。
MODEL_A = ('G-Full-CAWR', '渐进式解冻 (G-Full-CAWR)', '#4c72b0')   # 蓝
MODEL_B = ('S-NoProg',    '全解冻 (S-NoProg)',      '#8172b3')   # 紫

# —— 模型链接自检：id 必须在 model_registry.py 注册（数据/中文全称由 chart_data 按 id 解析）——
for _m in (MODEL_A[0], MODEL_B[0]):
    if not is_known(_m):
        raise SystemExit(f'[plot] 模型 {_m} 未在 model_registry.py 注册：请先在注册表加条目再画')
MAX_EPOCH = 99             # 截取公共轮数上限（此脚本仅 100 轮协议对比）
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
    # 图 1：训练/验证损失曲线（2 子图并排：左=MODEL_A 右=MODEL_B）
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
    fig1.suptitle(f'{cn_a.replace(" (", "（").replace(")", "）")} 与 {cn_b.replace(" (", "（").replace(")", "）")}'
                  f' 训练过程损失曲线对比（各100轮）', fontsize=13, y=1.02)
    fig1.tight_layout()
    out1 = fig_out('loss_curve', name_a, name_b)
    fig1.savefig(out1, bbox_inches='tight', dpi=300)
    plt.close(fig1)
    print(f'已生成: {out1}')

    # ============================================================
    # 图 2：验证集准确率曲线对比（同图双线 + best 峰标注，白底防穿字）
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(7.5, 5))
    ax2.plot(ep, df_a['val_acc'], label=cn_a, color=color_a, linewidth=1.8, alpha=0.9)
    ax2.plot(ep, df_b['val_acc'], label=cn_b, color=color_b, linewidth=1.8, alpha=0.9)
    ba, bb = df_a['val_acc'].idxmax(), df_b['val_acc'].idxmax()
    ax2.plot(ep[ba], df_a['val_acc'][ba], 'o', color=color_a, markersize=6, zorder=6)
    ax2.plot(ep[bb], df_b['val_acc'][bb], 's', color=color_b, markersize=6, zorder=6)
    # best 标注：文字精简为"数值@ep"（模型全名已在图例），两框按峰高错开 + 细引线指向峰点，
    # 避免长文本框互相重叠或整片盖住曲线（2026-09-09 视觉诊断后重构）
    best_a = (int(ep[ba]), df_a['val_acc'][ba], color_a)
    best_b = (int(ep[bb]), df_b['val_acc'][bb], color_b)
    high, low = sorted([best_a, best_b], key=lambda t: -t[1])
    for (x, y, color), dy in ((high, 3.8), (low, 3.4)):
        xl = x + (0 if dy == 3.8 else 0.6)   # 低峰框略右移，避免贴其右侧上行曲线
        ax2.annotate(f'{y:.2f}%@ep{x}',
                     xy=(x, y), xytext=(xl, y + dy),
                     fontsize=10, fontweight='bold', va='bottom', ha='center', color=color,
                     bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor=color, alpha=0.92),
                     arrowprops=dict(arrowstyle='-', color=color, lw=0.9, alpha=0.5))
    ax2.set_xlabel('训练轮次 (Epoch)')
    ax2.set_ylabel('验证集准确率 (%)')
    ax2.set_ylim(10, 90)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.legend(frameon=True, loc='lower right')
    ax2.set_title(f'验证集准确率对比（254类，CAWR调度，各100轮）', fontsize=13, pad=14)
    fig2.tight_layout()
    out2 = fig_out('val_acc', name_a, name_b)
    fig2.savefig(out2, bbox_inches='tight', dpi=300)
    plt.close(fig2)
    print(f'已生成: {out2}')

    # ============================================================
    # 图 3：耗时-准确率（收敛效率视角）
    # ============================================================
    cum_a = df_a['time_spend_s'].cumsum() / 60.0
    cum_b = df_b['time_spend_s'].cumsum() / 60.0
    fig3, ax3 = plt.subplots(figsize=(7.5, 5))
    ax3.plot(cum_a, df_a['val_acc'], label=cn_a, color=color_a, linewidth=1.8, alpha=0.9)
    ax3.plot(cum_b, df_b['val_acc'], label=cn_b, color=color_b, linewidth=1.8, alpha=0.9)
    ax3.set_xlabel('累计训练耗时 (分钟)')
    ax3.set_ylabel('验证集准确率 (%)')
    ax3.grid(True, alpha=0.4, linestyle='--')
    ax3.legend(frameon=True, loc='lower right')
    ax3.set_title('耗时-准确率曲线（收敛效率对比，各100轮）', fontsize=13)
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
