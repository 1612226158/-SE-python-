# -*- coding: utf-8 -*-
"""
hl_common.py —— human_loop 共用工具（2026-09-12 建）

作用：把"类别体系映射 / 短名解析 / 图片清单 / 结果表输出"这类重复逻辑集中一处，
供 make_kappa_pack.py 与 kappa_report.py 共用。

口径：与训练/评估完全一致——**252 合并类**（merged_dict 的 9 组合并；2026-09-26 起，
     由 7 组 18 子类 / 254 类并入「蔬菜+菜根菜叶→菜叶」「插头电线+充电线→插头充电线」而来），
     文件夹 id → 合并类名 用 calculate.generate_mappings；图片扩展名与训练一致（不含 avif）。
纪律：只读 train/val 数据集；产物只写 human_loop/exports/ 下；不改数据集、不碰训练。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))      # src_v2\human_loop
_SRC = os.path.dirname(_HERE)                           # src_v2
_ROOT = os.path.dirname(_SRC)                           # 项目根
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src_v2 import merged_dict
from calculate import generate_mappings

# 与训练 SafeImageFolder._scan_disk 一致的扩展名（avif 不计）
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif')
VAL_ROOT = os.path.join(_ROOT, 'val')
TRAIN_ROOT = os.path.join(_ROOT, 'train')
EXPORT_ROOT = os.path.join(_HERE, 'exports')

# 默认待核验类对（来自 09-12 混淆矩阵分析：top 双向/高单向混淆对）
#
# ⚠️ 2026-09-26 槽位替换（方案 A，保留 8 个槽位）：
#   原第 3/4 位 —— P03「插头电线:充电线」、P04「蔬菜:菜根菜叶」——
#   已随 merged_dict 并入同一合并类（可回收物-插头充电线 / 厨余垃圾-菜叶），
#   不再是"两类之间的二选一"盲标问题 ⇒ 必须替换。
#
#   **为什么保留 8 个槽位、而不是删掉两行**：pair_id 由本列表的**位置**生成
#   （见下方 parse_pairs 的 `f'P{i:02d}'`）。直接删行会让 鞋/水杯/蛋糕/玻璃制品类
#   从 P05~P08 **错位成 P03~P06**，而已完成的 A 轮盲标包
#   `exports/20260912_220250_kappa_pack/`（images/P01..P08、sheetA.csv、_key.csv）
#   与论文 `tab:kappa-results` / `tab:kappa-human` 都是按旧 P0N 引用的。
#   这是一处**不报错的错位**，故刻意保持 8 槽位不变。
#
#   替换依据（同为"未被占用 + 不受本次合并影响 + 同一父类"，按 09-12 混淆矩阵计数降序）：
#     P03 衣服:裤子 —— 未用类对中混淆最高（衣服→裤子 25 张，占该类错误 0.51），均属可回收物
#     P04 饼干:蛋糕 —— 双向合计 23 张（饼干→蛋糕 12、蛋糕→饼干 11），均属厨余垃圾
#   ⚠️ 注意：这两个新类对**只有"模型混淆"证据，尚无人工盲标证据**；
#      原 P03/P04 的"人机对照"结论（含论文 L1942 蔬菜/菜根菜叶 0.345 vs 0.700）已作废。
DEFAULT_PAIRS = [
    '锅:电饭煲',
    '盒子:纸箱',
    '衣服:裤子',      # P03（2026-09-26 替换原「插头电线:充电线」）
    '饼干:蛋糕',      # P04（2026-09-26 替换原「蔬菜:菜根菜叶」）
    '鞋:拖鞋',
    '水杯:保温杯',
    '蛋糕:面包',
    '玻璃制品类:水杯',
]


def merged_class_index():
    """返回 (fid2cls, cls2fids, renew)：文件夹 id→合并类名、合并类名→[文件夹id]、类名→索引。"""
    id2main, renew = generate_mappings(merged_dict=merged_dict)
    fid2cls = {}
    for k, v in id2main.items():
        try:
            fid2cls[int(k)] = v
        except (TypeError, ValueError):
            continue
    cls2fids = {}
    for fid, cls in fid2cls.items():
        cls2fids.setdefault(cls, []).append(fid)
    for cls in cls2fids:
        cls2fids[cls].sort()
    return fid2cls, cls2fids, renew


def short_name(full):
    """'私有类-厨余垃圾-蔬菜' → '蔬菜'；'公共类-可回收物-玻璃制品类' → '玻璃制品类'。"""
    parts = full.split('-')
    return parts[2] if len(parts) > 2 else full


def resolve_class(token, cls2fids):
    """把短名或全名解析为唯一的合并类全名；多个候选时报错并列出。"""
    if token in cls2fids:
        return token
    cands = [c for c in cls2fids if short_name(c) == token]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise SystemExit(f'[hl] 找不到类「{token}」（请检查是否拼写正确，或该类是否被合并）')
    raise SystemExit(f'[hl] 类名「{token}」有多个候选，请写全名：{cands}')


def parse_pairs(pairs_arg, cls2fids):
    """'锅:电饭煲,盒子:纸箱' → [(pair_id, clsA_full, clsB_full), ...]（pair_id 形如 P01）。"""
    tokens = [t for t in (pairs_arg or '').replace('，', ',').split(',') if t.strip()]
    if not tokens:
        tokens = DEFAULT_PAIRS
    out = []
    for i, tk in enumerate(tokens, 1):
        if ':' not in tk and '：' not in tk:
            raise SystemExit(f'[hl] 类对格式应为 A:B，收到「{tk}」')
        a, b = tk.replace('：', ':').split(':', 1)
        cls_a = resolve_class(a.strip(), cls2fids)
        cls_b = resolve_class(b.strip(), cls2fids)
        out.append((f'P{i:02d}', cls_a, cls_b))
    return out


def list_images(root, fid):
    """某文件夹下的图片绝对路径（按文件名排序）。"""
    folder = os.path.join(root, str(fid))
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, n) for n in sorted(os.listdir(folder))
            if n.lower().endswith(VALID_EXTS)]


def class_images(cls, cls2fids, root=VAL_ROOT):
    """某合并类的全部图片（该类的所有成员文件夹）。"""
    imgs = []
    for fid in cls2fids.get(cls, []):
        imgs += list_images(root, fid)
    return imgs


def cohens_kappa(a, b):
    """两个标注序列（元素为可比标签）→ (kappa, p_o, p_e)。空序列返回 (None, None, None)。

    κ = (p_o - p_e) / (1 - p_e)；p_e 由两侧标签的边缘分布计算（含随机一致的期望）。
    """
    n = len(a)
    if n == 0:
        return None, None, None
    labels = sorted(set(a) | set(b))
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    p_e = 0.0
    for lb in labels:
        ca = sum(1 for x in a if x == lb) / n
        cb = sum(1 for y in b if y == lb) / n
        p_e += ca * cb
    kappa = None if abs(1 - p_e) < 1e-12 else (p_o - p_e) / (1 - p_e)
    return kappa, p_o, p_e


def kappa_level(k):
    """按 Landis & Koch 常用分级给出中文档位。"""
    if k is None:
        return '不可计算'
    if k < 0:
        return '差于随机'
    if k < 0.21:
        return '极弱（0-0.20）'
    if k < 0.41:
        return '一般（0.21-0.40）'
    if k < 0.61:
        return '中等（0.41-0.60）'
    if k < 0.81:
        return '较好（0.61-0.80）'
    return '几乎完全一致（0.81-1.00）'
