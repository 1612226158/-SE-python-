# -*- coding: utf-8 -*-
"""
make_kappa_pack.py —— 生成"标注一致性检验"抽样包（2026-09-12 建，human_loop）

用途：为"哪些类对的标签边界本身不可靠（任务内在歧义）"提供硬证据——
      从易混淆类对中平衡抽样、打乱文件名，生成**盲标**用的图片包与两张空白标注表；
      标注完成后用 kappa_report.py 计算 Cohen's κ。

用法：
  python make_kappa_pack.py                                  # 默认 8 组易混淆对，每类 15 张
  python make_kappa_pack.py --pairs "锅:电饭煲,盒子:纸箱" --n-per-class 20
  python make_kappa_pack.py --seed 20260912 --out <自定义目录>

产出（默认 human_loop/exports/<时间戳>_kappa_pack/）：
  images/P01/P01_001.jpg ...   打乱顺序的图片副本（文件名不含类别信息，保证盲标）
  pairs.csv                    类对清单（pair_id, 两个类全名/短名, 各类张数）
  sheetA.csv / sheetB.csv      两张空白标注表（option_1/option_2 为随机顺序的两个候选类）
  _key.csv                     ⚠ 答案键（图片→真实类别），标注者不得打开
  README_ANNOTATOR.txt         标注者须知

纪律：只读 val（或 --source train）；不改数据集；产物仅写 exports/。
"""
import os
import sys
import csv
import random
import shutil
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hl_common import (VALID_EXTS, VAL_ROOT, TRAIN_ROOT, EXPORT_ROOT, DEFAULT_PAIRS,
                       merged_class_index, short_name, parse_pairs, class_images)

ANNOTATOR_README = """【标注须知】请严格按以下规则操作，不要打开 _key.csv（那是答案，打开会让本批数据作废）

1. 打开 sheetA.csv（或 sheetB.csv），只填 choice 一列：
   - 看 images/<pair_id>/<image> 这张图；
   - 在两个候选类 option_1 / option_2 中二选一，choice 填 1 或 2（不要填类名）；
   - 若实在无法判断，choice 填 0，并在 notes 里写一句原因。
2. 只能依据图片内容判断；可以放大看细节，但不要搜索、不要与人讨论（两人须独立标注）。
3. 每张图都必须给出选择（允许 0，但请尽量少用）。
4. 同一 pair_id 内的图片来自两个相似类、顺序已随机打乱，不要试图猜测规律。
5. 建议节奏：每次 20-30 分钟，避免疲劳导致的系统性偏差；如中途休息，保存 CSV 后继续。

文件说明：
  images/       图片副本（文件名已打乱，不含类别线索）
  sheetX.csv    你的标注表：pair_id,image,option_1,option_2,choice,notes
  pairs.csv     本次类对清单（仅说明比对的是哪两类）
  _key.csv      ⚠ 答案键：pair_id,image,true_class,true_short,option_1,option_2,source_path

标注完成后把 sheetA.csv / sheetB.csv 交回，用 kappa_report.py 计算 κ。
"""


def main():
    ap = argparse.ArgumentParser(description='生成标注一致性检验抽样包（盲标）')
    ap.add_argument('--pairs', default=None, help='类对，如 "锅:电饭煲,盒子:纸箱"；缺省用内置 8 组易混淆对')
    ap.add_argument('--n-per-class', type=int, default=15, help='每个类抽多少张（默认 15 → 每对 30 张）')
    ap.add_argument('--seed', type=int, default=20260912, help='抽样与打乱随机种子（可复现）')
    ap.add_argument('--source', choices=['val', 'train'], default='val',
                    help='图片来源（默认 val；train 图更多但会与训练集重叠，一般不用）')
    ap.add_argument('--out', default=None, help='输出目录（默认 exports/<时间戳>_kappa_pack）')
    a = ap.parse_args()

    root = VAL_ROOT if a.source == 'val' else TRAIN_ROOT
    rng = random.Random(a.seed)
    _fid2cls, cls2fids, _renew = merged_class_index()
    pairs = parse_pairs(a.pairs, cls2fids)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = a.out or os.path.join(EXPORT_ROOT, f'{ts}_kappa_pack')
    img_root = os.path.join(out_dir, 'images')
    os.makedirs(img_root, exist_ok=True)

    rows, key_rows, pair_rows = [], [], []
    for pid, cls_a, cls_b in pairs:
        pool_a = class_images(cls_a, cls2fids, root)
        pool_b = class_images(cls_b, cls2fids, root)
        if len(pool_a) < a.n_per_class or len(pool_b) < a.n_per_class:
            print(f'[hl] 警告 {pid} {short_name(cls_a)}/{short_name(cls_b)}：可用图片 '
                  f'{len(pool_a)}/{len(pool_b)} 张，不足 {a.n_per_class}，按较小值抽取')
        k = min(a.n_per_class, len(pool_a), len(pool_b))
        if k == 0:
            print(f'[hl] 跳过 {pid}：某一类没有可用图片')
            continue
        picked = ([(p, cls_a) for p in rng.sample(pool_a, k)] +
                  [(p, cls_b) for p in rng.sample(pool_b, k)])
        rng.shuffle(picked)
        os.makedirs(os.path.join(img_root, pid), exist_ok=True)
        for i, (src, cls) in enumerate(picked, 1):
            ext = os.path.splitext(src)[1].lower() or '.jpg'
            name = f'{pid}_{i:03d}{ext}'
            shutil.copy2(src, os.path.join(img_root, pid, name))
            # 每张图的两个候选顺序独立随机，避免位置偏置；两张标注表共用同一顺序
            opts = [cls_a, cls_b]
            rng.shuffle(opts)
            o1, o2 = short_name(opts[0]), short_name(opts[1])
            rows.append(dict(pair_id=pid, image=f'{pid}/{name}', option_1=o1, option_2=o2,
                             choice='', notes=''))
            key_rows.append(dict(pair_id=pid, image=f'{pid}/{name}', true_class=cls,
                                 true_short=short_name(cls), option_1=o1, option_2=o2,
                                 source_path=src))
        pair_rows.append(dict(pair_id=pid, class_A=cls_a, class_B=cls_b,
                              short_A=short_name(cls_a), short_B=short_name(cls_b),
                              n_A=k, n_B=k, source=a.source))
        print(f'[hl] {pid} {short_name(cls_a)} vs {short_name(cls_b)}：各抽 {k} 张（源 {a.source}）')

    def write_csv(path, fieldnames, data, encoding='utf-8-sig'):
        with open(path, 'w', encoding=encoding, newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(data)

    for sh in ('sheetA.csv', 'sheetB.csv'):
        write_csv(os.path.join(out_dir, sh), ['pair_id', 'image', 'option_1', 'option_2', 'choice', 'notes'], rows)
    write_csv(os.path.join(out_dir, 'pairs.csv'),
              ['pair_id', 'class_A', 'class_B', 'short_A', 'short_B', 'n_A', 'n_B', 'source'], pair_rows)
    write_csv(os.path.join(out_dir, '_key.csv'),
              ['pair_id', 'image', 'true_class', 'true_short', 'option_1', 'option_2', 'source_path'], key_rows)
    with open(os.path.join(out_dir, 'README_ANNOTATOR.txt'), 'w', encoding='utf-8') as f:
        f.write(ANNOTATOR_README)

    total = len(rows)
    print(f'\n[hl] 抽样包已生成：{out_dir}')
    print(f'     类对 {len(pair_rows)} 组，图片 {total} 张（每张二选一）')
    print('     标注表 sheetA.csv / sheetB.csv（两位标注者各一份，独立盲标）')
    print('     ⚠ _key.csv 为答案键，标注期间不要打开')


if __name__ == '__main__':
    main()
