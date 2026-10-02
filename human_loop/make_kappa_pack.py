# -*- coding: utf-8 -*-
"""
make_kappa_pack.py —— 生成"标注一致性检验"抽样包（2026-09-12 建，human_loop）

用途：从252类标签协议下仍相互独立的易混淆类对中平衡抽样、打乱文件名，
      生成**盲标**用的图片包与两张空白标注表；
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
  protocol.json                标签协议、定义哈希、随机种子和类对清单
  public_manifest.json         不含答案的公开完整性清单（图片/CSV哈希）
  README_ANNOTATOR.txt         标注者须知

纪律：只读 val（或 --source train）；不改数据集；产物仅写 exports/。
"""
import os
import sys
import csv
import json
import random
import shutil
import argparse
from datetime import datetime

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hl_common import (VALID_EXTS, VAL_ROOT, TRAIN_ROOT, EXPORT_ROOT, DEFAULT_PAIRS,
                       merged_class_index, short_name, parse_pairs, class_images,
                       current_protocol_identity, validate_protocol, sha256_file,
                       canonical_rows_hash, PUBLIC_MANIFEST_FILE,
                       SHEET_IMMUTABLE_FIELDS)

ANNOTATOR_README = """【标注须知】请严格按以下规则操作，不要打开 _key.csv（那是答案，打开会让本批数据作废）

1. 打开 sheetA.csv（或 sheetB.csv），只填 choice 一列：
   - 看 images/<pair_id>/<image> 这张图；
   - 在两个候选类 option_1 / option_2 中二选一，choice 填 1 或 2（不要填类名）；
   - 若实在无法判断，choice 填 0，并在 notes 里写一句原因。
1a. 图形界面会对公共合并类显示其原始子类参考；该提示来自固定标签定义，A/B两轮必须使用同一界面版本。
1b. 【重要】**同一个类对（pair_id）内，左右两个候选类的顺序是固定的**（例如 P01 左永远是"锅"、右永远是"电饭煲"），
    不会逐张颠倒；不同类对之间的左右顺序是随机分配的（用于抵消位置偏好）。所以请**按图片内容判断，不要按位置猜**。
    若在界面上填错：回到该题重新点另一个选项即可覆盖（界面会显示"↻ 已改答"），或 Delete 清除本题（Ctrl+Z 撤销）。
2. 只能依据图片内容判断；可以放大看细节，但不要搜索、不要与人讨论。本文研究设计为同一标注者的两轮独立盲标：
   先完成轮次 A，间隔至少 12 小时后再完成轮次 B；第二轮前不得查看 sheetA.csv。
2b. 两个候选类都不属于、或图片本身有问题时：按 X（或点「图外·异常」按钮）写入标记 [图外]，
    该标记不改 choice；报告会将图外样本单独计数并排除出κ与准确率，再按一次 X 可取消。
3. 每张图都必须给出选择（允许 0，但请尽量少用）。
4. 同一 pair_id 内图片的排列顺序已打乱（与真实类别无关），不要试图按图片先后顺序猜测规律。
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
    ap.add_argument('--exclude-pack', action='append', default=[],
                    help='排除既有包public_manifest.json中的图片内容哈希；可重复指定')
    ap.add_argument('--prefer-unseen-from-pack', action='append', default=[],
                    help='优先避开既有包图片；某类数量不足时仅回用达到样本量所需的最少图片')
    ap.add_argument('--per-image-order', action='store_true',
                    help='旧行为：逐张随机候选顺序（不推荐——标注者会看错位置；默认改为“类对内固定、类对间随机”）')
    a = ap.parse_args()

    if a.n_per_class <= 0:
        raise SystemExit('[hl] --n-per-class 必须是正整数')

    root = VAL_ROOT if a.source == 'val' else TRAIN_ROOT
    rng = random.Random(a.seed)
    _fid2cls, cls2fids, _renew = merged_class_index()
    pairs = parse_pairs(a.pairs, cls2fids)

    def load_public_hashes(pack_paths, purpose):
        loaded_hashes, pack_names = set(), []
        for pack_path in pack_paths:
            pack_path = os.path.abspath(pack_path)
            manifest_path = os.path.join(pack_path, PUBLIC_MANIFEST_FILE)
            try:
                with open(manifest_path, encoding='utf-8') as f:
                    source_manifest = json.load(f)
                hashes = {item.get('sha256') for item in source_manifest.get('images', [])
                          if isinstance(item, dict) and isinstance(item.get('sha256'), str)}
            except (OSError, ValueError, TypeError) as e:
                raise SystemExit(f'[hl] 无法读取{purpose}包公开清单：{manifest_path}（{e}）')
            if not hashes:
                raise SystemExit(f'[hl] {purpose}包公开清单没有有效图片哈希：{manifest_path}')
            loaded_hashes.update(hashes)
            pack_names.append(os.path.basename(pack_path))
        return loaded_hashes, pack_names

    excluded_hashes, excluded_pack_names = load_public_hashes(a.exclude_pack, '排除')
    preferred_unseen_hashes, preferred_pack_names = load_public_hashes(
        a.prefer_unseen_from_pack, '优先避开')
    source_hash_cache = {}

    def source_hash(path):
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized not in source_hash_cache:
            source_hash_cache[normalized] = sha256_file(path)
        return source_hash_cache[normalized]

    def source_allowed(path):
        return source_hash(path) not in excluded_hashes if excluded_hashes else True

    def pick_prefer_unseen(pool, count):
        if not preferred_unseen_hashes:
            return rng.sample(pool, count)
        unseen = [path for path in pool if source_hash(path) not in preferred_unseen_hashes]
        seen = [path for path in pool if source_hash(path) in preferred_unseen_hashes]
        take_unseen = min(count, len(unseen))
        picked = rng.sample(unseen, take_unseen)
        if take_unseen < count:
            picked.extend(rng.sample(seen, count - take_unseen))
        return picked

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = a.out or os.path.join(EXPORT_ROOT, f'{ts}_kappa_pack')
    output_created_by_run = not os.path.exists(out_dir)
    if os.path.exists(out_dir):
        if not os.path.isdir(out_dir):
            raise SystemExit(f'[hl] 输出路径已存在且不是目录：{out_dir}')
        if os.listdir(out_dir):
            raise SystemExit(f'[hl] 输出目录非空，拒绝覆盖或混入旧抽样包：{out_dir}')
    img_root = os.path.join(out_dir, 'images')
    os.makedirs(img_root, exist_ok=True)

    rows, key_rows, pair_rows = [], [], []
    used_source_paths = set()
    pair_ids = [pid for pid, _cls_a, _cls_b in pairs]
    order_rng = random.Random(f'{a.seed}-pair-order')
    order_rng.shuffle(pair_ids)
    a_left_pair_ids = set(pair_ids[:(len(pair_ids) + 1) // 2])
    for pid, cls_a, cls_b in pairs:
        pool_a = [path for path in class_images(cls_a, cls2fids, root)
                  if os.path.normcase(os.path.abspath(path)) not in used_source_paths
                  and source_allowed(path)]
        pool_b = [path for path in class_images(cls_b, cls2fids, root)
                  if os.path.normcase(os.path.abspath(path)) not in used_source_paths
                  and source_allowed(path)]
        if len(pool_a) < a.n_per_class or len(pool_b) < a.n_per_class:
            if output_created_by_run and os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            raise SystemExit(
                f'[hl] {pid} {short_name(cls_a)}/{short_name(cls_b)} 在整包去重后仅有 '
                f'{len(pool_a)}/{len(pool_b)} 张，不足每类 {a.n_per_class} 张；'
                '为保持预注册样本量，拒绝静默缩减')
        k = a.n_per_class
        picked = ([(p, cls_a) for p in pick_prefer_unseen(pool_a, k)] +
                  [(p, cls_b) for p in pick_prefer_unseen(pool_b, k)])
        rng.shuffle(picked)
        used_source_paths.update(os.path.normcase(os.path.abspath(path)) for path, _cls in picked)
        os.makedirs(os.path.join(img_root, pid), exist_ok=True)
        # 候选顺序策略（2026-09-12 用户反馈后修改）：
        #   默认：**类对内固定**左右顺序；类对之间按种子随机后做整体平衡
        #         （默认8组恰好4组A左、4组B左），降低位置偏差。
        #   --per-image-order：旧行为，逐张随机（不推荐）。
        pair_opts = [cls_a, cls_b] if pid in a_left_pair_ids else [cls_b, cls_a]
        for i, (src, cls) in enumerate(picked, 1):
            ext = os.path.splitext(src)[1].lower() or '.jpg'
            name = f'{pid}_{i:03d}{ext}'
            copied = os.path.join(img_root, pid, name)
            shutil.copy2(src, copied)
            source_sha256 = sha256_file(src)
            image_sha256 = sha256_file(copied)
            if image_sha256 != source_sha256:
                raise SystemExit(f'[hl] 图片复制后哈希不一致：{src}')
            if a.per_image_order:
                opts = [cls_a, cls_b]
                rng.shuffle(opts)
                o1, o2 = short_name(opts[0]), short_name(opts[1])
            else:
                o1, o2 = short_name(pair_opts[0]), short_name(pair_opts[1])
            rows.append(dict(pair_id=pid, image=f'{pid}/{name}', option_1=o1, option_2=o2,
                             choice='', notes=''))
            key_rows.append(dict(pair_id=pid, image=f'{pid}/{name}', true_class=cls,
                                 true_short=short_name(cls), option_1=o1, option_2=o2,
                                 source_path=os.path.abspath(src), image_sha256=image_sha256,
                                 source_sha256=source_sha256))
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
              ['pair_id', 'image', 'true_class', 'true_short', 'option_1', 'option_2',
               'source_path', 'image_sha256', 'source_sha256'], key_rows)
    with open(os.path.join(out_dir, 'README_ANNOTATOR.txt'), 'w', encoding='utf-8') as f:
        f.write(ANNOTATOR_README)
    manifest = {
        'manifest_version': 1,
        'protocol_identity': current_protocol_identity(),
        'pairs_csv_sha256': sha256_file(os.path.join(out_dir, 'pairs.csv')),
        'sheet_immutable_sha256': canonical_rows_hash(rows, SHEET_IMMUTABLE_FIELDS),
        'image_count': len(rows),
        'images': [{'image': r['image'], 'sha256': r['image_sha256']} for r in key_rows],
    }
    manifest_path = os.path.join(out_dir, PUBLIC_MANIFEST_FILE)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    protocol = {
        **current_protocol_identity(),
        'rounds': ['A', 'B'],
        'minimum_round_interval_hours': 12,
        'seed': a.seed,
        'source': a.source,
        'n_per_class': a.n_per_class,
        'global_source_paths_unique': True,
        'excluded_public_packs': excluded_pack_names,
        'excluded_image_sha256_count': len(excluded_hashes),
        'prefer_unseen_public_packs': preferred_pack_names,
        'previously_seen_hash_count': len(preferred_unseen_hashes),
        'reused_previously_seen_images': sum(
            1 for row in key_rows if row['image_sha256'] in preferred_unseen_hashes),
        'option_order_mode': 'per_image_random' if a.per_image_order else 'pair_fixed_balanced',
        'pair_order_balance': None if a.per_image_order else {
            'A_left': len(a_left_pair_ids),
            'B_left': len(pair_ids) - len(a_left_pair_ids)},
        'public_manifest_file': PUBLIC_MANIFEST_FILE,
        'public_manifest_sha256': sha256_file(manifest_path),
        'pairs': pair_rows,
    }
    validate_protocol(protocol, cls2fids)
    with open(os.path.join(out_dir, 'protocol.json'), 'w', encoding='utf-8') as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)

    total = len(rows)
    print(f'\n[hl] 抽样包已生成：{out_dir}')
    print(f'     类对 {len(pair_rows)} 组，图片 {total} 张（每张二选一）')
    print('     标注表 sheetA.csv / sheetB.csv（同一标注者的两次独立盲标轮次）')
    print('     ⚠ _key.csv 为答案键，标注期间不要打开')


if __name__ == '__main__':
    main()
