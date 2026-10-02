# -*- coding: utf-8 -*-
"""
hl_common.py —— human_loop 共用工具（2026-09-12 建）

作用：把"类别体系映射 / 短名解析 / 图片清单 / 结果表输出"这类重复逻辑集中一处，
供 make_kappa_pack.py 与 kappa_report.py 共用。

口径：与训练/评估完全一致——**252 合并类**（merged_dict 的 9 组合并；2026-09-26 起，
     「菜根菜叶+蔬菜→厨余垃圾-菜叶」
     「插头电线+充电线→可回收物-插头充电线」），
     文件夹 id → 合并类名 用 calculate.generate_mappings；图片扩展名与训练一致（含 avif）。
纪律：只读 train/val 数据集；产物只写 human_loop/exports/ 下；不改数据集、不碰训练。
"""
import os
import sys
import csv
import json
import hashlib


def configure_utf8_console():
    """让命令行工具在 Windows 控制台安全输出中文和符号。

    pythonw 或被 GUI 启动时 stdout/stderr 可能为 None；不支持 reconfigure 的
    环境保持原样。errors='replace' 可避免日志字符反过来中断标注或自检流程。
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is None or not hasattr(stream, 'reconfigure'):
            continue
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, OSError, ValueError):
            pass

_HERE = os.path.dirname(os.path.abspath(__file__))      # src_v2\human_loop
_SRC = os.path.dirname(_HERE)                           # src_v2
_ROOT = os.path.dirname(_SRC)                           # 项目根
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from taxonomy import (EXPECTED_NUM_CLASSES, LABEL_SCHEMA_ID, LABEL_SCHEMA_VERSION,
                      LABEL_SCHEMA_DEFINITION_HASH, load_current_label_schema,
                      merged_dict)

# 与训练 SafeImageFolder / WRS 白名单一致的扩展名。
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif', '.avif')
VAL_ROOT = os.path.join(_ROOT, 'val')
TRAIN_ROOT = os.path.join(_ROOT, 'train')
EXPORT_ROOT = os.path.join(_HERE, 'exports')

HUMAN_LOOP_PROTOCOL_ID = 'waste252-kappa-test-retest'
HUMAN_LOOP_PROTOCOL_VERSION = 3
STUDY_DESIGN = 'intra-annotator-test-retest'
ANNOTATION_UI_MODE = 'public_merged_members_visible'
PUBLIC_MANIFEST_FILE = 'public_manifest.json'
SHEET_IMMUTABLE_FIELDS = ('pair_id', 'image', 'option_1', 'option_2')
PAIR_FIELDS = ('pair_id', 'class_A', 'class_B', 'short_A', 'short_B',
               'n_A', 'n_B', 'source')

# 252 类标签协议下的默认待核验类对。P03/P04 使用仍为独立输出类的替代类对；
# 任何旧 254 类抽样包都不得继续生成论文结果。
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

MERGED_MEMBER_HINTS = {
    '菜根菜叶': '厨余垃圾-菜叶',
    '蔬菜': '厨余垃圾-菜叶',
    '插头电线': '可回收物-插头充电线',
    '充电线': '可回收物-插头充电线',
}


def public_class_member_hint(class_name):
    """返回公共合并类包含的原始子类名称；私有类或歧义短名返回空字符串。

    标注表使用短名（如“玻璃制品类”），协议内部使用完整名
    （如“公共类-可回收物-玻璃制品类”），两种形式均支持。提示只描述
    标签定义，不读取图片答案，因此可安全显示在盲标界面中。
    """
    matches = []
    for parent_name, members in merged_dict.items():
        full_name = f'公共类-{parent_name}'
        if class_name in (full_name, short_name(full_name)):
            matches.append('、'.join(child_name for _class_id, child_name in members))
    return matches[0] if len(matches) == 1 else ''


def merged_class_index():
    """返回 (fid2cls, cls2fids, renew)：文件夹 id→合并类名、合并类名→[文件夹id]、类名→索引。"""
    id2main, renew, _mapping_hash = load_current_label_schema(_ROOT)
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
    if len(renew) != EXPECTED_NUM_CLASSES:
        raise RuntimeError(
            f'盲标标签映射必须是 {LABEL_SCHEMA_ID} v{LABEL_SCHEMA_VERSION} / '
            f'{EXPECTED_NUM_CLASSES} 类，当前得到 {len(renew)} 类')
    return fid2cls, cls2fids, renew


def current_protocol_identity():
    """返回盲标包必须绑定的研究设计与当前完整252类映射身份。"""
    _id2main, renew, mapping_hash = load_current_label_schema(_ROOT)
    if len(renew) != EXPECTED_NUM_CLASSES:
        raise RuntimeError(f'当前标签映射不是 {EXPECTED_NUM_CLASSES} 类')
    return {
        'human_loop_protocol_id': HUMAN_LOOP_PROTOCOL_ID,
        'human_loop_protocol_version': HUMAN_LOOP_PROTOCOL_VERSION,
        'study_design': STUDY_DESIGN,
        'annotation_ui_mode': ANNOTATION_UI_MODE,
        'label_schema_id': LABEL_SCHEMA_ID,
        'label_schema_version': LABEL_SCHEMA_VERSION,
        'label_schema_definition_hash': LABEL_SCHEMA_DEFINITION_HASH,
        'label_mapping_hash': mapping_hash,
        'num_classes': EXPECTED_NUM_CLASSES,
    }


def validate_protocol(protocol, cls2fids=None):
    """校验协议身份与类对语义；失败抛 ValueError，供 GUI/CLI 共用。"""
    if not isinstance(protocol, dict):
        raise ValueError('protocol.json 顶层必须是对象')
    expected = current_protocol_identity()
    mismatches = {key: (protocol.get(key), value) for key, value in expected.items()
                  if protocol.get(key) != value}
    if mismatches:
        raise ValueError(f'标签/盲标协议身份不一致：{mismatches}')
    if protocol.get('rounds') != ['A', 'B']:
        raise ValueError('protocol.json rounds 必须严格为 ["A", "B"]')
    interval = protocol.get('minimum_round_interval_hours')
    if not isinstance(interval, int) or isinstance(interval, bool) or interval < 12:
        raise ValueError('minimum_round_interval_hours 必须是至少12的整数')
    if protocol.get('source') not in ('val', 'train'):
        raise ValueError('source 必须为 val 或 train')
    n_per_class = protocol.get('n_per_class')
    if not isinstance(n_per_class, int) or isinstance(n_per_class, bool) or n_per_class <= 0:
        raise ValueError('n_per_class 必须是正整数')
    if not isinstance(protocol.get('seed'), int) or isinstance(protocol.get('seed'), bool):
        raise ValueError('seed 必须是整数')
    if protocol.get('global_source_paths_unique') is not True:
        raise ValueError('global_source_paths_unique 必须为 true')
    order_mode = protocol.get('option_order_mode')
    balance = protocol.get('pair_order_balance')
    if order_mode == 'pair_fixed_balanced':
        if (not isinstance(balance, dict) or set(balance) != {'A_left', 'B_left'} or
                any(not isinstance(balance.get(k), int) or isinstance(balance.get(k), bool) or
                    balance.get(k) < 0 for k in ('A_left', 'B_left'))):
            raise ValueError('pair_order_balance 必须精确声明非负整数 A_left/B_left')
    elif order_mode == 'per_image_random':
        if balance is not None:
            raise ValueError('逐图随机模式的 pair_order_balance 必须为 null')
    else:
        raise ValueError('option_order_mode 必须为 pair_fixed_balanced 或 per_image_random')
    if protocol.get('public_manifest_file') != PUBLIC_MANIFEST_FILE:
        raise ValueError(f'public_manifest_file 必须为 {PUBLIC_MANIFEST_FILE}')
    manifest_hash = protocol.get('public_manifest_sha256')
    if (not isinstance(manifest_hash, str) or len(manifest_hash) != 64 or
            any(ch not in '0123456789abcdef' for ch in manifest_hash.lower())):
        raise ValueError('public_manifest_sha256 必须是64位SHA256')
    pairs = protocol.get('pairs')
    if not isinstance(pairs, list) or not pairs:
        raise ValueError('protocol.json 缺少非空 pairs')
    if cls2fids is None:
        _fid2cls, cls2fids, _renew = merged_class_index()
    seen = set()
    for index, row in enumerate(pairs, 1):
        if not isinstance(row, dict):
            raise ValueError(f'协议类对第 {index} 项不是对象')
        pid = row.get('pair_id')
        cls_a, cls_b = row.get('class_A'), row.get('class_B')
        if pid != f'P{index:02d}' or pid in seen:
            raise ValueError(f'协议类对编号无效或重复：{pid}')
        if cls_a not in cls2fids or cls_b not in cls2fids or cls_a == cls_b:
            raise ValueError(f'{pid} 含不存在、已合并或相同的输出类：{cls_a} / {cls_b}')
        if row.get('short_A') != short_name(cls_a) or row.get('short_B') != short_name(cls_b):
            raise ValueError(f'{pid} 的短名与完整输出类不一致')
        if row.get('source') != protocol['source']:
            raise ValueError(f'{pid} 的 source 与协议不一致')
        for field in ('n_A', 'n_B'):
            value = row.get(field)
            if (not isinstance(value, int) or isinstance(value, bool) or
                    value != n_per_class):
                raise ValueError(f'{pid} 的 {field} 必须等于 n_per_class')
        seen.add(pid)
    if order_mode == 'pair_fixed_balanced':
        if balance['A_left'] + balance['B_left'] != len(pairs):
            raise ValueError('pair_order_balance 总数与 pairs 数量不一致')
        if abs(balance['A_left'] - balance['B_left']) > 1:
            raise ValueError('pair_order_balance 未实现整体平衡')
    return expected


def sha256_file(path, chunk_size=1024 * 1024):
    """流式计算文件SHA256（小写十六进制）。"""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_rows_hash(rows, fields):
    """仅对指定不可变字段做稳定序列化后计算SHA256。"""
    payload = [{field: str(row.get(field, '')) for field in fields} for row in rows]
    raw = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _read_csv(path):
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def validate_public_pack(pack, protocol=None):
    """不读取答案键，校验协议、公开清单、类对、标注表和图片内容。

    GUI与报告共用；任何旧包嫁接、CSV不可变字段篡改、图片替换或缺失都会失败。
    返回已验证的 protocol、pairs、sheet 行与 manifest。
    """
    pack = os.path.abspath(pack)
    if protocol is None:
        with open(os.path.join(pack, 'protocol.json'), encoding='utf-8') as f:
            protocol = json.load(f)
    validate_protocol(protocol)

    manifest_path = os.path.join(pack, PUBLIC_MANIFEST_FILE)
    if sha256_file(manifest_path) != protocol['public_manifest_sha256'].lower():
        raise ValueError('公开清单SHA256与protocol.json不一致')
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict) or manifest.get('manifest_version') != 1:
        raise ValueError('公开清单版本无效')
    expected_identity = current_protocol_identity()
    if manifest.get('protocol_identity') != expected_identity:
        raise ValueError('公开清单的协议/标签映射身份不一致')

    pairs_path = os.path.join(pack, 'pairs.csv')
    sheet_a_path = os.path.join(pack, 'sheetA.csv')
    sheet_b_path = os.path.join(pack, 'sheetB.csv')
    pair_rows = _read_csv(pairs_path)
    sheet_a = _read_csv(sheet_a_path)
    sheet_b = _read_csv(sheet_b_path)
    if not pair_rows or not sheet_a or not sheet_b:
        raise ValueError('pairs.csv、sheetA.csv、sheetB.csv 均必须非空')
    if sha256_file(pairs_path) != manifest.get('pairs_csv_sha256'):
        raise ValueError('pairs.csv 内容哈希与公开清单不一致')
    immutable_a = canonical_rows_hash(sheet_a, SHEET_IMMUTABLE_FIELDS)
    immutable_b = canonical_rows_hash(sheet_b, SHEET_IMMUTABLE_FIELDS)
    if immutable_a != immutable_b or immutable_a != manifest.get('sheet_immutable_sha256'):
        raise ValueError('两轮标注表的不可变字段或公开清单不一致')

    proto_pairs = protocol['pairs']
    if len(pair_rows) != len(proto_pairs):
        raise ValueError('pairs.csv 行数与protocol.json不一致')
    pair_map = {}
    for csv_row, proto_row in zip(pair_rows, proto_pairs):
        for field in PAIR_FIELDS:
            if str(csv_row.get(field, '')) != str(proto_row.get(field, '')):
                raise ValueError(f'pairs.csv 与protocol.json不一致：{field}')
        pair_map[csv_row['pair_id']] = csv_row

    manifest_images = manifest.get('images')
    if not isinstance(manifest_images, list) or not manifest_images:
        raise ValueError('公开清单缺少非空 images')
    image_hashes = {}
    for entry in manifest_images:
        if not isinstance(entry, dict):
            raise ValueError('公开清单 images 项必须是对象')
        rel, digest = entry.get('image'), entry.get('sha256')
        if not rel or rel in image_hashes or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f'公开清单图片项无效或重复：{rel!r}')
        image_hashes[rel] = digest.lower()
    if manifest.get('image_count') != len(image_hashes):
        raise ValueError('公开清单 image_count 不一致')
    sheet_images = [row.get('image', '') for row in sheet_a]
    if len(sheet_images) != len(set(sheet_images)) or set(sheet_images) != set(image_hashes):
        raise ValueError('标注表图片集合与公开清单不一致')

    disk_images = set()
    image_root = os.path.abspath(os.path.join(pack, 'images'))
    for root, _dirs, files in os.walk(image_root):
        for name in files:
            disk_images.add(os.path.relpath(os.path.join(root, name), image_root).replace(os.sep, '/'))
    if disk_images != set(image_hashes):
        raise ValueError('磁盘图片集合与公开清单不一致')
    for rel, expected_hash in image_hashes.items():
        path = os.path.abspath(os.path.join(image_root, rel.replace('/', os.sep)))
        if os.path.commonpath([image_root, path]) != image_root or sha256_file(path) != expected_hash:
            raise ValueError(f'图片内容哈希不一致或路径越界：{rel}')

    orientation = {'A_left': 0, 'B_left': 0}
    by_pair = {}
    for row in sheet_a:
        pid = row.get('pair_id')
        pair = pair_map.get(pid)
        if pair is None or set((row.get('option_1'), row.get('option_2'))) != {
                pair['short_A'], pair['short_B']}:
            raise ValueError(f'标注表含未知类对或候选语义错误：{pid}')
        by_pair.setdefault(pid, set()).add((row.get('option_1'), row.get('option_2')))
    if protocol['option_order_mode'] == 'pair_fixed_balanced':
        for pid, orders in by_pair.items():
            if len(orders) != 1:
                raise ValueError(f'{pid} 未保持类对内固定候选顺序')
            first = next(iter(orders))[0]
            orientation['A_left' if first == pair_map[pid]['short_A'] else 'B_left'] += 1
        if orientation != protocol['pair_order_balance']:
            raise ValueError(f'实际候选方向与pair_order_balance不一致：{orientation}')
    return {'protocol': protocol, 'pairs': pair_rows, 'sheetA': sheet_a,
            'sheetB': sheet_b, 'manifest': manifest, 'image_hashes': image_hashes}


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
        if token in MERGED_MEMBER_HINTS:
            raise SystemExit(
                f'[hl] 原始类别「{token}」已合并为「{MERGED_MEMBER_HINTS[token]}」，'
                '不能再作为独立盲标候选')
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
        if cls_a == cls_b:
            raise SystemExit(f'[hl] 类对「{tk}」在252类标签映射中属于同一输出类，不能做二选一盲标')
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
