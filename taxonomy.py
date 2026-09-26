# -*- coding: utf-8 -*-
"""训练标签体系的唯一事实来源。

原始数据目录仍保留 265 个细类；本模块只定义训练时的逻辑合并、最终索引和可复核指纹。
该模块不依赖 torch/torchvision，训练队列和轻量自检可以直接导入。
"""
from __future__ import annotations

import hashlib
import json
import os


LABEL_SCHEMA_ID = "waste265-merge9-v2"
LABEL_SCHEMA_VERSION = 2
RAW_CLASS_COUNT = 265
EXPECTED_NUM_CLASSES = 252


merged_dict = {
    '可回收物-玻璃制品类': [('57', '玻璃器皿'),
                            ('56', '玻璃瓶'),
                            ('55', '玻璃壶'),
                            ('59', '玻璃制品')],
    '可回收物-金属制品': [('117', '金属罐'), ('118', '金属制品'), ('60', '不锈钢制品')],
    '厨余垃圾-哈密瓜': [('26', '哈密瓜'), ('49', '哈密瓜2')],
    '其他垃圾-抹布毛巾': [('213', '厨房抹布'), ('232', '毛巾')],
    '厨余垃圾-果壳': [('24', '果壳'), ('225', '果壳2')],
    '可回收物-鞋': [('183', '鞋'), ('184', '鞋子')],
    '其他垃圾-陶瓷': [('250', '陶瓷'), ('234', '破碎陶瓷'), ('209', '茶壶碎片')],
    # 2026-09-26：消除数据集中仍然存在的两组语义重叠。
    '厨余垃圾-菜叶': [('6', '菜根菜叶'), ('46', '蔬菜')],
    '可回收物-插头充电线': [('64', '插头电线'), ('69', '充电线')],
}


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


LABEL_SCHEMA_DEFINITION_HASH = hashlib.sha256(
    _canonical_json(merged_dict).encode('utf-8')
).hexdigest()


def generate_label_mappings(merged_groups, class_labels, dataset_class_ids):
    """由原始类别名和目录 ID 生成训练使用的逻辑类别映射。"""
    id_to_main_class = {}
    seen_ids = set()

    for main_class in sorted(merged_groups):
        for item_id, child_name in merged_groups[main_class]:
            item_id = str(item_id)
            if item_id in seen_ids:
                raise ValueError(f'原始类别 ID {item_id} 被重复放入多个合并组')
            if not item_id.isdigit() or int(item_id) >= len(class_labels):
                raise ValueError(f'合并组 {main_class} 含无效类别 ID {item_id}')
            actual_name = class_labels[int(item_id)]
            actual_leaf_name = actual_name.split('-', 1)[-1]
            if actual_leaf_name != child_name:
                raise ValueError(
                    f'合并组 {main_class} 的 ID {item_id} 名称不一致：'
                    f'配置={child_name}，classname.txt={actual_leaf_name}')
            seen_ids.add(item_id)
            id_to_main_class[item_id] = f'公共类-{main_class}'

    for class_id in sorted(int(value) for value in dataset_class_ids):
        class_id_text = str(class_id)
        if class_id_text not in id_to_main_class:
            id_to_main_class[class_id_text] = f'私有类-{class_labels[class_id]}'

    sorted_labels = sorted(set(id_to_main_class.values()))
    renew_class_to_index = {
        main_class: str(index)
        for index, main_class in enumerate(sorted_labels)
    }
    return id_to_main_class, renew_class_to_index


def mapping_fingerprint(id_to_main_class, renew_class_to_index):
    """对最终 ID→逻辑类和逻辑类→输出索引同时取指纹。"""
    payload = {
        'schema_id': LABEL_SCHEMA_ID,
        'schema_version': LABEL_SCHEMA_VERSION,
        'definition_hash': LABEL_SCHEMA_DEFINITION_HASH,
        'id_to_main_class': id_to_main_class,
        'renew_class_to_index': renew_class_to_index,
    }
    return hashlib.sha256(_canonical_json(payload).encode('utf-8')).hexdigest()


def load_current_label_schema(project_root):
    """从项目的 classname.txt 与 train/ 目录构建当前完整映射。"""
    with open(os.path.join(project_root, 'classname.txt'), encoding='utf-8') as file:
        class_labels = file.read().splitlines()
    train_root = os.path.join(project_root, 'train')
    dataset_class_ids = [name for name in os.listdir(train_root) if name.isdigit()]
    mappings = generate_label_mappings(merged_dict, class_labels, dataset_class_ids)
    id_to_main_class, renew_class_to_index = mappings
    if len(class_labels) != RAW_CLASS_COUNT:
        raise ValueError(f'classname.txt 应有 {RAW_CLASS_COUNT} 类，实际 {len(class_labels)} 类')
    if len(id_to_main_class) != RAW_CLASS_COUNT:
        raise ValueError(f'train/ 应覆盖 {RAW_CLASS_COUNT} 个原始类，实际 {len(id_to_main_class)} 类')
    if len(renew_class_to_index) != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f'合并后应有 {EXPECTED_NUM_CLASSES} 类，实际 {len(renew_class_to_index)} 类')
    return id_to_main_class, renew_class_to_index, mapping_fingerprint(*mappings)
