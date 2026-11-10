# -*- coding: utf-8 -*-
"""V3 NoTF实验注册表：配置ID、骨干和解冻时点的唯一来源。

保留原八项，新增ResNet18/ShuffleNetV2x1.0的SE[1]全解冻配置。
不构造骨干与时点的笛卡尔积。model/arch='v2'是内部模型实现键。
当前十项共同使用NoTF；首轮两个SE[1]配置seed1、100轮，任务以QUEUE为准。
Shuffle已接入，RegNetX400MF仍待适配；增强研究候选仅在QUEUE注释展示。
"""
from copy import deepcopy
import hashlib
import json

FULL_UNFREEZE_ID = 'FullUnfreeze'
PROGRESSIVE_UNFREEZE_ID = 'ProgressiveUnfreeze'
EARLY_UNFREEZE_ID = 'ProgressiveUnfreeze-E15-40'
LATE_UNFREEZE_ID = 'ProgressiveUnfreeze-E45-75'
RESNET18_ID = 'ResNet18-FullUnfreeze'
RESNET34_ID = 'ResNet34-FullUnfreeze'
REGNETX400MF_ID = 'RegNetX400MF-FullUnfreeze'
SHUFFLENETV2_X1_0_ID = 'ShuffleNetV2x1.0-FullUnfreeze'
RESNET18_SE1_ID = 'ResNet18-SE1-FullUnfreeze'
SHUFFLENETV2_X1_0_SE1_ID = 'ShuffleNetV2x1.0-SE1-FullUnfreeze'
BACKBONE_FEATURE_DIMS = {
    'resnext50_32x4d': 2048, 'resnet18': 512, 'resnet34': 512,
    'regnet_x_400mf': 400, 'shufflenet_v2_x1_0': 1024,
}
TRAINING_POLICY = {
    'optimizer': 'adamw_all_parameter_groups', 'backbone_lr': 1e-4,
    'head_lr': 1e-4, 'weight_decay': 1e-2,
    'scheduler': 'cawr_global_epoch', 't_0': 5, 't_mult': 2, 'eta_min': 1e-6,
    'backbone_bn_policy': 'frozen_stats',
    'optimizer_state_on_unfreeze': 'preserve',
}


def _make_entry(backbone, unfreeze_epochs, short, phase, regions=(1, 2)):
    epochs = list(unfreeze_epochs)
    return {
        'params': dict(model='v2', backbone=backbone, pretrained=True,
                       regions=list(regions), transformer_layers=0, use_decouple=True,
                       unfreeze='progressive' if epochs else 'none',
                       unfreeze_epochs=epochs, scheduler='cawr'),
        'arch': 'v2', 'short': short,
        'cn': short + '（SE[' + ','.join(map(str, regions)) + ']）',
        'target_epochs': 100, 'phase': phase, 'protocol_target': 3,
        'status': 'registered',
        'note': '首轮seed1；统一全局lr/正则化/BN规则，准备与短运行通过后执行。',
    }


MODELS = {
    FULL_UNFREEZE_ID: _make_entry('resnext50_32x4d', [], '全程全解冻', 'baseline'),
    PROGRESSIVE_UNFREEZE_ID: _make_entry('resnext50_32x4d', [30, 60], '渐进式30/60', 'baseline'),
    EARLY_UNFREEZE_ID: _make_entry('resnext50_32x4d', [15, 40], '较早解冻15/40', 'unfreeze_timing'),
    LATE_UNFREEZE_ID: _make_entry('resnext50_32x4d', [45, 75], '较晚解冻45/75', 'unfreeze_timing'),
    RESNET18_ID: _make_entry('resnet18', [], 'ResNet18全解冻', 'backbone'),
    RESNET34_ID: _make_entry('resnet34', [], 'ResNet34全解冻', 'backbone'),
    REGNETX400MF_ID: {
        **_make_entry('regnet_x_400mf', [], 'RegNetX400MF全解冻', 'backbone'),
        'status': 'pending_backbone_adapter',
        'note': '参数已登记；stem/trunk特征提取待适配，未加入QUEUE或执行训练。',
    },
    SHUFFLENETV2_X1_0_ID: {
        **_make_entry('shufflenet_v2_x1_0', [], 'ShuffleNetV2x1.0全解冻', 'backbone'),
        'note': '骨干已接入并保留conv5；SE[1,2]对照在QUEUE中注释，未执行训练。',
    },
    RESNET18_SE1_ID: _make_entry('resnet18', [], 'ResNet18单区域全解冻', 'backbone', regions=[1]),
    SHUFFLENETV2_X1_0_SE1_ID: _make_entry(
        'shufflenet_v2_x1_0', [], 'ShuffleNetV2x1.0单区域全解冻', 'backbone', regions=[1]),
}
DEFAULT_MODEL = FULL_UNFREEZE_ID
DEFAULT_SMOKE_EPOCHS = 3


# P07 层级 SE 方案。固定融合项已接入，实际队列由 QUEUE 显式指定。
# 网格为整图的 (行, 列)。子区在直属父区内细分，边界对齐。
# 全部候选共用骨干字段。实际开跑后换骨干时，应保留旧运行的配置身份。
HIERARCHICAL_SE_PLAN = {
    'backbone': 'shufflenet_v2_x1_0',
    'pretrained': True,
    'transformer_layers': 0,
    'unfreeze_epochs': (),
    'seed': 1,
    'target_epochs': 100,
    # 全局与局部平分原 SE[1] 隐藏宽度。局部 MLP 在区域及层间共享。
    'se_reduction': 16,
    'residual_limit': 0.5,
    # 各区域池化成同维向量并共享分类头。层内等权汇总区域概率，再做层间融合。
    # 同一检查点另取父层和局部层的预测，以统计纠错、干扰和置信度变化。
    'variants': {
        'HierSE-Parallel-L2': {
            'level_grids': ((1, 1), (1, 2)),
            'parent_modulation': False,
            'level_weights': (0.6, 0.4),
        },
        'HierSE-Nested-L2': {
            'level_grids': ((1, 1), (1, 2)),
            'parent_modulation': True,
            'level_weights': (0.6, 0.4),
        },
        'HierSE-Parallel-L3': {
            'level_grids': ((1, 1), (1, 2), (2, 2)),
            'parent_modulation': False,
            'level_weights': (0.6, 0.2, 0.2),
        },
        'HierSE-Nested-L3': {
            'level_grids': ((1, 1), (1, 2), (2, 2)),
            'parent_modulation': True,
            'level_weights': (0.6, 0.2, 0.2),
        },
    },
    # 后续冻结固定版分支，用单独划分的数据拟合逐图输出层间 Softmax 比例的融合器。
    'adaptive_fusion': {'reference': 'HierSE-Nested-L2'},
}


for name, variant in HIERARCHICAL_SE_PLAN['variants'].items():
    plan = HIERARCHICAL_SE_PLAN
    mode = '父子嵌套' if variant['parent_modulation'] else '并行叠加'
    title = f"HierSE {len(variant['level_grids'])}级{mode}"
    entry = _make_entry(plan['backbone'], plan['unfreeze_epochs'], title, 'hierarchical_se', regions=())
    entry['cn'] = title
    entry['target_epochs'] = plan['target_epochs']
    entry['params'].update(pretrained=plan['pretrained'], transformer_layers=plan['transformer_layers'],
                           hierarchical_se={
                               **variant,
                               'level_grids': [list(grid) for grid in variant['level_grids']],
                               'level_weights': list(variant['level_weights']),
                               'se_reduction': plan['se_reduction'],
                               'residual_limit': plan['residual_limit'],
                           })
    MODELS[name] = entry

ORDER = list(MODELS)


def normalize(name: str) -> str:
    return name


def _entry(name: str) -> dict:
    if name in MODELS:
        return MODELS[name]
    from run_queue import long_entry
    return long_entry(name)


def known_models() -> list:
    from run_queue import LONG_CONFIGS
    return list(ORDER) + list(LONG_CONFIGS)


def is_known(name: str) -> bool:
    return name in MODELS or name in known_models()


def params_of(name: str) -> dict:
    return deepcopy(_entry(name)['params'])


def cn_of(name: str) -> str:
    return _entry(name)['cn']


def short_of(name: str) -> str:
    return _entry(name)['short']


def arch_of(name: str) -> str:
    return _entry(name)['arch']


def regions_of(name: str):
    return params_of(name)['regions']


def tf_of(name: str) -> int:
    return params_of(name)['transformer_layers']


def dec_of(name: str) -> bool:
    return params_of(name)['use_decouple']


def target_of(name: str) -> int:
    return _entry(name)['target_epochs']


def backbone_of(name: str) -> str:
    return params_of(name)['backbone']


def unfreeze_epochs_of(name: str) -> list:
    return params_of(name)['unfreeze_epochs']


def training_policy_of(name: str) -> dict:
    _entry(name)
    return deepcopy(TRAINING_POLICY)


def experiment_signature_of(name: str) -> str:
    payload = {'params': params_of(name), 'epochs': target_of(name),
               'protocol_target': _entry(name)['protocol_target'],
               'training_policy': training_policy_of(name)}
    source = _entry(name).get('long_source')
    if source:
        payload['long_source'] = source
    text = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def long_source_of(name: str, seed: int):
    """返回Long来源身份；普通实验返回None。"""
    source = _entry(name).get('long_source')
    return dict(source, seed=seed) if source else None


def config_for_run(name: str, seed: int, resume=False, smoke=False, *, smoke_epochs=None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed必须为非负整数')
    epochs = unfreeze_epochs_of(name)
    source = long_source_of(name, seed) if not smoke else None
    max_epochs = target_of(name)
    if smoke:
        max_epochs = DEFAULT_SMOKE_EPOCHS if smoke_epochs is None else smoke_epochs
    config = {'CONFIG_ID': name, 'SEED': seed, 'SMOKE': bool(smoke),
            'MAX_EPOCHS': max_epochs, 'RESUME': bool(resume or source),
            'AUTO': True, 'BACKBONE': backbone_of(name),
            'UNFREEZE1_EPOCH': epochs[0] if epochs else None,
            'UNFREEZE2_EPOCH': epochs[1] if epochs else None,
            'EXPERIMENT_SIGNATURE': experiment_signature_of(name)}
    if source:
        config['LONG_SOURCE'] = source
    return config


def validate_config_binding(name: str, config: dict):
    expected = config_for_run(name, config.get('SEED', 1),
                              config.get('RESUME', False), config.get('SMOKE', False))
    for key in ('BACKBONE', 'UNFREEZE1_EPOCH', 'UNFREEZE2_EPOCH', 'EXPERIMENT_SIGNATURE'):
        if key in config and config[key] != expected[key]:
            raise ValueError(f'{key}与注册表冲突：{config[key]!r} != {expected[key]!r}')
    if config.get('SMOKE', False):
        max_epochs = config.get('MAX_EPOCHS', DEFAULT_SMOKE_EPOCHS)
        if type(max_epochs) is not int or max_epochs < 1:
            raise ValueError('冒烟 MAX_EPOCHS 必须为正整数')
    elif config.get('MAX_EPOCHS', target_of(name)) != target_of(name):
        raise ValueError('正式预算与注册表不一致')
    if 'LONG_SOURCE' in expected:
        if config.get('LONG_SOURCE') != expected['LONG_SOURCE'] or not config.get('RESUME'):
            raise ValueError('Long必须携带登记来源并使用续跑模式')


def validate_registry():
    for name in ORDER:
        p = params_of(name); epochs = p['unfreeze_epochs']
        if p['backbone'] not in BACKBONE_FEATURE_DIMS:
            raise ValueError(f'未知骨干: {p["backbone"]}')
        if p['unfreeze'] == 'progressive':
            if len(epochs) != 2 or not all(type(e) is int for e in epochs) or not 0 < epochs[0] < epochs[1] < target_of(name):
                raise ValueError(f'非法解冻时点: {name}: {epochs}')
        elif epochs:
            raise ValueError(f'全解冻不应设置阶段时点: {name}')


validate_registry()

if __name__ == '__main__':
    print(f'V3注册表：{len(MODELS)}项，默认{DEFAULT_MODEL}，状态registered')
    for config_id in ORDER:
        print(config_id, params_of(config_id), target_of(config_id))
