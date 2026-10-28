# -*- coding: utf-8 -*-
"""V3 NoTF实验注册表：配置ID、骨干和解冻时点的唯一来源。

保留两个已命名基线；新增早/晚解冻及ResNet18/34全解冻。
不构造骨干与时点的笛卡尔积。model/arch='v2'是内部模型实现键。
当前六项共同使用NoTF，不在ID中重复写系列/基线前缀。首轮均seed1、100轮。
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
BACKBONE_FEATURE_DIMS = {'resnext50_32x4d': 2048, 'resnet18': 512, 'resnet34': 512}
TRAINING_POLICY = {
    'optimizer': 'adamw_all_parameter_groups', 'backbone_lr': 1e-4,
    'head_lr': 1e-4, 'weight_decay': 1e-2,
    'scheduler': 'cawr_global_epoch', 't_0': 5, 't_mult': 2, 'eta_min': 1e-6,
    'backbone_bn_policy': 'frozen_stats',
    'optimizer_state_on_unfreeze': 'preserve',
}


def _make_entry(backbone, unfreeze_epochs, short, phase):
    epochs = list(unfreeze_epochs)
    return {
        'params': dict(model='v2', backbone=backbone, pretrained=True,
                       regions=[1, 2], transformer_layers=0, use_decouple=True,
                       unfreeze='progressive' if epochs else 'none',
                       unfreeze_epochs=epochs, scheduler='cawr'),
        'arch': 'v2', 'short': short, 'cn': short + '（SE[1,2]）',
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
}
ORDER = list(MODELS)
ALIASES = {}
DEFAULT_MODEL = FULL_UNFREEZE_ID
MODEL_PARAMS = {key: deepcopy(value['params']) for key, value in MODELS.items()}
MODEL_CN = {key: value['cn'] for key, value in MODELS.items()}
MODEL_SHORT = {key: value['short'] for key, value in MODELS.items()}
MODEL_ARCH = {key: value['arch'] for key, value in MODELS.items()}
MODEL_TARGET_EPOCHS = {key: value['target_epochs'] for key, value in MODELS.items()}


def normalize(name: str) -> str:
    return name


def _entry(name: str) -> dict:
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(f'未注册的V3实验ID: {name!r}; 可用ID: {ORDER}') from None


def known_models() -> list:
    return list(ORDER)


def is_known(name: str) -> bool:
    return name in MODELS


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
    text = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def config_for_run(name: str, seed: int, resume=False, smoke=False) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed必须为非负整数')
    epochs = unfreeze_epochs_of(name)
    return {'CONFIG_ID': name, 'SEED': seed, 'SMOKE': bool(smoke),
            'MAX_EPOCHS': 3 if smoke else target_of(name), 'RESUME': bool(resume),
            'AUTO': True, 'BACKBONE': backbone_of(name),
            'UNFREEZE1_EPOCH': epochs[0] if epochs else None,
            'UNFREEZE2_EPOCH': epochs[1] if epochs else None,
            'EXPERIMENT_SIGNATURE': experiment_signature_of(name)}


def validate_config_binding(name: str, config: dict):
    expected = config_for_run(name, config.get('SEED', 1),
                              config.get('RESUME', False), config.get('SMOKE', False))
    for key in ('BACKBONE', 'UNFREEZE1_EPOCH', 'UNFREEZE2_EPOCH', 'EXPERIMENT_SIGNATURE'):
        if key in config and config[key] != expected[key]:
            raise ValueError(f'{key}与注册表冲突：{config[key]!r} != {expected[key]!r}')
    if not config.get('SMOKE', False) and config.get('MAX_EPOCHS', target_of(name)) != target_of(name):
        raise ValueError('正式预算与注册表不一致')


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
