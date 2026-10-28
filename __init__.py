"""V3轻量包入口：标签/模型导入不初始化训练设备或输出。"""
from .taxonomy import (
    EXPECTED_NUM_CLASSES, LABEL_SCHEMA_ID, LABEL_SCHEMA_VERSION,
    LABEL_SCHEMA_DEFINITION_HASH, mapping_fingerprint, merged_dict,
)


def __getattr__(name):
    # 仅显式请求设备或保存工具时加载运行时，纯标签导入保持轻量。
    runtime_exports = {
        'device', 'DynamicMeanPad', 'pth_file_name', 'save_model_after_epoch',
        'extensions_suffix', 'ACC_STOP', 'ERROR_RATE', 'TRAIN_NUM_WORKERS',
        'VAL_NUM_WORKERS', 'metrics_file',
    }
    if name not in runtime_exports:
        raise AttributeError(f'{__name__}没有属性{name!r}')
    from importlib import import_module
    runtime = import_module(".runtime", __name__)
    try:
        return getattr(runtime, name)
    except AttributeError:
        raise AttributeError(f'{__name__}没有属性{name!r}') from None
