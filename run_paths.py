"""V3配置与运行目录；导入不创建目录、不读图片、不初始化设备。"""
from pathlib import Path
import hashlib
import json
import os
from tempfile import NamedTemporaryFile
import time

if __package__:
    from . import model_registry as registry
else:
    import model_registry as registry

V3_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = V3_ROOT.parent
RUNS_ROOT = V3_ROOT / 'runs'
CONFIG_FILE = V3_ROOT / 'experiment_config.json'

# 唯一逐轮指标格式；恢复和队列入口校验，writer按此顺序输出。
METRICS_COLUMNS = (
    'config_id', 'seed', 'epoch', 'sampler_seed',
    'label_schema_version', 'label_mapping_hash', 'num_classes',
    'train_acc', 'train_loss', 'val_acc', 'val_loss',
    'val_macro_f1', 'val_weighted_f1', 'lr', 'gradient_norm',
    'time_spend_s', 'state', 'save_time',
    'val_weighted_precision', 'val_weighted_recall',
)


def load_config(path=None):
    source = Path(path or os.environ.get('V3_CONFIG_FILE', CONFIG_FILE))
    with source.open(encoding='utf-8') as handle:
        config = json.load(handle)
    registry.validate_config_binding(config['CONFIG_ID'], config)
    normalization = config.get('NORMALIZATION')
    if not isinstance(normalization, dict):
        raise ValueError('配置缺少冻结的NORMALIZATION.mean/std')
    mean, std = normalization.get('mean', []), normalization.get('std', [])
    if (len(mean) != 3 or len(std) != 3 or
            not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in mean) or
            not all(isinstance(x, (int, float)) and 0 < x <= 1 for x in std)):
        raise ValueError('NORMALIZATION必须提供三个有效mean和正std')
    return config


def normalization_signature(config):
    text = json.dumps(config['NORMALIZATION'], sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def run_directory(config_id, seed, smoke=False, runs_root=None):
    registry.config_for_run(config_id, seed, smoke=smoke)
    root = Path(runs_root or RUNS_ROOT)
    tag = f'{config_id}_seed{seed}'
    return root / '_smoke' / tag if smoke else root / tag


def result_path(config_id, seed, runs_root=None):
    tag = f'{config_id}_seed{seed}'
    return run_directory(config_id, seed, runs_root=runs_root) / f'results_{tag}.json'


def checkpoint_path(config_id, seed, which='last', runs_root=None):
    if which not in ('last', 'best'):
        raise ValueError('checkpoint类型必须是last或best')
    tag = f'{config_id}_seed{seed}'
    return run_directory(config_id, seed, runs_root=runs_root) / f'checkpoint_{tag}_{which}.pth'


def formal_run_directories(runs_root=None):
    root = Path(runs_root or RUNS_ROOT)
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir()
                  if path.is_dir() and path.name not in ('analyses', '_smoke')
                  and not path.name.startswith('.'))


def configure_process(config=None, *, runs_root=None):
    config = dict(config or load_config())
    directory = run_directory(config['CONFIG_ID'], config['SEED'], config.get('SMOKE', False),
                              runs_root=runs_root)
    if config.get('SMOKE', False):
        directory = directory.with_name(directory.name + '_' + str(time.time_ns()))
    elif config.get('RESUME', False):
        last = checkpoint_path(config['CONFIG_ID'], config['SEED'])
        if not last.is_file():
            raise FileNotFoundError(f'要求续跑但缺少last断点：{last}；不能回退为新训练')
    elif not config.get('RESUME', False) and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'已有运行产物：{directory}；请选择续跑，不能静默覆盖')
    directory.mkdir(parents=True, exist_ok=runs_root is None)
    os.environ['V3_RUN_DIR'] = str(directory)
    os.environ['V3_NORMALIZATION_SIGNATURE'] = normalization_signature(config)
    snapshot = dict(config, RUN_DIRECTORY=str(directory))
    snapshot_path = directory / 'config.json'
    if not config.get('RESUME', False) or not snapshot_path.exists():
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return directory


def register_mysql_run(snapshot_path, *, model, run, database):
    """Register a fresh run and bind its committed database identity to config.json."""
    if __package__:
        from .mysql_store import register_new_run
    else:
        from mysql_store import register_new_run

    path = Path(snapshot_path)
    snapshot = json.loads(path.read_text(encoding='utf-8'))
    if snapshot.get('SMOKE') or snapshot.get('RESUME') or snapshot.get('LONG_SOURCE'):
        raise ValueError('当前仅登记新的普通正式运行')
    if 'MYSQL_RUN_ID' in snapshot or 'MYSQL_DATABASE' in snapshot:
        raise ValueError(f'运行配置已有 MySQL 绑定：{path}')
    if (snapshot['CONFIG_ID'] != model['config_id'] or snapshot['SEED'] != run['seed']
            or snapshot.get('MAX_EPOCHS', model['target_epochs']) != model['target_epochs']):
        raise ValueError(f'运行配置与待登记的模型、种子或预算不一致：{path}')
    run_id = register_new_run(model=model, run=run, database=database)
    snapshot.update(MYSQL_RUN_ID=run_id, MYSQL_DATABASE=database)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                prefix='.config-', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(snapshot, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError as error:
        raise RuntimeError(f'MySQL 已登记 run_id={run_id}，但无法绑定 {path}，请先核对该运行') from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return run_id
