"""V3配置与运行目录；导入不创建目录、不读图片、不初始化设备。"""
from contextlib import suppress
from pathlib import Path
import hashlib
import json
import os
from pickle import UnpicklingError
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
    _bind_mysql_snapshot(path, snapshot, run_id, database)
    return run_id


def _bind_mysql_snapshot(path, snapshot, run_id, database):
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


def register_mysql_long(config, parent_dir, output_root):
    """Verify an explicit source, then bind a new Long test run without training."""
    if __package__:
        from .mysql_store import TEST_DATABASE, cli_password, register_test_long_run
        from .run_queue import SAMPLING_PROTOCOL_VERSION, TRAINING_PROTOCOL_VERSION
    else:
        from mysql_store import TEST_DATABASE, cli_password, register_test_long_run
        from run_queue import SAMPLING_PROTOCOL_VERSION, TRAINING_PROTOCOL_VERSION

    config_id, seed = config['CONFIG_ID'], config['SEED']
    registry.validate_config_binding(config_id, config)
    source = registry.long_source_of(config_id, seed)
    if config.get('SMOKE') or source is None:
        raise ValueError('当前配置不是已登记的正式 Long 实验')
    if not str(parent_dir).strip() or not str(output_root).strip():
        raise ValueError('Long 登记需要指定来源运行目录和独立输出目录')
    root = Path(output_root).resolve()
    if root.is_relative_to(RUNS_ROOT.resolve()):
        raise ValueError('Long 登记请使用 runs 以外的独立输出目录')
    parent_dir = Path(parent_dir).resolve()
    signature = normalization_signature(config)
    with cli_password():
        parent_config, last, parent_run_id, action = inspect_mysql_run(parent_dir)
        if action != 'done':
            raise ValueError(f'Long 来源 run_id={parent_run_id} 尚未完成')
        if (parent_config['CONFIG_ID'] != source['config_id'] or parent_config['SEED'] != seed
                or last['experiment_signature'] != source['experiment_signature']
                or last['normalization_signature'] != signature):
            raise ValueError('Long 来源的模型、seed 或配置签名不匹配')
        start_epoch = registry.target_of(source['config_id'])
        target_epochs = registry.target_of(config_id)
        params = registry.params_of(config_id)
        if (source['epochs'] != [0, start_epoch - 1] or target_epochs <= start_epoch
                or params != registry.params_of(source['config_id'])
                or registry.training_policy_of(config_id) != registry.training_policy_of(source['config_id'])):
            raise ValueError('Long 必须延长来源预算并沿用模型与训练策略')
        required = ('state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict',
                    'best_epoch', 'best_val_acc', 'train_acc_at_best', 'sampler_seed', 'state')
        missing = [key for key in required if key not in last]
        if missing:
            raise ValueError(f'Long 来源 last 缺少恢复状态：{missing}')
        if (last.get('training_protocol_version') != TRAINING_PROTOCOL_VERSION
                or last.get('sampling_protocol_version') != SAMPLING_PROTOCOL_VERSION):
            raise ValueError('Long 来源 last 的训练或采样协议不兼容')
        metadata = last.get('model_metadata')
        payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        if (not isinstance(metadata, dict)
                or hashlib.sha256(payload.encode('utf-8')).hexdigest() != last.get('model_signature_sha256')):
            raise ValueError('Long 来源 last 的模型结构签名不匹配')
        directory = run_directory(config_id, seed, runs_root=root)
        directory.mkdir(parents=True, exist_ok=False)
        try:
            run_id = register_test_long_run(
                model=dict(config_id=config_id, backbone=params['backbone'], model_params=params,
                           target_epochs=target_epochs),
                parent_run_id=parent_run_id, parent_config_id=source['config_id'], seed=seed,
                start_epoch=start_epoch, normalization_signature=signature, checkpoint=last,
            )
        except ValueError:
            with suppress(OSError):
                directory.rmdir()
            raise
    snapshot = dict(config, RUN_DIRECTORY=str(directory), MYSQL_PARENT_RUN_DIRECTORY=str(parent_dir))
    _bind_mysql_snapshot(directory / 'config.json', snapshot, run_id, TEST_DATABASE)
    return run_id, parent_run_id, start_epoch


def load_mysql_binding(run_dir):
    """Read an ordinary run's original database binding."""
    if __package__:
        from .mysql_store import DATABASES
    else:
        from mysql_store import DATABASES

    directory = Path(run_dir).resolve()
    config = load_config(directory / 'config.json')
    if config.get('SMOKE') or config.get('LONG_SOURCE'):
        raise ValueError('MySQL 断点检查仅接受普通运行')
    run_id = config.get('MYSQL_RUN_ID')
    if (config.get('MYSQL_DATABASE') not in DATABASES
            or type(run_id) is not int or not 1 <= run_id <= 4_294_967_295):
        raise ValueError('config.json 缺少有效的 MySQL 运行绑定')
    bound_directory = config.get('RUN_DIRECTORY')
    if not isinstance(bound_directory, str) or Path(bound_directory).resolve() != directory:
        raise ValueError('当前目录与 config.json 绑定的原运行目录不一致')
    return config, run_id


def _load_mysql_checkpoint(run_dir):
    directory = Path(run_dir).resolve()
    config, run_id = load_mysql_binding(directory)
    config_id, seed = config['CONFIG_ID'], config['SEED']
    budget = registry.target_of(config_id)
    signature = normalization_signature(config)
    last_path = checkpoint_path(config_id, seed, runs_root=directory.parent)
    if last_path.parent != directory:
        raise ValueError(f'运行目录与配置的模型和 seed 不一致：{directory}')
    if not last_path.is_file():
        raise FileNotFoundError(f'缺少 last 断点：{last_path}')

    import torch

    try:
        checkpoint = torch.load(last_path, map_location='cpu')
    except (EOFError, UnpicklingError) as error:
        raise ValueError(f'last 断点无法读取：{last_path}') from error
    if not isinstance(checkpoint, dict):
        raise ValueError(f'last 断点格式无效：{last_path}')
    if __package__:
        from .mysql_store import TEST_DATABASE
    else:
        from mysql_store import TEST_DATABASE
    # 旧 MySQL 断点只有 run_id，当时只能写测试库。
    if (checkpoint.get('mysql_database', TEST_DATABASE), checkpoint.get('mysql_run_id')) != (
            config['MYSQL_DATABASE'], run_id):
        raise ValueError('config.json 与 last 的 MySQL 库名或 run_id 不一致')
    if (checkpoint.get('config_id') != config_id or checkpoint.get('seed') != seed
            or checkpoint.get('experiment_signature') != registry.experiment_signature_of(config_id)
            or checkpoint.get('normalization_signature') != signature):
        raise ValueError('config.json 与 last 的模型、seed 或配置签名不一致')
    epoch = checkpoint.get('epoch')
    if type(epoch) is not int or not 0 <= epoch < budget:
        raise ValueError(f'last 的 epoch 不在训练预算内：{epoch!r}')
    return config, checkpoint, run_id


def inspect_mysql_run(run_dir):
    """Read the bound run and decide whether to resume, finalize, or skip."""
    if __package__:
        from .mysql_store import cli_password, read_run_progress
    else:
        from mysql_store import cli_password, read_run_progress

    config, checkpoint, run_id = _load_mysql_checkpoint(run_dir)
    config_id, seed = config['CONFIG_ID'], config['SEED']
    budget = registry.target_of(config_id)
    signature = normalization_signature(config)
    epoch = checkpoint['epoch']
    with cli_password():
        rows = read_run_progress(run_id, database=config['MYSQL_DATABASE'])
    if not rows:
        raise ValueError(f"{config['MYSQL_DATABASE']} 中找不到 run_id={run_id}")
    run = rows[0]
    if run['is_valid'] != 1 or run['parent_run_id'] is not None:
        raise ValueError(f'run_id={run_id} 不是有效的普通运行')
    if (run['run_id'], run['config_id'], run['seed'], run['target_epochs'],
            run['normalization_signature']) != (run_id, config_id, seed, budget, signature):
        raise ValueError('MySQL 与本地的运行身份、模型、seed、预算或归一化签名不一致')
    epochs = [row['epoch'] for row in rows if row['epoch'] is not None]
    if epochs and epochs[-1] > epoch:
        raise ValueError(f'MySQL 已有 epoch={epochs[-1]}，领先于 last 的 epoch={epoch}')
    if epochs != list(range(epoch + 1)):
        missing = sorted(set(range(epoch + 1)).difference(epochs))
        raise ValueError(f'MySQL 指标未连续覆盖到 last 的 epoch={epoch}，缺失轮次：{missing}')
    if (checkpoint.get('sampler_seed'), checkpoint.get('state')) != (
            rows[-1]['sampler_seed'], rows[-1]['state']):
        raise ValueError('last 与同轮 MySQL 指标的采样种子或解冻状态不一致')
    if run['status'] == 'completed':
        best = max(rows, key=lambda row: row['val_acc'])
        if (epoch + 1 != budget or run['ended_at'] is None or run['result_run_id'] != run_id
                or (run['final_epoch'], run['final_val_acc']) != (epoch, rows[-1]['val_acc'])
                or (run['best_val_epoch'], run['best_val_acc']) != (best['epoch'], best['val_acc'])):
            raise ValueError(f'run_id={run_id} 的完成标记、指标、结果或末轮断点不一致')
        action = 'done'
    else:
        if run['status'] != 'incomplete' or run['result_run_id'] is not None or run['ended_at'] is not None:
            raise ValueError(f'run_id={run_id} 的未完成状态与结果不一致')
        action = 'finalize' if epoch + 1 == budget else 'resume'
    return config, checkpoint, run_id, action


def load_mysql_resume(run_dir):
    """Load a checkpoint once and confirm that a training epoch can follow it."""
    config, checkpoint, run_id, action = inspect_mysql_run(run_dir)
    epoch = checkpoint['epoch']
    if action != 'resume':
        raise ValueError(f'run_id={run_id} 已达到 {epoch + 1} 轮预算，没有下一轮可续跑')
    return config, checkpoint, run_id, epoch + 1


def check_mysql_resume(run_dir):
    """Report the next epoch without starting training."""
    _, _, run_id, next_epoch = load_mysql_resume(run_dir)
    return run_id, next_epoch


def finalize_mysql_run(run_dir):
    """Finalize the bound run using the existing last checkpoint as evidence."""
    if __package__:
        from .mysql_store import cli_password, finalize_run
    else:
        from mysql_store import cli_password, finalize_run

    config, checkpoint, run_id = _load_mysql_checkpoint(run_dir)
    with cli_password():
        finalize_run(
            run_id, database=config['MYSQL_DATABASE'], config_id=config['CONFIG_ID'], seed=config['SEED'],
            target_epochs=registry.target_of(config['CONFIG_ID']),
            normalization_signature=checkpoint['normalization_signature'],
            checkpoint_epoch=checkpoint['epoch'], sampler_seed=checkpoint['sampler_seed'],
            state=checkpoint['state'],
        )
    return run_id
