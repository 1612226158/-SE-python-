"""命令行帮助与只读启动检查；不启动训练、读图片或下载权重。"""
import argparse
import importlib.util
import json

try:
    from . import model_registry as registry
    from .run_paths import PROJECT_ROOT, RUNS_ROOT, load_config, run_directory
except ImportError:
    import model_registry as registry
    from run_paths import PROJECT_ROOT, RUNS_ROOT, load_config, run_directory


def handle_cli(kind, queue_tasks=None):
    parser = argparse.ArgumentParser(description=f'V3 {kind}：默认运行；--check只检查，不训练')
    parser.add_argument('--check', action='store_true', help='只读核对配置、路径、依赖及任务，不读图片')
    parser.add_argument('--list', action='store_true', help='列出当前注册的实验')
    args = parser.parse_args()
    if kind == 'queue' and queue_tasks is None:
        raise ValueError('队列入口必须传入run_queue.QUEUE中的显式任务')
    if args.list:
        if kind == 'queue':
            for config_id, seed in queue_tasks:
                registry.config_for_run(config_id, seed)
                print(f'{config_id} seed={seed}', registry.params_of(config_id))
        else:
            for config_id in registry.known_models():
                print(config_id, registry.params_of(config_id))
        return True
    if not args.check:
        return False
    config = load_config()
    paths = {name: (PROJECT_ROOT / name).exists() for name in ('train', 'val', 'classname.txt')}
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ('torch', 'torchvision', 'numpy', 'pandas', 'sklearn',
                                 'transformers', 'PIL', 'cv2', 'pillow_avif', 'psutil', 'tqdm')}
    tasks = ([(config_id, seed, False) for config_id, seed in queue_tasks]
             if kind == 'queue' else
             [(config['CONFIG_ID'], config['SEED'], config.get('SMOKE', False))])
    print(json.dumps({'check_only': True, 'data_paths_exist': paths, 'dependencies_found': dependencies,
                      'runs_root': str(RUNS_ROOT),
                      'tasks': [registry.config_for_run(cid, seed, smoke=smoke)
                                for cid, seed, smoke in tasks],
                      'selected_train_smoke': config.get('SMOKE', False),
                      'selected_output_preview': str(run_directory(config['CONFIG_ID'], config['SEED'], config.get('SMOKE', False))),
                      'image_read': False, 'weights_downloaded': False, 'training_started': False},
                     ensure_ascii=False, indent=2))
    if not all(paths.values()) or not all(dependencies.values()):
        raise SystemExit(1)
    return True
