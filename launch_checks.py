"""解析启动选项；检查/列表只读，返回解析结果供入口选择模式。"""
import argparse
import importlib.util
import json

if __package__:
    from . import model_registry as registry
    from .run_paths import PROJECT_ROOT, RUNS_ROOT, load_config, run_directory
else:
    import model_registry as registry
    from run_paths import PROJECT_ROOT, RUNS_ROOT, load_config, run_directory


def handle_cli(kind, queue_tasks=None):
    parser = argparse.ArgumentParser(description=f'V3 {kind}：默认运行；--check只检查，不训练')
    parser.add_argument('--check', action='store_true', help='只读核对配置、路径、依赖及任务，不读图片')
    parser.add_argument('--list', action='store_true', help='列出当前注册的实验')
    if kind == 'queue':
        parser.add_argument('--smoke', action='store_true', help='队列各任务固定3轮冒烟，写入runs/_smoke')
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
        return args
    if not args.check:
        return args
    config = load_config()
    paths = {name: (PROJECT_ROOT / name).exists() for name in ('train', 'val', 'classname.txt')}
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ('torch', 'torchvision', 'numpy', 'pandas', 'sklearn',
                                 'PIL', 'pillow_avif', 'psutil', 'tqdm')}
    tasks = ([(config_id, seed, args.smoke) for config_id, seed in queue_tasks]
             if kind == 'queue' else
             [(config['CONFIG_ID'], config['SEED'], config.get('SMOKE', False))])
    preview = str(run_directory(*tasks[0])) if tasks else None
    print(json.dumps({'check_only': True, 'data_paths_exist': paths, 'dependencies_found': dependencies,
                      'runs_root': str(RUNS_ROOT),
                      'tasks': [registry.config_for_run(cid, seed, smoke=smoke)
                                for cid, seed, smoke in tasks],
                      'active_config_smoke': config.get('SMOKE', False),
                      'selected_train_smoke': tasks[0][2] if tasks else None,
                      'selected_output_preview': preview,
                      'image_read': False, 'weights_downloaded': False, 'training_started': False},
                     ensure_ascii=False, indent=2))
    if not all(paths.values()) or not all(dependencies.values()):
        raise SystemExit(1)
    return args
