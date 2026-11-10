"""解析启动选项；检查/列表只读，返回解析结果供入口选择模式。"""
import argparse
import importlib.util
import json
from pathlib import Path

if __package__:
    from . import model_registry as registry
    from .run_paths import PROJECT_ROOT, RUNS_ROOT, load_config, run_directory
else:
    import model_registry as registry
    from run_paths import PROJECT_ROOT, RUNS_ROOT, load_config, run_directory


def mysql_verification_input():
    config = load_config()
    config_id = config['CONFIG_ID']
    if config.get('SMOKE') or config.get('LONG_SOURCE'):
        raise ValueError('MySQL 首轮联机验证请选用普通正式训练配置')
    params = registry.params_of(config_id)
    return {
        'model': {
            'config_id': config_id,
            'backbone': params['backbone'],
            'model_params': params,
            'target_epochs': registry.target_of(config_id),
        },
        'seed': config['SEED'],
        'training_params': {
            'training_policy': registry.training_policy_of(config_id),
            'normalization': config['NORMALIZATION'],
            'purpose': '数据库读写验证',
        },
    }


def handle_cli(kind, queue_tasks=None):
    parser = argparse.ArgumentParser(description=f'V3 {kind}：默认运行；--check只检查，不训练')
    parser.add_argument('--check', action='store_true', help='只读核对配置、路径、依赖及任务，不读图片')
    parser.add_argument('--list', action='store_true', help='列出当前注册的实验')
    if kind == 'train':
        parser.add_argument('--verify-mysql', action='store_true',
                            help='用当前注册配置验证测试库读写，不启动训练')
        parser.add_argument('--register-mysql', metavar='输出根目录',
                            help='准备新的普通训练，在测试库登记并绑定 config.json 后退出')
        parser.add_argument('--register-mysql-long', nargs=2, metavar=('来源运行目录', '输出根目录'),
                            help='核对已完成来源，为当前 Long 配置登记测试库运行后退出')
        parser.add_argument('--mysql-first-epoch', metavar='输出根目录',
                            help='登记新运行，训练并向测试库提交 epoch 0 后退出')
        parser.add_argument('--train-mysql', metavar='输出根目录',
                            help='按 V3_MYSQL_DATABASE 登记并训练新的普通运行，默认测试库')
        parser.add_argument('--check-mysql-resume', metavar='运行目录',
                            help='只读核对 last、运行配置和绑定的 MySQL 库')
        parser.add_argument('--resume-mysql-one-epoch', metavar='运行目录',
                            help='从已绑定的普通运行断点续跑一轮后退出')
        parser.add_argument('--resume-mysql', metavar='运行目录',
                            help='沿用普通运行的数据库绑定及断点，续跑到预算')
        parser.add_argument('--finalize-mysql-run', metavar='运行目录',
                            help='根据 last 和完整指标在绑定库补做收尾，不启动训练')
    if kind == 'queue':
        parser.add_argument('--smoke', action='store_true', help='队列各任务冒烟，写入runs/_smoke')
        parser.add_argument('--mysql', dest='mysql_output', metavar='输出根目录',
                            help='指定 MySQL 队列输出根目录；无参数时使用 runs 和正式库')
    parser.add_argument('--smoke-epochs', type=int, metavar='N',
                        help='指定冒烟轮数，默认3轮。队列需加 --smoke，单独训练需配置 SMOKE=true')
    args = parser.parse_args()
    if kind == 'queue' and args.mysql_output is not None:
        if args.smoke or args.smoke_epochs is not None:
            parser.error('MySQL 队列与冒烟模式互斥')
        if not args.mysql_output.strip():
            parser.error('MySQL 队列需要指定独立输出目录')
        args.mysql_output = Path(args.mysql_output).resolve()
        if args.mysql_output != RUNS_ROOT and args.mysql_output.is_relative_to(RUNS_ROOT):
            parser.error('MySQL 队列请使用 runs 本身或 runs 以外的独立输出目录')
    elif kind == 'queue' and not args.smoke:
        args.mysql_output = RUNS_ROOT
    if args.smoke_epochs is not None:
        if args.smoke_epochs < 1:
            parser.error('--smoke-epochs 必须为正整数')
        if kind == 'queue' and not args.smoke:
            parser.error('--smoke-epochs 需要与 --smoke 一起使用')
    args.smoke_config = None
    if kind == 'train':
        mysql_modes = [args.verify_mysql, args.register_mysql is not None,
                       args.mysql_first_epoch is not None, args.check_mysql_resume is not None,
                       args.resume_mysql_one_epoch is not None, args.finalize_mysql_run is not None,
                       args.train_mysql is not None, args.resume_mysql is not None,
                       args.register_mysql_long is not None]
        if sum(mysql_modes) > 1 or (any(mysql_modes) and (
                args.check or args.list or args.smoke_epochs is not None)):
            parser.error('MySQL 模式请单独使用')
        if args.smoke_epochs is not None:
            args.smoke_config = load_config()
            if not args.smoke_config.get('SMOKE', False):
                parser.error('--smoke-epochs 需要配置 SMOKE=true')
            args.smoke_config['MAX_EPOCHS'] = args.smoke_epochs
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
    config = args.smoke_config or load_config()
    paths = {name: (PROJECT_ROOT / name).exists() for name in ('train', 'val', 'classname.txt')}
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ('torch', 'torchvision', 'numpy', 'pandas', 'sklearn',
                                 'PIL', 'pillow_avif', 'psutil', 'tqdm')}
    tasks = ([(config_id, seed, args.smoke) for config_id, seed in queue_tasks]
             if kind == 'queue' else
             [(config['CONFIG_ID'], config['SEED'], config.get('SMOKE', False))])
    output_root = args.mysql_output if kind == 'queue' and args.mysql_output is not None else RUNS_ROOT
    preview = str(run_directory(*tasks[0], runs_root=output_root)) if tasks else None
    smoke_epochs = args.smoke_epochs if kind == 'queue' else config.get('MAX_EPOCHS')
    print(json.dumps({'check_only': True, 'data_paths_exist': paths, 'dependencies_found': dependencies,
                      'runs_root': str(output_root),
                      'tasks': [registry.config_for_run(cid, seed, smoke=smoke, smoke_epochs=smoke_epochs)
                                for cid, seed, smoke in tasks],
                      'active_config_smoke': config.get('SMOKE', False),
                      'selected_train_smoke': tasks[0][2] if tasks else None,
                      'selected_output_preview': preview,
                      'image_read': False, 'weights_downloaded': False, 'training_started': False},
                     ensure_ascii=False, indent=2))
    if not all(paths.values()) or not all(dependencies.values()):
        raise SystemExit(1)
    return args
