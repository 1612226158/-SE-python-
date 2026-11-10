"""MySQL persistence for v3. Importing this module does not open a connection."""

import argparse
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from getpass import getpass
import json
import logging
from math import fsum, isfinite
import os
from pathlib import Path
import subprocess

import pymysql
from pymysql.cursors import DictCursor
from pymysql.err import IntegrityError, MySQLError


DATABASE = "waste_classification_v3"
TEST_DATABASE = "waste_classification_v3_test"
DATABASES = (DATABASE, TEST_DATABASE)
TRAINING_DATABASE_ENV = "V3_MYSQL_DATABASE"
STORE_USER = "waste_v3_store"
PASSWORD_ENV = "V3_MYSQL_STORE_PASSWORD"
_session_password = ContextVar("mysql_store_password", default=None)


def _read_windows_store_password():
    local_app_data = os.environ.get("LOCALAPPDATA")
    if os.name != "nt" or not local_app_data:
        return None
    path = Path(local_app_data) / "WasteClassificationV3" / "mysql-store.credential.xml"
    if not path.is_file():
        return None
    powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    script = """
        $ErrorActionPreference = 'Stop'
        [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
        $credential = Import-Clixml -LiteralPath $env:V3_MYSQL_STORE_CREDENTIAL_FILE
        if ($credential -isnot [PSCredential] -or $credential.UserName -cne 'waste_v3_store') { exit 1 }
        [Console]::Write($credential.GetNetworkCredential().Password)
    """
    try:
        result = subprocess.run(
            [str(powershell), "-NoProfile", "-NonInteractive", "-Command", script],
            env=dict(os.environ, V3_MYSQL_STORE_CREDENTIAL_FILE=str(path)),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError(f"无法读取 Windows MySQL 凭据：{path}") from None
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"Windows MySQL 凭据无法解密或账号不匹配：{path}")
    return result.stdout


def training_database(*, default=TEST_DATABASE):
    """Select the target for new ordinary training runs."""
    database = os.environ.get(TRAINING_DATABASE_ENV, default)
    if database not in DATABASES:
        raise ValueError(f"{TRAINING_DATABASE_ENV} 必须是 {DATABASE} 或 {TEST_DATABASE}")
    return database


def _connect(database, *, autocommit):
    if database not in DATABASES:
        raise ValueError("未知的 v3 MySQL 库名")
    password = os.environ.get(PASSWORD_ENV) or _session_password.get() or _read_windows_store_password()
    if not password:
        raise RuntimeError(f"请设置 {PASSWORD_ENV}")
    return pymysql.connect(
        host="127.0.0.1",
        port=int(os.environ.get("V3_MYSQL_PORT", "3306")),
        user=STORE_USER,
        password=password,
        database=database,
        charset="utf8mb4",
        collation="utf8mb4_0900_bin",
        init_command="SET SESSION time_zone = '+08:00'",
        autocommit=autocommit,
        cursorclass=DictCursor,
        connect_timeout=5,
        read_timeout=10,
        write_timeout=10,
    )


@contextmanager
def transaction(database=DATABASE):
    """Yield a cursor. An exception aborts the transaction; commit errors propagate."""
    connection = _connect(database, autocommit=False)
    try:
        connection.begin()
        try:
            with connection.cursor() as cursor:
                yield cursor
        except BaseException:
            with suppress(MySQLError):
                connection.rollback()
            raise
        connection.commit()
    finally:
        if connection.open:
            connection.close()


@contextmanager
def cli_password():
    if os.environ.get(PASSWORD_ENV) or _session_password.get():
        yield
        return
    password = _read_windows_store_password() or getpass(f"MySQL password for {STORE_USER}: ")
    token = _session_password.set(password)
    try:
        yield
    finally:
        _session_password.reset(token)


class _TrainingLogHandler(logging.Handler):
    """Mirror marked events to the database that owns the run."""

    def __init__(self, run_id, database):
        super().__init__()
        self.run_id = run_id
        self.database = database
        self.epoch = None
        self.reported_error = None

    def emit(self, record):
        log_type = getattr(record, 'db_log_type', None)
        if log_type is None:
            return
        run_id = getattr(record, 'db_run_id', self.run_id)
        database = getattr(record, 'db_database', self.database)
        epoch = getattr(record, 'db_epoch', None)
        if epoch is not None:
            self.epoch = epoch
        traceback = None
        if record.exc_info and record.exc_info[0] is not None:
            self.reported_error = record.exc_info[1]
            traceback = logging.Formatter().formatException(record.exc_info)
        level = record.levelname if record.levelname in (
            'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL') else None
        log_time = datetime.fromtimestamp(record.created, timezone(timedelta(hours=8))).replace(tzinfo=None)
        try:
            with transaction(database) as cursor:
                cursor.execute("""
                    INSERT INTO training_logs
                        (run_id, log_time, log_type, level, epoch, message, traceback)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (run_id, log_time, log_type, level, epoch, record.getMessage(), traceback))
        except Exception as error:
            logging.getLogger(record.name).warning(
                'MySQL 训练日志写入失败：run_id=%s epoch=%s：%s', run_id, epoch, error)


@contextmanager
def queue_log_context(logger, *, database):
    """Route global queue events to the queue's target database."""
    handler = _TrainingLogHandler(None, database)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()


@contextmanager
def training_log_context(run_id, *, database, smoke=False):
    """Enable database logging for the bound ordinary training invocation."""
    if smoke or run_id is None:
        yield
        return
    logger = logging.getLogger()
    handler = _TrainingLogHandler(run_id, database)
    logger.addHandler(handler)
    try:
        yield
    except Exception as error:
        if handler.reported_error is not error:
            logger.exception('训练运行中止', extra={'db_log_type': 'train', 'db_epoch': handler.epoch})
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()


def write_temperature_log(*, database, run_id, epoch, log_time, disk_sample_time, gpu_temp_c, cpu_temp_c,
                          disk1_temp_c, disk2_temp_c, level, message):
    """Save at most one temperature summary for a completed epoch."""
    temperatures = tuple(float(value) if value is not None and isfinite(value) else 0.0
                         for value in (gpu_temp_c, cpu_temp_c, disk1_temp_c, disk2_temp_c))
    try:
        with transaction(database) as cursor:
            cursor.execute("""
                SELECT r.run_id FROM training_runs AS r
                JOIN epoch_metrics AS e ON e.run_id = r.run_id AND e.epoch = %s
                WHERE r.run_id = %s AND r.is_valid = 1
                FOR UPDATE
            """, (epoch, run_id))
            if cursor.fetchone() is None:
                raise ValueError('运行无效或对应的 epoch 指标尚未保存')
            cursor.execute("""
                SELECT log_id FROM temperature_logs WHERE run_id = %s AND epoch = %s FOR UPDATE
            """, (run_id, epoch))
            if cursor.fetchone() is not None:
                return
            cursor.execute("""
                INSERT INTO temperature_logs
                    (run_id, log_time, disk_sample_time, gpu_temp_c, cpu_temp_c,
                     disk1_temp_c, disk2_temp_c, level, epoch, message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (run_id, log_time, disk_sample_time, *temperatures, level, epoch, message))
    except Exception as error:
        logging.warning('MySQL 温控摘要写入失败：run_id=%s epoch=%s：%s', run_id, epoch, error)


def _encode_params(params):
    if not isinstance(params, dict) or not params:
        raise ValueError("模型或训练参数必须是非空 JSON 对象")
    return json.dumps(params, ensure_ascii=False, allow_nan=False, sort_keys=True)


def register_model(cursor, *, config_id, backbone, model_params, target_epochs,
                   num_classes=None, parameter_count=None):
    """Return the model ID, reusing an existing compatible model definition."""
    params = _encode_params(model_params)
    lookup = """
        SELECT model_id, backbone, model_params, target_epochs, num_classes
        FROM experiment_models WHERE config_id = %s
    """
    cursor.execute(lookup, (config_id,))
    existing = cursor.fetchone()
    if existing is None:
        try:
            cursor.execute("""
                INSERT INTO experiment_models
                    (config_id, backbone, model_params, target_epochs,
                     num_classes, parameter_count)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (config_id, backbone, params, target_epochs, num_classes, parameter_count))
            return cursor.lastrowid
        except IntegrityError as error:
            if error.args[0] != 1062:
                raise
            # A concurrent insert may have committed after our snapshot was taken.
            cursor.execute(lookup + " FOR SHARE", (config_id,))
            existing = cursor.fetchone()
            if existing is None:
                raise
    if (
        existing["backbone"] != backbone
        or _encode_params(json.loads(existing["model_params"])) != params
        or existing["target_epochs"] != target_epochs
        or (num_classes is not None and existing["num_classes"] not in (None, num_classes))
    ):
        raise ValueError(f"同名模型的定义或训练预算不同：{config_id}")
    return existing["model_id"]


def create_run(cursor, *, model_id, seed, training_params, batch_size=None,
               train_samples=None, val_samples=None, started_at=None, parent_run_id=None):
    """Register a new run. Any resume must use its existing run ID."""
    cursor.execute("""
        INSERT INTO training_runs
            (model_id, seed, training_params, batch_size, train_samples, val_samples, started_at, parent_run_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (model_id, seed, _encode_params(training_params), batch_size,
          train_samples, val_samples, started_at, parent_run_id))
    return cursor.lastrowid


def register_new_run(*, model, run, database):
    """Register the model and run in one transaction, returning only after commit."""
    with transaction(database) as cursor:
        model_id = register_model(cursor, **model)
        run_id = create_run(cursor, model_id=model_id, **run)
    return run_id


def register_test_long_run(*, model, parent_run_id, parent_config_id, seed, start_epoch,
                           normalization_signature, checkpoint):
    """Register a Long child without inserting any inherited epoch rows."""
    if start_epoch < 1:
        raise ValueError('Long 来源预算必须大于 0')
    with transaction(TEST_DATABASE) as cursor:
        cursor.execute("""
            SELECT r.seed, r.batch_size, r.train_samples, r.val_samples, r.training_params,
                   m.config_id, m.model_params, m.target_epochs, m.num_classes, m.parameter_count,
                   t.final_epoch, t.best_val_epoch, t.best_val_acc
            FROM training_runs AS r
            JOIN experiment_models AS m ON m.model_id = r.model_id
            JOIN training_results AS t ON t.run_id = r.run_id
            WHERE r.run_id = %s AND r.status = 'completed' AND r.is_valid = 1
                AND r.parent_run_id IS NULL AND r.ended_at IS NOT NULL
            FOR SHARE
        """, (parent_run_id,))
        parent = cursor.fetchone()
        if parent is None:
            raise ValueError(f'run_id={parent_run_id} 不是已完成的有效普通来源')
        training_params = json.loads(parent['training_params'])
        if (parent['config_id'] != parent_config_id or parent['seed'] != seed
                or parent['target_epochs'] != start_epoch or model['target_epochs'] <= start_epoch
                or training_params.get('normalization_signature') != normalization_signature
                or _encode_params(json.loads(parent['model_params'])) != _encode_params(model['model_params'])):
            raise ValueError('Long 目标与来源的模型、种子、预算或归一化配置不一致')
        cursor.execute("""
            SELECT epoch, sampler_seed, state, val_acc, train_acc FROM epoch_metrics
            WHERE run_id = %s ORDER BY epoch FOR SHARE
        """, (parent_run_id,))
        rows = cursor.fetchall()
        if [row['epoch'] for row in rows] != list(range(start_epoch)):
            raise ValueError('Long 来源指标未完整覆盖原预算')
        last = rows[-1]
        best = max(rows, key=lambda row: row['val_acc'])
        if ((parent['final_epoch'], last['sampler_seed'], last['state']) !=
                (checkpoint['epoch'], checkpoint['sampler_seed'], checkpoint['state'])
                or (parent['best_val_epoch'], parent['best_val_acc']) != (best['epoch'], best['val_acc'])
                or (checkpoint['best_epoch'], checkpoint['best_val_acc'], checkpoint['train_acc_at_best']) !=
                (best['epoch'], best['val_acc'], best['train_acc'])):
            raise ValueError('Long 来源末轮断点、指标或结果不一致')
        model_id = register_model(cursor, **model, num_classes=parent['num_classes'],
                                  parameter_count=parent['parameter_count'])
        cursor.execute("""
            SELECT run_id FROM training_runs WHERE model_id = %s AND seed = %s AND is_valid = 1
            FOR UPDATE
        """, (model_id, seed))
        if cursor.fetchone() is not None:
            raise ValueError('测试库已存在有效的同模型、同 seed Long 运行')
        run_id = create_run(cursor, model_id=model_id, seed=seed, training_params=training_params,
                            batch_size=parent['batch_size'], train_samples=parent['train_samples'],
                            val_samples=parent['val_samples'], parent_run_id=parent_run_id)
    return run_id


def insert_epoch(cursor, *, run_id, epoch, train_acc, train_loss, val_acc, val_loss,
                 lr, time_spend_s, sampler_seed=None, val_macro_f1=None,
                 val_weighted_f1=None, val_weighted_precision=None,
                 val_weighted_recall=None, gradient_norm=None, state=None, save_time=None):
    """Insert one epoch. A duplicate key is reported to the caller."""
    cursor.execute("""
        INSERT INTO epoch_metrics
            (run_id, epoch, train_acc, train_loss, val_acc, val_loss, lr, time_spend_s,
             sampler_seed, val_macro_f1, val_weighted_f1, val_weighted_precision,
             val_weighted_recall, gradient_norm, state, save_time)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (run_id, epoch, train_acc, train_loss, val_acc, val_loss, lr, time_spend_s,
          sampler_seed, val_macro_f1, val_weighted_f1, val_weighted_precision,
          val_weighted_recall, gradient_norm, state, save_time))


def read_epoch(run_id, epoch, *, database=DATABASE):
    """Read an epoch using a fresh connection and return None if it is absent."""
    with _connect(database, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM epoch_metrics WHERE run_id = %s AND epoch = %s",
                           (run_id, epoch))
            return cursor.fetchone()


def read_run_results(run_id, *, database=TEST_DATABASE):
    """Read one ordinary run, its ordered metrics, and its result in a read-only snapshot."""
    if type(run_id) is not int or not 1 <= run_id <= 4_294_967_295:
        raise ValueError('run_id 必须是 INT UNSIGNED 范围内的正整数')
    with _connect(database, autocommit=False) as connection:
        with connection.cursor() as cursor:
            cursor.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
            connection.begin()
            cursor.execute("""
                SELECT r.*, m.config_id, m.backbone, m.num_classes, m.model_params,
                       m.parameter_count, m.target_epochs
                FROM training_runs AS r
                JOIN experiment_models AS m ON m.model_id = r.model_id
                WHERE r.run_id = %s
            """, (run_id,))
            run = cursor.fetchone()
            if run is None:
                raise ValueError(f'找不到 run_id={run_id}')
            if run['is_valid'] != 1:
                raise ValueError(f'run_id={run_id} 已作废')
            if run['parent_run_id'] is not None:
                raise ValueError(f'run_id={run_id} 属于 Long，当前只读取普通运行')
            cursor.execute('SELECT * FROM epoch_metrics WHERE run_id = %s ORDER BY epoch', (run_id,))
            metrics = list(cursor.fetchall())
            cursor.execute('SELECT * FROM training_results WHERE run_id = %s', (run_id,))
            result = cursor.fetchone()

    budget = run['target_epochs']
    if metrics and metrics[-1]['epoch'] >= budget:
        raise ValueError(f'run_id={run_id} 的指标超出 {budget} 轮预算')
    if run['status'] == 'completed':
        if (result is None or run['ended_at'] is None or result['final_epoch'] != budget - 1
                or [row['epoch'] for row in metrics] != list(range(budget))):
            raise ValueError(f'run_id={run_id} 的完成状态、逐轮指标或最终结果不一致')
    elif run['status'] != 'incomplete' or result is not None or run['ended_at'] is not None:
        raise ValueError(f'run_id={run_id} 的未完成状态与最终结果不一致')
    for key in ('model_params', 'training_params'):
        run[key] = json.loads(run[key])
    return {'run': run, 'metrics': metrics, 'result': result}


def read_run_progress(run_id, *, database):
    """Read run identity and all its epoch rows in one consistent SELECT."""
    with _connect(database, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT r.run_id, r.seed, r.status, r.is_valid, r.parent_run_id, r.ended_at,
                       m.config_id, m.target_epochs,
                       JSON_UNQUOTE(JSON_EXTRACT(r.training_params, '$.normalization_signature'))
                           AS normalization_signature,
                       e.epoch, e.sampler_seed, e.state, e.val_acc,
                       t.run_id AS result_run_id, t.final_epoch, t.final_val_acc,
                       t.best_val_epoch, t.best_val_acc
                FROM training_runs AS r
                JOIN experiment_models AS m ON m.model_id = r.model_id
                LEFT JOIN epoch_metrics AS e ON e.run_id = r.run_id
                LEFT JOIN training_results AS t ON t.run_id = r.run_id
                WHERE r.run_id = %s
                ORDER BY e.epoch
            """, (run_id,))
            return cursor.fetchall()


def has_run(config_id, seed, *, database):
    """Detect a valid run whose binding may have been moved or lost."""
    with _connect(database, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT r.run_id FROM training_runs AS r
                JOIN experiment_models AS m ON m.model_id = r.model_id
                WHERE m.config_id = %s AND r.seed = %s AND r.is_valid = 1
                LIMIT 1
            """, (config_id, seed))
            return cursor.fetchone() is not None


def _lock_open_run(cursor, run_id):
    cursor.execute("""
        SELECT r.seed, m.config_id, m.target_epochs,
               JSON_UNQUOTE(JSON_EXTRACT(r.training_params, '$.normalization_signature'))
                   AS normalization_signature
        FROM training_runs AS r
        JOIN experiment_models AS m ON m.model_id = r.model_id
        WHERE r.run_id = %s AND r.status = 'incomplete'
            AND r.is_valid = 1 AND r.parent_run_id IS NULL
        FOR UPDATE
    """, (run_id,))
    run = cursor.fetchone()
    if run is None:
        raise ValueError(f'run_id={run_id} 不是有效且未完成的普通运行')
    return run


def write_epoch(run_id, *, database, epoch, sampler_seed, state, **metrics):
    """Append the next epoch within the bound ordinary run's budget."""
    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch 必须是非负整数")
    values = {key: float(value) if value is not None else None for key, value in metrics.items()}
    for key, value in values.items():
        if value is not None and not isfinite(value):
            raise ValueError(f"epoch {epoch} 指标 {key} 不是有限数值")
    save_time = datetime.now(timezone(timedelta(hours=8))).replace(tzinfo=None)
    with transaction(database) as cursor:
        budget = _lock_open_run(cursor, run_id)['target_epochs']
        if epoch >= budget:
            raise ValueError(f'run_id={run_id} 的 epoch={epoch} 超出 {budget} 轮预算')
        cursor.execute("""
            SELECT epoch FROM epoch_metrics WHERE run_id = %s ORDER BY epoch FOR UPDATE
        """, (run_id,))
        existing_epochs = [row['epoch'] for row in cursor.fetchall()]
        if epoch in existing_epochs:
            raise ValueError(f'run_id={run_id} 的 epoch={epoch} 已存在')
        if existing_epochs != list(range(epoch)):
            raise ValueError(f'run_id={run_id} 的已有指标未连续覆盖到 epoch={epoch - 1}，拒绝写入 epoch={epoch}')
        insert_epoch(cursor, run_id=run_id, epoch=epoch, sampler_seed=sampler_seed,
                     state=state, save_time=save_time, **values)


def finalize_run(run_id, *, database, config_id, seed, target_epochs, normalization_signature,
                      checkpoint_epoch, sampler_seed, state):
    """Commit a complete ordinary run's results and completion status together."""
    if not 0 <= checkpoint_epoch == target_epochs - 1:
        raise ValueError(f'last 的 epoch={checkpoint_epoch} 未到末轮 epoch={target_epochs - 1}')
    with transaction(database) as cursor:
        run = _lock_open_run(cursor, run_id)
        if (run['config_id'], run['seed'], run['target_epochs'], run['normalization_signature']) != (
                config_id, seed, target_epochs, normalization_signature):
            raise ValueError(f'run_id={run_id} 的模型、seed、预算或归一化签名不一致')
        cursor.execute("""
            SELECT epoch, sampler_seed, state, train_acc, val_acc, val_loss,
                   val_macro_f1, val_weighted_f1, time_spend_s
            FROM epoch_metrics WHERE run_id = %s ORDER BY epoch FOR UPDATE
        """, (run_id,))
        metrics = cursor.fetchall()
        if [row['epoch'] for row in metrics] != list(range(target_epochs)):
            raise ValueError(f'run_id={run_id} 的指标未完整覆盖 epoch=0 至 {target_epochs - 1}')
        final = metrics[-1]
        if (final['sampler_seed'], final['state']) != (sampler_seed, state):
            raise ValueError('last 与末轮指标的采样种子或解冻状态不一致')
        best = max(metrics, key=lambda row: row['val_acc'])
        training_time = fsum(row['time_spend_s'] for row in metrics)
        save_time = datetime.now(timezone(timedelta(hours=8))).replace(tzinfo=None)
        cursor.execute("""
            INSERT INTO training_results
                (run_id, best_val_epoch, best_val_acc, train_acc_at_best,
                 final_epoch, final_val_acc, final_val_loss, final_val_macro_f1,
                 final_val_weighted_f1, training_time_s, save_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (run_id, best['epoch'], best['val_acc'], best['train_acc'],
              final['epoch'], final['val_acc'], final['val_loss'], final['val_macro_f1'],
              final['val_weighted_f1'], training_time, save_time))
        cursor.execute("""
            UPDATE training_runs SET status = 'completed', ended_at = %s
            WHERE run_id = %s AND status = 'incomplete' AND is_valid = 1
        """, (save_time, run_id))
        if cursor.rowcount != 1:
            raise RuntimeError(f'run_id={run_id} 的完成状态未写入')


def verify_test_database(*, model, seed, training_params):
    """Save the caller's registry configuration and a synthetic metric to the test database."""
    metrics = {
        "epoch": 0, "train_acc": 42.5, "train_loss": 0.5, "val_acc": 40.25,
        "val_loss": 0.625, "lr": 0.0001, "time_spend_s": 0.3359583647195782,
    }
    with cli_password():
        with transaction(TEST_DATABASE) as cursor:
            model_id = register_model(cursor, **model)
            if register_model(cursor, **model) != model_id:
                raise RuntimeError("同名模型登记未复用 model_id")
            run_id = create_run(cursor, model_id=model_id, seed=seed, training_params=training_params)
            insert_epoch(cursor, run_id=run_id, **metrics)
        row = read_epoch(run_id, 0, database=TEST_DATABASE)
        if row is None or row["run_id"] != run_id or any(row[key] != value for key, value in metrics.items()):
            raise RuntimeError("提交后读回的逐轮指标与写入值不一致")
    print(f"{TEST_DATABASE}: {model['config_id']}, model_id={model_id}, run_id={run_id}, epoch=0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v3 MySQL 测试库查询与验证")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--verify-test", action="store_true",
                       help="向独立测试库提交一组新记录并用新连接读回")
    modes.add_argument("--read-run", type=int, metavar="RUN_ID",
                       help="只读输出一个普通运行的身份、指标及结果")
    args = parser.parse_args()
    if args.read_run is not None:
        try:
            with cli_password():
                data = read_run_results(args.read_run)
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False))
        except (OSError, ValueError, RuntimeError, MySQLError) as error:
            parser.exit(1, f"MySQL 读取失败：{error}\n")
        raise SystemExit(0)
    if __package__:
        from .launch_checks import mysql_verification_input
    else:
        from launch_checks import mysql_verification_input
    verify_test_database(**mysql_verification_input())
