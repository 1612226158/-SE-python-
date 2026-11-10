"""MySQL persistence for v3. Importing this module does not open a connection."""

import argparse
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta, timezone
from getpass import getpass
import json
from math import isfinite
import os

import pymysql
from pymysql.cursors import DictCursor
from pymysql.err import IntegrityError, MySQLError


DATABASE = "waste_classification_v3"
TEST_DATABASE = "waste_classification_v3_test"
STORE_USER = "waste_v3_store"
PASSWORD_ENV = "V3_MYSQL_STORE_PASSWORD"


def _connect(database, *, autocommit):
    if database not in (DATABASE, TEST_DATABASE):
        raise ValueError("未知的 v3 MySQL 库名")
    password = os.environ.get(PASSWORD_ENV)
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
    if os.environ.get(PASSWORD_ENV):
        yield
        return
    os.environ[PASSWORD_ENV] = getpass(f"MySQL password for {STORE_USER}: ")
    try:
        yield
    finally:
        del os.environ[PASSWORD_ENV]


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
               train_samples=None, val_samples=None, started_at=None):
    """Register a new ordinary run. Any resume must use its existing run ID."""
    cursor.execute("""
        INSERT INTO training_runs
            (model_id, seed, training_params, batch_size, train_samples, val_samples, started_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (model_id, seed, _encode_params(training_params), batch_size,
          train_samples, val_samples, started_at))
    return cursor.lastrowid


def register_new_run(*, model, run, database):
    """Register the model and run in one transaction, returning only after commit."""
    with transaction(database) as cursor:
        model_id = register_model(cursor, **model)
        run_id = create_run(cursor, model_id=model_id, **run)
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


def write_first_epoch(run_id, *, epoch, sampler_seed, state, **metrics):
    """Commit the first actual training epoch to the test database."""
    if epoch != 0:
        raise ValueError("当前测试入口仅允许写入 epoch 0")
    values = {key: float(value) if value is not None else None for key, value in metrics.items()}
    for key, value in values.items():
        if value is not None and not isfinite(value):
            raise ValueError(f"epoch 0 指标 {key} 不是有限数值")
    save_time = datetime.now(timezone(timedelta(hours=8))).replace(tzinfo=None)
    with transaction(TEST_DATABASE) as cursor:
        insert_epoch(cursor, run_id=run_id, epoch=epoch, sampler_seed=sampler_seed,
                     state=state, save_time=save_time, **values)


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
    parser = argparse.ArgumentParser(description="v3 MySQL 独立读写验证")
    parser.add_argument("--verify-test", action="store_true", required=True,
                        help="向独立测试库提交一组新记录并用新连接读回")
    parser.parse_args()
    if __package__:
        from .launch_checks import mysql_verification_input
    else:
        from launch_checks import mysql_verification_input
    verify_test_database(**mysql_verification_input())
