# -*- coding: utf-8 -*-
"""管理员磁盘温度采样器，供普通权限训练进程读取盘温日志。

默认每60秒向src_v3/runs/disk_temp.log写入采样记录，间隔最小5秒。
在项目根目录的管理员终端执行：python src_v3/disk_temp_sampler.py
--interval设置间隔；--once只采一次；Ctrl+C停止循环。
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import temp_guard as tg  # 复用同一读取命令与日志格式，避免两处口径漂移

MAX_LINES = 20000      # 日志超过这么多行就裁剪
KEEP_LINES = 5000      # 裁剪后保留最后这么多行
_logger = logging.getLogger('src_v3.disk_temp_sampler')


class _DiskTempFileHandler(logging.FileHandler):
    """让现有sample_once异常处理继续返回写入失败。"""
    def handleError(self, record):
        raise


def _ensure_log():
    """首次采样时配置文件日志，返回路径。"""
    path = tg.DISK_TEMP_LOG
    if not _logger.handlers:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _logger.setLevel(logging.INFO)
        _logger.propagate = False
        handler = _DiskTempFileHandler(path, mode='a', encoding='utf-8')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s,%(levelname)s,%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        _logger.addHandler(handler)
    return path


def _trim(path):
    """日志过大时只保留最后KEEP_LINES条已有记录。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= MAX_LINES:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines[-KEEP_LINES:])
    except Exception:
        pass


def sample_once(verbose=True):
    """采一次：读全部物理盘温度 → 追加一行到日志。成功返回该行文本，失败返回 None。"""
    temps = tg.read_disk_temps_direct()
    if not temps:
        if verbose:
            print("[采样] 读取失败（权限不足，或 Get-PhysicalDisk 不可用）")
        return None
    line = tg.format_disk_temp_log_line(temps)
    path = _ensure_log()
    try:
        _logger.info(line.split(',', 1)[1])
    except Exception as e:
        if verbose:
            print("[采样] 写日志失败: %s" % e)
        return None
    _trim(path)
    if verbose:
        print("[采样] " + line)
    return line


def main():
    ap = argparse.ArgumentParser(description="磁盘温度采样器（需管理员）")
    ap.add_argument("--interval", type=int, default=60, help="采样间隔秒（默认 60）")
    ap.add_argument("--once", action="store_true", help="只采一次就退出")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(errors="replace")  # GBK 控制台下避免 UnicodeEncodeError
    except Exception:
        pass

    print("=" * 70)
    if not tg.is_elevated():
        print("[磁盘温度采样器] 当前【不是】管理员权限 —— 读盘温会返回“拒绝访问”。")
        print("请改用【管理员】终端运行本脚本（命令见下），然后把该窗口留着：")
        print(f'  "{sys.executable}" "{os.path.abspath(__file__)}"')
        print("=" * 70)
        return 2

    path = _ensure_log()
    print("[磁盘温度采样器] 已以管理员权限运行")
    print("  日志文件: %s" % path)
    print("  采样间隔: %d 秒（Ctrl+C 停止）" % args.interval)

    temps = tg.read_disk_temps_direct()
    if temps:
        for dev in sorted(temps):
            info = temps[dev]
            print("  物理盘 %d: %-28s 温度=%s°C%s" % (
                dev, info["name"],
                "?" if info["temp_c"] is None else "%.0f" % info["temp_c"],
                "   <== E:（数据集盘）所在" if info.get("is_E") else ""))
        if not any(i.get("is_E") for i in temps.values()):
            print("  [注意] 未识别出 E: 所在物理盘 —— 请检查 E: 盘符是否存在")
    else:
        print("  [警告] 首次读取失败，请确认确实是以管理员身份运行")
    print("=" * 70)

    if args.once:
        sample_once()
        return 0

    try:
        while True:
            sample_once()
            time.sleep(max(5, args.interval))
    except KeyboardInterrupt:
        print("\n[磁盘温度采样器] 已停止（日志已保留）")
        return 0


if __name__ == "__main__":
    sys.exit(main())
