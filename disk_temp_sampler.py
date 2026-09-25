# -*- coding: utf-8 -*-
"""
disk_temp_sampler.py —— 磁盘温度采样器（**必须用管理员权限运行**）

【为什么需要它】2026-09-22 本机实测：
  - E: = Samsung SSD 990 PRO 2TB（NVMe，数据集盘），系统盘 = SAMSUNG MZVL21T0HCLR；
  - 管理员下 `Get-StorageReliabilityCounter` 能读到温度（E: 53~54°C、系统盘 41°C）；
  - 但**非管理员一律"拒绝访问"**，而训练进程是普通权限。
  → 所以把"提权"限制在这个**只读传感器**的小脚本里：它每 60 s 采一次温度写
    `runs\\disk_temp.log`，`temp_guard` 只读那个日志文件（免提权）。
    训练进程权限完全不变；本脚本崩了也不影响训练（guard 读不到就记 `?`）。

【用法】在【管理员】PowerShell / 终端里跑一次，窗口留着即可：
    D:\\Python\\Python3.10.7\\python.exe E:\\DataSet\\垃圾分类图片-2\\src_v2\\disk_temp_sampler.py
停止：Ctrl+C（或直接关窗口）

【可选参数】
    --interval 60   采样间隔秒（默认 60，最小 5）
    --once          只采一次就退出（自检用）
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import temp_guard as tg  # 复用同一读取命令与日志格式，避免两处口径漂移

MAX_LINES = 20000      # 日志超过这么多行就裁剪
KEEP_LINES = 5000      # 裁剪后保留最后这么多行


def _ensure_log():
    """确保日志文件存在且带表头；返回其路径。"""
    path = tg.DISK_TEMP_LOG
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(tg.DISK_TEMP_LOG_HEADER + "\n")
    return path


def _trim(path):
    """日志过大时裁剪：保留表头 + 最后 KEEP_LINES 行。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= MAX_LINES:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(lines[0])
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
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
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
        print(r"  D:\Python\Python3.10.7\python.exe E:\DataSet\垃圾分类图片-2\src_v2\disk_temp_sampler.py")
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
