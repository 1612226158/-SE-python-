# -*- coding: utf-8 -*-
"""训练温度守护：检查CPU/GPU及可选磁盘指标，按配置告警或休眠。

GPU温度来自nvidia-smi，CPU温度来自Windows热区性能计数器。
盘温优先读取管理员采样器写入的日志；日志不可用时，仅管理员进程可备用直读。
关闭DISK_MONITOR_ENABLED后仅检查CPU/GPU；阈值与休眠时长由本模块常量控制。
"""
import ctypes
import logging
import os
import subprocess
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(_HERE, "runs")

# ==================== 阈值与休息时长（按需修改） ====================
GPU_MAX_C = 85.0      # GPU温度阈值（摄氏度）。
CPU_MAX_C = 93.0      # CPU温度阈值（摄氏度）。
REST_SECONDS = 600    # 休息 10 分钟
# ====================================================================

# ==================== 磁盘监测总开关 ====================
DISK_MONITOR_ENABLED = True   # 是否检查磁盘I/O、空间与温度。
# ======================================================

# ==================== 磁盘守护（数据集盘；免管理员；默认仅告警不休息） ====================
# 通过性能计数器和磁盘空间接口读取I/O与剩余空间，默认仅告警。
DISK_MOUNT = "E:\\"            # 监视哪个卷：数据集 train/val 与 runs/ 产物都在 E:
DISK_LETTER = "e:"             # 用于在性能计数器实例名（形如 "1 e:"）里定位该盘
DISK_QUEUE_WARN = 2.0          # 磁盘队列长度告警阈值。
DISK_LATENCY_WARN_MS = 20.0    # 平均传输延时告警阈值（毫秒）。
DISK_FREE_MIN_GB = 50.0        # 剩余空间告警阈值（GB）。
DISK_REST_ON_HIGH = False      # I/O或空间告警是否计入休眠，默认仅告警。
# =====================================================================================

# ==================== 磁盘温度：采样器日志优先 ====================
# 管理员单独运行disk_temp_sampler.py，普通训练进程读取其日志。
# 日志不可用时，仅允许且具有管理员权限的进程备用直读。
DISK_TEMP_LOG = os.path.join(RUNS_DIR, "disk_temp.log")
DISK_TEMP_MAX_AGE_S = 180       # 日志新鲜度上限；超过视为"采样器没在跑"
DISK_TEMP_WARN_C = 70.0         # 盘温超过此值 → 告警
DISK_TEMP_REST_C = 80.0         # 盘温达到此值时可计入休眠（摄氏度）。
DISK_TEMP_REST_ON_HIGH = True   # 盘温过高是否触发休息（与 I/O 不同：休息确实能让 SSD 降温）
DISK_TEMP_SENTINEL_MAX = 120    # 温度 > 此值视为哨兵/不支持（如 255）
# =====================================================================================


def _no_window_flag():
    """Windows 下隐藏子进程控制台窗口；非 Windows 返回 0。"""
    try:
        return subprocess.CREATE_NO_WINDOW
    except AttributeError:
        return 0


def get_gpu_temp_c():
    """读取 GPU 温度（摄氏度）；读不到返回 None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=_no_window_flag(),
        )
        line = out.stdout.strip().splitlines()[0].strip()
        return float(line)
    except Exception:
        return None


def get_cpu_temp_c():
    """读取 CPU 温度（摄氏度）；读不到返回 None。"""
    try:
        cmd = (
            "Get-Counter '\\Thermal Zone Information(*)\\Temperature' "
            "-ErrorAction SilentlyContinue | "
            "ForEach-Object { $_.CounterSamples.CookedValue }"
        )
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, timeout=20,
            creationflags=_no_window_flag(),
        )
        vals = []
        for tok in out.stdout.split():
            try:
                vals.append(float(tok))
            except ValueError:
                continue
        if not vals:
            return None
        # 计数器 cooked value 单位=开尔文 → 摄氏；只保留合理区间，否则视为读不到
        temps = [v - 273.15 for v in vals if 0.0 <= v - 273.15 <= 150.0]
        return max(temps) if temps else None
    except Exception:
        return None


_PS_DISK = (
    r"$ErrorActionPreference='SilentlyContinue';"
    r"$p=@('\PhysicalDisk(*)\Avg. Disk sec/Transfer','\PhysicalDisk(*)\Current Disk Queue Length');"
    r"$c=Get-Counter -Counter $p;"
    # 使用|分隔实例名、计数器名和数值，供下方解析。
    r"foreach($s in $c.CounterSamples){ $s.InstanceName + '|' + ($s.Path -replace '.*\\','') + '|' + $s.CookedValue }"
)


def get_disk_io(letter=DISK_LETTER):
    """读取指定盘符的平均传输延时（毫秒）与队列长度；读不到返回None。"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _PS_DISK],
            capture_output=True, text=True, timeout=30,
            creationflags=_no_window_flag(),
        )
        latency_ms = queue = None
        for line in out.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            inst, name, raw = parts[0].strip().lower(), parts[1].strip().lower(), parts[2].strip()
            if letter not in inst:          # 只要目标盘那个实例
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if "disk sec/transfer" in name:
                latency_ms = val * 1000.0   # 秒 → 毫秒
            elif "queue length" in name:
                queue = val
        if latency_ms is None and queue is None:
            return None
        return {"latency_ms": latency_ms, "queue": queue}
    except Exception:
        return None


def get_disk_free_gb(mount=DISK_MOUNT):
    """返回该卷的 (总容量GB, 剩余GB)；读不到返回 (None, None)。纯标准库，免管理员。"""
    try:
        import shutil
        u = shutil.disk_usage(mount)
        return u.total / 2 ** 30, u.free / 2 ** 30
    except Exception:
        return None, None


# 采样器与 temp_guard 共用这一条命令，保证"写日志"和"直读"口径一致。
# 使用|分隔设备ID、名称、当前/最大温度与盘符标记。
_PS_DISK_TEMP = (
    r"$ErrorActionPreference='SilentlyContinue';"
    r"$hd=(Get-Partition | Where-Object { $_.DriveLetter -eq 'E' } | Select-Object -First 1).DiskNumber;"
    r"foreach($d in Get-PhysicalDisk){"
    r"$c=$d | Get-StorageReliabilityCounter;"
    r"$n='';"
    r"if($hd -ne $null -and $d.DeviceId -eq $hd){$n='E'};"
    r"$d.DeviceId.ToString() + '|' + $d.FriendlyName + '|' + $c.Temperature + '|' + $c.TemperatureMax + '|' + $n}"
)

def is_elevated():
    """当前进程是否具备管理员权限。"""
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _parse_temp(raw):
    """把温度字符串转成摄氏度；空值/非数/哨兵值(>120，如 255)一律返回 None。"""
    try:
        v = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if v < -40.0 or v > DISK_TEMP_SENTINEL_MAX:
        return None
    return v


def read_disk_temps_direct():
    """**需要管理员**：直接读所有物理盘温度。

    返回 {device_id: {"name","temp_c","temp_max_c","is_E"}}；整条命令失败返回 None。
    非管理员时 `Get-StorageReliabilityCounter` 会拒绝访问 → 各项 temp_c 为 None。
    """
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _PS_DISK_TEMP],
            capture_output=True, text=True, timeout=30,
            creationflags=_no_window_flag(),
        )
        res = {}
        for line in out.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) != 5:
                continue
            dev_s, name, t, tmax, is_e = (p.strip() for p in parts)
            try:
                dev = int(dev_s)
            except ValueError:
                continue
            res[dev] = {"name": name, "temp_c": _parse_temp(t),
                        "temp_max_c": _parse_temp(tmax), "is_E": (is_e == "E")}
        return res or None
    except Exception:
        return None


def format_disk_temp_log_line(temps, when=None):
    """采样器的控制台/返回文本；文件记录另由logging增加时间和等级。"""
    when = when or datetime.now()
    e_dev = e_temp = ""
    detail = []
    for dev in sorted(temps):
        info = temps[dev]
        t = info.get("temp_c")
        detail.append("%d=%s" % (dev, "" if t is None else "%.0f" % t))
        if info.get("is_E"):
            e_dev = str(dev)
            e_temp = "" if t is None else "%.0f" % t
    return "%s,%s,%s,%s" % (when.strftime("%Y-%m-%d %H:%M:%S"), e_dev, e_temp, ";".join(detail))


def read_disk_temp_log(path=DISK_TEMP_LOG, max_age_s=DISK_TEMP_MAX_AGE_S):
    """读采样器日志的最新一行 → (E盘温度 or None, 秒龄 or None)。

    温度读不到时 age 仍可能返回（用于区分"采样器没跑"与"日志过期"）。
    """
    try:
        last = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    last = line
        if not last:
            return None, None
        fields = last.split(",")
        if len(fields) != 5:
            return None, None
        temp = _parse_temp(fields[3])
        try:
            age = (datetime.now() - datetime.strptime(fields[0].strip(), "%Y-%m-%d %H:%M:%S")).total_seconds()
        except ValueError:
            return temp, None
        if age < 0 or age > max_age_s:
            return None, age
        return temp, age
    except Exception:
        return None, None


def get_disk_temp(use_direct=True):
    """取 E: 盘温度。返回 {"temp_c": float|None, "src": 说明字符串}。

    优先级：① 采样器日志（新鲜）→ ② 本进程是管理员则直读 → ③ 记 ? 并给出原因。
    """
    temp, age = read_disk_temp_log()
    if temp is not None:
        return {"temp_c": temp, "src": "采样器日志"}
    if use_direct and is_elevated():
        temps = read_disk_temps_direct()
        if temps:
            for info in temps.values():
                if info.get("is_E") and info.get("temp_c") is not None:
                    return {"temp_c": info["temp_c"], "src": "管理员直读"}
            return {"temp_c": None, "src": "管理员直读无温度"}
        return {"temp_c": None, "src": "管理员直读失败"}
    if age is not None:
        if age < 0:
            return {"temp_c": None, "src": "日志时间戳异常(未来)"}
        if age <= DISK_TEMP_MAX_AGE_S:
            return {"temp_c": None, "src": "采样器在跑但该盘未报温度"}
        return {"temp_c": None, "src": "日志过期%.0fs" % age}
    return {"temp_c": None, "src": "采样器未运行"}


def get_disk_temp_c():
    """便捷包装：只取数值（读不到返回 None）。"""
    return get_disk_temp()["temp_c"]


def _fmt(v):
    return "?" if v is None else f"{v:.1f}"


def _read_twice(read_fn, interval=3.0):
    """连读两次并按误差决定最终值，降低单次偶然误差、偏向保守：
    - 两次差 < 5°C       → 取较高值（基本一致，宁高勿低）
    - 5°C ≤ 差 < 10°C    → 取平均值（平滑波动）
    - 差 ≥ 10°C          → 取较高值（疑似毛刺，宁高勿低）
    读不到：一次 None 用另一次；两次都 None 返回 None。"""
    vals = []
    for i in range(2):
        v = read_fn()
        if v is not None:
            if i == 0:
                time.sleep(3)  # 暂停3s
            vals.append(v)
        if i == 0:
            time.sleep(interval)  # 两次测量之间隔一下，避免读到同一瞬时的抖动
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    lo, hi = sorted(vals)
    diff = hi - lo
    if diff < 5.0:
        return hi
    if diff < 10.0:
        return (lo + hi) / 2.0
    return hi


def _check_temps_only(tag="", log_info=logging.info, log_warn=logging.warning, count=0):
    """关闭磁盘监测时只检查GPU/CPU温度，返回是否触发休眠。"""
    gpu = _read_twice(get_gpu_temp_c)
    cpu = _read_twice(get_cpu_temp_c)
    log_info(f"[温度守护{tag}] GPU={_fmt(gpu)}°C, CPU={_fmt(cpu)}°C（双读判定；阈值 GPU>{GPU_MAX_C} / CPU>{CPU_MAX_C}）")

    over = []
    if gpu is not None and gpu > GPU_MAX_C:
        over.append(f"GPU {gpu:.1f}°C > {GPU_MAX_C}°C")
    if cpu is not None and cpu > CPU_MAX_C:
        over.append(f"CPU {cpu:.1f}°C > {CPU_MAX_C}°C")

    if over:
        if count >= 2:
            log_warn(f"[温度守护][多次温度异常, 延长时间] ⚠️ {'、'.join(over)}，休息 {REST_SECONDS*2 // 60} 分钟…")
            time.sleep(REST_SECONDS * 2)
            _check_temps_only(tag, log_info, log_warn, count+1)
        else:
            log_warn(f"[温度守护] ⚠️ {'、'.join(over)}，休息 {REST_SECONDS // 60} 分钟…")
            time.sleep(REST_SECONDS)
            _check_temps_only(tag, log_info, log_warn, count+1)
        log_info("[温度守护] 休息结束，继续")
        return True
    return False


def check_temps_and_rest(tag="", log_info=logging.info, log_warn=logging.warning):
    """检查CPU/GPU及可选磁盘指标，按现有开关告警或休眠。

    磁盘I/O和空间由DISK_REST_ON_HIGH控制是否计入休眠，
    盘温由DISK_TEMP_REST_ON_HIGH控制；总开关关闭时只检查CPU/GPU。
    """
    if not DISK_MONITOR_ENABLED:
        return _check_temps_only(tag, log_info, log_warn)

    gpu = _read_twice(get_gpu_temp_c)
    cpu = _read_twice(get_cpu_temp_c)
    io = get_disk_io()
    _total_gb, free_gb = get_disk_free_gb()
    dt = get_disk_temp()
    dtemp = dt["temp_c"]

    lat = _fmt(io["latency_ms"] if io else None)
    que = _fmt(io["queue"] if io else None)
    free_s = "?" if free_gb is None else f"{free_gb:.0f}"
    dtemp_s = "?" if dtemp is None else f"{dtemp:.0f}"
    log_info(f"[温度守护{tag}] GPU={_fmt(gpu)}°C, CPU={_fmt(cpu)}°C（双读判定；阈值 GPU>{GPU_MAX_C} / CPU>{CPU_MAX_C}）"
             f" | 磁盘{DISK_MOUNT[0]}: 盘温={dtemp_s}°C({dt['src']}) 队列={que} 延时={lat}ms 剩余={free_s}GB")

    over = []
    if gpu is not None and gpu > GPU_MAX_C:
        over.append(f"GPU {gpu:.1f}°C > {GPU_MAX_C}°C")
    if cpu is not None and cpu > CPU_MAX_C:
        over.append(f"CPU {cpu:.1f}°C > {CPU_MAX_C}°C")

    # 磁盘：先收集告警，再决定哪些计入休息条件
    #   - I/O（队列/延时）默认只告警（休息不能缓解 I/O）
    #   - 盘温默认计入休息（休息确实能让 SSD 降温），可用 DISK_TEMP_REST_ON_HIGH 关闭
    disk_alerts, disk_rest = [], []
    if io and io["queue"] is not None and io["queue"] > DISK_QUEUE_WARN:
        disk_alerts.append(f"队列 {io['queue']:.2f} > {DISK_QUEUE_WARN}")
    if io and io["latency_ms"] is not None and io["latency_ms"] > DISK_LATENCY_WARN_MS:
        disk_alerts.append(f"传输延时 {io['latency_ms']:.1f}ms > {DISK_LATENCY_WARN_MS}ms")
    if free_gb is not None and free_gb < DISK_FREE_MIN_GB:
        disk_alerts.append(f"剩余空间 {free_gb:.0f}GB < {DISK_FREE_MIN_GB:.0f}GB")
    if dtemp is not None and dtemp >= DISK_TEMP_REST_C:
        disk_alerts.append(f"盘温 {dtemp:.0f}°C >= {DISK_TEMP_REST_C:.0f}°C")
        if DISK_TEMP_REST_ON_HIGH:
            disk_rest.append(f"磁盘{DISK_MOUNT[0]} 盘温 {dtemp:.0f}°C >= {DISK_TEMP_REST_C:.0f}°C")
    elif dtemp is not None and dtemp > DISK_TEMP_WARN_C:
        disk_alerts.append(f"盘温 {dtemp:.0f}°C > {DISK_TEMP_WARN_C:.0f}°C")
    if disk_alerts:
        if DISK_REST_ON_HIGH:
            disk_rest.extend([a for a in disk_alerts if a not in disk_rest])
        log_warn(f"[磁盘守护{tag}] ⚠️ {DISK_MOUNT[0]} {'、'.join(disk_alerts)}"
                 f"（{'已计入休息条件' if disk_rest else '仅告警，不休息'}）")
        over.extend(disk_rest)

    if over:
        log_warn(f"[温度守护] ⚠️ {'、'.join(over)}，休息 {REST_SECONDS // 60} 分钟…")
        time.sleep(REST_SECONDS)
        log_info("[温度守护] 休息结束，继续")
        return True
    return False


if __name__ == '__main__':
    gpu = get_gpu_temp_c()
    cpu = get_cpu_temp_c()
    print(f"[温度守护] GPU={_fmt(gpu)}°C, CPU={_fmt(cpu)}°C（阈值 GPU>{GPU_MAX_C} / CPU>{CPU_MAX_C}）；"
          f"本进程管理员权限={is_elevated()}；磁盘监测开关={DISK_MONITOR_ENABLED}")
    if not DISK_MONITOR_ENABLED:
        print("[磁盘守护] 已关闭（DISK_MONITOR_ENABLED=False）——守护点不做任何磁盘采样，等价原逻辑")
    else:
        io = get_disk_io()
        total_gb, free_gb = get_disk_free_gb()
        dt = get_disk_temp()
        print(f"[磁盘守护] {DISK_MOUNT[0]} 盘温={_fmt(dt['temp_c'])}°C（来源: {dt['src']}；"
              f"阈值 告警>{DISK_TEMP_WARN_C:.0f} / 休息>{DISK_TEMP_REST_C:.0f}；"
              f"计入休息={'是' if DISK_TEMP_REST_ON_HIGH else '否'}）")
        print(f"[磁盘守护] {DISK_MOUNT[0]}: 队列={_fmt(io['queue'] if io else None)} "
              f"延时={_fmt(io['latency_ms'] if io else None)}ms "
              f"剩余={'?' if free_gb is None else format(free_gb, '.0f')}GB / "
              f"总容量={'?' if total_gb is None else format(total_gb, '.0f')}GB"
              f"（阈值 队列>{DISK_QUEUE_WARN} / 延时>{DISK_LATENCY_WARN_MS}ms / 剩余<{DISK_FREE_MIN_GB:.0f}GB；"
              f"仅告警={'是' if not DISK_REST_ON_HIGH else '否，会计入休息'}）")
