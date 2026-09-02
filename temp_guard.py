# -*- coding: utf-8 -*-
"""
temp_guard.py —— 温度守护

在训练关键节点（epoch 30/50/80、每个任务跑完切换下一个前）检测 CPU/GPU 温度，
任一超过阈值就让电脑休息 REST_SECONDS 秒，避免过热降频或损伤硬件。

温度来源：
  - GPU：nvidia-smi（无需管理员）
  - CPU：Windows「Thermal Zone Information」性能计数器（cooked value 单位=开尔文，减 273.15 得摄氏）

阈值与休息时长改下面常量即可。
"""
import logging
import subprocess
import time

# ==================== 阈值与休息时长（按需修改） ====================
GPU_MAX_C = 85.0      # GPU 超过此值 → 休息（RTX 4060 Laptop 降频线约 87°C）
CPU_MAX_C = 90.0      # CPU 超过此值 → 休息（笔记本 CPU 降频线约 95~100°C）
REST_SECONDS = 600    # 休息 10 分钟
# ====================================================================


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


def check_temps_and_rest(tag="", log_info=logging.info, log_warn=logging.warning):
    """检测温度（GPU/CPU 各读两次并按误差取值），任一超过阈值则休息 REST_SECONDS 秒。返回 True 表示休息过。"""
    gpu = _read_twice(get_gpu_temp_c)
    cpu = _read_twice(get_cpu_temp_c)
    log_info(f"[温度守护{tag}] GPU={_fmt(gpu)}°C, CPU={_fmt(cpu)}°C（双读判定；阈值 GPU>{GPU_MAX_C} / CPU>{CPU_MAX_C}）")

    over = []
    if gpu is not None and gpu > GPU_MAX_C:
        over.append(f"GPU {gpu:.1f}°C > {GPU_MAX_C}°C")
    if cpu is not None and cpu > CPU_MAX_C:
        over.append(f"CPU {cpu:.1f}°C > {CPU_MAX_C}°C")

    if over:
        log_warn(f"[温度守护] ⚠️ {'、'.join(over)}，休息 {REST_SECONDS // 60} 分钟…")
        time.sleep(REST_SECONDS)
        log_info("[温度守护] 休息结束，继续")
        return True
    return False


if __name__ == '__main__':
    gpu = get_gpu_temp_c()
    cpu = get_cpu_temp_c()
    print(f"[温度守护] GPU={_fmt(gpu)}°C, CPU={_fmt(cpu)}°C（阈值 GPU>{GPU_MAX_C} / CPU>{CPU_MAX_C}）")
