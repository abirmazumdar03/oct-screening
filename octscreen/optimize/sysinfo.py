# octscreen/optimize/sysinfo.py
"""Describe the machine a benchmark ran on, so every latency number is traceable."""
import platform


def system_info():
    import onnxruntime as ort
    import psutil
    try:
        import cpuinfo
        cpu = cpuinfo.get_cpu_info().get("brand_raw") or platform.processor()
    except Exception:
        cpu = platform.processor()
    battery = None
    try:
        b = psutil.sensors_battery()
        battery = None if b is None else {"percent": b.percent, "plugged_in": bool(b.power_plugged)}
    except Exception:
        pass
    return {
        "machine": platform.node(),
        "cpu": cpu,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "battery": battery,
        "on_battery": bool(battery) and not battery["plugged_in"],
    }