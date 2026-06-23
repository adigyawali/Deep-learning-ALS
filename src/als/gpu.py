"""GPU / host-memory monitoring shown live during training.

Everything degrades gracefully: no CUDA, no ``psutil``, no ``nvidia-smi`` all
just produce shorter strings rather than errors, so the same calls work on the
lab box and on a laptop.
"""

from __future__ import annotations

import shutil
import subprocess

import torch

try:
    import psutil  # optional; nicer host-RAM numbers
except ImportError:  # pragma: no cover - psutil is an optional extra
    psutil = None


def _bytes_to_gib(n: float) -> float:
    return n / (1024 ** 3)


def host_ram_used_total_gib() -> tuple[float, float]:
    """(used, total) host RAM in GiB. Uses psutil if present, else /proc/meminfo."""
    if psutil is not None:
        vm = psutil.virtual_memory()
        return _bytes_to_gib(vm.total - vm.available), _bytes_to_gib(vm.total)
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, val = line.split(":", 1)
                info[key.strip()] = int(val.strip().split()[0]) * 1024  # kB → B
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        return _bytes_to_gib(total - avail), _bytes_to_gib(total)
    except (OSError, ValueError, KeyError):
        return float("nan"), float("nan")


def _nvidia_smi_line(device: torch.device) -> str | None:
    """One-line GPU utilization/temperature via nvidia-smi, or None if unavailable."""
    if device.type != "cuda" or shutil.which("nvidia-smi") is None:
        return None
    idx = device.index if device.index is not None else torch.cuda.current_device()
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits", "-i", str(idx)],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        util, temp, mem_used, mem_total = (x.strip() for x in out.stdout.strip().split(","))
        return f"util {util}% temp {temp}C smi_mem {mem_used}/{mem_total}MiB"
    except (subprocess.SubprocessError, ValueError):
        return None


def device_report(device: torch.device) -> str:
    """Human-readable one-time banner: device, GPU name, VRAM, host RAM."""
    parts = [f"device={device.type}"]
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        parts.append(f"gpu='{props.name}'")
        parts.append(f"vram={_bytes_to_gib(props.total_memory):.1f}GiB")
        parts.append(f"sm_{props.major}{props.minor}")
    used, total = host_ram_used_total_gib()
    parts.append(f"host_ram={used:.1f}/{total:.1f}GiB")
    return "  ".join(parts)


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def step_report(device: torch.device) -> str:
    """Per-epoch memory line: peak GPU alloc/reserved + host RAM (+ nvidia-smi)."""
    used, total = host_ram_used_total_gib()
    if device.type != "cuda":
        return f"ram {used:.1f}/{total:.1f}GiB"
    alloc = _bytes_to_gib(torch.cuda.max_memory_allocated(device))
    reserved = _bytes_to_gib(torch.cuda.max_memory_reserved(device))
    line = f"gpu_alloc {alloc:.2f}GiB gpu_reserved {reserved:.2f}GiB ram {used:.1f}/{total:.1f}GiB"
    smi = _nvidia_smi_line(device)
    return f"{line} {smi}" if smi else line
