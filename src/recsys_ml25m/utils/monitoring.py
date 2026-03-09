from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class ResourceSnapshot:
    cpu_percent: float | None = None
    ram_percent: float | None = None
    gpu_util_percent: float | None = None
    gpu_mem_percent: float | None = None
    gpu_temp_c: float | None = None


def _fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0f}%"


def _fmt_temp(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0f}C"


class ResourceMonitor:
    def __init__(self, enabled: bool = True, gpu_index: int = 0):
        self.enabled = enabled
        self.gpu_index = int(gpu_index)
        self._psutil = None
        if self.enabled:
            try:
                import psutil

                self._psutil = psutil
            except Exception:
                self._psutil = None

    def snapshot(self) -> ResourceSnapshot:
        if not self.enabled:
            return ResourceSnapshot()

        snap = ResourceSnapshot()
        if self._psutil is not None:
            try:
                snap.cpu_percent = float(self._psutil.cpu_percent(interval=None))
                snap.ram_percent = float(self._psutil.virtual_memory().percent)
            except Exception:
                pass

        gpu = self._gpu_snapshot_via_nvidia_smi()
        if gpu is not None:
            snap.gpu_util_percent = gpu.gpu_util_percent
            snap.gpu_mem_percent = gpu.gpu_mem_percent
            snap.gpu_temp_c = gpu.gpu_temp_c
        return snap

    def summary(self) -> str:
        s = self.snapshot()
        return (
            f"cpu={_fmt_pct(s.cpu_percent)} "
            f"ram={_fmt_pct(s.ram_percent)} "
            f"gpu={_fmt_pct(s.gpu_util_percent)} "
            f"vram={_fmt_pct(s.gpu_mem_percent)} "
            f"gpu_temp={_fmt_temp(s.gpu_temp_c)}"
        )

    def _gpu_snapshot_via_nvidia_smi(self) -> ResourceSnapshot | None:
        query = "utilization.gpu,memory.used,memory.total,temperature.gpu"
        cmd = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            f"--id={self.gpu_index}",
        ]
        try:
            out = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=1.0,
                check=False,
            )
            if out.returncode != 0:
                return None
            line = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                return None
            util = float(parts[0])
            mem_used = float(parts[1])
            mem_total = float(parts[2])
            temp = float(parts[3])
            mem_pct = 100.0 * mem_used / max(mem_total, 1.0)
            return ResourceSnapshot(gpu_util_percent=util, gpu_mem_percent=mem_pct, gpu_temp_c=temp)
        except Exception:
            return None

