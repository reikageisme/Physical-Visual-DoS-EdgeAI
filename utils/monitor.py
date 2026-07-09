"""
monitor.py — Resource Monitor with Latency Breakdown

Changes from original (per REVIEWED.md Major Issue 8):
  - Added per-stage latency logging: preproc / forward / conf_filter / nms / total
  - NMS time is now logged separately to verify NMS-as-bottleneck claim
  - CSV header extended with latency breakdown columns
  - Added get_latency_summary() for statistical reporting
"""

import psutil
import time
import os
import csv
import statistics
from datetime import datetime


class EdgeMonitor:
    """
    Resource monitor for Edge-AI simulation.

    Logs per-frame:
      - CPU%, RAM%, RAM_MB, Temperature
      - FPS (actual)
      - Per-stage latency: preproc_ms, forward_ms, conf_filter_ms, nms_ms, total_ms
    """

    CSV_HEADER = [
        "Timestamp",
        "CPU_Percent",
        "RAM_Percent",
        "RAM_Used_MB",
        "Temperature_C",
        "FPS_Actual",
        "Preproc_ms",
        "Forward_ms",
        "ConfFilter_ms",
        "NMS_ms",
        "Render_ms",
        "TotalFrame_ms",
        "Num_Raw_Boxes",
        "Num_Final_Boxes",
        "Scenario",
    ]

    def __init__(self, log_dir: str = "logs", scenario: str = "unknown"):
        self.log_dir  = log_dir
        self.scenario = scenario

        os.makedirs(self.log_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            self.log_dir, f"resource_log_{scenario}_{ts}.csv"
        )

        with open(self.log_file, mode='w', newline='') as f:
            csv.writer(f).writerow(self.CSV_HEADER)

        # Warm up psutil
        psutil.cpu_percent(interval=None)

        # ── Internal history for statistics ──────────────────────────────────
        self._nms_history   : list[float] = []
        self._fps_history   : list[float] = []
        self._cpu_history   : list[float] = []
        self._total_history : list[float] = []

    # ─────────────────────────────────────────────────────────────────────────
    def get_cpu_load(self) -> float:
        return psutil.cpu_percent(interval=None)

    def get_ram_usage(self) -> tuple[float, float]:
        mem = psutil.virtual_memory()
        return mem.percent, mem.used / (1024 * 1024)

    def get_temperature(self) -> float:
        """Read CPU temperature (Raspberry Pi / Linux). Returns 0 on Windows."""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return float(f.read()) / 1000.0
        except (FileNotFoundError, PermissionError):
            return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    def log_status(
        self,
        current_fps: float,
        latency_ms: dict = None,
        num_raw_boxes: int = 0,
        num_final_boxes: int = 0,
        render_ms: float = 0.0,
    ) -> dict:
        """
        Log one frame's metrics to CSV and return dict for display.

        Args:
            current_fps     : measured FPS for this frame
            latency_ms      : dict from VictimModel.get_predictions_with_nms()
                              keys: preproc_ms, forward_ms, conf_filter_ms, nms_ms, total_ms
            num_raw_boxes   : boxes BEFORE NMS (attack metric)
            num_final_boxes : boxes AFTER NMS (observable metric)
            render_ms       : time spent rendering/displaying (ms)
        """
        cpu             = self.get_cpu_load()
        ram_pct, ram_mb = self.get_ram_usage()
        temp            = self.get_temperature()
        ts              = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Unpack latency dict (or zeros if not profiling)
        lms = latency_ms or {}
        preproc_ms    = lms.get('preproc_ms',    0.0)
        forward_ms    = lms.get('forward_ms',    0.0)
        conf_ms       = lms.get('conf_filter_ms', 0.0)
        nms_ms        = lms.get('nms_ms',        0.0)
        total_ms      = lms.get('total_ms',      0.0)

        # ── CSV row ───────────────────────────────────────────────────────────
        with open(self.log_file, mode='a', newline='') as f:
            csv.writer(f).writerow([
                ts, cpu, ram_pct, ram_mb, temp, current_fps,
                f"{preproc_ms:.2f}", f"{forward_ms:.2f}",
                f"{conf_ms:.2f}", f"{nms_ms:.2f}",
                f"{render_ms:.2f}", f"{total_ms:.2f}",
                num_raw_boxes, num_final_boxes,
                self.scenario,
            ])

        # ── Update internal history ───────────────────────────────────────────
        self._nms_history.append(nms_ms)
        self._fps_history.append(current_fps)
        self._cpu_history.append(cpu)
        self._total_history.append(total_ms)

        return {
            "cpu"           : cpu,
            "ram_pct"       : ram_pct,
            "ram_mb"        : ram_mb,
            "temp"          : temp,
            "fps"           : current_fps,
            "preproc_ms"    : preproc_ms,
            "forward_ms"    : forward_ms,
            "conf_ms"       : conf_ms,
            "nms_ms"        : nms_ms,
            "total_ms"      : total_ms,
            "num_raw_boxes" : num_raw_boxes,
            "num_final_boxes": num_final_boxes,
        }

    # ─────────────────────────────────────────────────────────────────────────
    def get_latency_summary(self) -> dict:
        """
        Return statistical summary of logged latencies.
        Useful for paper Table reporting (mean ± std).
        """
        def _stats(data: list[float]) -> dict:
            if not data:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'n': 0}
            return {
                'mean': statistics.mean(data),
                'std' : statistics.stdev(data) if len(data) > 1 else 0,
                'min' : min(data),
                'max' : max(data),
                'n'   : len(data),
            }

        return {
            'nms_ms'   : _stats(self._nms_history),
            'fps'      : _stats(self._fps_history),
            'cpu'      : _stats(self._cpu_history),
            'total_ms' : _stats(self._total_history),
        }

    def print_summary(self):
        """Print a formatted summary table to console."""
        s = self.get_latency_summary()
        print(f"\n{'─'*60}")
        print(f"  Run Summary — Scenario: {self.scenario}")
        print(f"{'─'*60}")
        print(f"  FPS       : {s['fps']['mean']:.2f} ± {s['fps']['std']:.2f}  "
              f"[min={s['fps']['min']:.2f}, max={s['fps']['max']:.2f}]")
        print(f"  CPU%      : {s['cpu']['mean']:.1f} ± {s['cpu']['std']:.1f}")
        print(f"  NMS (ms)  : {s['nms_ms']['mean']:.2f} ± {s['nms_ms']['std']:.2f}  "
              f"[max={s['nms_ms']['max']:.2f}]")
        print(f"  Total (ms): {s['total_ms']['mean']:.2f} ± {s['total_ms']['std']:.2f}")
        print(f"  Frames logged: {s['fps']['n']}")
        print(f"  Log file: {self.log_file}")
        print(f"{'─'*60}\n")
