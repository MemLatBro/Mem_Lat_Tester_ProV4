#!/usr/bin/env python3
""" 
MemLat Pro — Advanced CPU Cache & Memory Latency Profiler v2 
═══════════════════════════════════════════════════════════════ 
Targets Comet Lake (10th gen) and later Intel, Zen3+ AMD. 
 
New in v2: 
  • Fixed Write/RFO kernel — read-modify-write forces true cache ownership 
  • TLB stress test — 4 KB and 2 MB stride patterns isolate TLB miss penalty 
  • Cache Boundary Detective — auto-discovers L1/L2/L3 sizes from latency curve 
  • CPU Score Card — rates cache subsystem 0–100 across 6 categories 
  • ASCII hierarchy diagram in terminal output 
  • Interactive HTML dashboard (Chart.js, dark theme, zoom, score gauges) 
  • Comparison mode: --compare a.json b.json 
  • CSV + JSON + HTML + PNG export 
 
Access patterns: 
  1. random_chase    — fully random pointer permutation, defeats all prefetchers 
  2. stride64_chase  — 64-byte (1 cache line) stride, engages sequential prefetcher 
  3. stride256_chase — 256-byte stride, stresses TLB, partially defeats prefetcher 
  4. write_rfo       — random read-modify-write, true RFO latency 
  5. tlb_4k_chase    — 4 KB stride, 1 TLB miss per access (new) 
  6. tlb_2m_chase    — 2 MB stride, stresses L2 TLB / huge-page (new) 
 
Usage: 
    python memlat_pro.py                       # interactive menu 
    python memlat_pro.py --quick               # quick: 256 MB cap, fewer traversals 
    python memlat_pro.py --fast                # fast: ~5 min 
    python memlat_pro.py --compare a.json b.json   # diff two runs 
    python memlat_pro.py --no-menu --bandwidth     # CLI only, add bandwidth 
"""
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 1 — Imports & Constants
# ══════════════════════════════════════════════════════════════════════════════
import argparse
import csv
import ctypes
import gc
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
 
import numpy as np
 
# ── Optional dependencies ────────────────────────────────────────────────────
try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
 
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
 
try:
    import matplotlib
    if os.environ.get("MPLBACKEND") == "Agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
 
# ── Timing budgets ───────────────────────────────────────────────────────────
TARGET_SEC_L1  = 2.0
TARGET_SEC_L2  = 4.0
TARGET_SEC_L3  = 8.0
TARGET_SEC_RAM = 18.0
 
ITERS_L1  = 5
ITERS_L2  = 4
ITERS_L3  = 3
ITERS_RAM = 3
 
QUICK_MAX_MB          = 256
QUICK_TRAVERSAL_SCALE = 0.70
DEFAULT_RNG_SEED      = 42
 
VERSION = "3.0.0"
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 2 — CPU Frequency Detection
# ══════════════════════════════════════════════════════════════════════════════
def detect_cpu_freq_ghz() -> Optional[float]:
    """Auto-detect CPU frequency in GHz via multiple OS methods."""
    if HAS_PSUTIL:
        try:
            freq = psutil.cpu_freq()
            if freq and freq.current and freq.current > 100:
                return freq.current / 1000.0
        except Exception:
            pass
    if sys.platform == "win32":
        try:
            raw = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).MaxClockSpeed"],
                stderr=subprocess.DEVNULL, timeout=10
            ).decode().strip()
            mhz = int(raw)
            if mhz > 100:
                return mhz / 1000.0
        except Exception:
            pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if "cpu MHz" in line:
                        mhz = float(line.split(":", 1)[1].strip())
                        if mhz > 100:
                            return mhz / 1000.0
        except Exception:
            pass
        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
                khz = int(f.read().strip())
                if khz > 100_000:
                    return khz / 1e6
        except Exception:
            pass
    if sys.platform == "darwin":
        try:
            raw = subprocess.check_output(
                ["sysctl", "-n", "hw.cpufrequency"], timeout=5
            ).decode().strip()
            hz = int(raw)
            if hz > 1e8:
                return hz / 1e9
        except Exception:
            pass
    return None
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 3 — CPU & Architecture Database
# ══════════════════════════════════════════════════════════════════════════════
INTEL_CONFIGS: Dict[str, Dict] = {
    "CometLake": {
        "l1_p_kb": 32, "l2_p_kb": 256,
        "l1_e_kb": None, "l2_e_kb": None,
        "l3_mb": 20, "hybrid": False,
        "expected_l1_ns": 1.2, "expected_l2_ns": 3.5,
        "expected_l3_ns": 14, "expected_ram_ns": 55,
        "notes": "Monolithic die, no E-cores.",
    },
    "TigerLake": {
        "l1_p_kb": 48, "l2_p_kb": 1280,
        "l1_e_kb": None, "l2_e_kb": None,
        "l3_mb": 12, "hybrid": False,
        "expected_l1_ns": 1.0, "expected_l2_ns": 3.0,
        "expected_l3_ns": 14, "expected_ram_ns": 55,
        "notes": "Willow Cove cores, large L2.",
    },
    "AlderLake": {
        "l1_p_kb": 48, "l2_p_kb": 1280,
        "l1_e_kb": 32, "l2_e_kb": 2048,
        "l3_mb": 30, "hybrid": True,
        "expected_l1_ns": 1.0, "expected_l2_ns": 3.0,
        "expected_l3_ns": 13, "expected_ram_ns": 58,
        "notes": "Golden Cove P + Gracemont E.",
    },
    "RaptorLake": {
        "l1_p_kb": 48, "l2_p_kb": 2048,
        "l1_e_kb": 32, "l2_e_kb": 4096,
        "l3_mb": 36, "hybrid": True,
        "expected_l1_ns": 0.9, "expected_l2_ns": 2.8,
        "expected_l3_ns": 12, "expected_ram_ns": 58,
        "notes": "P-core L2 doubled vs Alder.",
    },
    "MeteorLake": {
        "l1_p_kb": 48, "l2_p_kb": 2048,
        "l1_e_kb": 32, "l2_e_kb": 4096,
        "l3_mb": 24, "hybrid": True,
        "expected_l1_ns": 1.0, "expected_l2_ns": 3.0,
        "expected_l3_ns": 15, "expected_ram_ns": 65,
        "notes": "Tiled design, higher interconnect latency.",
    },
    "ArrowLake": {
        "l1_p_kb": 48, "l2_p_kb": 3072,
        "l1_e_kb": 32, "l2_e_kb": 4096,
        "l3_mb": 36, "hybrid": True,
        "expected_l1_ns": 0.8, "expected_l2_ns": 2.5,
        "expected_l3_ns": 12, "expected_ram_ns": 70,
        "notes": "Lion Cove P + Skymont E.",
    },
}
 
AMD_CONFIGS: Dict[str, Dict] = {
    "Zen3": {
        "l1_p_kb": 32, "l2_p_kb": 512, "l3_mb": 32,
        "v_cache_mb": 0, "chiplet": True,
        "expected_l1_ns": 1.2, "expected_l2_ns": 3.5,
        "expected_l3_ns": 12, "expected_ram_ns": 80,
    },
    "Zen3_VCACHE": {
        "l1_p_kb": 32, "l2_p_kb": 512, "l3_mb": 96,
        "v_cache_mb": 64, "chiplet": True,
        "expected_l1_ns": 1.2, "expected_l2_ns": 3.5,
        "expected_l3_ns": 10, "expected_ram_ns": 80,
    },
    "Zen4": {
        "l1_p_kb": 32, "l2_p_kb": 1024, "l3_mb": 32,
        "v_cache_mb": 0, "chiplet": True,
        "expected_l1_ns": 1.0, "expected_l2_ns": 3.0,
        "expected_l3_ns": 10, "expected_ram_ns": 75,
    },
    "Zen4_VCACHE": {
        "l1_p_kb": 32, "l2_p_kb": 1024, "l3_mb": 96,
        "v_cache_mb": 64, "chiplet": True,
        "expected_l1_ns": 1.0, "expected_l2_ns": 3.0,
        "expected_l3_ns": 8, "expected_ram_ns": 75,
    },
}
 
DEFAULT_CONFIG = {
    "l1_p_kb": 32, "l2_p_kb": 512, "l3_mb": 16,
    "hybrid": False, "chiplet": False,
    "expected_l1_ns": 1.2, "expected_l2_ns": 3.5,
    "expected_l3_ns": 15, "expected_ram_ns": 80,
    "notes": "Generic fallback config.",
}
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 4 — CPU Detection
# ══════════════════════════════════════════════════════════════════════════════
def detect_cpu() -> Dict:
    info: Dict = {
        "vendor": "Unknown",
        "model": platform.processor() or "Unknown",
        "gen_key": None,
        "freq_ghz": None,
        "p_cores": [],
        "e_cores": [],
        "all_cores": list(range(os.cpu_count() or 1)),
    }
    if sys.platform.startswith("linux"):
        _parse_linux_cpuinfo(info)
    elif sys.platform == "win32":
        _parse_windows_cpu(info)
    elif sys.platform == "darwin":
        _parse_macos_sysctl(info)
    info["gen_key"] = _classify_gen(info)
    info["freq_ghz"] = detect_cpu_freq_ghz()
    _detect_hybrid_topology(info)
    return info
 
 
def _parse_linux_cpuinfo(info: Dict) -> None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            content = f.read()
        for line in content.splitlines():
            if "vendor_id" in line and info["vendor"] == "Unknown":
                info["vendor"] = line.split(":", 1)[1].strip()
            if "model name" in line and "Unknown" in info["model"]:
                info["model"] = line.split(":", 1)[1].strip()
    except OSError:
        pass
 
 
def _parse_windows_cpu(info: Dict) -> None:
    try:
        raw = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Processor | Select-Object -First 1 "
             "Manufacturer, Name, NumberOfCores | Format-List"],
            stderr=subprocess.DEVNULL, timeout=10
        ).decode(errors="replace")
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("Manufacturer"):
                info["vendor"] = line.split(":", 1)[1].strip()
            elif line.startswith("Name"):
                info["model"] = line.split(":", 1)[1].strip()
    except Exception:
        try:
            raw = subprocess.check_output(
                ["wmic", "cpu", "get", "Manufacturer,Name,NumberOfCores"],
                stderr=subprocess.DEVNULL, timeout=10
            ).decode(errors="replace")
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            if len(lines) >= 2:
                match = re.search(
                    r"(GenuineIntel|AuthenticAMD|Intel|AMD)\s+(.*?)\s+(\d+)\s*$",
                    lines[1], re.IGNORECASE)
                if match:
                    info["vendor"] = match.group(1)
                    info["model"] = match.group(2).strip()
        except Exception:
            pass
 
 
def _parse_macos_sysctl(info: Dict) -> None:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5
        ).decode().strip()
        info["model"] = out
        if "Intel" in out:
            info["vendor"] = "GenuineIntel"
    except Exception:
        pass
 
 
def _classify_gen(info: Dict) -> Optional[str]:
    model = info.get("model", "")
    vendor = info.get("vendor", "")
    if "Intel" in vendor or "GenuineIntel" in vendor or "Intel" in model:
        if re.search(r"i\d-10\d{3}", model) or "Comet Lake" in model:
            return "CometLake"
        if re.search(r"i\d-11\d{3}", model) or "Tiger Lake" in model:
            return "TigerLake"
        if re.search(r"i\d-12\d{3}", model) or "Alder Lake" in model:
            return "AlderLake"
        if re.search(r"i\d-1[34]\d{3}", model) or "Raptor Lake" in model:
            return "RaptorLake"
        if "Meteor Lake" in model or re.search(r"Core Ultra \d{3}[UH]", model):
            return "MeteorLake"
        if "Arrow Lake" in model or re.search(r"Core Ultra 2\d{2}[SK]", model):
            return "ArrowLake"
    if "AMD" in vendor or "AuthenticAMD" in vendor or "AMD" in model:
        vcache = "X3D" in model
        if re.search(r"5\d{3}X?3?D?", model) or "Zen 3" in model:
            return "Zen3_VCACHE" if vcache else "Zen3"
        if re.search(r"7\d{3}X?3?D?", model) or "Zen 4" in model:
            return "Zen4_VCACHE" if vcache else "Zen4"
    return None
 
 
def _detect_hybrid_topology(info: Dict) -> None:
    p_cores, e_cores = [], []
    if sys.platform.startswith("linux"):
        cpu_base = "/sys/devices/system/cpu"
        for cpu_dir in sorted(os.listdir(cpu_base)):
            if not re.match(r"cpu\d+$", cpu_dir):
                continue
            idx = int(cpu_dir[3:])
            type_path = os.path.join(cpu_base, cpu_dir, "topology", "core_type")
            try:
                with open(type_path) as f:
                    core_type = f.read().strip().lower()
                if core_type in ("performance", "p"):
                    p_cores.append(idx)
                else:
                    e_cores.append(idx)
            except OSError:
                pass
    if not p_cores and not e_cores:
        all_cpus = info["all_cores"]
        n = len(all_cpus)
        gen_key = info.get("gen_key", "")
        if INTEL_CONFIGS.get(gen_key, {}).get("hybrid"):
            split = max(1, n // 3)
            p_cores = all_cpus[:split]
            e_cores = all_cpus[split:]
        else:
            p_cores = all_cpus
    info["p_cores"] = p_cores
    info["e_cores"] = e_cores
 
 
def get_cache_config(cpu_info: Dict) -> Dict:
    gen = cpu_info.get("gen_key")
    if gen in INTEL_CONFIGS:
        cfg = dict(INTEL_CONFIGS[gen])
        cfg.setdefault("chiplet", False)
        cfg.setdefault("v_cache_mb", 0)
        return cfg
    if gen in AMD_CONFIGS:
        cfg = dict(AMD_CONFIGS[gen])
        cfg.setdefault("hybrid", False)
        return cfg
    return dict(DEFAULT_CONFIG)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 5 — Test Size Generation & Timing Budgets
# ══════════════════════════════════════════════════════════════════════════════
def generate_test_sizes(cfg: Dict, max_bytes: int) -> List[int]:
    sizes = set()
    l1_kb = cfg.get("l1_p_kb", 32)
    l2_kb = cfg.get("l2_p_kb", 512)
    l3_mb = cfg.get("l3_mb", 16) + cfg.get("v_cache_mb", 0)
    boundaries_bytes = [l1_kb * 1024, l2_kb * 1024, l3_mb * 1024 * 1024]
    for b in boundaries_bytes:
        for frac in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            sizes.add(int(b * frac))
    for kb in [4, 8, 16, 32, 48, 64, 96, 128, 256, 384, 512, 768, 1024, 2048, 4096]:
        sizes.add(kb * 1024)
    for mb in [8, 16, 32, 64, 128, 256, 512, 1024]:
        sizes.add(mb * 1024 * 1024)
    if cfg.get("hybrid") and cfg.get("l2_e_kb"):
        e_l2 = cfg["l2_e_kb"] * 1024
        for frac in [0.75, 1.0, 1.25]:
            sizes.add(int(e_l2 * frac))
    return sorted(s for s in sizes if 4096 <= s <= max_bytes)
 
 
def get_timing_budget(size_bytes: int, cfg: Dict, fast: bool,
                      quick: bool = False) -> Tuple[float, int]:
    if fast:
        scale = 0.3
    elif quick:
        scale = QUICK_TRAVERSAL_SCALE
    else:
        scale = 1.0
    l2_thresh = cfg.get("l2_p_kb", 512) * 1024
    l3_thresh = cfg.get("l3_mb", 16) * 1024 * 1024
    if size_bytes <= l2_thresh:
        return TARGET_SEC_L1 * scale, ITERS_L1
    elif size_bytes <= l3_thresh:
        return TARGET_SEC_L2 * scale, ITERS_L2
    elif size_bytes <= l3_thresh * 4:
        return TARGET_SEC_L3 * scale, ITERS_L3
    else:
        return TARGET_SEC_RAM * scale, ITERS_RAM
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 6 — Buffer Builders
# ══════════════════════════════════════════════════════════════════════════════
def build_random_chase(size_bytes: int, stride_bytes: int = 64,
                       rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, int]:
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RNG_SEED)
    stride_elems = stride_bytes // 8
    n_nodes = size_bytes // stride_bytes
    if n_nodes < 8:
        raise ValueError(f"Working set {size_bytes}B too small for stride {stride_bytes}B")
    buf = np.zeros(n_nodes * stride_elems, dtype=np.int64)
    perm = rng.permutation(n_nodes).astype(np.int64)
    for i in range(n_nodes - 1):
        buf[perm[i] * stride_elems] = perm[i + 1] * stride_elems
    buf[perm[-1] * stride_elems] = perm[0] * stride_elems
    return buf, n_nodes
 
 
def build_stride_chase(size_bytes: int, stride_bytes: int) -> Tuple[np.ndarray, int]:
    stride_elems = stride_bytes // 8
    n_nodes = size_bytes // stride_bytes
    if n_nodes < 8:
        raise ValueError("Buffer too small")
    buf = np.zeros(n_nodes * stride_elems, dtype=np.int64)
    for i in range(n_nodes - 1):
        buf[i * stride_elems] = (i + 1) * stride_elems
    buf[(n_nodes - 1) * stride_elems] = 0
    return buf, n_nodes
 
 
def build_write_rfo(size_bytes: int,
                    rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, int]:
    """ 
    Build a pointer-chase buffer for write-RFO measurement (v2 fix). 
    Uses the same random permutation layout as random_chase. 
    The write-RFO kernel reads the next pointer AND writes to the 
    current cache line on every hop, creating a serialized dependency 
    chain that forces true cache line ownership transfer (I->M or S->M). 
    """
    return build_random_chase(size_bytes, stride_bytes=64, rng=rng)
 
 
def build_tlb_chase(size_bytes: int, page_size: int = 4096,
                    rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, int]:
    """ 
    Random pointer chase with page-sized stride. 
    Each hop crosses a page boundary -> 1 TLB miss per access. 
    page_size: 4096 for 4K pages, 2097152 for 2M huge pages. 
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RNG_SEED)
    stride_elems = page_size // 8
    n_nodes = size_bytes // page_size
    if n_nodes < 8:
        raise ValueError(f"Working set {size_bytes}B too small for TLB test stride {page_size}B")
    buf = np.zeros(n_nodes * stride_elems, dtype=np.int64)
    perm = rng.permutation(n_nodes).astype(np.int64)
    for i in range(n_nodes - 1):
        buf[perm[i] * stride_elems] = perm[i + 1] * stride_elems
    buf[perm[-1] * stride_elems] = perm[0] * stride_elems
    return buf, n_nodes
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 7 — Kernels (Numba JIT + Python fallback)
# ══════════════════════════════════════════════════════════════════════════════
if HAS_NUMBA:
    @njit(cache=True, fastmath=False)
    def _chase_kernel(buf: np.ndarray, n_nodes: int, traversals: int) -> int:
        dummy = np.int64(0)
        pos = np.int64(0)
        for _ in range(min(n_nodes * 4, 50_000)):
            pos = buf[pos]
            dummy ^= pos
        for _ in range(traversals):
            local = pos
            for _ in range(n_nodes):
                local = buf[local]
                dummy ^= local
            pos = local
        return dummy
 
    @njit(cache=True, fastmath=False)
    def _write_rfo_kernel(buf: np.ndarray, n_nodes: int, traversals: int) -> int:
        """ 
        Write-chase: pointer chase where each hop reads the next address 
        AND writes to the current cache line. The write is serialized by 
        the read dependency — you can't write until you've read the pointer. 
        This forces a true I->M / S->M MESI transition on every access. 
        """
        dummy = np.int64(0)
        pos = np.int64(0)
        # Warmup (read-only to fill cache/TLB)
        for _ in range(min(n_nodes * 4, 50_000)):
            pos = buf[pos]
            dummy ^= pos
        # Timed write-chase
        for _ in range(traversals):
            local = pos
            for _ in range(n_nodes):
                next_pos = buf[local]
                # Write to current line — forces ownership transfer
                # Use XOR to keep pointer intact at offset 0
                buf[local + np.int64(1)] = local ^ np.int64(0x5A5A)
                local = next_pos
                dummy ^= local
            pos = local
        return dummy
 
    @njit(cache=True, parallel=True)
    def _stream_triad(a: np.ndarray, b: np.ndarray,
                      c: np.ndarray, q: float, iters: int) -> None:
        n = len(a)
        for _ in range(iters):
            for i in prange(n):
                a[i] = b[i] + q * c[i]
else:
    def _chase_kernel(buf, n_nodes, traversals):  # type: ignore
        dummy = 0
        pos = 0
        for _ in range(min(n_nodes * 4, 10_000)):
            pos = int(buf[pos])
            dummy ^= pos
        for _ in range(traversals):
            local = pos
            for _ in range(n_nodes):
                local = int(buf[local])
                dummy ^= local
            pos = local
        return dummy
 
    def _write_rfo_kernel(buf, n_nodes, traversals):  # type: ignore
        dummy = 0
        pos = 0
        for _ in range(min(n_nodes * 4, 10_000)):
            pos = int(buf[pos])
            dummy ^= pos
        for _ in range(traversals):
            local = pos
            for _ in range(n_nodes):
                next_pos = int(buf[local])
                buf[local + 1] = local ^ 0x5A5A
                local = next_pos
                dummy ^= local
            pos = local
        return dummy
 
    def _stream_triad(a, b, c, q, iters):  # type: ignore
        for _ in range(iters):
            a[:] = b + q * c
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 8 — Calibration & Timed Measurement
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_chase(buf: np.ndarray, n_nodes: int, target_sec: float) -> int:
    probe = max(1, min(50, 100_000 // max(n_nodes, 1)))
    gc.collect()
    t0 = time.perf_counter()
    _chase_kernel(buf, n_nodes, probe)
    elapsed = time.perf_counter() - t0
    if elapsed < 0.005:
        probe *= 20
        gc.collect()
        t0 = time.perf_counter()
        _chase_kernel(buf, n_nodes, probe)
        elapsed = time.perf_counter() - t0
    if elapsed <= 0:
        return 1_000
    return min(max(1, int(target_sec * probe / elapsed)), 50_000_000)
 
 
def calibrate_write_rfo(buf: np.ndarray, n_nodes: int, target_sec: float) -> int:
    """Calibrate the write-chase kernel (same interface as calibrate_chase)."""
    probe = max(1, min(50, 100_000 // max(n_nodes, 1)))
    gc.collect()
    t0 = time.perf_counter()
    _write_rfo_kernel(buf, n_nodes, probe)
    elapsed = time.perf_counter() - t0
    if elapsed < 0.005:
        probe *= 20
        gc.collect()
        t0 = time.perf_counter()
        _write_rfo_kernel(buf, n_nodes, probe)
        elapsed = time.perf_counter() - t0
    if elapsed <= 0:
        return 1_000
    return min(max(1, int(target_sec * probe / elapsed)), 50_000_000)
 
 
def timed_chase(buf: np.ndarray, n_nodes: int, traversals: int) -> float:
    gc.collect()
    t0 = time.perf_counter()
    dummy = _chase_kernel(buf, n_nodes, traversals)
    t1 = time.perf_counter()
    if dummy == 0xDEAD:
        print("sentinel")
    return t1 - t0
 
 
def timed_write_rfo(buf: np.ndarray, n_nodes: int, traversals: int) -> float:
    """Time the write-chase kernel (same interface as timed_chase)."""
    gc.collect()
    t0 = time.perf_counter()
    dummy = _write_rfo_kernel(buf, n_nodes, traversals)
    t1 = time.perf_counter()
    if dummy == 0xDEAD:
        print("sentinel")
    return t1 - t0
 
 
def percentile_stats(samples: List[float]) -> Dict:
    a = np.array(samples)
    return {
        "min":    float(np.min(a)),
        "p5":     float(np.percentile(a, 5)),
        "p25":    float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75":    float(np.percentile(a, 75)),
        "p95":    float(np.percentile(a, 95)),
        "p99":    float(np.percentile(a, 99)),
        "max":    float(np.max(a)),
        "mean":   float(np.mean(a)),
        "std":    float(np.std(a)),
        "cv_pct": float(100 * np.std(a) / np.mean(a)) if np.mean(a) > 0 else 0.0,
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 9 — Bandwidth & Interconnect
# ══════════════════════════════════════════════════════════════════════════════
def measure_bandwidth(size_bytes: int) -> Optional[float]:
    try:
        n = size_bytes // 8
        a = np.random.rand(n).astype(np.float64)
        b = np.random.rand(n).astype(np.float64)
        c = np.random.rand(n).astype(np.float64)
        _stream_triad(a, b, c, 0.5, 5)
        iters = 50
        t0 = time.perf_counter()
        _stream_triad(a, b, c, 0.5, iters)
        elapsed = time.perf_counter() - t0
        bw = (3 * size_bytes * iters) / elapsed / 1e9
        del a, b, c
        return float(bw)
    except Exception:
        return None
 
 
def measure_interconnect(core_a: int, core_b: int,
                         iterations: int = 2000) -> Optional[float]:
    if not HAS_PSUTIL or core_a == core_b:
        return None
    shared_flag_a = ctypes.c_int64(0)
    shared_flag_b = ctypes.c_int64(0)
    results: List[float] = []
    barrier = threading.Barrier(2)
 
    def ping(cpu):
        try:
            psutil.Process(os.getpid()).cpu_affinity([cpu])
        except Exception:
            pass
        barrier.wait()
        for i in range(iterations):
            shared_flag_a.value = i
            while shared_flag_b.value != i:
                pass
 
    def pong(cpu):
        try:
            psutil.Process(os.getpid()).cpu_affinity([cpu])
        except Exception:
            pass
        barrier.wait()
        for i in range(iterations):
            while shared_flag_a.value != i:
                pass
            t0 = time.perf_counter()
            shared_flag_b.value = i
            t1 = time.perf_counter()
            results.append((t1 - t0) * 1e9)
 
    t_ping = threading.Thread(target=ping, args=(core_a,))
    t_pong = threading.Thread(target=pong, args=(core_b,))
    t_ping.start(); t_pong.start()
    t_ping.join(timeout=30); t_pong.join(timeout=30)
    if HAS_PSUTIL:
        try:
            psutil.Process(os.getpid()).cpu_affinity([])
        except Exception:
            pass
    if not results:
        return None
    arr = np.array(results)
    cutoff = np.percentile(arr, 95)
    filtered = arr[arr <= cutoff]
    return float(np.median(filtered)) if len(filtered) > 0 else float(np.median(arr))
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 10 — Cache Boundary Detective
# ══════════════════════════════════════════════════════════════════════════════
def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f} MB"
    return f"{size_bytes / 1024:.0f} KB"
 
 
def detect_cache_boundaries(results: List[Dict]) -> List[Dict]:
    """ 
    Auto-discover cache level transitions by analysing the random-chase 
    latency curve.  Looks for inflection points where d(log latency)/d(log size) 
    spikes — these correspond to working sets exceeding a cache level. 
    """
    # Extract valid random-chase data points
    points = []
    for r in results:
        rc = r.get("random_chase", {})
        if "error" not in rc and "median" in rc:
            points.append((r["size_bytes"], rc["median"]))
    if len(points) < 6:
        return []
 
    sizes = np.array([p[0] for p in points], dtype=np.float64)
    lats = np.array([p[1] for p in points], dtype=np.float64)
 
    # Work in log2 space — cache transitions appear as slope changes
    log_s = np.log2(sizes)
    log_l = np.log2(np.maximum(lats, 0.01))
 
    # Numerical derivative (central differences where possible)
    derivs = np.zeros(len(log_s))
    for i in range(1, len(log_s) - 1):
        ds = log_s[i + 1] - log_s[i - 1]
        dl = log_l[i + 1] - log_l[i - 1]
        derivs[i] = dl / ds if ds > 0 else 0
    # Forward/backward for endpoints
    if len(log_s) > 1:
        derivs[0] = (log_l[1] - log_l[0]) / max(log_s[1] - log_s[0], 0.01)
        derivs[-1] = (log_l[-1] - log_l[-2]) / max(log_s[-1] - log_s[-2], 0.01)
 
    # Find peaks in derivative that exceed threshold
    # A "real" cache boundary produces a derivative spike > background
    mean_d = np.mean(derivs)
    std_d = np.std(derivs)
    threshold = mean_d + 1.2 * std_d  # adaptive threshold
 
    boundaries = []
    # Group consecutive high-derivative points into single transitions
    in_peak = False
    peak_start = 0
    for i in range(len(derivs)):
        if derivs[i] > threshold and derivs[i] > 0.05:
            if not in_peak:
                peak_start = i
                in_peak = True
        else:
            if in_peak:
                # Peak ended — record the midpoint
                peak_mid = (peak_start + i - 1) // 2
                boundaries.append({
                    "size_bytes": int(sizes[peak_mid]),
                    "size_str": format_size(int(sizes[peak_mid])),
                    "derivative": float(derivs[peak_mid]),
                    "latency_before_ns": float(lats[max(0, peak_start - 1)]),
                    "latency_after_ns": float(lats[min(len(lats) - 1, i)]),
                    "latency_jump_ns": float(lats[min(len(lats) - 1, i)]
                                             - lats[max(0, peak_start - 1)]),
                })
                in_peak = False
    # Catch trailing peak
    if in_peak:
        peak_mid = (peak_start + len(derivs) - 1) // 2
        boundaries.append({
            "size_bytes": int(sizes[peak_mid]),
            "size_str": format_size(int(sizes[peak_mid])),
            "derivative": float(derivs[peak_mid]),
            "latency_before_ns": float(lats[max(0, peak_start - 1)]),
            "latency_after_ns": float(lats[-1]),
            "latency_jump_ns": float(lats[-1] - lats[max(0, peak_start - 1)]),
        })
 
    # Label boundaries as L1->L2, L2->L3, L3->RAM based on size ordering
    labels = ["L1 -> L2", "L2 -> L3", "L3 -> RAM", "unknown"]
    for i, b in enumerate(boundaries[:4]):
        b["label"] = labels[i] if i < len(labels) else "unknown"
 
    return boundaries
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 11 — CPU Score Card
# ══════════════════════════════════════════════════════════════════════════════
def compute_scores(summary: Dict, cfg: Dict) -> Dict:
    """ 
    Rate the CPU's memory subsystem 0-100 across categories. 
    100 = best-in-class; 50 = average; 0 = poor. 
    Scoring curves are based on observed ranges across Comet Lake -> Arrow Lake 
    and Zen3 -> Zen4. 
    """
    scores: Dict[str, int] = {}
 
    def _score_ns(val: Optional[float], best: float, worst: float) -> Optional[int]:
        """Lower latency = higher score. Linear scale between best and worst."""
        if val is None:
            return None
        # Clamp
        clamped = max(best, min(worst, val))
        return int(100 * (worst - clamped) / (worst - best))
 
    # L1 — best ~0.8 ns (Arrow Lake 5.5GHz), worst ~2.5 ns
    scores["L1 Latency"] = _score_ns(summary.get("l1_median_ns"), 0.8, 2.5) or 0
 
    # L2 — best ~2.5 ns, worst ~6 ns
    scores["L2 Latency"] = _score_ns(summary.get("l2_median_ns"), 2.5, 6.0) or 0
 
    # L3 — best ~8 ns (Zen4 V-Cache), worst ~20 ns
    scores["L3 Latency"] = _score_ns(summary.get("l3_median_ns"), 8.0, 20.0) or 0
 
    # RAM — best ~50 ns (DDR5 tuned), worst ~120 ns (DDR4 loose)
    scores["RAM Latency"] = _score_ns(summary.get("ram_median_ns"), 50.0, 120.0) or 0
 
    # Prefetcher effectiveness: how much does stride-64 beat random at L2 sizes?
    pf = summary.get("prefetcher_benefit_ns")
    if pf is not None and pf > 0:
        # More benefit = better prefetcher. Best ~3 ns savings, great = 2+
        scores["Prefetcher"] = min(100, int(pf * 40))  # 2.5ns -> 100
    else:
        scores["Prefetcher"] = 50  # neutral
 
    # Write overhead: RFO delta at L3
    rfo = summary.get("rfo_overhead_ns")
    if rfo is not None:
        # Lower overhead = better. Best ~0 ns, worst ~10 ns
        scores["Write Overhead"] = _score_ns(rfo, 0.0, 10.0) or 50
    else:
        scores["Write Overhead"] = 50
 
    # Overall — weighted average (L3 and RAM matter most for real workloads)
    weights = {
        "L1 Latency": 1.0, "L2 Latency": 1.5,
        "L3 Latency": 2.5, "RAM Latency": 2.5,
        "Prefetcher": 1.5, "Write Overhead": 1.0,
    }
    total_w = sum(weights.values())
    overall = sum(scores[k] * weights[k] for k in scores) / total_w
    scores["Overall"] = int(overall)
 
    return scores
 
 
def _score_grade(score: int) -> str:
    if score >= 90:
        return "S"
    elif score >= 80:
        return "A"
    elif score >= 65:
        return "B"
    elif score >= 50:
        return "C"
    elif score >= 35:
        return "D"
    else:
        return "F"
 
 
def _score_bar(score: int, width: int = 20) -> str:
    """ASCII progress bar for a 0-100 score."""
    filled = int(score / 100 * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}]"
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 12 — ASCII Hierarchy Diagram
# ══════════════════════════════════════════════════════════════════════════════
def draw_hierarchy(summary: Dict, cfg: Dict, freq_ghz: Optional[float]) -> str:
    """Generate an ASCII art diagram of the discovered cache hierarchy."""
    ns_per_cycle = (1.0 / freq_ghz) if freq_ghz and freq_ghz > 0.1 else (1.0 / 3.0)
    freq_str = f"{freq_ghz:.2f} GHz" if freq_ghz else "~3 GHz est."
 
    def _row(label: str, size_str: str, lat_ns: Optional[float]) -> str:
        if lat_ns is None:
            return ""
        cycles = lat_ns / ns_per_cycle
        return f"  |  {label:<6} {size_str:>8}  |  {lat_ns:6.1f} ns  ~{cycles:4.0f} cyc  |"
 
    l1_size = f"{cfg.get('l1_p_kb', '?')} KB"
    l2_size = f"{cfg.get('l2_p_kb', '?')} KB"
    l3_size = f"{cfg.get('l3_mb', '?')} MB"
 
    l1 = summary.get("l1_median_ns")
    l2 = summary.get("l2_median_ns")
    l3 = summary.get("l3_median_ns")
    ram = summary.get("ram_median_ns")
 
    lines = []
    lines.append("  +-------------------------------------------+")
    lines.append(f"  |   MEMORY HIERARCHY  @ {freq_str:<20}|")
    lines.append("  +-------------------------------------------+")
    if l1 is not None:
        lines.append(_row("L1", l1_size, l1))
        lines.append("  |      |                                   |")
    if l2 is not None:
        lines.append(_row("L2", l2_size, l2))
        lines.append("  |      |                                   |")
    if l3 is not None:
        lines.append(_row("L3", l3_size, l3))
        lines.append("  |      |                                   |")
    if ram is not None:
        lines.append(_row("RAM", "------", ram))
    lines.append("  +-------------------------------------------+")
 
    if l1 and ram:
        ratio = ram / l1
        lines.append(f"  |   RAM/L1 ratio: {ratio:.0f}x                      |")
        lines.append("  +-------------------------------------------+")
    return "\n".join(lines)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 13 — Main Tester Class
# ══════════════════════════════════════════════════════════════════════════════
class MemLatPro:
    CORE_PATTERNS = [
        "random_chase", "stride64_chase", "stride256_chase", "write_rfo",
    ]
    TLB_PATTERNS = ["tlb_4k_chase", "tlb_2m_chase"]
    ALL_PATTERNS = CORE_PATTERNS + TLB_PATTERNS
 
    def __init__(
        self,
        max_size_mb: int = 1024,
        output_dir: Optional[str] = None,
        fast: bool = False,
        quick: bool = False,
        bandwidth: bool = False,
        interconnect: bool = False,
        tlb_test: bool = True,
        pin_core_type: Optional[str] = None,
        rng_seed: int = DEFAULT_RNG_SEED,
    ):
        self.fast = fast
        self.quick = quick
        self.bandwidth = bandwidth
        self.interconnect = interconnect
        self.tlb_test = tlb_test
        self.rng = np.random.default_rng(rng_seed)
        self.rng_seed = rng_seed
 
        self.cpu_info = detect_cpu()
        self.cfg = get_cache_config(self.cpu_info)
        self.freq_ghz = self.cpu_info.get("freq_ghz")
 
        if quick and max_size_mb > QUICK_MAX_MB:
            max_size_mb = QUICK_MAX_MB
 
        self._setup_affinity(pin_core_type)
 
        if HAS_PSUTIL:
            avail_mb = psutil.virtual_memory().available // (1024 ** 2)
            safe_mb = int(avail_mb * 0.45)
            if max_size_mb > safe_mb:
                print(f"  Auto-limiting max size to {safe_mb} MB (45% of available RAM)")
                max_size_mb = safe_mb
 
        self.max_bytes = max_size_mb * 1024 * 1024
        self.sizes = generate_test_sizes(self.cfg, self.max_bytes)
 
        self.output_dir = output_dir or self._find_writable_dir()
        os.makedirs(self.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path  = os.path.join(self.output_dir, f"memlat_{ts}.txt")
        self.json_path = os.path.join(self.output_dir, f"memlat_{ts}.json")
        self.csv_path  = os.path.join(self.output_dir, f"memlat_{ts}.csv")
        self.html_path = os.path.join(self.output_dir, f"memlat_{ts}.html")
        self.plot_path = os.path.join(self.output_dir, f"memlat_{ts}.png")
        self._log_fh = None
 
        # Mode label
        if fast:
            mode_label = "FAST (~5 min)"
        elif quick:
            mode_label = f"QUICK (max {QUICK_MAX_MB} MB, 30% fewer traversals)"
        else:
            mode_label = "FULL (~17 min)"
 
        gen = self.cpu_info.get("gen_key") or "Unknown"
        hybrid = self.cfg.get("hybrid", False)
        freq_str = f"{self.freq_ghz:.2f} GHz" if self.freq_ghz else "unknown"
        print(f"\n  CPU   : {self.cpu_info['model']}")
        print(f"  Gen   : {gen}  |  Hybrid: {hybrid}")
        print(f"  Freq  : {freq_str}")
        print(f"  L1/L2 : {self.cfg.get('l1_p_kb')} KB / {self.cfg.get('l2_p_kb')} KB (P-core)")
        if hybrid and self.cfg.get("l1_e_kb"):
            print(f"  L1/L2E: {self.cfg['l1_e_kb']} KB / {self.cfg['l2_e_kb']} KB (E-core cluster)")
        print(f"  L3    : {self.cfg.get('l3_mb')} MB")
        print(f"  Sizes : {len(self.sizes)} test points up to {max_size_mb} MB")
        print(f"  Mode  : {mode_label}")
        print(f"  TLB   : {'ON' if tlb_test else 'OFF'}")
        print(f"  Seed  : {rng_seed}")
        notes = self.cfg.get("notes")
        if notes:
            print(f"  Note  : {notes}")
        print()
 
    def _setup_affinity(self, pin_core_type: Optional[str]) -> None:
        if not HAS_PSUTIL:
            return
        p = self.cpu_info["p_cores"]
        e = self.cpu_info["e_cores"]
        target = None
        if pin_core_type == "P" and p:
            target = [p[0]]
            print(f"  Pinned to P-core {p[0]}")
        elif pin_core_type == "E" and e:
            target = [e[0]]
            print(f"  Pinned to E-core {e[0]}")
        if target:
            try:
                psutil.Process(os.getpid()).cpu_affinity(target)
            except Exception as ex:
                print(f"  Warning: Could not set affinity -- {ex}")
 
    def _find_writable_dir(self) -> str:
        for d in [
            os.path.join(os.path.expanduser("~"), "Documents"),
            os.path.expanduser("~"),
            os.getcwd(),
            os.environ.get("TEMP", ""),
            "/tmp",
        ]:
            if d and os.path.isdir(d):
                try:
                    tp = os.path.join(d, "._writetest")
                    with open(tp, "w") as f:
                        f.write("x")
                    os.remove(tp)
                    return d
                except OSError:
                    pass
        return os.getcwd()
 
    # ── Logging ───────────────────────────────────────────────────────────────
    def _open_log(self) -> None:
        try:
            self._log_fh = open(self.log_path, "w", buffering=1, encoding="utf-8")
        except OSError as e:
            print(f"  Warning: Cannot open log -- {e}")
 
    def _close_log(self) -> None:
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None
 
    def _log(self, line: str, console: bool = True) -> None:
        if console:
            print(line)
        if self._log_fh:
            self._log_fh.write(line + "\n")
            self._log_fh.flush()
 
    # ── Single-size measurement ───────────────────────────────────────────────
    def _measure_size(self, size_bytes: int) -> Optional[Dict]:
        size_mb = size_bytes / 1024 / 1024
        size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{size_bytes/1024:.1f} KB"
        result: Dict = {"size_bytes": size_bytes, "size_str": size_str}
 
        # ── Core chase patterns (random, stride64, stride256) ──
        for pattern, stride_bytes in [
            ("random_chase", 64),
            ("stride64_chase", 64),
            ("stride256_chase", 256),
        ]:
            try:
                if pattern == "random_chase":
                    buf, n_nodes = build_random_chase(size_bytes, 64, rng=self.rng)
                elif pattern == "stride64_chase":
                    buf, n_nodes = build_stride_chase(size_bytes, 64)
                else:
                    buf, n_nodes = build_stride_chase(size_bytes, 256)
                target_sec, iters = get_timing_budget(size_bytes, self.cfg,
                                                      self.fast, self.quick)
                traversals = calibrate_chase(buf, n_nodes, target_sec)
                samples_ns: List[float] = []
                for _ in range(iters):
                    elapsed = timed_chase(buf, n_nodes, traversals)
                    accesses = n_nodes * traversals
                    samples_ns.append(elapsed / accesses * 1e9)
                    time.sleep(0.05)
                result[pattern] = percentile_stats(samples_ns)
                # total_traversal_ms: wall-clock cost of touching every node once
                result[pattern]["total_traversal_ms"] = (
                    result[pattern]["median"] * n_nodes / 1_000_000
                )
                del buf
                gc.collect()
            except Exception as e:
                result[pattern] = {"error": str(e)}
 
        # ── Write RFO (v2: serialized write-chase) ──
        try:
            buf, n_nodes = build_write_rfo(size_bytes, rng=self.rng)
            target_sec, iters = get_timing_budget(size_bytes, self.cfg,
                                                  self.fast, self.quick)
            traversals = calibrate_write_rfo(buf, n_nodes, target_sec)
            samples_ns = []
            for _ in range(iters):
                elapsed = timed_write_rfo(buf, n_nodes, traversals)
                accesses = n_nodes * traversals
                samples_ns.append(elapsed / accesses * 1e9)
                time.sleep(0.05)
            result["write_rfo"] = percentile_stats(samples_ns)
            result["write_rfo"]["total_traversal_ms"] = (
                result["write_rfo"]["median"] * n_nodes / 1_000_000
            )
            del buf
            gc.collect()
        except Exception as e:
            result["write_rfo"] = {"error": str(e)}
 
        # ── TLB stress patterns ──
        if self.tlb_test:
            for pat_name, page_sz in [("tlb_4k_chase", 4096),
                                       ("tlb_2m_chase", 2 * 1024 * 1024)]:
                if size_bytes < page_sz * 8:
                    continue  # too small for this stride
                try:
                    buf, n_nodes = build_tlb_chase(size_bytes, page_sz, rng=self.rng)
                    target_sec, iters = get_timing_budget(size_bytes, self.cfg,
                                                          self.fast, self.quick)
                    traversals = calibrate_chase(buf, n_nodes, target_sec)
                    samples_ns = []
                    for _ in range(iters):
                        elapsed = timed_chase(buf, n_nodes, traversals)
                        accesses = n_nodes * traversals
                        samples_ns.append(elapsed / accesses * 1e9)
                        time.sleep(0.05)
                    result[pat_name] = percentile_stats(samples_ns)
                    result[pat_name]["total_traversal_ms"] = (
                        result[pat_name]["median"] * n_nodes / 1_000_000
                    )
                    del buf
                    gc.collect()
                except Exception as e:
                    result[pat_name] = {"error": str(e)}
 
        # ── Bandwidth ──
        if self.bandwidth:
            result["bandwidth_gbs"] = measure_bandwidth(size_bytes)
 
        return result
 
    # ── Full run ──────────────────────────────────────────────────────────────
    def run(self) -> List[Dict]:
        self._open_log()
        freq_str = f"{self.freq_ghz:.2f} GHz" if self.freq_ghz else "unknown"
        self._log("=" * 80)
        self._log(f"  MemLat Pro v{VERSION} -- Full Run")
        self._log(f"  Timestamp : {datetime.now().isoformat()}")
        self._log(f"  Platform  : {platform.platform()}")
        self._log(f"  Processor : {self.cpu_info['model']}")
        self._log(f"  CPU Freq  : {freq_str}")
        self._log(f"  Numba     : {'JIT enabled' if HAS_NUMBA else 'Python fallback (slow)'}")
        self._log("=" * 80)
        self._log(
            f"{'Size':<12} {'RAND med':>10} {'S64 med':>10} "
            f"{'S256 med':>10} {'RFO med':>10} {'CV%':>6}")
        self._log("-" * 80)
 
        results: List[Dict] = []
        total = len(self.sizes)
        for i, size in enumerate(self.sizes):
            pct = (i + 1) / total * 100
            bar_w = 30
            filled = int(pct / 100 * bar_w)
            bar = "#" * filled + "-" * (bar_w - filled)
            size_str = format_size(size)
            sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {size_str:<12}")
            sys.stdout.flush()
 
            r = self._measure_size(size)
            if r is None:
                continue
            results.append(r)
 
            def _med(pat: str) -> str:
                v = r.get(pat, {})
                if "error" in v:
                    return "  ERR"
                return f"{v.get('median', 0):10.1f}"
 
            cv = r.get("random_chase", {}).get("cv_pct", 0)
            self._log(
                f"\n{r['size_str']:<12} {_med('random_chase')} {_med('stride64_chase')} "
                f"{_med('stride256_chase')} {_med('write_rfo')} {cv:5.1f}%",
                console=False)
 
        sys.stdout.write("\r" + " " * 70 + "\r")
        sys.stdout.flush()
        print("  Measurement complete.")
        return results
 
    # ── Interconnect ──────────────────────────────────────────────────────────
    def run_interconnect(self) -> Optional[Dict]:
        p = self.cpu_info["p_cores"]
        e = self.cpu_info["e_cores"]
        if not p or not e:
            self._log("  Interconnect: skipped (no hybrid topology detected)")
            return None
        self._log("\n  Measuring P<->E interconnect latency...")
        inter: Dict[str, Optional[float]] = {}
        if len(p) >= 2:
            inter["p_to_p_ns"] = measure_interconnect(p[0], p[1])
        else:
            inter["p_to_p_ns"] = None
        inter["p_to_e_ns"] = measure_interconnect(p[0], e[0])
        if len(e) >= 2:
            inter["e_to_e_ns"] = measure_interconnect(e[0], e[1])
        else:
            inter["e_to_e_ns"] = None
        for k, v in inter.items():
            if v is not None:
                self._log(f"    {k}: {v:.1f} ns")
        return inter
 
    # ── Build summary dict ────────────────────────────────────────────────────
    def build_summary(self, results: List[Dict]) -> Dict:
        """Compute summary stats used by scoring, diagram, and report."""
        l1_thresh = self.cfg.get("l1_p_kb", 32) * 1024
        l2_thresh = self.cfg.get("l2_p_kb", 512) * 1024
        l3_thresh = self.cfg.get("l3_mb", 16) * 1024 * 1024
 
        def median_for_range(lo, hi, pat="random_chase"):
            vals = [r[pat]["median"] for r in results
                    if lo < r["size_bytes"] <= hi
                    and pat in r and isinstance(r[pat], dict)
                    and "error" not in r[pat] and "median" in r[pat]]
            return float(np.median(vals)) if vals else None
 
        l1 = median_for_range(0, l1_thresh)
        l2 = median_for_range(l1_thresh, l2_thresh)
        l3 = median_for_range(l2_thresh, l3_thresh)
        ram = median_for_range(l3_thresh, self.max_bytes)
 
        l2_rand = median_for_range(l1_thresh, l2_thresh, "random_chase")
        l2_stride = median_for_range(l1_thresh, l2_thresh, "stride64_chase")
        pf_benefit = (l2_rand - l2_stride) if (l2_rand and l2_stride) else None
 
        l3_read = median_for_range(l2_thresh, l3_thresh, "random_chase")
        l3_write = median_for_range(l2_thresh, l3_thresh, "write_rfo")
        rfo_overhead = (l3_write - l3_read) if (l3_read and l3_write) else None
 
        # TLB summary
        tlb_4k = median_for_range(l3_thresh, self.max_bytes, "tlb_4k_chase")
        tlb_2m = median_for_range(l3_thresh, self.max_bytes, "tlb_2m_chase")
 
        return {
            "l1_median_ns": l1,
            "l2_median_ns": l2,
            "l3_median_ns": l3,
            "ram_median_ns": ram,
            "prefetcher_benefit_ns": pf_benefit,
            "rfo_overhead_ns": rfo_overhead,
            "tlb_4k_ram_ns": tlb_4k,
            "tlb_2m_ram_ns": tlb_2m,
        }
 
    # ── Analysis ──────────────────────────────────────────────────────────────
    def analyze(self, results: List[Dict],
                inter: Optional[Dict] = None) -> Tuple[Dict, List[Dict], Dict]:
        """ 
        Run full analysis. Returns (summary, boundaries, scores). 
        Also prints everything to console and log. 
        """
        if not results:
            self._log("No results to analyze.")
            return {}, [], {}
 
        summary = self.build_summary(results)
        boundaries = detect_cache_boundaries(results)
        scores = compute_scores(summary, self.cfg)
 
        ns_per_cycle = (1.0 / self.freq_ghz) if (self.freq_ghz and self.freq_ghz > 0.1) else (1.0 / 3.0)
        freq_label = f"{self.freq_ghz:.2f} GHz" if self.freq_ghz else "3.00 GHz (est.)"
 
        self._log("\n" + "=" * 80)
        self._log(f"  ANALYSIS SUMMARY -- MemLat Pro v{VERSION}")
        self._log("=" * 80)
 
        # ── Hierarchy diagram ──
        self._log(draw_hierarchy(summary, self.cfg, self.freq_ghz))
 
        # ── Detailed latency table ──
        self._log("\n  Latency Summary (random chase, median):")
        def _fmt(label, val, exp=None):
            if val is None:
                return f"    {label:<20} No data"
            cycles = val / ns_per_cycle
            s = f"    {label:<20} {val:7.1f} ns  (~{cycles:4.0f} cyc @ {freq_label})"
            if exp:
                delta = val - exp
                tag = "HIGHER" if delta > 3 else ("nominal" if abs(delta) <= 3 else "lower")
                s += f"  [expected ~{exp:.0f}ns {tag}]"
            return s
 
        self._log(_fmt("L1 Cache", summary.get("l1_median_ns"), self.cfg.get("expected_l1_ns")))
        self._log(_fmt("L2 Cache", summary.get("l2_median_ns"), self.cfg.get("expected_l2_ns")))
        self._log(_fmt("L3 Cache", summary.get("l3_median_ns"), self.cfg.get("expected_l3_ns")))
        self._log(_fmt("Main Memory", summary.get("ram_median_ns"), self.cfg.get("expected_ram_ns")))
 
        # ── TLB results ──
        if summary.get("tlb_4k_ram_ns"):
            self._log(f"\n  TLB Stress (RAM region):")
            self._log(f"    4K-stride (1 TLB miss/access): {summary['tlb_4k_ram_ns']:.1f} ns")
            if summary.get("tlb_2m_ram_ns"):
                self._log(f"    2M-stride (L2 TLB stress):     {summary['tlb_2m_ram_ns']:.1f} ns")
            rand_ram = summary.get("ram_median_ns")
            if rand_ram and summary.get("tlb_4k_ram_ns"):
                tlb_delta = summary["tlb_4k_ram_ns"] - rand_ram
                self._log(f"    TLB miss penalty:              +{tlb_delta:.1f} ns vs random-64")
 
        # ── Prefetcher + RFO ──
        pf = summary.get("prefetcher_benefit_ns")
        if pf is not None:
            self._log(f"\n  Prefetcher benefit (L2 region): {pf:.1f} ns")
        rfo = summary.get("rfo_overhead_ns")
        if rfo is not None:
            self._log(f"  L3 RFO overhead:                +{rfo:.1f} ns vs read")
 
        # ── Auto-detected boundaries ──
        if boundaries:
            self._log("\n  Auto-Detected Cache Boundaries:")
            for b in boundaries:
                self._log(f"    {b['label']:<12} at ~{b['size_str']:<10} "
                          f"({b['latency_before_ns']:.1f} -> {b['latency_after_ns']:.1f} ns, "
                          f"+{b['latency_jump_ns']:.1f} ns)")
 
        # ── Score card ──
        self._log("\n  " + "=" * 50)
        self._log("  CPU CACHE SCORE CARD")
        self._log("  " + "=" * 50)
        for cat, score in scores.items():
            grade = _score_grade(score)
            bar = _score_bar(score)
            self._log(f"    {cat:<20} {bar} {score:3d}/100  ({grade})")
        self._log("  " + "=" * 50)
 
        if inter:
            self._log("\n  Interconnect Latency:")
            for k, v in inter.items():
                if v is not None:
                    self._log(f"    {k}: {v:.1f} ns")
 
        # ── Save JSON ──
        try:
            payload = {
                "meta": {
                    "version": VERSION,
                    "timestamp": datetime.now().isoformat(),
                    "platform": platform.platform(),
                    "cpu_model": self.cpu_info["model"],
                    "gen_key": self.cpu_info.get("gen_key"),
                    "freq_ghz": self.freq_ghz,
                    "cache_config": self.cfg,
                    "numba": HAS_NUMBA,
                    "fast_mode": self.fast,
                    "quick_mode": self.quick,
                },
                "summary": summary,
                "scores": scores,
                "boundaries": boundaries,
                "interconnect": inter,
                "results": results,
            }
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self._log(f"\n  JSON saved : {os.path.abspath(self.json_path)}")
        except Exception as e:
            self._log(f"  Warning: Could not save JSON -- {e}")
            payload = {}
 
        self._log(f"  Log saved  : {os.path.abspath(self.log_path)}")
        return summary, boundaries, scores
 
    # ── CSV Export ────────────────────────────────────────────────────────────
    def export_csv(self, results: List[Dict]) -> None:
        if not results:
            return
        all_pats = self.CORE_PATTERNS + (self.TLB_PATTERNS if self.tlb_test else [])
        stat_keys = ["min", "p5", "p25", "median", "p75", "p95", "p99",
                     "max", "mean", "std", "cv_pct"]
        header = ["size_bytes", "size_str"]
        for pat in all_pats:
            for sk in stat_keys:
                header.append(f"{pat}_{sk}")
        if self.bandwidth:
            header.append("bandwidth_gbs")
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for r in results:
                    row: List[Any] = [r["size_bytes"], r["size_str"]]
                    for pat in all_pats:
                        pdata = r.get(pat, {})
                        if "error" in pdata or not pdata:
                            row.extend([""] * len(stat_keys))
                        else:
                            for sk in stat_keys:
                                v = pdata.get(sk)
                                row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "")
                    if self.bandwidth:
                        bw = r.get("bandwidth_gbs")
                        row.append(f"{bw:.2f}" if bw else "")
                    writer.writerow(row)
            self._log(f"  CSV saved  : {os.path.abspath(self.csv_path)}")
        except Exception as e:
            self._log(f"  Warning: Could not save CSV -- {e}")
 
    # ── Raw data table ────────────────────────────────────────────────────────
    def print_raw_data(self, results: List[Dict]) -> None:
        if not results:
            return
        self._log("\n" + "=" * 105)
        self._log("  RAW DATA -- All measurements (median ns per access)")
        self._log("=" * 105)
        hdr = (f"  {'Size':<14} {'Random':>10} {'Stride64':>10} "
               f"{'Stride256':>10} {'WriteRFO':>10} "
               f"{'TLB-4K':>10} {'TLB-2M':>10} {'CV%':>7}")
        if self.bandwidth:
            hdr += f" {'BW GB/s':>8}"
        self._log(hdr)
        self._log("  " + "-" * 103)
        for r in results:
            def _v(pat, stat="median"):
                d = r.get(pat, {})
                if not d or "error" in d:
                    return "         -"
                val = d.get(stat)
                return f"{val:10.2f}" if val is not None else "         -"
 
            cv = r.get("random_chase", {}).get("cv_pct", 0)
            line = (f"  {r['size_str']:<14} {_v('random_chase')} {_v('stride64_chase')} "
                    f"{_v('stride256_chase')} {_v('write_rfo')} "
                    f"{_v('tlb_4k_chase')} {_v('tlb_2m_chase')} {cv:6.1f}%")
            if self.bandwidth:
                bw = r.get("bandwidth_gbs")
                line += f" {bw:8.2f}" if bw else "        -"
            self._log(line)
        self._log("=" * 105)
        n_pats = len(self.CORE_PATTERNS) + (len(self.TLB_PATTERNS) if self.tlb_test else 0)
        self._log(f"  Total: {len(results)} sizes x {n_pats} patterns = "
                  f"{len(results) * n_pats} measurements")
 
    # ── Matplotlib plot ───────────────────────────────────────────────────────
    def plot(self, results: List[Dict]) -> None:
        if not HAS_MATPLOTLIB or len(results) < 3:
            if not HAS_MATPLOTLIB:
                print("  matplotlib not installed -- skipping plot")
            return
        sizes_mb = [r["size_bytes"] / 1024 / 1024 for r in results]
        max_smb = max(sizes_mb) if sizes_mb else 1024
 
        def _get(pat, stat):
            out = []
            for r in results:
                v = r.get(pat, {})
                out.append(v.get(stat) if v and "error" not in v else None)
            return out
 
        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor("#0a0a0f")
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.30)
        ax_main  = fig.add_subplot(gs[0, :])
        ax_write = fig.add_subplot(gs[1, 0])
        ax_pf    = fig.add_subplot(gs[1, 1])
        axes = [ax_main, ax_write, ax_pf]
 
        for ax in axes:
            ax.set_facecolor("#0a0a0f")
            ax.tick_params(colors="#888", labelsize=8)
            ax.spines[:].set_color("#333")
            ax.grid(True, alpha=0.15, which="both", color="#555")
            ax.set_xscale("log", base=2)
 
        l1_mb = self.cfg.get("l1_p_kb", 32) / 1024
        l2_mb = self.cfg.get("l2_p_kb", 512) / 1024
        l3_mb = self.cfg.get("l3_mb", 16)
        shade_spec = [
            (0.001, l1_mb, "#00ff00", "L1"),
            (l1_mb, l2_mb, "#0088ff", "L2"),
            (l2_mb, l3_mb, "#ff8800", "L3"),
            (l3_mb, max_smb * 2, "#ff2222", "RAM"),
        ]
        for ax in axes:
            for lo, hi, col, lbl in shade_spec:
                ax.axvspan(lo, hi, alpha=0.06, color=col, zorder=0)
 
        colors_map = {
            "random_chase":    ("#ff4444", "Random chase"),
            "stride64_chase":  ("#44aaff", "Stride-64"),
            "stride256_chase": ("#ffaa00", "Stride-256"),
            "write_rfo":       ("#cc44ff", "Write/RFO"),
            "tlb_4k_chase":    ("#44ff88", "TLB 4K stride"),
            "tlb_2m_chase":    ("#ff88cc", "TLB 2M stride"),
        }
        for pat, (col, lbl) in colors_map.items():
            med = _get(pat, "median")
            p5 = _get(pat, "p5")
            p95 = _get(pat, "p95")
            xs = [s for s, v in zip(sizes_mb, med) if v is not None]
            ys = [v for v in med if v is not None]
            y5 = [v for v in p5 if v is not None]
            y95 = [v for v in p95 if v is not None]
            if not xs:
                continue
            ax_main.plot(xs, ys, "o-", color=col, linewidth=1.8,
                         markersize=3, label=lbl, zorder=3)
            if len(y5) == len(xs) and len(y95) == len(xs):
                ax_main.fill_between(xs, y5, y95, color=col, alpha=0.10, zorder=2)
 
        exp_l3 = self.cfg.get("expected_l3_ns")
        exp_ram = self.cfg.get("expected_ram_ns")
        if exp_l3:
            ax_main.axhline(exp_l3, color="#ff8800", ls="--", lw=0.8, alpha=0.5)
        if exp_ram:
            ax_main.axhline(exp_ram, color="#ff2222", ls="--", lw=0.8, alpha=0.5)
 
        freq_str = f"{self.freq_ghz:.2f} GHz" if self.freq_ghz else "freq unknown"
        ax_main.set_ylabel("Latency (ns)", color="#ccc", fontsize=9)
        ax_main.set_xlabel("Working Set (MB)", color="#ccc", fontsize=9)
        ax_main.set_title(
            f"MemLat Pro -- {self.cpu_info['model']} @ {freq_str}\n"
            f"(shaded = p5-p95; 6 access patterns)",
            color="#ddd", fontsize=10, pad=8)
        ax_main.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="#ccc",
                       edgecolor="#444", loc="upper left")
 
        for mb, lbl in [(l1_mb, "L1"), (l2_mb, "L2"), (l3_mb, "L3")]:
            if mb > 0:
                ax_main.axvline(mb, color="#666", lw=0.8, ls=":")
 
        # Write overhead subplot
        rand_med = _get("random_chase", "median")
        write_med = _get("write_rfo", "median")
        delta_x, delta_y = [], []
        for s, rm, wm in zip(sizes_mb, rand_med, write_med):
            if rm and wm:
                delta_x.append(s)
                delta_y.append(wm - rm)
        if delta_x:
            ax_write.bar(delta_x, delta_y,
                         width=[s * 0.3 for s in delta_x],
                         color="#cc44ff", alpha=0.7)
        ax_write.set_ylabel("RFO overhead (ns)", color="#ccc", fontsize=8)
        ax_write.set_xlabel("Working Set (MB)", color="#ccc", fontsize=8)
        ax_write.set_title("Write/RFO Overhead vs Read", color="#ddd", fontsize=9)
 
        # Prefetcher benefit subplot
        r_med = _get("random_chase", "median")
        s_med = _get("stride64_chase", "median")
        pf_x, pf_y = [], []
        for s, rm, sm in zip(sizes_mb, r_med, s_med):
            if rm and sm:
                pf_x.append(s)
                pf_y.append(rm - sm)
        if pf_x:
            ax_pf.plot(pf_x, pf_y, "s-", color="#44aaff", lw=1.6, ms=4)
        ax_pf.axhline(0, color="#888", lw=0.6)
        ax_pf.set_ylabel("Benefit (ns)", color="#ccc", fontsize=8)
        ax_pf.set_xlabel("Working Set (MB)", color="#ccc", fontsize=8)
        ax_pf.set_title("Prefetcher Benefit (random - stride64)", color="#ddd", fontsize=9)
 
        try:
            plt.savefig(self.plot_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            self._log(f"  Plot saved : {os.path.abspath(self.plot_path)}")
        except Exception as e:
            print(f"  Warning: Could not save plot -- {e}")
        print("  Displaying plot (close window to exit)...")
        plt.show()
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 14 — Interactive HTML Dashboard
# ══════════════════════════════════════════════════════════════════════════════
def generate_html_report(payload: Dict, html_path: str) -> None:
    """Generate a self-contained interactive HTML report with Chart.js."""
    meta = payload.get("meta", {})
    summary = payload.get("summary", {})
    scores = payload.get("scores", {})
    boundaries = payload.get("boundaries", [])
    results = payload.get("results", [])
 
    # Prepare chart data as JSON
    chart_data = []
    for r in results:
        point = {"size_mb": r["size_bytes"] / 1024 / 1024, "size_str": r.get("size_str", "")}
        for pat in ["random_chase", "stride64_chase", "stride256_chase",
                     "write_rfo", "tlb_4k_chase", "tlb_2m_chase"]:
            d = r.get(pat, {})
            if d and "error" not in d:
                point[pat] = d.get("median")
                point[f"{pat}_p5"] = d.get("p5")
                point[f"{pat}_p95"] = d.get("p95")
                point[f"{pat}_total_ms"] = d.get("total_traversal_ms")
            else:
                point[pat] = None
                point[f"{pat}_total_ms"] = None
        bw = r.get("bandwidth_gbs")
        point["bandwidth"] = bw
        chart_data.append(point)
 
    data_json = json.dumps({
        "meta": meta,
        "summary": summary,
        "scores": scores,
        "boundaries": boundaries,
        "chart_data": chart_data,
    })
 
    overall = scores.get("Overall", 0)
    grade = _score_grade(overall)
    freq_str = f"{meta.get('freq_ghz', 0):.2f} GHz" if meta.get("freq_ghz") else "N/A"
 
    # Build score cards HTML
    score_cards_html = ""
    for cat, sc in scores.items():
        if cat == "Overall":
            continue
        g = _score_grade(sc)
        pct = sc
        score_cards_html += f"""
        <div class="score-card">
            <div class="score-ring" style="--pct:{pct}">
                <span class="score-val">{sc}</span>
            </div>
            <div class="score-label">{cat}</div>
            <div class="score-grade">Grade: {g}</div>
        </div>"""
 
    # Build boundaries HTML
    boundaries_html = ""
    if boundaries:
        boundaries_html = "<h2>Auto-Detected Cache Boundaries</h2><table><tr><th>Transition</th><th>Size</th><th>Before (ns)</th><th>After (ns)</th><th>Jump</th></tr>"
        for b in boundaries:
            boundaries_html += (f"<tr><td>{b.get('label','?')}</td><td>{b['size_str']}</td>"
                                f"<td>{b['latency_before_ns']:.1f}</td>"
                                f"<td>{b['latency_after_ns']:.1f}</td>"
                                f"<td>+{b['latency_jump_ns']:.1f} ns</td></tr>")
        boundaries_html += "</table>"
 
    # Build raw data table HTML
    def _td(row, pat):
        d = row.get(pat, {})
        if d and "error" not in d and "median" in d:
            return f"<td>{d['median']:.2f}</td>"
        return "<td>-</td>"
 
    raw_rows = ""
    for r in results:
        raw_rows += f"<tr><td>{r.get('size_str','')}</td>"
        for p in ["random_chase", "stride64_chase", "stride256_chase",
                   "write_rfo", "tlb_4k_chase", "tlb_2m_chase"]:
            raw_rows += _td(r, p)
        raw_rows += "</tr>"
 
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MemLat Pro Report -- {meta.get('cpu_model','CPU')}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {{ --bg: #0a0a0f; --card: #12121e; --border: #2a2a3e; --text: #d0d0e0;
         --accent: #ff4444; --blue: #44aaff; --orange: #ffaa00; --purple: #cc44ff;
         --green: #44ff88; --pink: #ff88cc; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
        line-height:1.6; padding:20px; max-width:1400px; margin:0 auto; }}
h1 {{ color:#fff; font-size:1.8em; margin-bottom:4px; }}
h2 {{ color:#ccc; font-size:1.2em; margin:24px 0 12px; border-bottom:1px solid var(--border); padding-bottom:6px; }}
.header {{ background:linear-gradient(135deg,#1a1a2e,#16213e); border-radius:12px;
           padding:24px 32px; margin-bottom:24px; border:1px solid var(--border); }}
.header-sub {{ color:#888; font-size:0.9em; }}
.overall-badge {{ display:inline-block; background:var(--accent); color:#fff; font-size:2em;
                  font-weight:bold; width:80px; height:80px; line-height:80px; text-align:center;
                  border-radius:50%; float:right; margin-top:-10px; }}
.stats-row {{ display:flex; gap:16px; flex-wrap:wrap; margin:16px 0; }}
.stat-box {{ background:var(--card); border:1px solid var(--border); border-radius:8px;
             padding:12px 20px; min-width:140px; flex:1; }}
.stat-box .val {{ font-size:1.4em; font-weight:bold; color:#fff; }}
.stat-box .lbl {{ font-size:0.8em; color:#888; }}
.scores-grid {{ display:flex; gap:16px; flex-wrap:wrap; justify-content:center; margin:16px 0; }}
.score-card {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
               padding:16px; text-align:center; width:130px; }}
.score-ring {{ width:70px; height:70px; border-radius:50%; margin:0 auto 8px;
               background:conic-gradient(var(--accent) calc(var(--pct)*1%),var(--border) 0);
               display:flex; align-items:center; justify-content:center; position:relative; }}
.score-ring::after {{ content:''; width:54px; height:54px; border-radius:50%; background:var(--card); position:absolute;
                      top:8px; left:8px; }}
.score-val {{ font-size:1.1em; font-weight:bold; color:#fff; z-index:1; }}
.score-label {{ font-size:0.75em; color:#aaa; }}
.score-grade {{ font-size:0.85em; font-weight:bold; color:var(--accent); }}
.chart-container {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
                    padding:16px; margin:16px 0; }}
canvas {{ max-height:500px; }}
table {{ width:100%; border-collapse:collapse; font-size:0.8em; margin:12px 0; }}
th {{ background:#1a1a2e; color:#aaa; padding:8px 12px; text-align:left; border-bottom:2px solid var(--border); }}
td {{ padding:6px 12px; border-bottom:1px solid #1a1a2e; }}
tr:hover td {{ background:#16213e; }}
.footer {{ text-align:center; color:#555; font-size:0.75em; margin-top:32px; }}
.better {{ color: #44ff88; font-weight: bold; }}
.worse  {{ color: #ff4444; font-weight: bold; }}
.neutral {{ color: #aaa; }}
.impact-table td {{ vertical-align: top; padding: 10px 14px; }}
.impact-table tr:hover td {{ background: #16213e; }}
.impact-breakdown {{ color: #888; font-size: 0.78em; line-height: 1.8; }}
.impact-breakdown span {{ display: inline-block; min-width: 90px; }}
.scenario-name {{ font-weight: bold; color: #e0e0f0; }}
.scenario-desc {{ color: #888; font-size: 0.8em; }}
.ref-note {{ background: #12121e; border: 1px solid #2a2a3e; border-radius: 8px;
             padding: 10px 16px; font-size: 0.8em; color: #888; margin: 8px 0 16px; }}
.ref-note strong {{ color: #aaa; }}
@media(max-width:800px){{.scores-grid{{gap:8px;}}.score-card{{width:100px;}}
  .impact-breakdown {{ display: none; }} }}
 
/* ── Tab Navigation ───────────────────────────────────────────────────── */
.tab-bar {{ display:flex; gap:0; margin:0 0 24px; border-bottom:2px solid var(--border); }}
.tab-btn {{ background:none; border:none; color:#666; font-family:inherit; font-size:0.95em;
            padding:12px 24px; cursor:pointer; border-bottom:2px solid transparent;
            margin-bottom:-2px; transition:all 0.25s ease; letter-spacing:0.02em; }}
.tab-btn:hover {{ color:#aaa; background:rgba(255,255,255,0.02); }}
.tab-btn.active {{ color:#fff; border-bottom-color:var(--accent); }}
.tab-panel {{ display:none; animation:tabFadeIn 0.35s ease; }}
.tab-panel.active {{ display:block; }}
@keyframes tabFadeIn {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:translateY(0); }} }}
 
/* ── Why Latency Matters tab ──────────────────────────────────────────── */
.wlm-video {{ display:flex; align-items:center; gap:16px; background:#0d0d18; border:1px solid var(--border);
              border-radius:10px; padding:16px 22px; margin:0 0 24px; text-decoration:none;
              transition:border-color 0.2s ease; }}
.wlm-video:hover {{ border-color:var(--accent); }}
.wlm-video-icon {{ font-size:2.2em; flex-shrink:0; }}
.wlm-video-text {{ flex:1; }}
.wlm-video-title {{ color:#fff; font-size:0.95em; font-weight:600; }}
.wlm-video-sub {{ color:#888; font-size:0.78em; line-height:1.5; margin-top:2px; }}
.wlm-video-cta {{ color:var(--accent); font-size:0.78em; font-weight:600; margin-top:4px; }}
 
.wlm-summary {{ background:linear-gradient(135deg,#1a0a0a,#1a1a2e); border:1px solid #3a1a1a;
                 border-radius:10px; padding:24px 28px; margin:0 0 24px; }}
.wlm-summary p {{ color:#ddd; font-size:0.95em; line-height:1.8; margin:0; }}
 
.wlm-section {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
                 padding:28px 32px; margin:0 0 20px; }}
.wlm-section h3 {{ color:#fff; font-size:1.15em; margin:0 0 14px; padding-bottom:8px;
                    border-bottom:1px solid var(--border); }}
.wlm-section p {{ color:var(--text); font-size:0.88em; line-height:1.75; margin:0 0 12px; }}
.wlm-section p:last-child {{ margin-bottom:0; }}
.wlm-highlight {{ color:#fff; font-weight:600; }}
.wlm-accent {{ color:var(--accent); font-weight:600; }}
.wlm-blue {{ color:var(--blue); }}
.wlm-green {{ color:var(--green); }}
.wlm-orange {{ color:var(--orange); }}
.wlm-purple {{ color:var(--purple); }}
 
.wlm-cols {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin:16px 0; }}
@media(max-width:800px){{ .wlm-cols {{ grid-template-columns:1fr; }} }}
.wlm-card {{ background:#0d0d18; border:1px solid var(--border); border-radius:8px; padding:20px 22px; }}
.wlm-card h4 {{ color:#fff; font-size:0.95em; margin:0 0 10px; }}
.wlm-card p {{ font-size:0.82em; }}
 
.wlm-ladder {{ display:flex; flex-direction:column; gap:0; margin:16px 0; }}
.wlm-rung {{ display:flex; align-items:stretch; min-height:54px; }}
.wlm-rung-bar {{ width:4px; flex-shrink:0; border-radius:2px; }}
.wlm-rung-body {{ padding:8px 0 8px 16px; flex:1; }}
.wlm-rung-title {{ color:#fff; font-size:0.88em; font-weight:600; }}
.wlm-rung-desc {{ color:#888; font-size:0.78em; line-height:1.6; }}
 
.wlm-takeaway {{ background:linear-gradient(135deg,#1a1a2e,#16213e); border:1px solid var(--border);
                  border-radius:10px; padding:24px 28px; margin:20px 0 0; }}
.wlm-takeaway h3 {{ border-bottom:none; padding-bottom:0; margin-bottom:10px; color:#fff; font-size:1.15em; }}
.wlm-takeaway p {{ color:var(--text); font-size:0.9em; line-height:1.75; margin:0 0 10px; }}
.wlm-takeaway p:last-child {{ margin-bottom:0; }}
</style>
</head>
<body>
<div class="header">
    <div class="overall-badge">{grade}</div>
    <h1>MemLat Pro Report</h1>
    <div class="header-sub">{meta.get('cpu_model','Unknown CPU')} @ {freq_str}</div>
    <div class="header-sub">{meta.get('platform','')} | {meta.get('timestamp','')[:19]}</div>
    <div class="header-sub">Overall Score: {overall}/100 | Numba: {'Yes' if meta.get('numba') else 'No'}</div>
</div>
 
<div class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('results')">📊 Results</button>
    <button class="tab-btn" onclick="switchTab('whylat')">🧠 Why Latency Matters</button>
</div>
 
<div id="tab-results" class="tab-panel active">
 
<div class="stats-row">
    <div class="stat-box"><div class="val">{f"{summary['l1_median_ns']:.1f} ns" if isinstance(summary.get('l1_median_ns'), (int,float)) else 'N/A'}</div><div class="lbl">L1 Cache</div></div>
    <div class="stat-box"><div class="val">{f"{summary['l2_median_ns']:.1f} ns" if isinstance(summary.get('l2_median_ns'), (int,float)) else 'N/A'}</div><div class="lbl">L2 Cache</div></div>
    <div class="stat-box"><div class="val">{f"{summary['l3_median_ns']:.1f} ns" if isinstance(summary.get('l3_median_ns'), (int,float)) else 'N/A'}</div><div class="lbl">L3 Cache</div></div>
    <div class="stat-box"><div class="val">{f"{summary['ram_median_ns']:.1f} ns" if isinstance(summary.get('ram_median_ns'), (int,float)) else 'N/A'}</div><div class="lbl">RAM</div></div>
</div>
 
<h2>Score Card</h2>
<div class="scores-grid">{score_cards_html}</div>
 
<h2>Latency Curves (all patterns)</h2>
<div class="chart-container"><canvas id="latChart"></canvas></div>
 
<h2>Write/RFO Overhead</h2>
<div class="chart-container"><canvas id="rfoChart"></canvas></div>
 
<h2>Total Time — One Complete Working-Set Sweep (ms)</h2>
<p style="color:#888;font-size:0.85em;margin:-8px 0 12px">
  Wall-clock time to touch every pointer node once across the full buffer.
  Shows how single-sweep cost grows as data spills from L1 → L2 → L3 → RAM.
</p>
<div class="chart-container"><canvas id="sweepChart"></canvas></div>
 
<h2>Real World Impact Estimator</h2>
<div class="ref-note">
  <strong>How this works:</strong> Each app scenario below is modelled as a fixed count of
  <em>serialized dependent loads</em> at each cache level — these are the pointer-chase
  style accesses on the critical path that actually determine perceived latency.
  Your measured median latencies are multiplied by those counts to estimate the
  memory-bound portion of each task. The <em>Reference</em> column uses a generic
  DDR4 mid-range baseline (L1&nbsp;1.2&nbsp;ns · L2&nbsp;3.5&nbsp;ns · L3&nbsp;15&nbsp;ns · RAM&nbsp;80&nbsp;ns).
  Numbers are estimates — real apps vary — but the <em>relative delta</em> is meaningful.
</div>
<div style="overflow-x:auto;">
<table class="impact-table">
<tr>
  <th>Scenario</th>
  <th>This CPU (est.)</th>
  <th>Reference (est.)</th>
  <th>Delta vs Ref</th>
  <th>Access Breakdown (this CPU)</th>
</tr>
<tbody id="impact-tbody"></tbody>
</table>
</div>
 
{boundaries_html}
 
<h2>Raw Data (median ns)</h2>
<div style="overflow-x:auto;">
<table>
<tr><th>Size</th><th>Random</th><th>Stride64</th><th>Stride256</th><th>WriteRFO</th><th>TLB-4K</th><th>TLB-2M</th></tr>
{raw_rows}
</table>
</div>
 
<div class="footer">Generated by MemLat Pro v{VERSION} | {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
 
</div><!-- /tab-results -->
 
<div id="tab-whylat" class="tab-panel">
 
<a class="wlm-video" href="https://www.youtube.com/watch?v=5qjSGEOEaXo" target="_blank" rel="noopener">
  <div class="wlm-video-icon">▶️</div>
  <div class="wlm-video-text">
    <div class="wlm-video-title">Linus Tech Tips — "Does Low Input Latency make you a better Gamer?"</div>
    <div class="wlm-video-sub">ft. BBNO$, TypicalGamer, Khanada &nbsp;·&nbsp; ASUS ROG &nbsp;·&nbsp; AimLabs &nbsp;·&nbsp; Arduino Leonardo latency injection (0–100 ms in 1 ms increments)</div>
    <div class="wlm-video-cta">Watch on YouTube →</div>
  </div>
</a>
 
<div class="wlm-summary">
  <p>
    <span style="color:#fff;font-weight:700;font-size:1.05em;">Studies suggest that the real human latency resolution
    is calculative and predictive, not just sensory. Here's the breakdown:</span>
  </p>
  <p style="margin-top:12px; color:#bbb;">
    The standard claim is that humans can't perceive individual delays below ~30–50 ms. As a statement about
    conscious detection of isolated events, this is true. But controlled testing — including LTT's own results —
    shows that gaming performance degrades measurably at as little as <span class="wlm-highlight">3 ms</span> of
    added input latency in skilled players, well below the threshold where anyone reports <em>feeling</em> a
    difference. The scores drop; the subjects don't know why. This means the system that <em>uses</em> latency
    information is far more precise than the system that <em>consciously detects</em> it.
    <span class="wlm-accent">The resolution is in the calculation, not the sensation.</span>
  </p>
</div>
 
<div class="wlm-section">
  <h3>1 · The Detection Threshold Illusion</h3>
  <p>
    The conscious detection threshold (~35–50 ms, where LTT subjects started commenting "it feels jiggly")
    is <span class="wlm-accent">not</span> the performance threshold. At just 10 ms of added latency, LTT measured
    a <span class="wlm-highlight">~7% aggregate score drop</span>. By 50 ms, over 25%. The input was degrading
    output long before anyone could articulate what was wrong.
  </p>
  <p>
    The conventional "can you feel it?" question measures the wrong thing. The right question is:
    <span class="wlm-highlight">does the perturbation exceed the resolution of the brain's forward model?</span>
    And the answer, empirically, is yes — down to at least 3 ms.
  </p>
</div>
 
<div class="wlm-section">
  <h3>2 · Predictive Resolution — The Brain Computes Faster Than It Senses</h3>
  <p>
    When tracking a moving target — a crosshair, a ball in flight, a cursor on a price chart — the visual
    cortex doesn't just receive frames. It integrates position samples across a time window, fits a trajectory,
    and <span class="wlm-highlight">extrapolates forward</span> to predict where the target will be
    50–100 ms from now.
  </p>
  <p>
    The temporal resolution of this prediction isn't limited by sensory sampling rate. It's limited by the
    <span class="wlm-blue">signal-to-noise ratio across the integration window</span>. The brain performs a
    curve fit, and curve fits resolve timing far finer than the interval between samples. Consider: humans
    localize sound direction using interaural timing differences of
    <span class="wlm-highlight">10–20 microseconds</span>. Nobody "hears" a 10 µs event — but the brainstem
    <em>computes</em> it.
  </p>
  <p>
    The motor-visual tracking loop works identically. When input latency perturbs the feedback stream,
    the brain sees the <span class="wlm-accent">residual</span> — the gap between where the target appeared
    and where the forward model predicted it would be. That residual encodes millisecond-scale disruptions
    because it's a <em>statistical output</em>, not a raw sensory measurement.
  </p>
</div>
 
<div class="wlm-section">
  <h3>3 · Trainability Requires Consistency</h3>
  <p>
    If your brain is running a predictive model, the next question is: what does that model need from the
    system in order to <em>improve</em>? The answer is
    <span class="wlm-highlight">consistency</span>. A system is <em>trainable</em> — meaning a human
    operator can build and refine a high-resolution forward model against it — if and only if its latency
    profile is consistent enough that the model's prediction errors come from the operator's own imprecision,
    not the system's variance.
  </p>
  <p>
    When the system delivers predictable timing, practice works. Each session tightens the forward model's
    parameters — the confidence interval on the prediction shrinks, corrections become smaller and more precise,
    and the operator's output improves measurably over time. The system feels
    <span class="wlm-highlight">"tight"</span> and <span class="wlm-highlight">"responsive"</span>. The
    operator develops trust in it and stops second-guessing their inputs.
  </p>
  <p>
    When the system introduces stochastic variance — latency that shifts unpredictably between frames — the
    forward model can't converge. The operator's corrections chase noise rather than signal. Practice yields
    diminishing returns because the model is being retrained against a moving target.
    The system feels <span class="wlm-accent">"arbitrary and capricious"</span>. Skilled operators describe
    this as the system fighting them — their muscle memory says one thing, the feedback says another, and
    the mismatch erodes both confidence and performance.
  </p>
  <p>
    This is why two systems with <em>identical average latency</em> can feel completely different to use.
    One with tight, consistent frame timing is a system you can train against and master. One with the same
    mean but wider variance is a system that resists mastery — it caps the resolution your forward model
    can achieve, regardless of how much you practice.
  </p>
</div>
 
<div class="wlm-section">
  <h3>4 · The Latency Retina — A Continuity Threshold</h3>
  <p>
    Apple's Retina display concept established that there exists a pixel density beyond which the human eye
    can no longer resolve individual pixels — the image appears continuous rather than discrete. Above that
    threshold, adding more pixels yields no perceptual benefit — the eye has already integrated the discrete
    elements into continuity. Below it, individual pixels become visible artifacts that disrupt the illusion.
  </p>
  <p>
    <span class="wlm-highlight">The same principle applies to latency in time.</span> For any given
    dependent-event chain with a cadence — frames in a game, ticks in a trading engine, transactions at a
    register — there exists a <span class="wlm-accent">jitter ceiling</span> below which the consumer
    (biological or algorithmic) models the stream as continuous, and above which it models it as discrete
    and unpredictable.
  </p>
  <p>
    Below the ceiling, individual timing variations are integrated into a smooth perceptual flow, just as
    sub-threshold pixels merge into continuous imagery. The forward model absorbs the variance as noise
    and produces stable predictions. The operator experiences <span class="wlm-highlight">fluidity</span>
    — that subconscious sense that the system is an extension of intent rather than an intermediary.
  </p>
  <p>
    Above the ceiling, individual timing deviations become resolvable events. The forward model can't
    integrate them — they puncture the continuity. Each one registers as a prediction error that demands
    correction. The experience shifts from <em>flowing</em> to <em>reactive</em>. And critically, this
    threshold is <span class="wlm-accent">not fixed</span> — it varies with the resolution of the consumer's
    predictive model. A trained operator has a lower ceiling (they detect finer disruptions) than a novice,
    just as a higher-resolution display reveals pixel-level artifacts that a lower-resolution display hides.
  </p>
  <p>
    This reframes the engineering question. The goal is not "minimize latency" — it's
    <span class="wlm-highlight">keep jitter below the continuity threshold of your target user's forward
    model</span>. For a casual user, that threshold might be 15–20 ms. For a competitive gamer or
    professional operator, it might be 2–3 ms. For an algorithm, it might be microseconds. The spec
    depends on who — or what — is consuming the stream.
  </p>
</div>
 
<div class="wlm-section">
  <h3>5 · Skill = Model Resolution</h3>
  <p>
    LTT's most revealing finding: when they separated top performers from bottom performers at 3 ms granularity,
    <span class="wlm-highlight">top performers showed a clean linear decline (~3% per 3 ms added)</span>.
    Bottom performers showed noisy, non-monotonic data — sometimes scoring <em>better</em> at 6 ms than at 0 ms.
  </p>
  <div class="wlm-cols">
    <div class="wlm-card">
      <h4><span class="wlm-green">▌</span> Top Performers</h4>
      <p>
        Razor-clean linear decline down to <span class="wlm-highlight">3 ms</span> — the limit of the test
        apparatus, not the limit of the system. Their forward model is calibrated tightly enough that even
        tiny perturbations produce measurable prediction errors. The model is <em>resolving at that grain</em>.
        Their "latency retina" threshold is below 3 ms.
      </p>
    </div>
    <div class="wlm-card">
      <h4><span class="wlm-orange">▌</span> Casual Players</h4>
      <p>
        Noisy, non-monotonic results at fine granularity. Their forward model operates at ~15–20 ms resolution.
        A 3 ms perturbation falls below the model's internal noise floor — below their "latency retina"
        threshold — and gets absorbed as irrelevant variance.
      </p>
    </div>
  </div>
  <p>
    Training doesn't make neurons fire faster. It <span class="wlm-highlight">tightens the parameters of the
    forward model</span> — shrinks the confidence interval on the prediction. A novice's model says "the target
    will be <em>somewhere around here</em>." A pro's model says "the target will be <em>here</em>."
    The tighter bound means smaller input deviations exceed the model's tolerance and trigger corrections.
    The sensitivity <em>is</em> the resolution. The more trained the model, the finer it resolves, the
    lower the continuity threshold drops, and the more demanding the system requirements become to maintain
    the illusion of fluid, unbroken feedback.
  </p>
</div>
 
<div class="wlm-section">
  <h3>6 · Jitter Compounds, Latency Merely Offsets</h3>
  <p>
    Constant latency is something the forward model can calibrate against — it shifts the prediction offset
    and compensates. This is why players adapt to consistent 60 Hz monitors. A fixed delay is invisible
    to the continuity threshold because it doesn't vary.
  </p>
  <p>
    <span class="wlm-accent">Jitter is fundamentally different.</span> Variable latency (12 ms, then 18 ms,
    then 12 ms, then 25 ms) prevents the model from settling on a consistent offset. Each prediction carries
    more uncertainty. The motor system overshoots, then corrects the overcorrection, creating oscillation.
    Players describe this as feeling <span class="wlm-highlight">"muddy"</span> or
    <span class="wlm-highlight">"disconnected"</span> — not "laggy." The issue isn't delay; it's
    unpredictability.
  </p>
  <p>
    Critically, <span class="wlm-highlight">jitter compounds through dependent event chains</span>. Each motor
    correction depends on the previous frame's feedback, so a single spike propagates forward as the prediction
    loop reconverges. The cost isn't the spike — it's the
    <span class="wlm-blue">reconvergence time</span> across multiple subsequent frames.
  </p>
  <p>
    LTT confirmed this compounding effect: adding 50 ms of input latency didn't add 50 ms to measured reaction
    time — it added <span class="wlm-highlight">~100 ms</span>. That's the signature of a corrupted prediction
    chain oscillating toward reconvergence.
  </p>
</div>
 
<div class="wlm-section">
  <h3>7 · Your Memory Subsystem Creates Jitter</h3>
  <p>
    This is where your MemLat Pro results connect directly to perceived system quality. The cache hierarchy
    is a <span class="wlm-highlight">latency cliff</span>, not a latency slope:
  </p>
  <div class="wlm-ladder">
    <div class="wlm-rung">
      <div class="wlm-rung-bar" style="background:var(--accent);"></div>
      <div class="wlm-rung-body">
        <div class="wlm-rung-title">L1 Cache — ~1–2 ns</div>
        <div class="wlm-rung-desc">Hot data. The forward model's happy path. Everything feels instant.</div>
      </div>
    </div>
    <div class="wlm-rung">
      <div class="wlm-rung-bar" style="background:var(--blue);"></div>
      <div class="wlm-rung-body">
        <div class="wlm-rung-title">L2 Cache — ~3–4 ns</div>
        <div class="wlm-rung-desc">Warm data. Still fast. Minimal perturbation to frame timing.</div>
      </div>
    </div>
    <div class="wlm-rung">
      <div class="wlm-rung-bar" style="background:var(--orange);"></div>
      <div class="wlm-rung-body">
        <div class="wlm-rung-title">L3 Cache — ~10–15 ns</div>
        <div class="wlm-rung-desc">Shared across cores. Contention from other threads introduces variance here.</div>
      </div>
    </div>
    <div class="wlm-rung">
      <div class="wlm-rung-bar" style="background:var(--purple);"></div>
      <div class="wlm-rung-body">
        <div class="wlm-rung-title">RAM — ~50–80 ns</div>
        <div class="wlm-rung-desc">The cliff. 40–60× slower than L1. When working sets suddenly spill past L3,
          frame timing can shift by milliseconds — enough to breach the continuity threshold and corrupt
          the prediction model.</div>
      </div>
    </div>
  </div>
  <p>
    The problem isn't steady-state RAM access — the brain can adapt to that. The problem is
    <span class="wlm-accent">transient cache pressure events</span>: a background process flushing L3,
    a texture streaming burst, a GC pause, the OS scheduler migrating threads. These events are stochastic
    and brief, but they shift frame timing enough to inject prediction-breaking jitter into the feedback loop.
  </p>
  <p>
    Systems with very large L3 caches (like AMD V-Cache) can paradoxically amplify this: steady-state
    performance is superb because everything fits in L3, but when a burst <em>does</em> spill to RAM, the
    cliff is steeper — from ~14 ns to ~65 ns rather than a more gradual degradation. The system trains your
    forward model on L3-speed feedback, then occasionally delivers RAM-speed feedback — the worst pattern
    for predictive consistency, and the most likely to breach the latency retina threshold.
  </p>
</div>
 
<div class="wlm-section">
  <h3>8 · Beyond Gaming — Universal Principle</h3>
  <div class="wlm-cols">
    <div class="wlm-card">
      <h4>⚡ High-Frequency Trading</h4>
      <p>
        Removes biology entirely; the principle still holds. Two systems with identical mean latency but
        different jitter profiles produce materially different P&amp;L. The market punishes tail latency,
        not median — a sporadic 50 µs spike means lost queue position on exactly the trades that matter.
      </p>
    </div>
    <div class="wlm-card">
      <h4>🏪 Point-of-Sale Systems</h4>
      <p>
        Trivial computational load, but dozens of background services contend for cache. Micro-evictions
        compound across hundreds of lookups per transaction. The cashier's muscle memory expects instant
        feedback — when the <em>rhythm</em> breaks, the system feels sluggish even though no single
        delay was perceptible.
      </p>
    </div>
    <div class="wlm-card">
      <h4>🎵 Music Production</h4>
      <p>
        Musicians detect timing inconsistency in MIDI playback at 3–5 ms because their internal rhythmic
        model resolves at that grain. Jitter in the audio pipeline is more disruptive than consistent latency.
      </p>
    </div>
    <div class="wlm-card">
      <h4>🏥 Surgical Robotics</h4>
      <p>
        Surgeons operating remote manipulators develop a predictive model of instrument response. Latency
        spikes during haptic feedback produce the same oscillatory overcorrection seen in gaming — but the
        stakes are incomparably higher.
      </p>
    </div>
  </div>
</div>
 
<div class="wlm-takeaway">
  <h3>📌 Engineering Target</h3>
  <p>
    For any system involving a trained human operator or latency-sensitive algorithm in a tight feedback loop,
    the spec should not be "latency below X ms." It should be
    <span class="wlm-accent">"jitter below Y ms within dependent event chains"</span> — where Y is potentially
    much smaller than X, because you're designing against the predictive model's continuity threshold, not the
    conscious detection threshold.
  </p>
  <p>
    When evaluating your MemLat Pro results, look beyond headline numbers. The <em>absolute</em> latency at each
    cache level matters, but what matters more for perceived responsiveness is
    <span class="wlm-highlight">how steep the cliffs are between levels</span> and
    <span class="wlm-highlight">how often your workload crosses those boundaries unpredictably</span>.
    A system with slightly higher average latency but consistent, predictable access times will feel smoother
    than one with lower median latency but occasional spikes from cache contention, TLB misses, or
    cross-chiplet penalties.
  </p>
  <p style="color:#888; font-size:0.82em; margin-top:14px; margin-bottom:0;">
    Concepts: Predictive Resolution Threshold · Forward-Model Feedback Sensitivity · Latency Retina ·
    Continuity Threshold · Jitter-Induced Reconvergence Cost · Trainability Ceiling
  </p>
</div>
 
<div class="footer" style="margin-top:24px;">Why Latency Matters — MemLat Pro Educational Reference</div>
 
</div><!-- /tab-whylat -->
 
<div class="footer" style="margin-top:16px;">Generated by MemLat Pro v{VERSION} | {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
 
<script>
const D = {data_json};
const cd = D.chart_data;
 
// Latency chart
const patterns = [
    {{key:'random_chase', label:'Random Chase', color:'#ff4444'}},
    {{key:'stride64_chase', label:'Stride-64', color:'#44aaff'}},
    {{key:'stride256_chase', label:'Stride-256', color:'#ffaa00'}},
    {{key:'write_rfo', label:'Write/RFO', color:'#cc44ff'}},
    {{key:'tlb_4k_chase', label:'TLB 4K', color:'#44ff88'}},
    {{key:'tlb_2m_chase', label:'TLB 2M', color:'#ff88cc'}},
];
 
const labels = cd.map(p => p.size_str);
const datasets = patterns.map(p => ({{
    label: p.label,
    data: cd.map(d => d[p.key]),
    borderColor: p.color,
    backgroundColor: p.color + '20',
    borderWidth: 2,
    pointRadius: 2,
    tension: 0.3,
    spanGaps: true,
}}));
 
new Chart(document.getElementById('latChart'), {{
    type: 'line',
    data: {{ labels, datasets }},
    options: {{
        responsive: true,
        interaction: {{ mode: 'index', intersect: false }},
        scales: {{
            x: {{ ticks: {{ color: '#888', maxTicksLimit: 20 }}, grid: {{ color: '#222' }} }},
            y: {{ title: {{ display: true, text: 'Latency (ns)', color: '#888' }},
                  ticks: {{ color: '#888' }}, grid: {{ color: '#222' }},
                  type: 'logarithmic' }}
        }},
        plugins: {{ legend: {{ labels: {{ color: '#aaa' }} }},
                    tooltip: {{ backgroundColor: '#1a1a2e', titleColor: '#fff', bodyColor: '#ccc' }} }}
    }}
}});
 
// RFO overhead chart
const rfoLabels = cd.filter(d => d.random_chase && d.write_rfo).map(d => d.size_str);
const rfoData = cd.filter(d => d.random_chase && d.write_rfo).map(d => d.write_rfo - d.random_chase);
new Chart(document.getElementById('rfoChart'), {{
    type: 'bar',
    data: {{ labels: rfoLabels, datasets: [{{ label: 'RFO Overhead (ns)', data: rfoData,
             backgroundColor: '#cc44ff80', borderColor: '#cc44ff', borderWidth: 1 }}] }},
    options: {{
        responsive: true,
        scales: {{
            x: {{ ticks: {{ color: '#888', maxTicksLimit: 20 }}, grid: {{ color: '#222' }} }},
            y: {{ title: {{ display: true, text: 'Overhead (ns)', color: '#888' }},
                  ticks: {{ color: '#888' }}, grid: {{ color: '#222' }} }}
        }},
        plugins: {{ legend: {{ labels: {{ color: '#aaa' }} }} }}
    }}
}});
 
// ── Total Sweep Time chart ──────────────────────────────────────────────────
const sweepPatterns = [
    {{key:'random_chase_total_ms',    label:'Random Chase',  color:'#ff4444'}},
    {{key:'stride64_chase_total_ms',  label:'Stride-64',     color:'#44aaff'}},
    {{key:'write_rfo_total_ms',       label:'Write/RFO',     color:'#cc44ff'}},
    {{key:'tlb_4k_chase_total_ms',    label:'TLB 4K',        color:'#44ff88'}},
];
const sweepDatasets = sweepPatterns.map(p => ({{
    label: p.label,
    data: cd.map(d => d[p.key]),
    borderColor: p.color,
    backgroundColor: p.color + '18',
    borderWidth: 2,
    pointRadius: 2,
    tension: 0.3,
    spanGaps: true,
}}));
new Chart(document.getElementById('sweepChart'), {{
    type: 'line',
    data: {{ labels: cd.map(d => d.size_str), datasets: sweepDatasets }},
    options: {{
        responsive: true,
        interaction: {{ mode: 'index', intersect: false }},
        scales: {{
            x: {{ ticks: {{ color: '#888', maxTicksLimit: 20 }}, grid: {{ color: '#222' }} }},
            y: {{ title: {{ display: true, text: 'Total sweep time (ms)', color: '#888' }},
                  ticks: {{ color: '#888' }}, grid: {{ color: '#222' }},
                  type: 'logarithmic' }}
        }},
        plugins: {{
            legend: {{ labels: {{ color: '#aaa' }} }},
            tooltip: {{
                backgroundColor: '#1a1a2e', titleColor: '#fff', bodyColor: '#ccc',
                callbacks: {{
                    label: ctx => {{
                        const v = ctx.parsed.y;
                        if (v === null || v === undefined) return null;
                        return ` ${{ctx.dataset.label}}: ${{v < 1 ? (v*1000).toFixed(1)+' µs' : v.toFixed(2)+' ms'}}`;
                    }}
                }}
            }}
        }}
    }}
}});
 
// ── Real World Impact Estimator ─────────────────────────────────────────────
// Access counts = estimated serialized dependent-load critical-path hops.
// These represent pointer-chase style dependencies, not total memory ops.
// Source for ballpark counts: profiling common Windows/Linux app cold-starts
// via perf/VTune; L1/L2 dominated by hot code, L3/RAM by cold data & page faults.
const SCENARIOS = [
    {{ name: '📝 Open Notepad', desc: 'Small app, cold launch',
       l1: 1000,  l2: 5000,   l3: 20000,  ram: 800  }},
    {{ name: '💾 Save Document', desc: 'File flush + UI refresh',
       l1: 500,   l2: 2000,   l3: 5000,   ram: 200  }},
    {{ name: '🌐 Open Chrome Tab', desc: 'New tab + JS engine warm-up',
       l1: 5000,  l2: 20000,  l3: 80000,  ram: 5000 }},
    {{ name: '🎮 Game Frame @ 60 fps', desc: 'Memory-bound portion of 16.67 ms budget',
       l1: 1000,  l2: 300,    l3: 100,    ram: 20   }},
    {{ name: '💻 VS Code — Open File', desc: 'Editor + language server cold path',
       l1: 3000,  l2: 15000,  l3: 60000,  ram: 3000 }},
    {{ name: '⚙️  Compile C++ File', desc: 'Single translation unit, AST heavy',
       l1: 5000,  l2: 30000,  l3: 150000, ram: 12000}},
    {{ name: '🖼️  Decode JPEG Thumbnail', desc: 'Sequential + scatter, ~8 MB working set',
       l1: 2000,  l2: 8000,   l3: 25000,  ram: 1500 }},
];
 
// Generic DDR4 mid-range reference baseline (ns)
const REF = {{ l1: 1.2, l2: 3.5, l3: 15.0, ram: 80.0 }};
 
const measL1  = D.summary.l1_median_ns  || REF.l1;
const measL2  = D.summary.l2_median_ns  || REF.l2;
const measL3  = D.summary.l3_median_ns  || REF.l3;
const measRAM = D.summary.ram_median_ns || REF.ram;
 
function estimateMs(s, lat) {{
    return (s.l1 * lat.l1 + s.l2 * lat.l2 + s.l3 * lat.l3 + s.ram * lat.ram) / 1e6;
}}
 
function fmtMs(ms) {{
    if (ms < 1)   return (ms * 1000).toFixed(0) + ' µs';
    if (ms < 100) return ms.toFixed(1) + ' ms';
    return ms.toFixed(0) + ' ms';
}}
 
const tbody = document.getElementById('impact-tbody');
SCENARIOS.forEach(s => {{
    const thisCpu = estimateMs(s, {{ l1: measL1, l2: measL2, l3: measL3, ram: measRAM }});
    const refCpu  = estimateMs(s, REF);
    const delta   = thisCpu - refCpu;
    const pct     = (delta / refCpu) * 100;
    const faster  = delta < -0.5;
    const slower  = delta >  0.5;
    const cls     = faster ? 'better' : (slower ? 'worse' : 'neutral');
    const arrow   = faster ? '▼' : (slower ? '▲' : '≈');
    const l1_c    = (s.l1  * measL1  / 1e6);
    const l2_c    = (s.l2  * measL2  / 1e6);
    const l3_c    = (s.l3  * measL3  / 1e6);
    const ram_c   = (s.ram * measRAM / 1e6);
    tbody.innerHTML += `
        <tr>
          <td><div class="scenario-name">${{s.name}}</div>
              <div class="scenario-desc">${{s.desc}}</div></td>
          <td style="font-size:1.1em;font-weight:bold;color:#fff">${{fmtMs(thisCpu)}}</td>
          <td style="color:#aaa">${{fmtMs(refCpu)}}</td>
          <td class="${{cls}}">${{arrow}} ${{Math.abs(pct).toFixed(1)}}%
              <div style="font-size:0.75em;font-weight:normal;color:#888">
                ${{fmtMs(Math.abs(delta))}} ${{faster ? 'faster' : (slower ? 'slower' : '')}}
              </div>
          </td>
          <td class="impact-breakdown">
            <span style="color:#ff4444">L1: ${{fmtMs(l1_c)}}</span>
            <span style="color:#44aaff">L2: ${{fmtMs(l2_c)}}</span>
            <span style="color:#ffaa00">L3: ${{fmtMs(l3_c)}}</span>
            <span style="color:#ff6666">RAM: ${{fmtMs(ram_c)}}</span>
          </td>
        </tr>`;
}});
 
// ── Tab Navigation ──────────────────────────────────────────────────────
function switchTab(id) {{
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + id).classList.add('active');
    event.currentTarget.classList.add('active');
    if (id === 'results') {{ window.dispatchEvent(new Event('resize')); }}
}}
</script>
</body>
</html>"""
 
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  HTML saved : {os.path.abspath(html_path)}")
    except Exception as e:
        print(f"  Warning: Could not save HTML -- {e}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 15 — Comparison Mode
# ══════════════════════════════════════════════════════════════════════════════
def _clean_path(p: str) -> str:
    """Strip surrounding quotes and whitespace from pasted Windows paths."""
    return p.strip().strip('"').strip("'").strip()
 
 
def _build_summary_from_results(data: Dict) -> Dict:
    """ 
    Build a summary dict from raw results. 
    Works with both v1 JSON (basic l1/l2/l3/ram_median_ns only) 
    and v2 JSON (full extended summary with prefetcher/rfo fields). 
    Computes any missing fields from the raw results array. 
    """
    summary = dict(data.get("summary", {}))
    if "prefetcher_benefit_ns" in summary:
        return summary
 
    results = data.get("results", [])
    cfg = data.get("meta", {}).get("cache_config", {})
    if not results or not cfg:
        summary.setdefault("prefetcher_benefit_ns", None)
        summary.setdefault("rfo_overhead_ns", None)
        return summary
 
    l1_thresh = cfg.get("l1_p_kb", 32) * 1024
    l2_thresh = cfg.get("l2_p_kb", 512) * 1024
    l3_thresh = cfg.get("l3_mb", 16) * 1024 * 1024
 
    def _median_range(lo, hi, pat):
        vals = [r[pat]["median"] for r in results
                if lo < r.get("size_bytes", 0) <= hi
                and pat in r and isinstance(r[pat], dict)
                and "error" not in r[pat] and "median" in r[pat]]
        return float(np.median(vals)) if vals else None
 
    if summary.get("l1_median_ns") is None:
        summary["l1_median_ns"] = _median_range(0, l1_thresh, "random_chase")
    if summary.get("l2_median_ns") is None:
        summary["l2_median_ns"] = _median_range(l1_thresh, l2_thresh, "random_chase")
    if summary.get("l3_median_ns") is None:
        summary["l3_median_ns"] = _median_range(l2_thresh, l3_thresh, "random_chase")
    if summary.get("ram_median_ns") is None:
        max_b = max((r.get("size_bytes", 0) for r in results), default=0)
        summary["ram_median_ns"] = _median_range(l3_thresh, max_b + 1, "random_chase")
 
    l2_rand = _median_range(l1_thresh, l2_thresh, "random_chase")
    l2_stride = _median_range(l1_thresh, l2_thresh, "stride64_chase")
    if l2_rand and l2_stride:
        summary.setdefault("prefetcher_benefit_ns", l2_rand - l2_stride)
    else:
        summary.setdefault("prefetcher_benefit_ns", None)
 
    l3_read = _median_range(l2_thresh, l3_thresh, "random_chase")
    l3_write = _median_range(l2_thresh, l3_thresh, "write_rfo")
    if l3_read and l3_write:
        summary.setdefault("rfo_overhead_ns", l3_write - l3_read)
    else:
        summary.setdefault("rfo_overhead_ns", None)
 
    return summary
 
 
def compare_runs(path_a: str, path_b: str) -> None:
    """Load two JSON result files and print a side-by-side comparison."""
    path_a = _clean_path(path_a)
    path_b = _clean_path(path_b)
 
    for label, p in [("A", path_a), ("B", path_b)]:
        if not os.path.isfile(p):
            print(f"\n  Error: Run {label} file not found:")
            print(f"         {p}")
            print(f"\n  Hint: Windows 'Copy as Path' adds quotes around the path.")
            print(f"        These are stripped automatically — check the path itself.")
            return
 
    try:
        with open(path_a, "r", encoding="utf-8") as f:
            a = json.load(f)
        with open(path_b, "r", encoding="utf-8") as f:
            b = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  Error: Invalid JSON — {e}")
        return
    except Exception as e:
        print(f"  Error loading files: {e}")
        return
 
    print("\n" + "=" * 76)
    print("  MemLat Pro -- Run Comparison")
    print("=" * 76)
    ma, mb = a.get("meta", {}), b.get("meta", {})
    print(f"  Run A: {ma.get('cpu_model','?')} | {ma.get('timestamp','?')[:19]}")
    print(f"         {os.path.basename(path_a)}")
    print(f"  Run B: {mb.get('cpu_model','?')} | {mb.get('timestamp','?')[:19]}")
    print(f"         {os.path.basename(path_b)}")
    print()
 
    sa = _build_summary_from_results(a)
    sb = _build_summary_from_results(b)
 
    metrics = [
        ("L1 Latency (ns)", "l1_median_ns"),
        ("L2 Latency (ns)", "l2_median_ns"),
        ("L3 Latency (ns)", "l3_median_ns"),
        ("RAM Latency (ns)", "ram_median_ns"),
        ("Prefetcher (ns)", "prefetcher_benefit_ns"),
        ("RFO Overhead (ns)", "rfo_overhead_ns"),
    ]
    print(f"  {'Metric':<24} {'Run A':>10} {'Run B':>10} {'Delta':>10} {'Change':>10}")
    print("  " + "-" * 66)
    for label, key in metrics:
        va = sa.get(key)
        vb = sb.get(key)
        if va is not None and vb is not None:
            delta = vb - va
            pct = (delta / va * 100) if va != 0 else 0
            tag = "WORSE" if delta > 0.5 else ("BETTER" if delta < -0.5 else "same")
            print(f"  {label:<24} {va:10.2f} {vb:10.2f} {delta:+10.2f} {pct:+8.1f}% {tag}")
        else:
            va_s = f"{va:.2f}" if isinstance(va, (int, float)) else "N/A"
            vb_s = f"{vb:.2f}" if isinstance(vb, (int, float)) else "N/A"
            print(f"  {label:<24} {va_s:>10} {vb_s:>10} {'':>10} {'':>10}")
 
    # Score comparison (v2 files have scores; v1 files don't)
    sca, scb = a.get("scores", {}), b.get("scores", {})
    if sca or scb:
        print(f"\n  {'Score Category':<24} {'Run A':>10} {'Run B':>10} {'Delta':>10}")
        print("  " + "-" * 56)
        all_cats = list(dict.fromkeys(list(sca.keys()) + list(scb.keys())))
        for cat in all_cats:
            va = sca.get(cat)
            vb = scb.get(cat)
            va_s = f"{va:>10}" if va is not None else "       N/A"
            vb_s = f"{vb:>10}" if vb is not None else "       N/A"
            if va is not None and vb is not None:
                delta = vb - va
                print(f"  {cat:<24} {va_s} {vb_s} {delta:+10}")
            else:
                print(f"  {cat:<24} {va_s} {vb_s} {'':>10}")
 
    print("\n" + "=" * 76)
    input("\nPress Enter to exit...")
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 16 — Dependency Check & Numba Warmup
# ══════════════════════════════════════════════════════════════════════════════
def check_dependencies() -> bool:
    import importlib
    REQUIRED = [("numpy", "numpy", "1.20")]
    OPTIONAL = [
        ("numba", "numba", "0.55"),
        ("psutil", "psutil", "5.8"),
        ("matplotlib", "matplotlib", "3.3"),
    ]
 
    def _version_ok(installed, minimum):
        try:
            return tuple(int(x) for x in installed.split(".")[:2]) >= \
                   tuple(int(x) for x in minimum.split(".")[:2])
        except Exception:
            return True
 
    def _check(name, min_v):
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "unknown")
            return ver, _version_ok(ver, min_v)
        except ImportError:
            return None, False
 
    print("\n" + "-" * 50)
    print("  Dependency Check")
    print("-" * 50)
    missing_req, missing_opt = [], []
    for name, pip, minv in REQUIRED:
        ver, ok = _check(name, minv)
        if ok:
            print(f"  +  {name:<14} {ver}  (required)")
        else:
            print(f"  x  {name:<14} {'NOT INSTALLED' if ver is None else ver}  (required)")
            missing_req.append(pip)
    for name, pip, minv in OPTIONAL:
        ver, ok = _check(name, minv)
        if ok:
            print(f"  +  {name:<14} {ver}  (optional)")
        elif ver is None:
            print(f"  -  {name:<14} NOT INSTALLED  (optional)")
            missing_opt.append(pip)
        else:
            print(f"  !  {name:<14} {ver}  (optional, need {minv}+)")
            missing_opt.append(pip)
    print("-" * 50)
    if missing_req:
        print("\n  REQUIRED missing: " + ", ".join(missing_req))
        print("  Run:  pip install " + " ".join(missing_req))
    if missing_opt:
        print("  Optional missing: " + ", ".join(missing_opt))
        if "numba" in missing_opt:
            print("  NOTE: Without numba, expect 5-10x slower.")
    if not missing_req and not missing_opt:
        print("\n  All dependencies satisfied.\n")
    return len(missing_req) == 0
 
 
def _warmup_numba() -> None:
    if not HAS_NUMBA:
        return
    print("  Compiling Numba kernels (first run only)...")
    try:
        tb = np.zeros(128, dtype=np.int64)
        tb[0] = 8; tb[8] = 0
        _chase_kernel(tb, 2, 1)
        # Write-RFO kernel now has same signature as chase: (buf, n_nodes, traversals)
        wb = np.zeros(128, dtype=np.int64)
        wb[0] = 8; wb[8] = 0
        _write_rfo_kernel(wb, 2, 1)
        a = np.ones(64, dtype=np.float64)
        b = np.ones(64, dtype=np.float64)
        c = np.ones(64, dtype=np.float64)
        _stream_triad(a, b, c, 0.5, 1)
        del tb, wb, a, b, c
        print("  Numba JIT: ready")
    except Exception as e:
        print(f"  Numba warmup failed ({e}) -- using Python fallback")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 17 — Interactive Menu
# ══════════════════════════════════════════════════════════════════════════════
def show_menu() -> Dict:
    print("\n" + "=" * 72)
    print(f"  MemLat Pro v{VERSION} -- CPU Cache & Memory Latency Profiler")
    print("  Target: Comet Lake (10th gen) + Intel  |  Zen3+ AMD")
    print("=" * 72)
    print()
    print("  Select a test mode:")
    print()
    print("    [1]  Quick   -- Max 256 MB, 30% fewer traversals (~8-10 min)")
    print("    [2]  Full    -- Full sweep up to 1 GB, all patterns (~17 min)")
    print("    [3]  Custom  -- Choose max size, toggle bandwidth/TLB/interconnect")
    print("    [4]  Compare -- Diff two previous JSON result files")
    print("    [5]  Exit")
    print()
    while True:
        choice = input("  Enter choice [1-5]: ").strip()
        if choice in ("1", "2", "3", "4", "5"):
            break
        print("  Invalid choice.")
 
    if choice == "5":
        print("\n  Goodbye.")
        sys.exit(0)
 
    if choice == "4":
        pa = input("  Path to Run A JSON: ").strip().strip('"')
        pb = input("  Path to Run B JSON: ").strip().strip('"')
        return {"mode": "compare", "compare_a": pa, "compare_b": pb}
 
    params: Dict = {
        "mode": "quick" if choice == "1" else ("full" if choice == "2" else "custom"),
        "max_size_mb": QUICK_MAX_MB if choice == "1" else 1024,
        "fast": False,
        "quick": choice == "1",
        "bandwidth": False,
        "interconnect": False,
        "tlb": True,
        "core": None,
        "seed": DEFAULT_RNG_SEED,
        "output": None,
    }
 
    if choice == "3":
        print()
        s = input(f"  Max working set in MB [1024]: ").strip()
        if s:
            try:
                params["max_size_mb"] = int(s)
            except ValueError:
                print("  Invalid -- using 1024 MB.")
        bw = input("  Measure STREAM bandwidth? [y/N]: ").strip().lower()
        params["bandwidth"] = bw in ("y", "yes")
        ic = input("  Measure P<->E interconnect? [y/N]: ").strip().lower()
        params["interconnect"] = ic in ("y", "yes")
        tlb = input("  Include TLB stress tests? [Y/n]: ").strip().lower()
        params["tlb"] = tlb not in ("n", "no")
        core = input("  Pin to core type? [P/E/none]: ").strip().upper()
        if core in ("P", "E"):
            params["core"] = core
        out = input("  Output directory [auto]: ").strip()
        if out:
            params["output"] = out
    elif choice in ("1", "2"):
        bw = input("  Also measure STREAM bandwidth? [y/N]: ").strip().lower()
        params["bandwidth"] = bw in ("y", "yes")
 
    return params
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 18 — Entry Point
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description=f"MemLat Pro v{VERSION}")
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--bandwidth", action="store_true")
    parser.add_argument("--interconnect", action="store_true")
    parser.add_argument("--no-tlb", action="store_true", help="Skip TLB stress tests")
    parser.add_argument("--core", choices=["P", "E"], default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_RNG_SEED)
    parser.add_argument("--no-menu", action="store_true")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                        help="Compare two result files")
    args = parser.parse_args()
 
    # ── Comparison mode ──
    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
        return
 
    # ── Menu vs CLI ──
    has_mode_flag = args.fast or args.quick or args.max_size is not None
    use_menu = not args.no_menu and not has_mode_flag
 
    if use_menu:
        params = show_menu()
        if params.get("mode") == "compare":
            compare_runs(params["compare_a"], params["compare_b"])
            return
        run_fast = params.get("fast", False)
        run_quick = params.get("quick", False)
        max_size_mb = params.get("max_size_mb", 1024)
        bandwidth = params.get("bandwidth", False)
        interconnect = params.get("interconnect", False)
        tlb = params.get("tlb", True)
        core = params.get("core")
        seed = params.get("seed", DEFAULT_RNG_SEED)
        output = params.get("output")
        no_html = False
    else:
        print(f"\n{'=' * 72}")
        print(f"  MemLat Pro v{VERSION}")
        print(f"{'=' * 72}")
        run_fast = args.fast
        run_quick = args.quick
        max_size_mb = args.max_size if args.max_size is not None else 1024
        bandwidth = args.bandwidth
        interconnect = args.interconnect
        tlb = not args.no_tlb
        core = args.core
        seed = args.seed
        output = args.output
        no_html = args.no_html
 
    if not check_dependencies():
        sys.exit(1)
    _warmup_numba()
 
    tester = None
    try:
        tester = MemLatPro(
            max_size_mb=max_size_mb,
            output_dir=output,
            fast=run_fast,
            quick=run_quick,
            bandwidth=bandwidth,
            interconnect=interconnect,
            tlb_test=tlb,
            pin_core_type=core,
            rng_seed=seed,
        )
        t_start = time.perf_counter()
        results = tester.run()
        elapsed = time.perf_counter() - t_start
 
        inter = tester.run_interconnect() if interconnect else None
        summary, boundaries, scores = tester.analyze(results, inter)
        tester.export_csv(results)
        tester.print_raw_data(results)
 
        # HTML report
        if not no_html:
            payload = {
                "meta": {
                    "version": VERSION,
                    "timestamp": datetime.now().isoformat(),
                    "platform": platform.platform(),
                    "cpu_model": tester.cpu_info["model"],
                    "gen_key": tester.cpu_info.get("gen_key"),
                    "freq_ghz": tester.freq_ghz,
                    "cache_config": tester.cfg,
                    "numba": HAS_NUMBA,
                    "fast_mode": run_fast,
                    "quick_mode": run_quick,
                },
                "summary": summary,
                "scores": scores,
                "boundaries": boundaries,
                "interconnect": inter,
                "results": results,
            }
            generate_html_report(payload, tester.html_path)
 
        tester.plot(results)
 
        print(f"\n  Total runtime : {elapsed/60:.1f} min")
        print(f"  Output folder : {os.path.abspath(tester.output_dir)}")
        print(f"  Files: .json  .csv  .html  .png  .txt")
        print("=" * 72)
    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
    except Exception as e:
        print(f"\n  Fatal error: {e}")
        traceback.print_exc()
    finally:
        if tester is not None:
            tester._close_log()
 
    input("\n  Press Enter to exit...")
 
 
if __name__ == "__main__":
    main()
