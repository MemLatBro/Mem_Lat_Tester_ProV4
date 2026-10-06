#!/usr/bin/env python3
""" 
MemLat Pro — Advanced CPU Cache & Memory Latency Profiler v2 
═══════════════════════════════════════════════════════════════ 
Targets Comet Lake (10th gen) and later Intel, Zen3+ AMD. 
 
New in v2: 
  • Write kernel — read-modify-write chase (renamed dirty_writeback in 6.96:
    it measures dirty-line writeback cost, not RFO)
  • TLB patterns — one node per 4 KB / 2 MB page (see 6.98.0 for the layout)
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
  4. dirty_writeback — random chase that also dirties every line it visits;
                       its cost over random_chase is dirty-line writeback
                       (formerly "write_rfo"; it never measured RFO)
  5. tlb_4k_chase    — one node per 4 KB page, on forced 4 KB pages, random line in the page
  6. tlb_2m_chase    — one node per 2 MB page, on 2 MB pages, random line in the page
 
Usage: 
    python memlat_pro.py                       # interactive menu 
    python memlat_pro.py --quick               # quick: 256 MB cap, fewer traversals 
    python memlat_pro.py --fast                # fast: ~5 min 
    python memlat_pro.py --compare a.json b.json   # diff two runs 
    python memlat_pro.py --no-menu --bandwidth     # CLI only, add bandwidth
    python memlat_pro.py --no-menu --rfo           # CLI only, add cross-core RFO test

Changes in 6.96.0 (each checkpoint file carries VERSION "6.96.0-cpN"):
  cp1  Removed the P<->E interconnect test (it timed a single Python store,
       under the GIL, with process-wide affinity -- it measured nothing).
       Numba is now required for every measurement mode; without it the
       script refuses and prints install instructions (compare mode still
       works). The pure-Python fallback kernels were removed.
  cp2  "write_rfo" renamed "dirty_writeback" everywhere (pattern, CSV/JSON
       keys, charts, log). It measures dirty-line writeback cost, never RFO.
       Summary key rfo_overhead_ns -> writeback_overhead_ns; score
       "Write Overhead" -> "Writeback Overhead". Compare mode maps the old
       names, so pre-6.96 files still compare.
       Scoring fix: a worst-case writeback overhead now scores 0 (it used to
       become a neutral 50 through `... or 50`); only missing data is 50.
  cp3  Bandwidth-worker docstrings now describe what the kernel does
       (sequential streaming, not random gathers).
       Run context recorded in both JSON files (meta.run_context /
       run_context): pinning mode and CPUs actually used, SMT state, page
       size + THP settings + huge-page coverage of every latency buffer,
       and a measured core clock (rotate+add chain) before each size --
       flagged approximate when unpinned. Latency buffers now come from
       alloc_i64(): one anonymous mapping per buffer, same huge-page policy
       NumPy applied before, so coverage can be read back per buffer.
  cp4  Headline numbers come from plateau windows instead of every size
       between two boundaries: L1 <= L1/2, L2 2xL1..L2/2, L3 2xL2..L3/2,
       RAM >= 8xL3 (summary_windows / compute_summary). The analysis prints
       which sizes fed each number. When the size cap leaves fewer than two
       sizes >= 8xL3, random-chase-only "RAM plateau" points are added
       (1 GB + 2 GB on a 96 MB L3; respects the 45%-of-free-RAM limit).
       Compare mode recomputes summaries and scores from raw results with
       the same windows, so old and new files are summarised identically.
  cp5  New cross-core RFO test (Section 9A; menu prompt, --rfo). Core B
       dirties a random chain (128 B apart, <= L2/4); core A walks it once
       per round with plain loads and with `lock xadd`, against baselines
       over A's own lines. Reports cache-to-cache read, RFO / ownership
       transfer and the ownership premium. Threads are pinned per thread,
       B spins (no C-state cache flush), timing is TSC inside Numba. Adds a
       pair across CCDs on multi-CCD parts. In JSON as "cross_core_rfo",
       in the HTML, the log and compare mode.
  cp6  HTML reports are self-contained: Chart.js 4.4.0 ships inside this
       script (Section 17B, zlib+base64, SHA-256 checked, MIT notice kept)
       and is inlined into both report types instead of a CDN <script src>.
       Reports now draw their charts offline (each .html is ~205 KB larger).
  (final 6.96.0 = cp6 with VERSION "6.96.0")

Changes in 6.97.0:
  - Pointer-chase kernels keep the chain index unsigned (np.uint64). The
    signed index made Numba emit negative-index wraparound (sar+and) on the
    dependency chain: ~2 extra cycles per hop, ~40% inflation at L1.
  - Buffer builders vectorized (identical buffers, several times faster).
  - RAM headline (Section 5C): DRAM latency measured on 2 MB pages at one
    RAM-plateau size, comparable with MLC / AIDA64. The same size on forced
    4 KB pages is reported and graded separately ("4K Random Access", not in
    Overall); the difference is the page-walk cost. Windows needs the
    'Lock pages in memory' privilege (detected, never granted); without it
    the headline falls back to the ~256 MB point, labelled with its L3
    coverage and the reason. Stored as "page_modes" in the JSON. --no-page-modes skips it.
  - Summary windows respect 4 KB pages: L2 window ends at 256 KB (first-level
    DTLB reach) and, when the sweep runs on 4 KB pages (Windows, or THP off),
    the L3 window ends at 8 MB (second-level TLB reach).
  - Cycle counts use the measured clock (median of per-size probes) instead of
    the OS-reported frequency.

Changes in 6.97.1:
  - Loaded latency test: the latency thread's buffer now uses 2 MB pages by
    default, so each hop is one DRAM access and the result is comparable with
    MLC. 4 KB pages are an option (menu prompt, --ll-pages 4k); on 4 KB pages
    each loaded hop also pays a page walk that goes to DRAM once the
    bandwidth workers evict the page tables from L3. If 2 MB pages cannot be
    allocated the test falls back to 4 KB, says why, and records
    pages_requested / pages_used / fallback reason in the JSON and HTML.
    Bandwidth-worker buffers are unchanged (sequential streams barely touch
    the TLB). Severity wording is unchanged.

Changes in 6.98.0:
  - TLB patterns rebuilt. Before, every node sat at offset 0 of its page, so
    all nodes mapped to the same few cache sets and the two patterns measured
    cache-set conflict misses, not TLB cost (and on Linux the "4K" buffer was
    THP-backed from 4 MB up, so it took no 4 KB TLB misses at all). Now each
    node sits on a random cache line inside its page and the page size is
    forced: tlb_4k_chase on 4 KB pages, tlb_2m_chase on 2 MB pages (skipped
    with the reason when 2 MB pages are unavailable). The "TLB miss penalty"
    line (tlb_4k - random) is gone: both sides had the same TLB behaviour; the
    page-walk cost (4 KB - 2 MB, Section 5C) is the valid number.
    TLB-4K / TLB-2M values are NOT comparable with files from before 6.98;
    compare mode says so. Nothing else in the sweep changed.
  - Results are checkpointed to the JSON after every size and the final JSON
    is written before the analysis is printed, so an interruption or a later
    error can no longer lose a sweep. Every detected boundary gets a label
    (a 5th one used to crash the analysis). A closed stdin no longer crashes
    at the "Press Enter" prompts; menu inputs are validated; a failure in the
    loaded latency test is reported instead of closing the window.
  - Linux: cache sizes come from sysfs (the table is the fallback and still
    supplies the expected latencies); cores are grouped by the L3 they share
    (sysfs) instead of "8 cores per CCD"; the CPU model is read from
    /proc/cpuinfo even when platform.processor() returns "x86_64".
  - STREAM bandwidth: runs on the process's full affinity even when the sweep
    is pinned, builds its arrays without temporaries (3x the working set, was
    ~4x) and is skipped for a size whose arrays would not fit in free RAM.
  - Loaded latency: workers are started with "spawn" (no fork of a
    multi-threaded process), report every pass (was every third), and a worker
    that dies during start-up is detected at once instead of after 90 s.
"""
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 1 — Imports & Constants
# ══════════════════════════════════════════════════════════════════════════════
import argparse
import base64
import csv
import ctypes
import gc
import hashlib
import json
import mmap
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import threading
import zlib
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
 
VERSION = "6.98.0"


def _pause(prompt: str = "\n  Press Enter to exit...") -> None:
    """Keep a double-clicked console window open; a closed stdin (pipe, cron,
    IDE runner) is not an error (6.98)."""
    try:
        input(prompt)
    except EOFError:
        print()
 
 
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
        "l1_p_kb": 32, "l2_p_kb": 512,
        "l3_mb": 96,        # TOTAL L3 (32 base + 64 VCache) — do NOT add v_cache_mb
        "v_cache_mb": 64,   # informational: how much of l3_mb is VCache
        "chiplet": True,
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
        "l1_p_kb": 32, "l2_p_kb": 1024,
        "l3_mb": 96,        # TOTAL L3 (32 base + 64 VCache) — do NOT add v_cache_mb
        "v_cache_mb": 64,   # informational: how much of l3_mb is VCache
        "chiplet": True,
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
        got_model = False
        for line in content.splitlines():
            if "vendor_id" in line and info["vendor"] == "Unknown":
                info["vendor"] = line.split(":", 1)[1].strip()
            # (6.98) always prefer the kernel's model name: platform.processor()
            # returns just "x86_64" on Debian/Ubuntu/Fedora, which hid the model
            # and sent every CPU to the generic config.
            if line.startswith("model name") and not got_model:
                name = line.split(":", 1)[1].strip()
                if name:
                    info["model"] = name
                    got_model = True
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
 
 
def _parse_cache_size_kb(text: Optional[str]) -> Optional[int]:
    """'32K' / '1024K' / '96M' -> KB"""
    m = re.match(r"\s*(\d+)\s*([KMG]?)", text or "", re.IGNORECASE)
    if not m:
        return None
    mult = {"": 1 / 1024, "K": 1, "M": 1024, "G": 1024 * 1024}[m.group(2).upper()]
    kb = int(int(m.group(1)) * mult)
    return kb if kb > 0 else None


def read_sysfs_caches(cpu: int = 0) -> Dict:
    """(6.98) Linux: data/unified cache sizes of one logical CPU and the CPUs
    that share each level, from /sys/devices/system/cpu/cpuN/cache.
    Returns {} where sysfs is not available (other OSes, some containers)."""
    out: Dict[str, Any] = {}
    base = f"/sys/devices/system/cpu/cpu{cpu}/cache"
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        return out
    for idx in entries:
        if not idx.startswith("index"):
            continue
        level = _read_text(f"{base}/{idx}/level")
        ctype = (_read_text(f"{base}/{idx}/type") or "").lower()
        size_kb = _parse_cache_size_kb(_read_text(f"{base}/{idx}/size"))
        if not level or not level.isdigit() or size_kb is None or ctype == "instruction":
            continue
        lvl = int(level)
        if lvl in (1, 2, 3):
            out[f"l{lvl}_kb"] = size_kb
            try:
                out[f"l{lvl}_shared"] = _parse_cpu_list(_read_text(f"{base}/{idx}/shared_cpu_list"))
            except ValueError:
                out[f"l{lvl}_shared"] = []
    return out


def _table_cache_config(cpu_info: Dict) -> Dict:
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


def get_cache_config(cpu_info: Dict, use_sysfs: bool = True) -> Dict:
    """Cache sizes used for test sizes, summary windows and buffer sizing.
    (6.98) On Linux the L1d / L2 / L3 sizes are read from sysfs, so they are
    right for CPUs the table does not know (or misclassifies). The table is the
    fallback (Windows, macOS, no sysfs) and still provides the expected
    latencies, hybrid E-core sizes and notes. cfg["cache_source"] records which
    one was used; cfg["table_cache"] keeps the table values when they differ."""
    cfg = _table_cache_config(cpu_info)
    cfg["cache_source"] = "table"
    if not (use_sysfs and sys.platform.startswith("linux")):
        return cfg
    cpu = (cpu_info.get("p_cores") or [0])[0]
    sc = read_sysfs_caches(cpu)
    if not (sc.get("l1_kb") and sc.get("l2_kb")):
        return cfg
    table = {"l1_p_kb": cfg.get("l1_p_kb"), "l2_p_kb": cfg.get("l2_p_kb"), "l3_mb": cfg.get("l3_mb")}
    cfg["l1_p_kb"] = sc["l1_kb"]
    cfg["l2_p_kb"] = sc["l2_kb"]
    if sc.get("l3_kb"):
        mb = sc["l3_kb"] / 1024
        cfg["l3_mb"] = int(mb) if float(mb).is_integer() else round(mb, 2)
    cfg["cache_source"] = f"sysfs (cpu {cpu})"
    if cpu_info.get("gen_key") is None:
        cfg["notes"] = ("CPU not in the table: cache sizes from sysfs, "
                        "expected latencies are generic.")
    now = {"l1_p_kb": cfg["l1_p_kb"], "l2_p_kb": cfg["l2_p_kb"], "l3_mb": cfg["l3_mb"]}
    if now != table:
        cfg["table_cache"] = table
    return cfg
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 5 — Test Size Generation & Timing Budgets
# ══════════════════════════════════════════════════════════════════════════════
def generate_test_sizes(cfg: Dict, max_bytes: int) -> List[int]:
    sizes = set()
    l1_kb = cfg.get("l1_p_kb", 32)
    l2_kb = cfg.get("l2_p_kb", 512)
    # NOTE: l3_mb already includes V-Cache in the config database
    # (e.g., Zen4_VCACHE: l3_mb=96 = 32 base + 64 VCache)
    # Do NOT add v_cache_mb again — that double-counts.
    l3_mb = cfg.get("l3_mb", 16)
    boundaries_bytes = [l1_kb * 1024, l2_kb * 1024, int(l3_mb * 1024 * 1024)]
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
    l3_thresh = int(cfg.get("l3_mb", 16) * 1024 * 1024)
    if size_bytes <= l2_thresh:
        return TARGET_SEC_L1 * scale, ITERS_L1
    elif size_bytes <= l3_thresh:
        return TARGET_SEC_L2 * scale, ITERS_L2
    elif size_bytes <= l3_thresh * 4:
        return TARGET_SEC_L3 * scale, ITERS_L3
    else:
        return TARGET_SEC_RAM * scale, ITERS_RAM


# ── Summary windows (6.96) ──────────────────────────────────────────────────
# Each headline number is the median of the random chase over sizes that sit
# on that level's plateau, away from both neighbouring transitions:
#   L1  : up to L1/2           L2 : 2xL1 .. L2/2
#   L3  : 2xL2 .. L3/2         RAM: >= 8xL3
# Before 6.96 every size between two boundaries counted, so "RAM" included
# working sets just past L3 that L3 still largely held (on a 96 MB X3D the
# Quick-mode RAM median sat at 144 MB), which made RAM read too low.
RAM_PLATEAU_FACTOR = 8      # RAM figure only from working sets >= 8x L3
RAM_PLATEAU_MIN_POINTS = 2  # extra random-only sizes are added to reach this


L2_WINDOW_MAX_4K = 256 << 10   # first-level DTLB reach with 4 KB pages (6.97)
L3_WINDOW_MAX_4K = 8 << 20     # second-level TLB reach with 4 KB pages (6.97)


def summary_windows(cfg: Dict, small_pages: bool = False) -> Dict[str, Tuple[int, Optional[int]]]:
    """Inclusive (lo, hi) byte windows per level; hi=None means unbounded.
    Buffers in the L2 window are always on 4 KB pages, so it stops at the
    first-level DTLB reach; with small_pages (sweep on 4 KB pages) the L3 window
    also stops at the second-level TLB reach, so page walks stay out of L3."""
    l1 = cfg.get("l1_p_kb", 32) * 1024
    l2 = cfg.get("l2_p_kb", 512) * 1024
    l3 = int(cfg.get("l3_mb", 16) * 1024 * 1024)
    l3_hi = min(l3 // 2, L3_WINDOW_MAX_4K) if small_pages else l3 // 2
    win: Dict[str, Tuple[int, Optional[int]]] = {
        "L1": (1, l1 // 2),
        "L2": (2 * l1, min(l2 // 2, max(2 * l1, L2_WINDOW_MAX_4K))),
        "L3": (2 * l2, max(2 * l2, l3_hi)),
        "RAM": (RAM_PLATEAU_FACTOR * l3, None),
    }
    # Degenerate cache configs (window inverted): fall back to the whole level
    if win["L2"][0] > win["L2"][1]:
        win["L2"] = (l1 + 1, l2)
    if win["L3"][0] > win["L3"][1]:
        win["L3"] = (l2 + 1, l3)
    return win


def _in_window(size: int, win: Tuple[int, Optional[int]]) -> bool:
    lo, hi = win
    return size >= lo and (hi is None or size <= hi)


def _stat_ok(d: Any) -> bool:
    return isinstance(d, dict) and "error" not in d and "median" in d


def ram_plateau_sizes(cfg: Dict, grid_sizes: List[int],
                      limit_bytes: Optional[int]) -> Tuple[List[int], List[str]]:
    """
    Extra working-set sizes (random chase only) needed so the RAM window
    holds at least RAM_PLATEAU_MIN_POINTS sizes. Candidates are the first
    powers of two at or above 8xL3 (1 GB and 2 GB for a 96 MB L3).
    Returns (sizes, notes); sizes over limit_bytes are skipped with a note.
    """
    lo = summary_windows(cfg)["RAM"][0]
    have = [s for s in grid_sizes if s >= lo]
    p1 = 1 << max(0, (lo - 1).bit_length())
    extra: List[int] = []
    notes: List[str] = []
    for cand in (p1, 2 * p1, 4 * p1):
        if len(have) + len(extra) >= RAM_PLATEAU_MIN_POINTS:
            break
        if cand in have:
            continue
        if limit_bytes is not None and cand > limit_bytes:
            notes.append(f"RAM plateau point {cand >> 20} MB skipped: above the "
                         f"memory safety limit ({limit_bytes >> 20} MB)")
            continue
        extra.append(cand)
    return extra, notes


def compute_summary(results: List[Dict], cfg: Dict, small_pages: bool = False) -> Dict:
    """Headline numbers from plateau windows. Shared by runs and compare mode."""
    win = summary_windows(cfg, small_pages)

    def rows(level: str, pat: str) -> List[Dict]:
        return [r for r in results
                if _in_window(r.get("size_bytes", 0), win[level]) and _stat_ok(r.get(pat))]

    def med(rs: List[Dict], pat: str) -> Optional[float]:
        vals = [r[pat]["median"] for r in rs]
        return float(np.median(vals)) if vals else None

    def diff(level: str, pat_a: str, pat_b: str) -> Optional[float]:
        rs = [r for r in rows(level, pat_a) if _stat_ok(r.get(pat_b))]
        return (med(rs, pat_a) - med(rs, pat_b)) if rs else None

    tlb4 = rows("RAM", "tlb_4k_chase")
    return {
        "l1_median_ns": med(rows("L1", "random_chase"), "random_chase"),
        "l2_median_ns": med(rows("L2", "random_chase"), "random_chase"),
        "l3_median_ns": med(rows("L3", "random_chase"), "random_chase"),
        "ram_median_ns": med(rows("RAM", "random_chase"), "random_chase"),
        # same sizes on both sides of each difference
        "prefetcher_benefit_ns": diff("L2", "random_chase", "stride64_chase"),
        "writeback_overhead_ns": diff("L3", "dirty_writeback", "random_chase"),
        "tlb_4k_ram_ns": med(tlb4, "tlb_4k_chase"),
        "tlb_2m_ram_ns": med(rows("RAM", "tlb_2m_chase"), "tlb_2m_chase"),
        # (6.98) "tlb_4k_vs_random_ns" removed: both chases had the same TLB
        # behaviour, so it was not a TLB penalty. See page_walk_ns (Section 5C).
        "windows": {
            level: {"lo_bytes": lo, "hi_bytes": hi,
                    "sizes": [r.get("size_str") for r in rows(level, "random_chase")]}
            for level, (lo, hi) in win.items()
        },
    }

# ══════════════════════════════════════════════════════════════════════════════
#  Section 5B — Run Context: buffer allocation, page backing, SMT, affinity, clock
#  (6.96) Everything here exists so each result file records the conditions it
#  was measured under: page size and huge-page coverage per buffer, SMT state,
#  CPU affinity / the CPU actually used, and a measured core clock.
# ══════════════════════════════════════════════════════════════════════════════
# NumPy itself calls madvise(MADV_HUGEPAGE) on Linux for allocations of 4 MiB
# or more. alloc_i64() keeps exactly that policy so results stay comparable
# with earlier versions; it only adds a dedicated mapping per buffer so the
# huge-page coverage of each buffer can be read back from /proc/self/smaps.
HUGEPAGE_ADVISE_MIN_BYTES = 4 << 20


def alloc_i64(n_elems: int) -> np.ndarray:
    """Zero-filled int64 array backed by its own anonymous, page-aligned mapping."""
    n_elems = int(n_elems)
    nbytes = max(n_elems * 8, mmap.PAGESIZE)
    if sys.platform.startswith("linux"):
        # MAP_PRIVATE matters: Python's default for anonymous mmap is MAP_SHARED,
        # which is shmem-backed and ignores MADV_HUGEPAGE unless shmem THP is on.
        mm = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE)
        if nbytes >= HUGEPAGE_ADVISE_MIN_BYTES and hasattr(mmap, "MADV_HUGEPAGE"):
            try:
                mm.madvise(mmap.MADV_HUGEPAGE)
            except (OSError, AttributeError):
                pass
    else:
        mm = mmap.mmap(-1, nbytes)
    # The array keeps the mapping alive; it is unmapped when the array is freed.
    return np.frombuffer(mm, dtype=np.int64, count=n_elems)


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def _bracketed_choice(text: Optional[str]) -> Optional[str]:
    """'always [madvise] never' -> 'madvise'"""
    if not text:
        return None
    m = re.search(r"\[([^\]]+)\]", text)
    return m.group(1) if m else text


def memory_page_context() -> Dict:
    """System-wide page-size settings that affect every latency buffer."""
    ctx: Dict[str, Any] = {"base_page_kb": mmap.PAGESIZE // 1024}
    if sys.platform.startswith("linux"):
        thp = "/sys/kernel/mm/transparent_hugepage"
        ctx["thp_enabled"] = _bracketed_choice(_read_text(f"{thp}/enabled"))
        ctx["thp_defrag"] = _bracketed_choice(_read_text(f"{thp}/defrag"))
        pmd = _read_text(f"{thp}/hpage_pmd_size")
        ctx["huge_page_kb"] = int(pmd) // 1024 if pmd and pmd.isdigit() else 2048
        ctx["allocation"] = (f"one anonymous mapping per buffer; MADV_HUGEPAGE when "
                             f">= {HUGEPAGE_ADVISE_MIN_BYTES >> 20} MiB (same as NumPy)")
    elif sys.platform == "win32":
        ctx["allocation"] = "one anonymous mapping per buffer; standard pages (large pages not requested)"
    else:
        ctx["allocation"] = "one anonymous mapping per buffer; OS default pages"
    return ctx


def hugepage_coverage_pct(arr: np.ndarray) -> Optional[float]:
    """
    Percent of `arr` backed by transparent huge pages, summed from the
    AnonHugePages fields of the mapping(s) covering it in /proc/self/smaps.
    Linux only; 0.0 on Windows (large pages are never requested), None when
    it cannot be attributed (e.g. the buffer shares a mapping with other data,
    as NumPy-allocated arrays do). Call after the buffer has been written:
    pages are only allocated on first touch.
    """
    nbytes = int(arr.nbytes)
    if sys.platform == "win32":
        return 0.0
    if not sys.platform.startswith("linux"):
        return None
    if nbytes < (2 << 20):
        return 0.0          # smaller than one 2 MB page
    page = mmap.PAGESIZE
    lo_b = int(arr.ctypes.data)
    hi_b = lo_b + ((nbytes + page - 1) // page) * page
    total_kb, found, inside = 0, False, False
    try:
        with open("/proc/self/smaps", encoding="utf-8") as f:
            for line in f:
                head = line.split(" ", 1)[0]
                if "-" in head and line[0] in "0123456789abcdef":
                    s, e = (int(x, 16) for x in head.split("-"))
                    inside = s < hi_b and e > lo_b
                    if inside:
                        if s < lo_b or e > hi_b:
                            return None     # mapping extends past the buffer
                        found = True
                elif inside and line.startswith("AnonHugePages:"):
                    total_kb += int(line.split()[1])
    except Exception:
        return None
    if not found:
        return None
    return round(min(100.0, total_kb * 1024 * 100.0 / nbytes), 1)


def detect_smt_state() -> Dict:
    """Logical vs physical counts plus the kernel's SMT switch where exposed."""
    logical = os.cpu_count()
    physical = None
    if HAS_PSUTIL:
        try:
            physical = psutil.cpu_count(logical=False)
        except Exception:
            physical = None
    st: Dict[str, Any] = {"logical_cpus": logical, "physical_cores": physical,
                          "smt_active": None, "smt_control": None}
    if sys.platform.startswith("linux"):
        active = _read_text("/sys/devices/system/cpu/smt/active")
        if active in ("0", "1"):
            st["smt_active"] = active == "1"
        st["smt_control"] = _read_text("/sys/devices/system/cpu/smt/control")
    if st["smt_active"] is None and logical and physical:
        st["smt_active"] = logical > physical
    return st


def current_affinity() -> Optional[List[int]]:
    """CPUs the calling thread may run on."""
    try:
        if hasattr(os, "sched_getaffinity"):
            return sorted(os.sched_getaffinity(0))
        if HAS_PSUTIL:
            return sorted(psutil.Process().cpu_affinity())
    except Exception:
        pass
    return None


_GETCPU = None


def current_cpu() -> Optional[int]:
    """Logical CPU the calling thread is executing on right now (best effort)."""
    global _GETCPU
    try:
        if _GETCPU is None:
            if sys.platform.startswith("linux"):
                _GETCPU = ctypes.CDLL(None).sched_getcpu
            elif sys.platform == "win32":
                _GETCPU = ctypes.windll.kernel32.GetCurrentProcessorNumber
            else:
                _GETCPU = False
        if _GETCPU:
            cpu = int(_GETCPU())
            return cpu if cpu >= 0 else None
    except Exception:
        _GETCPU = False
    return None


# The probe kernel (Section 7) runs a dependent rotate -> add chain:
# two 1-cycle integer ops per iteration on every current x86 core.
CLOCK_PROBE_CYCLES_PER_ITER = 2


def measure_clock_ghz(n_iter: int = 30_000_000, repeats: int = 3) -> Optional[float]:
    """
    Measure the clock of the core the calling thread is on, by timing a chain
    of dependent 1-cycle ALU ops. Best of `repeats` (interrupts only slow it).
    ~10-20 ms per repeat. If the thread is not pinned, this is the clock of
    whichever core it happened to run on, so treat it as approximate.
    """
    if not HAS_NUMBA:
        return None
    try:
        _clock_probe_kernel(1000, np.uint64(1))          # compiled & warm
        best = 0.0
        for r in range(repeats):
            t0 = time.perf_counter()
            x = _clock_probe_kernel(n_iter, np.uint64(r + 2))
            dt = time.perf_counter() - t0
            if x == 0xDEAD:                               # keep x alive
                print("sentinel")
            if dt > 0:
                best = max(best, CLOCK_PROBE_CYCLES_PER_ITER * n_iter / dt / 1e9)
        return round(best, 3) if best > 0 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Section 5C — Page-size modes: explicit 4 KB vs 2 MB buffers, RAM headline (6.97)
#  The RAM headline is DRAM latency measured on 2 MB pages (no page walks), so it
#  is comparable with MLC/AIDA64. 4 KB-page random access (what big-footprint
#  software pays, page walks included) is reported and graded separately, and
#  the difference is the page-walk cost. Without large pages the headline falls
#  back to the ~256 MB point, labelled with how much of it L3 still covers.
# ══════════════════════════════════════════════════════════════════════════════
import weakref

PAGE_MODE_FALLBACK_BYTES = 256 << 20   # fallback headline working set
LINUX_2M_MIN_COVERAGE_PCT = 90.0       # below this a "2 MB" buffer is not trusted


class LargePageUnavailable(Exception):
    """Large pages could not be used. .reason is short, .hint says how to fix it."""
    def __init__(self, reason: str, hint: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.hint = hint


_WIN_LP_HINT_PRIV = ("Grant 'Lock pages in memory' to your account: secpol.msc -> Local "
                     "Policies -> User Rights Assignment -> Lock pages in memory -> add "
                     "your user, then sign out and back in. If it is already granted, run "
                     "MemLat from an elevated terminal (Run as administrator).")
_WIN_LP_HINT_FRAG = ("Windows could not find enough physically contiguous 2 MB blocks "
                     "(memory fragmentation). Run soon after a reboot, or close "
                     "memory-heavy applications, and try again.")


def _win_is_elevated() -> Optional[bool]:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return None


def _win_enable_lock_memory_privilege() -> Tuple[bool, str]:
    """Switch SeLockMemoryPrivilege on in this process's token. Only works if the
    account already holds the right ('Lock pages in memory'); never grants it."""
    from ctypes import wintypes
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class LUID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Luid", LUID), ("Attributes", wintypes.DWORD)]

    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [("PrivilegeCount", wintypes.DWORD),
                    ("Privileges", LUID_AND_ATTRIBUTES * 1)]

    TOKEN_ADJUST_PRIVILEGES, TOKEN_QUERY, SE_PRIVILEGE_ENABLED = 0x20, 0x8, 0x2
    ERROR_NOT_ALL_ASSIGNED = 1300
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                          ctypes.POINTER(wintypes.HANDLE)]
    advapi32.LookupPrivilegeValueW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR,
                                               ctypes.POINTER(LUID)]
    advapi32.AdjustTokenPrivileges.argtypes = [wintypes.HANDLE, wintypes.BOOL,
                                               ctypes.POINTER(TOKEN_PRIVILEGES),
                                               wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(),
                                     TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                                     ctypes.byref(token)):
        return False, f"OpenProcessToken failed (error {ctypes.get_last_error()})"
    try:
        luid = LUID()
        if not advapi32.LookupPrivilegeValueW(None, "SeLockMemoryPrivilege", ctypes.byref(luid)):
            return False, f"LookupPrivilegeValue failed (error {ctypes.get_last_error()})"
        tp = TOKEN_PRIVILEGES(1, (LUID_AND_ATTRIBUTES * 1)(
            LUID_AND_ATTRIBUTES(luid, SE_PRIVILEGE_ENABLED)))
        ok = advapi32.AdjustTokenPrivileges(token, False, ctypes.byref(tp), 0, None, None)
        err = ctypes.get_last_error()
        if not ok:
            return False, f"AdjustTokenPrivileges failed (error {err})"
        if err == ERROR_NOT_ALL_ASSIGNED:
            return False, "not_held"
        return True, "ok"
    finally:
        kernel32.CloseHandle(token)


def _win_alloc_large_i64(n_elems: int) -> Tuple[np.ndarray, Dict]:
    """int64 array on Windows large pages (VirtualAlloc MEM_LARGE_PAGES).
    Raises LargePageUnavailable with the reason and the fix."""
    ok, why = _win_enable_lock_memory_privilege()
    if not ok:
        if why == "not_held":
            elev = _win_is_elevated()
            reason = ("'Lock pages in memory' privilege not held"
                      + ("" if elev else " (process not elevated)" if elev is False else ""))
            raise LargePageUnavailable(reason, _WIN_LP_HINT_PRIV)
        raise LargePageUnavailable(why, _WIN_LP_HINT_PRIV)
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetLargePageMinimum.restype = ctypes.c_size_t
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                      wintypes.DWORD, wintypes.DWORD]
    kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD]
    MEM_COMMIT, MEM_RESERVE, MEM_LARGE_PAGES = 0x1000, 0x2000, 0x20000000
    PAGE_READWRITE, MEM_RELEASE = 0x04, 0x8000
    lp = int(kernel32.GetLargePageMinimum() or 0)
    if lp <= 0:
        raise LargePageUnavailable("large pages not supported by this system")
    nbytes = int(n_elems) * 8
    size = ((nbytes + lp - 1) // lp) * lp
    addr = kernel32.VirtualAlloc(None, size, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                                 PAGE_READWRITE)
    if not addr:
        err = ctypes.get_last_error()
        if err == 1314:      # ERROR_PRIVILEGE_NOT_HELD
            raise LargePageUnavailable("'Lock pages in memory' privilege not effective",
                                       _WIN_LP_HINT_PRIV)
        if err in (1450, 8, 1455):   # NO_SYSTEM_RESOURCES / NOT_ENOUGH_MEMORY / PAGEFILE_QUOTA
            raise LargePageUnavailable(f"no contiguous memory for {size >> 20} MB of large "
                                       f"pages (error {err})", _WIN_LP_HINT_FRAG)
        raise LargePageUnavailable(f"VirtualAlloc(MEM_LARGE_PAGES) failed (error {err})",
                                   _WIN_LP_HINT_FRAG)
    arr = np.ctypeslib.as_array((ctypes.c_int64 * int(n_elems)).from_address(addr))
    # VirtualAlloc memory is zero-filled; release it when the array is freed.
    weakref.finalize(arr, kernel32.VirtualFree, addr, 0, MEM_RELEASE)
    return arr, {"page_kb": lp // 1024, "bytes_allocated": size}


def _linux_alloc_i64(n_elems: int, huge: bool) -> np.ndarray:
    nbytes = max(int(n_elems) * 8, mmap.PAGESIZE)
    mm = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                   prot=mmap.PROT_READ | mmap.PROT_WRITE)
    advice = getattr(mmap, "MADV_HUGEPAGE" if huge else "MADV_NOHUGEPAGE", None)
    if advice is not None:
        try:
            mm.madvise(advice)
        except OSError:
            pass
    return np.frombuffer(mm, dtype=np.int64, count=int(n_elems))


def make_page_allocator(mode: str):
    """Allocator for build_random_chase. mode '4k' forces small pages,
    '2m' requires large pages (raises LargePageUnavailable when it cannot)."""
    if mode not in ("4k", "2m"):
        raise ValueError(mode)
    if sys.platform == "win32":
        if mode == "4k":
            return alloc_i64                       # standard 4 KB pages
        return lambda n: _win_alloc_large_i64(n)[0]
    if sys.platform.startswith("linux"):
        if mode == "2m":
            thp = _bracketed_choice(_read_text("/sys/kernel/mm/transparent_hugepage/enabled"))
            if thp == "never":
                raise LargePageUnavailable(
                    "transparent huge pages disabled (THP = never)",
                    "echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled")
        return lambda n: _linux_alloc_i64(n, huge=(mode == "2m"))
    raise LargePageUnavailable(f"page-size control not implemented on {sys.platform}")


def sweep_uses_small_pages(platform_str: Optional[str], pages_ctx: Optional[Dict]) -> bool:
    """True when the main sweep's multi-MB buffers sit on 4 KB pages (Windows,
    or Linux with THP disabled). Decides the 4 KB-page summary window caps."""
    pages_ctx = pages_ctx or {}
    plat = (platform_str or "").lower()
    if plat.startswith("windows") or plat == "win32":
        return True
    if pages_ctx.get("thp_enabled") == "never":
        return True
    return False


def apply_ram_headline(summary: Dict, page_modes: Optional[Dict], cfg: Dict) -> Dict:
    """
    Option C RAM headline:
      large pages measured  -> DRAM latency on 2 MB pages (graded "RAM Latency")
      otherwise             -> the ~256 MB random-chase point, labelled with its
                               L3 coverage and why large pages were unavailable
      no page-mode data     -> the plateau median, as in 6.96
    Also sets ram_4k_ns (graded separately) and page_walk_ns (= 4K - 2M).
    """
    s = dict(summary)
    s["ram_plateau_ns"] = summary.get("ram_median_ns")
    l3 = int(cfg.get("l3_mb", 16) * 1024 * 1024)
    pm = page_modes or {}
    m2, m4, fb = pm.get("2m") or {}, pm.get("4k") or {}, pm.get("fallback") or {}
    head: Dict[str, Any] = {"large_pages": {k: m2.get(k) for k in
                                            ("available", "reason", "hint", "page_kb",
                                             "hugepage_coverage_pct")}}
    if m2.get("available") and m2.get("median_ns") is not None:
        s["ram_median_ns"] = m2["median_ns"]
        head.update(source="large_pages", size_bytes=pm.get("size_bytes"),
                    label=f"DRAM latency, 2 MB pages, {format_size(pm['size_bytes'])} working set")
    elif fb.get("median_ns") is not None:
        s["ram_median_ns"] = fb["median_ns"]
        cov = round(min(100.0, l3 * 100.0 / fb["size_bytes"]), 0)
        head.update(source="fallback_256mb", size_bytes=fb["size_bytes"], l3_coverage_pct=cov,
                    label=(f"{format_size(fb['size_bytes'])} working set — L3 covers {cov:.0f}%; "
                           f"large pages unavailable"))
    else:
        head.update(source="plateau", label="median over the RAM plateau window (default pages)")
    if m4.get("available") and m4.get("median_ns") is not None:
        s["ram_4k_ns"] = m4["median_ns"]
    else:
        s["ram_4k_ns"] = None
    if s.get("ram_4k_ns") is not None and m2.get("available") and m2.get("median_ns") is not None:
        s["page_walk_ns"] = s["ram_4k_ns"] - m2["median_ns"]
    else:
        s["page_walk_ns"] = None
    s["ram_headline"] = head
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  Section 6 — Buffer Builders
# ══════════════════════════════════════════════════════════════════════════════
def build_random_chase(size_bytes: int, stride_bytes: int = 64,
                       rng: Optional[np.random.Generator] = None,
                       alloc=None) -> Tuple[np.ndarray, int]:
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RNG_SEED)
    stride_elems = stride_bytes // 8
    n_nodes = size_bytes // stride_bytes
    if n_nodes < 8:
        raise ValueError(f"Working set {size_bytes}B too small for stride {stride_bytes}B")
    buf = (alloc or alloc_i64)(n_nodes * stride_elems)   # own mapping -> page backing is reportable
    perm = rng.permutation(n_nodes).astype(np.int64)
    # (6.97) vectorized: identical buffer to the old Python loop, ~7x faster
    buf[perm * stride_elems] = np.roll(perm, -1) * stride_elems
    return buf, n_nodes
 
 
def build_stride_chase(size_bytes: int, stride_bytes: int) -> Tuple[np.ndarray, int]:
    stride_elems = stride_bytes // 8
    n_nodes = size_bytes // stride_bytes
    if n_nodes < 8:
        raise ValueError("Buffer too small")
    buf = alloc_i64(n_nodes * stride_elems)   # own mapping -> page backing is reportable
    idx = np.arange(n_nodes, dtype=np.int64)
    buf[idx * stride_elems] = ((idx + 1) % n_nodes) * stride_elems   # (6.97) vectorized
    return buf, n_nodes
 
 
def build_dirty_writeback(size_bytes: int,
                          rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, int]:
    """
    Build the buffer for the dirty-writeback chase (formerly "write_rfo").
    Same random permutation layout as random_chase; the kernel additionally
    stores into every line it visits, so every line it touches ends up dirty.

    What it measures: random-chase latency plus the cost of writing dirty
    lines back when they are evicted (L2->L3, and to DRAM for large sets).
    What it does NOT measure: RFO. The pointer load brings each line in
    first (Exclusive state, no other sharers), so the store is a local
    E->M upgrade with no ownership request, and the store is not on the
    dependency chain anyway. See the cross-core RFO test for real RFO cost.
    """
    return build_random_chase(size_bytes, stride_bytes=64, rng=rng)
 
 
def build_tlb_chase(size_bytes: int, page_size: int = 4096,
                    rng: Optional[np.random.Generator] = None,
                    alloc=None) -> Tuple[np.ndarray, int]:
    """
    Random pointer chase with one node per page: every hop lands on a
    different page, so the pattern loads the TLB with n_nodes pages while
    touching only n_nodes cache lines.
    page_size: 4096 for 4 KB pages, 2097152 for 2 MB pages. Pass an `alloc`
    that really gives that page size (make_page_allocator).

    (6.98) Each node sits on a random cache line inside its page. Before, every
    node was at offset 0 of its page; page-aligned addresses share their low
    address bits, so all nodes competed for the same few cache sets (one L1
    set; one L2 set for 2 MB strides) and the chase measured conflict misses
    instead of TLB cost. Node 0 stays at element 0, where the kernel starts.
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RNG_SEED)
    stride_elems = page_size // 8
    n_nodes = size_bytes // page_size
    if n_nodes < 8:
        raise ValueError(f"Working set {size_bytes}B too small for TLB test stride {page_size}B")
    buf = (alloc or alloc_i64)(n_nodes * stride_elems)   # own mapping -> page backing is reportable
    perm = rng.permutation(n_nodes).astype(np.int64)
    line_off = rng.integers(0, page_size // 64, n_nodes).astype(np.int64) * 8   # in int64 elements
    line_off[0] = 0
    pos = perm * stride_elems + line_off[perm]
    buf[pos] = np.roll(pos, -1)
    return buf, n_nodes
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 7 — Kernels (Numba JIT, required)
# ══════════════════════════════════════════════════════════════════════════════
if HAS_NUMBA:
    @njit(cache=True, fastmath=False)
    def _chase_kernel(buf: np.ndarray, n_nodes: int, traversals: int) -> int:
        # (6.97) The chain index is kept unsigned. With a signed int64 index Numba
        # emits Python's negative-index wraparound (sar+and) on every hop, which
        # sits on the dependency chain and added ~2 cycles per hop (~40% at L1).
        dummy = np.uint64(0)
        pos = np.uint64(0)
        for _ in range(min(n_nodes * 4, 50_000)):
            pos = np.uint64(buf[pos])
            dummy ^= pos
        for _ in range(traversals):
            local = pos
            for _ in range(n_nodes):
                local = np.uint64(buf[local])
                dummy ^= local
            pos = local
        return dummy
 
    @njit(cache=True, fastmath=False)
    def _dirty_writeback_kernel(buf: np.ndarray, n_nodes: int, traversals: int) -> int:
        """
        Dirty-writeback chase (formerly _write_rfo_kernel): a pointer chase
        that also stores into each visited line, leaving it Modified.
        The store does not gate the next load (only the pointer load does),
        and the line was already brought in Exclusive by that load, so no
        RFO is issued. The extra time vs _chase_kernel is the writeback of
        dirty victims (plus store-port pressure in cache-resident sizes).
        """
        dummy = np.uint64(0)
        pos = np.uint64(0)
        # Warmup (read-only to fill cache/TLB); unsigned index, see _chase_kernel
        for _ in range(min(n_nodes * 4, 50_000)):
            pos = np.uint64(buf[pos])
            dummy ^= pos
        # Timed chase: load pointer, dirty the line, follow pointer
        for _ in range(traversals):
            local = pos
            for _ in range(n_nodes):
                next_pos = np.uint64(buf[local])
                # Dirty the current line (word 1); pointer at word 0 stays intact
                buf[local + np.uint64(1)] = np.int64(local) ^ np.int64(0x5A5A)
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

    @njit(cache=True, fastmath=False)
    def _clock_probe_kernel(n_iter: int, seed: np.uint64) -> np.uint64:
        """Clock probe (6.96): x = rotr(x, 7) + K. LLVM emits rorx/ror + add,
        a strictly dependent chain of two 1-cycle ops = 2 core cycles per
        iteration, with no register copy in the chain (a shift/xor variant
        needs a mov, and imperfect move elimination made it read ~10% low).
        A multiply chain is unusable: LLVM folds the constant products."""
        x = np.uint64(seed)
        k = np.uint64(0x9E3779B97F4A7C15)
        r = np.uint64(7)
        l = np.uint64(57)
        for _ in range(n_iter):
            x = ((x >> r) | (x << l)) + k
        return x

    # ── Cross-core RFO kernels (6.96) ────────────────────────────────────────
    # Raw-address primitives built as LLVM intrinsics so the chase is a pure
    # pointer chase and the RMW is a real `lock xadd`.
    from numba import types as _nbt
    from numba.extending import intrinsic as _nb_intrinsic
    from numba.core import cgutils as _nb_cgutils
    from llvmlite import ir as _llir

    def _p64(builder, addr):
        return builder.inttoptr(addr, _llir.PointerType(_llir.IntType(64)))

    @_nb_intrinsic
    def _ptr_load(typingctx, addr):
        """Plain 64-bit load from a raw address."""
        def codegen(context, builder, sig, args):
            return builder.load(_p64(builder, args[0]))
        return _nbt.int64(_nbt.int64), codegen

    @_nb_intrinsic
    def _ptr_locked_fetch_add(typingctx, addr, val):
        """`lock xadd [addr], val` -> old value. Completes only once this
        core holds the line exclusively, so it carries the full RFO."""
        def codegen(context, builder, sig, args):
            return builder.atomic_rmw("add", _p64(builder, args[0]), args[1], "seq_cst")
        return _nbt.int64(_nbt.int64, _nbt.int64), codegen

    @_nb_intrinsic
    def _ptr_load_acquire(typingctx, addr):
        def codegen(context, builder, sig, args):
            return builder.load_atomic(_p64(builder, args[0]), "acquire", 8)
        return _nbt.int64(_nbt.int64), codegen

    @_nb_intrinsic
    def _ptr_store_release(typingctx, addr, val):
        def codegen(context, builder, sig, args):
            builder.store_atomic(args[1], _p64(builder, args[0]), "release", 8)
            return context.get_dummy_value()
        return _nbt.void(_nbt.int64, _nbt.int64), codegen

    @_nb_intrinsic
    def _rdtsc(typingctx):
        """x86 time-stamp counter (llvm.readcyclecounter)."""
        def codegen(context, builder, sig, args):
            fn = _nb_cgutils.get_or_insert_function(
                builder.module, _llir.FunctionType(_llir.IntType(64), []),
                "llvm.readcyclecounter")
            return builder.call(fn, [])
        return _nbt.int64(), codegen

    @njit(cache=True, nogil=True)
    def _rdtsc_now() -> int:
        return _rdtsc()

    @njit(cache=True, nogil=True)
    def _rfo_write_chain(region, nxt, step):
        """Store each node's successor address: every node line becomes
        Modified in the calling core's cache."""
        for i in range(nxt.shape[0]):
            region[i * step] = nxt[i]

    @njit(cache=True, nogil=True)
    def _rfo_chase(start, hops, locked, delta):
        """One traversal. locked=0: plain loads. locked=1: lock xadd with a
        runtime delta of 0 (value unchanged; LLVM cannot drop the RMW
        because delta is not a compile-time constant)."""
        p = start
        if locked:
            for _ in range(hops):
                p = _ptr_locked_fetch_add(p, delta)
        else:
            for _ in range(hops):
                p = _ptr_load(p)
        return p

    @njit(cache=True, nogil=True)
    def _rfo_local_rounds(region, nxt, step, start, hops, locked, delta, ticks, sink):
        """Core A alone: dirty the chain itself, then time one traversal."""
        acc = 0
        for r in range(ticks.shape[0]):
            _rfo_write_chain(region, nxt, step)
            t0 = _rdtsc()
            acc ^= _rfo_chase(start, hops, locked, delta)
            ticks[r] = _rdtsc() - t0
        sink[0] = acc
        return 0

    @njit(cache=True, nogil=True)
    def _rfo_owner_rounds(region, nxt, step, ready, done, rounds, timeout_ticks):
        """Core B: dirty the chain, publish round r, spin (staying in C0 so
        its caches keep the lines) until core A has finished round r."""
        for r in range(1, rounds + 1):
            _rfo_write_chain(region, nxt, step)
            _ptr_store_release(ready, r)
            deadline = _rdtsc() + timeout_ticks
            spins = 0
            while _ptr_load_acquire(done) < r:
                spins += 1
                if (spins & 1023) == 0 and _rdtsc() > deadline:
                    return -r
        return 0

    @njit(cache=True, nogil=True)
    def _rfo_taker_rounds(start, hops, locked, delta, ready, done, ticks, timeout_ticks, sink):
        """Core A: wait for B's round r, time one traversal over B's dirty
        lines, then release B."""
        acc = 0
        for r in range(1, ticks.shape[0] + 1):
            deadline = _rdtsc() + timeout_ticks
            spins = 0
            while _ptr_load_acquire(ready) < r:
                spins += 1
                if (spins & 1023) == 0 and _rdtsc() > deadline:
                    return -r
            t0 = _rdtsc()
            acc ^= _rfo_chase(start, hops, locked, delta)
            ticks[r - 1] = _rdtsc() - t0
            _ptr_store_release(done, r)
        sink[0] = acc
        return 0
# No pure-Python fallback kernels: interpreter overhead (~50-100 ns per hop)
# would swamp the 1-15 ns cache latencies being measured, producing wrong
# numbers rather than slow ones. check_dependencies() refuses to start any
# measurement mode when Numba is unavailable.


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
 
 
def calibrate_dirty_writeback(buf: np.ndarray, n_nodes: int, target_sec: float) -> int:
    """Calibrate the dirty-writeback kernel (same interface as calibrate_chase)."""
    probe = max(1, min(50, 100_000 // max(n_nodes, 1)))
    gc.collect()
    t0 = time.perf_counter()
    _dirty_writeback_kernel(buf, n_nodes, probe)
    elapsed = time.perf_counter() - t0
    if elapsed < 0.005:
        probe *= 20
        gc.collect()
        t0 = time.perf_counter()
        _dirty_writeback_kernel(buf, n_nodes, probe)
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
 
 
def timed_dirty_writeback(buf: np.ndarray, n_nodes: int, traversals: int) -> float:
    """Time the dirty-writeback kernel (same interface as timed_chase)."""
    gc.collect()
    t0 = time.perf_counter()
    dummy = _dirty_writeback_kernel(buf, n_nodes, traversals)
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
#  Section 9 — Bandwidth
# ══════════════════════════════════════════════════════════════════════════════
BANDWIDTH_ARRAYS = 3            # a, b, c of the triad, each as large as the working set
BANDWIDTH_MAX_FREE_FRACTION = 0.8


def bandwidth_fits(size_bytes: int) -> bool:
    """(6.98) The triad needs three arrays of the working-set size. The sweep's
    memory limit only covers one, so check free RAM before allocating."""
    if not HAS_PSUTIL:
        return True
    try:
        avail = psutil.virtual_memory().available
    except Exception:
        return True
    return BANDWIDTH_ARRAYS * size_bytes <= BANDWIDTH_MAX_FREE_FRACTION * avail


def measure_bandwidth(size_bytes: int,
                      wide_affinity: Optional[List[int]] = None) -> Optional[float]:
    """STREAM-style triad over three arrays of size_bytes each, on all Numba threads.
    (6.98) wide_affinity: the CPUs the process had before the sweep was pinned.
    A pinned sweep narrows the affinity (process-wide on Windows), which put
    every triad thread on the one pinned core; the affinity is widened for the
    triad and restored afterwards. Arrays are filled in place (no temporaries)."""
    saved = None
    proc = None
    try:
        n = size_bytes // 8
        if n < 1 or not bandwidth_fits(size_bytes):
            return None
        if wide_affinity and HAS_PSUTIL:
            try:
                proc = psutil.Process()
                cur = proc.cpu_affinity()
                if sorted(cur) != sorted(wide_affinity):
                    proc.cpu_affinity(list(wide_affinity))
                    saved = cur
            except Exception:
                saved = None
        a = np.empty(n, dtype=np.float64); a.fill(0.0)
        b = np.empty(n, dtype=np.float64); b.fill(1.0)
        c = np.empty(n, dtype=np.float64); c.fill(2.0)
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
    finally:
        if saved is not None and proc is not None:
            try:
                proc.cpu_affinity(saved)
            except Exception:
                pass

# ══════════════════════════════════════════════════════════════════════════════
#  Section 9A — Cross-Core RFO Test (ownership transfer)  [6.96]
#
#  A single-threaded write cannot show RFO cost: the pointer load brings the
#  line in Exclusive first, so the store is a silent local E->M upgrade (that
#  is why "write_rfo" was renamed dirty_writeback). A real Read-For-Ownership
#  happens when another core holds the line. So:
#    core B writes a random chain of lines (128 B apart, footprint <= L2/4),
#           leaving every line Modified in B's private cache, then spins;
#    core A walks that chain once, timed with the TSC, either with plain
#           loads (cache-to-cache read) or with `lock xadd` (each hop must
#           take exclusive ownership before the next address is known).
#  Baselines repeat both walks over lines A dirtied itself. Then:
#    c2c read        = remote load  - local load
#    RFO transfer    = remote RMW   - local RMW   (the true RFO cost)
#    ownership extra = RFO transfer - c2c read
#  B spins instead of sleeping so its core never enters a deep C-state,
#  which would flush its caches to L3 and turn the test into an L3 test.
#  128 B node spacing keeps the adjacent-line prefetcher from pulling a
#  future node along with the current one.
# ══════════════════════════════════════════════════════════════════════════════
RFO_NODE_STRIDE = 128
RFO_ROUNDS = 500
RFO_WARMUP_ROUNDS = 10


def _parse_cpu_list(text: Optional[str]) -> List[int]:
    """'0-7,16-23' -> [0..7, 16..23]"""
    cpus: List[int] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _shares_l3(cpu_a: int, cpu_b: int) -> Optional[bool]:
    """True/False from Linux sysfs cache topology; None where unavailable."""
    base = f"/sys/devices/system/cpu/cpu{cpu_a}/cache"
    try:
        for idx in sorted(os.listdir(base)):
            if _read_text(f"{base}/{idx}/level") == "3":
                return cpu_b in _parse_cpu_list(_read_text(f"{base}/{idx}/shared_cpu_list"))
    except Exception:
        pass
    return None


def _pin_current_thread(cpu: int) -> bool:
    """Pin the calling thread (not the process) to one logical CPU."""
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {cpu})          # pid 0 = calling thread on Linux
            return True
        if sys.platform == "win32":
            k32 = ctypes.windll.kernel32
            k32.GetCurrentThread.restype = ctypes.c_void_p
            k32.SetThreadAffinityMask.restype = ctypes.c_size_t
            k32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            ok = k32.SetThreadAffinityMask(k32.GetCurrentThread(), 1 << cpu) != 0
            time.sleep(0)                           # let the scheduler move us
            return ok
    except Exception:
        pass
    return False


def _rfo_core_pairs() -> List[Dict]:
    """(core A, core B) pairs: one sharing an L3, plus one across CCDs/L3s
    when the chip has more than one. Physical cores only (never SMT siblings)."""
    topo = _detect_core_topology()
    phys = topo.get("physical_cores") or []
    pairs: List[Dict] = []
    if len(phys) < 2:
        return pairs
    groups = [g for g in (topo.get("ccd_groups") or []) if g]
    g0 = groups[0] if groups else phys
    if len(g0) >= 3:
        a, b = g0[1], g0[2]                         # skip core 0 (interrupt-heavy)
    elif len(g0) == 2:
        a, b = g0[0], g0[1]
    else:
        a, b = phys[0], phys[1]
    pairs.append({"label": "same L3", "core_a": a, "core_b": b})
    if len(groups) >= 2 and groups[1]:
        g1 = groups[1]
        pairs.append({"label": "different L3 (cross-CCD)",
                      "core_a": a, "core_b": g1[1] if len(g1) >= 2 else g1[0]})
    for p in pairs:
        p["shares_l3"] = _shares_l3(p["core_a"], p["core_b"])
        if p["shares_l3"] is False and p["label"] == "same L3":
            p["label"] = "different L3"
    return pairs


def _calibrate_tsc_ghz() -> float:
    _rdtsc_now()
    c0, t0 = _rdtsc_now(), time.perf_counter()
    time.sleep(0.1)
    c1, t1 = _rdtsc_now(), time.perf_counter()
    return (c1 - c0) / (t1 - t0) / 1e9


def run_cross_core_rfo(cfg: Dict, rounds: int = RFO_ROUNDS) -> Dict:
    """Run the four variants for each core pair. Returns a JSON-ready dict."""
    out: Dict[str, Any] = {"test": "cross_core_rfo", "pairs": []}
    if not HAS_NUMBA:
        out["skipped"] = "numba required"
        return out
    if platform.machine().lower() not in ("x86_64", "amd64"):
        out["skipped"] = f"x86-64 only (TSC timing, lock xadd); this is {platform.machine()}"
        return out
    if not (hasattr(os, "sched_setaffinity") or sys.platform == "win32"):
        out["skipped"] = "per-thread CPU pinning not supported on this OS"
        return out
    pairs = _rfo_core_pairs()
    if not pairs:
        out["skipped"] = "needs at least two physical cores"
        return out

    l2_bytes = cfg.get("l2_p_kb", 512) * 1024
    n = int(min(4096, max(256, l2_bytes // 4 // RFO_NODE_STRIDE)))
    step = RFO_NODE_STRIDE // 8
    region = alloc_i64(n * step)
    base = int(region.ctypes.data)
    perm = np.random.default_rng(DEFAULT_RNG_SEED).permutation(n).astype(np.int64)
    nxt = np.empty(n, dtype=np.int64)
    nxt[perm] = base + np.roll(perm, -1) * RFO_NODE_STRIDE
    start = base + int(perm[0]) * RFO_NODE_STRIDE
    ctl = alloc_i64(64)                                 # two control words, 128 B apart
    ready, done = int(ctl.ctypes.data), int(ctl.ctypes.data) + 128
    sink = np.zeros(1, dtype=np.int64)

    # Compile everything on this thread first (rounds=0 -> loops do not run)
    z = np.zeros(0, dtype=np.int64)
    _rfo_write_chain(region, nxt, step)
    _rfo_chase(start, 1, 0, 0)
    _rfo_chase(start, 1, 1, 0)
    _rfo_local_rounds(region, nxt, step, start, n, 0, 0, z, sink)
    _rfo_owner_rounds(region, nxt, step, ready, done, 0, 1)
    _rfo_taker_rounds(start, n, 0, 0, ready, done, z, 1, sink)
    tsc_ghz = _calibrate_tsc_ghz()
    timeout_ticks = int(tsc_ghz * 1e9 * 5)              # 5 s per wait

    # On Windows a thread can only be pinned inside the process mask; widen
    # it for the test if the main run pinned the process, then restore.
    saved_aff = None
    if sys.platform == "win32" and HAS_PSUTIL:
        try:
            saved_aff = psutil.Process().cpu_affinity()
            psutil.Process().cpu_affinity([])
        except Exception:
            saved_aff = None

    out.update({
        "method": ("core B dirties a random chain, core A walks it once per round "
                   "(plain loads, or lock xadd = ownership); TSC-timed in Numba"),
        "nodes": n, "node_stride_bytes": RFO_NODE_STRIDE,
        "footprint_kb": n * RFO_NODE_STRIDE // 1024,
        "rounds": rounds, "warmup_rounds_dropped": RFO_WARMUP_ROUNDS,
        "tsc_ghz": round(tsc_ghz, 4),
    })

    def _stats(ticks: np.ndarray) -> Dict:
        per_hop = ticks[RFO_WARMUP_ROUNDS:].astype(np.float64) / tsc_ghz / n
        return percentile_stats(list(per_hop))

    try:
        for pair in pairs:
            a, b = pair["core_a"], pair["core_b"]
            rec: Dict[str, Any] = dict(pair, variants={}, errors=[])
            seen: Dict[str, Any] = {}

            for name, locked in (("local_read", 0), ("local_rmw", 1)):
                ticks = np.zeros(rounds, dtype=np.int64)
                st: Dict[str, Any] = {}

                def _local():
                    st["pinned"] = _pin_current_thread(a)
                    seen["cpu_a"] = current_cpu()
                    st["rc"] = _rfo_local_rounds(region, nxt, step, start, n, locked, 0, ticks, sink)
                th = threading.Thread(target=_local, daemon=True)
                th.start(); th.join(60)
                if st.get("rc") == 0 and st.get("pinned"):
                    rec["variants"][name] = _stats(ticks)
                else:
                    rec["errors"].append(f"{name}: pinned={st.get('pinned')} rc={st.get('rc')}")

            for name, locked in (("remote_read", 0), ("remote_rmw", 1)):
                ticks = np.zeros(rounds, dtype=np.int64)
                ctl[:] = 0
                st = {}

                def _owner():
                    st["pin_b"] = _pin_current_thread(b)
                    seen["cpu_b"] = current_cpu()
                    st["rc_b"] = _rfo_owner_rounds(region, nxt, step, ready, done,
                                                   rounds, timeout_ticks)

                def _taker():
                    st["pin_a"] = _pin_current_thread(a)
                    st["rc_a"] = _rfo_taker_rounds(start, n, locked, 0, ready, done,
                                                   ticks, timeout_ticks, sink)
                tb = threading.Thread(target=_owner, daemon=True)
                ta = threading.Thread(target=_taker, daemon=True)
                tb.start(); ta.start()
                ta.join(120); tb.join(120)
                if (st.get("rc_a") == 0 and st.get("rc_b") == 0
                        and st.get("pin_a") and st.get("pin_b")):
                    rec["variants"][name] = _stats(ticks)
                else:
                    rec["errors"].append(f"{name}: pin_a={st.get('pin_a')} pin_b={st.get('pin_b')} "
                                         f"rc_a={st.get('rc_a')} rc_b={st.get('rc_b')}")

            rec["cpu_a_seen"] = seen.get("cpu_a")
            rec["cpu_b_seen"] = seen.get("cpu_b")
            v = rec["variants"]

            def _diff(x: str, y: str) -> Optional[float]:
                return (v[x]["median"] - v[y]["median"]) if (x in v and y in v) else None
            c2c = _diff("remote_read", "local_read")
            rfo = _diff("remote_rmw", "local_rmw")
            rec["derived"] = {
                "c2c_read_ns": c2c,
                "rfo_transfer_ns": rfo,
                "ownership_premium_ns": (rfo - c2c) if (rfo is not None and c2c is not None) else None,
            }
            out["pairs"].append(rec)
    finally:
        if saved_aff is not None:
            try:
                psutil.Process().cpu_affinity(saved_aff)
            except Exception:
                pass
    return out


def format_rfo_report(rfo: Optional[Dict]) -> List[str]:
    """Console/log lines for a run_cross_core_rfo() result."""
    if not rfo:
        return []
    if rfo.get("skipped"):
        return [f"\n  Cross-core RFO test skipped: {rfo['skipped']}"]
    lines = [f"\n  Cross-Core RFO (ownership transfer)  "
             f"[{rfo.get('footprint_kb')} KB, {rfo.get('nodes')} lines "
             f"{rfo.get('node_stride_bytes')} B apart, {rfo.get('rounds')} rounds/variant]"]
    names = [("local_read", "A's own lines, plain load"),
             ("local_rmw", "A's own lines, locked RMW"),
             ("remote_read", "B's dirty lines, plain load"),
             ("remote_rmw", "B's dirty lines, locked RMW")]
    for p in rfo.get("pairs", []):
        rel = p.get("label", "")
        sl3 = p.get("shares_l3")
        rel += "" if sl3 is None else (" (sysfs: shared L3)" if sl3 else " (sysfs: separate L3)")
        lines.append(f"    Core A={p['core_a']} takes lines from core B={p['core_b']} -- {rel}")
        lines.append(f"      {'Variant':<30} {'median':>8} {'p5':>8} {'p95':>8}   ns/hop")
        for key, label in names:
            s = p.get("variants", {}).get(key)
            if s:
                lines.append(f"      {label:<30} {s['median']:8.1f} {s['p5']:8.1f} {s['p95']:8.1f}")
            else:
                lines.append(f"      {label:<30} {'n/a':>8}")
        d = p.get("derived", {})

        def _f(x: Optional[float]) -> str:
            return f"{x:+.1f} ns" if x is not None else "n/a"
        lines.append(f"      -> cache-to-cache read     : {_f(d.get('c2c_read_ns'))}   (remote load - local load)")
        lines.append(f"      -> RFO / ownership transfer: {_f(d.get('rfo_transfer_ns'))}   (remote RMW - local RMW)")
        lines.append(f"      -> ownership premium       : {_f(d.get('ownership_premium_ns'))}   (RFO transfer - c2c read)")
        for err in p.get("errors", []):
            lines.append(f"      ! {err}")
    return lines


# ══════════════════════════════════════════════════════════════════════════════
#  Section 9B — Loaded Latency Test (Infinity Fabric Stress)
#  Inspired by Chips & Cheese methodology: run a latency-sensitive pointer
#  chase on one core while progressively adding bandwidth-hungry threads
#  on other cores to expose XI queue, IFOP, and memory controller contention.
# ══════════════════════════════════════════════════════════════════════════════

import multiprocessing as _mp

# (6.98) "spawn" on every OS. The default on Linux was fork(), and the parent is
# multi-threaded by then (Numba's pool, BLAS), which Python warns can deadlock
# the child. Workers only receive plain arguments, so nothing depends on fork.
_MP_CTX = _mp.get_context("spawn")

def _bw_worker_process(array_mb: int, core_id: int, stop_flag, ready_flag, pass_counter):
    """Bandwidth worker process — Numba-compiled sequential streaming reader.

    Each worker pins itself to one core, allocates a float64 array far larger
    than L3, and repeatedly streams through it front to back with a
    @njit(nogil=True) kernel (8 independent accumulators). Sequential access
    lets the hardware prefetchers run at full rate, so a single core pulls
    roughly 30-50 GB/s from DRAM -- enough for a few workers to saturate the
    CCD's IFOP link and the memory controller queues.

    History: V5.0-5.4 used random gathers (numpy fancy indexing, later a
    random-index Numba kernel). Those topped out near 7-11 GB/s per core
    (DRAM row misses), far too little to load the fabric; the switch to
    sequential streaming is what made the loaded-latency test work.
    """
    # Pin to designated core FIRST, before any allocation
    try:
        import psutil as _ps
        _ps.Process().cpu_affinity([core_id])
    except Exception:
        try:
            import os as _os
            if hasattr(_os, 'sched_setaffinity'):
                _os.sched_setaffinity(0, {core_id})
        except Exception:
            pass

    import numpy as _np
    from numba import njit as _njit

    # ── Numba-compiled hot loop — this is the critical performance piece ──
    @_njit(cache=False, nogil=True)
    def _streaming_bw_kernel(arr, n_elems):
        """Sequential streaming read — maximizes memory bandwidth per core.

        KEY INSIGHT (fixing the fundamental design mistake in V5.0-5.4):
        Clamchowder's bandwidth threads used SEQUENTIAL streaming, not random.
        A single Zen 4 core can stream ~50 GB/s through a large array because
        the prefetcher serves sequential access at near-peak DRAM bandwidth.

        Random access was hitting ~11 GB/s per core — a DRAM row-open bottleneck
        that no amount of MLP optimization can fix. The 8-way unrolling didn't
        help because each random access opens a new DRAM row (~20ns overhead).

        Sequential access through a 960 MB array still misses L3 (960 >> 96 MB)
        on every new cache line — the prefetcher just makes it FAST, which is
        exactly what we need to saturate IFOP bandwidth.

        Uses 8-way unrolling to keep the add pipeline from being the bottleneck.
        """
        a0 = 0.0; a1 = 0.0; a2 = 0.0; a3 = 0.0
        a4 = 0.0; a5 = 0.0; a6 = 0.0; a7 = 0.0

        n_bulk = (n_elems // 8) * 8
        for i in range(0, n_bulk, 8):
            a0 += arr[i]
            a1 += arr[i + 1]
            a2 += arr[i + 2]
            a3 += arr[i + 3]
            a4 += arr[i + 4]
            a5 += arr[i + 5]
            a6 += arr[i + 6]
            a7 += arr[i + 7]

        for i in range(n_bulk, n_elems):
            a0 += arr[i]

        return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7

    # Allocate large array — must be >> L3 to guarantee misses
    n_elems = (array_mb * 1024 * 1024) // 8  # float64 = 8 bytes
    arr = _np.ones(n_elems, dtype=_np.float64)

    # Sequential streaming: every element is read in order.
    # Prefetcher serves this efficiently -> maximum bandwidth per core.
    # Array is >> L3, so every cache line still comes from DRAM.
    bytes_per_pass = n_elems * 8  # every float64 read = 8 bytes

    # JIT compile the kernel before signaling ready (first call triggers compile)
    _streaming_bw_kernel(arr, min(1000, n_elems))

    # Signal ready — kernel is compiled and array is allocated
    ready_flag.value = 1

    dummy = 0.0
    passes = 0
    while stop_flag.value == 0:
        # One sequential streaming pass over the whole array (prefetcher-friendly)
        dummy += _streaming_bw_kernel(arr, n_elems)
        passes += 1
        # (6.98) publish every pass. Publishing every third pass made the
        # parent's bandwidth figure jump in steps of three array sizes.
        pass_counter.value = passes

    # Prevent dead-code elimination
    if dummy == -999.999:
        print(dummy)


class _BandwidthWorkerMP:
    """Multiprocessing bandwidth worker using a sequential streaming read pattern.
    Each worker runs in a separate OS process (no GIL), pinned to its own core,
    and streams through an array much larger than L3, generating sustained
    DRAM read traffic (and continuously evicting L3 contents)."""

    def __init__(self, array_mb: int, core_id: int):
        self._array_mb = array_mb
        self._core_id = core_id
        self._stop_flag = _MP_CTX.RawValue('i', 0)     # no lock needed
        self._ready_flag = _MP_CTX.RawValue('i', 0)     # no lock needed
        self._pass_counter = _MP_CTX.RawValue('q', 0)   # 64-bit, no lock (single writer)
        self._proc = None
        # Bytes per pass = full array streamed sequentially
        n_elems = (array_mb * 1024 * 1024) // 8
        self._bytes_per_pass = n_elems * 8  # every float64 = 8 bytes

    def start(self):
        self._proc = _MP_CTX.Process(
            target=_bw_worker_process,
            args=(self._array_mb, self._core_id, self._stop_flag,
                  self._ready_flag, self._pass_counter),
            daemon=True,
        )
        self._proc.start()

    def wait_ready(self, timeout=90.0) -> bool:  # extra time for Numba JIT in child
        """True once the worker is streaming. (6.98) False as soon as the
        worker process has died (it used to wait out the whole timeout) or
        when the timeout passes."""
        t0 = time.time()
        while self._ready_flag.value == 0:
            if self._proc is not None and not self._proc.is_alive():
                print(f"  Warning: BW worker on core {self._core_id} exited during start-up "
                      f"(exit code {self._proc.exitcode}; out of memory?)")
                return False
            time.sleep(0.05)
            if time.time() - t0 > timeout:
                print(f"  Warning: BW worker on core {self._core_id} timed out waiting for ready")
                return False
        return True

    def stop(self):
        self._stop_flag.value = 1

    def join(self, timeout=10.0):
        if self._proc and self._proc.is_alive():
            self._proc.join(timeout=timeout)
            if self._proc.is_alive():
                self._proc.terminate()

    def snapshot_passes(self) -> int:
        """Get current pass count (lock-free read)."""
        return self._pass_counter.value

    def snapshot_bytes(self) -> int:
        """Get estimated bytes of DRAM traffic generated."""
        return self._pass_counter.value * self._bytes_per_pass

    @property
    def bytes_per_pass(self) -> int:
        return self._bytes_per_pass


def _sysfs_core_groups() -> Tuple[List[int], List[List[int]]]:
    """(6.98) Linux: (first logical CPU of every online physical core, those
    cores grouped by the L3 they share). ([], []) when sysfs has no topology;
    (cores, []) when it has no L3 sharing information."""
    base = "/sys/devices/system/cpu"
    try:
        cpus = sorted(int(d[3:]) for d in os.listdir(base) if re.match(r"cpu\d+$", d))
    except OSError:
        return [], []
    phys: List[int] = []
    for c in cpus:
        if _read_text(f"{base}/cpu{c}/online") == "0":
            continue
        try:
            sib = _parse_cpu_list(_read_text(f"{base}/cpu{c}/topology/core_cpus_list")
                                  or _read_text(f"{base}/cpu{c}/topology/thread_siblings_list"))
        except ValueError:
            sib = []
        if sib and min(sib) == c:
            phys.append(c)
    groups: Dict[int, List[int]] = {}
    for c in phys:
        shared = read_sysfs_caches(c).get("l3_shared")
        if not shared:
            return phys, []
        groups.setdefault(min(shared), []).append(c)
    return phys, [groups[k] for k in sorted(groups)]


def _detect_core_topology() -> Dict:
    """Detect CCX/CCD topology for AMD or core layout for Intel.
    Returns dict with 'physical_cores' list and 'ccx_groups' if detectable."""
    result = {
        "physical_cores": [],
        "ccx_groups": [],   # list of lists: [[core_ids in CCX0], [CCX1], ...]
        "ccd_groups": [],   # list of lists: [[core_ids in CCD0], [CCD1], ...]
        "vendor": "unknown",
    }
    n_logical = os.cpu_count() or 1

    # Try to get physical core mapping
    sysfs_groups: List[List[int]] = []
    if sys.platform.startswith("linux"):
        result["physical_cores"], sysfs_groups = _sysfs_core_groups()
    if sys.platform.startswith("linux") and not result["physical_cores"]:
        core_map = {}
        try:
            with open("/proc/cpuinfo") as f:
                current_proc = None
                for line in f:
                    if line.startswith("processor"):
                        current_proc = int(line.split(":")[1].strip())
                    elif line.startswith("core id") and current_proc is not None:
                        core_id = int(line.split(":")[1].strip())
                        if core_id not in core_map:
                            core_map[core_id] = current_proc
                            result["physical_cores"].append(current_proc)
        except OSError:
            pass
    elif sys.platform == "win32":
        # On Windows, even logical cores 0,2,4,... are typically physical
        # for SMT systems. Use every other core as approximation.
        if n_logical >= 2:
            result["physical_cores"] = list(range(0, n_logical, 2))
        else:
            result["physical_cores"] = [0]

    if not result["physical_cores"]:
        result["physical_cores"] = list(range(n_logical))

    # For AMD Zen: CCX = 8 cores sharing L3 on Zen3+, CCD = 1 CCX on Zen3+
    # We infer from core count and CPU model
    cpu_info = detect_cpu()
    model = cpu_info.get("model", "")
    gen_key = cpu_info.get("gen_key", "")

    if "AMD" in cpu_info.get("vendor", "") or "AMD" in model:
        result["vendor"] = "AMD"
    elif "Intel" in cpu_info.get("vendor", "") or "Intel" in model:
        result["vendor"] = "Intel"
    result["group_source"] = "heuristic"

    if sysfs_groups:
        # (6.98) cores grouped by the L3 they actually share (6+6 on a 7900X,
        # every CCD on Threadripper/EPYC) instead of "8 cores per CCD".
        result["ccx_groups"] = [g[:] for g in sysfs_groups]
        result["ccd_groups"] = [g[:] for g in sysfs_groups]
        result["group_source"] = "sysfs L3 sharing"
    elif result["vendor"] == "AMD":
        phys = result["physical_cores"]
        n_phys = len(phys)
        # Zen 3/4/5: 8 cores per CCX, 1 CCX per CCD
        ccx_size = 8
        if n_phys <= ccx_size:
            # Single CCX/CCD chip (e.g., 7800X3D, 7600X)
            result["ccx_groups"] = [phys[:]]
            result["ccd_groups"] = [phys[:]]
        else:
            # Multi-CCD (e.g., 7950X, 9950X)
            ccd0 = phys[:ccx_size]
            ccd1 = phys[ccx_size:ccx_size * 2] if n_phys > ccx_size else []
            result["ccx_groups"] = [g for g in [ccd0, ccd1] if g]
            result["ccd_groups"] = [g for g in [ccd0, ccd1] if g]
    elif result["vendor"] == "Intel":
        # Intel monolithic: all cores on one die, ring/mesh bus
        result["ccx_groups"] = [result["physical_cores"][:]]
        result["ccd_groups"] = [result["physical_cores"][:]]
    else:
        result["ccx_groups"] = [result["physical_cores"][:]]
        result["ccd_groups"] = [result["physical_cores"][:]]

    return result


def run_loaded_latency_test(
    latency_buf_mb: Optional[int] = None,  # None = auto from cache coverage target
    bw_buf_mb: Optional[int] = None,       # None = auto from cache coverage target
    cache_coverage_pct: float = 5.0,       # target: L3 covers this % of latency buffer
    measure_seconds: float = 5.0,
    warmup_seconds: float = 3.0,
    rng_seed: int = DEFAULT_RNG_SEED,
    output_dir: Optional[str] = None,
    page_mode: str = "2m",                 # 6.97.1: latency buffer pages, "2m" (default) or "4k"
) -> Dict:
    """
    Run the full loaded latency test suite.

    Methodology (after Chips & Cheese / clamchowder):
      1. Pin a latency thread to core 0, running a random pointer chase
         through a buffer large enough to miss L3 (forces DRAM access).
      2. Progressively add bandwidth threads on other physical cores.
      3. At each thread count, measure the latency thread's per-hop time
         and the aggregate bandwidth from the bandwidth threads.
      4. Report how latency degrades as bandwidth load increases.

    This exposes XI queue contention, IFOP link saturation, and
    memory controller scheduling under load — the chiplet penalties
    that single-threaded latency tests completely miss.
    """
    if not HAS_PSUTIL:
        print("  ERROR: psutil is required for loaded latency test (CPU pinning).")
        print("         Install with: pip install psutil")
        return {"error": "psutil required"}

    print("\n" + "=" * 72)
    print("  LOADED LATENCY TEST — Infinity Fabric / Memory Subsystem Stress")
    print("  Methodology: Chips & Cheese (clamchowder) loaded latency approach")
    print("=" * 72)

    # ── Detect topology ──
    topo = _detect_core_topology()
    cpu_info = detect_cpu()
    cfg = get_cache_config(cpu_info)
    freq_ghz = detect_cpu_freq_ghz()

    phys_cores = topo["physical_cores"]
    n_phys = len(phys_cores)
    vendor = topo["vendor"]

    # ── Auto-size buffers from L3 cache and coverage target ──
    # l3_mb already includes V-Cache (e.g., 96 MB = 32 base + 64 VC)
    l3_total_mb = cfg.get("l3_mb", 16)

    if latency_buf_mb is None:
        # buffer = L3 / (coverage/100)  e.g. 96MB / 0.05 = 1920 MB
        latency_buf_mb = max(256, int(l3_total_mb / (cache_coverage_pct / 100.0)))
        auto_lat = True
    else:
        auto_lat = False

    if bw_buf_mb is None:
        # BW buffers use 2x the coverage target (e.g., 10% vs 5%)
        # Still ~90% DRAM hit rate, but halves memory vs latency buffer.
        # This prevents OOM when spawning many worker processes.
        bw_coverage_target = cache_coverage_pct * 2.0
        bw_buf_mb = max(256, int(l3_total_mb / (bw_coverage_target / 100.0)))
        auto_bw = True
    else:
        auto_bw = False

    lat_coverage = (l3_total_mb / latency_buf_mb * 100) if latency_buf_mb > 0 else 0
    bw_coverage = (l3_total_mb / bw_buf_mb * 100) if bw_buf_mb > 0 else 0

    if HAS_PSUTIL:
        avail_mb = psutil.virtual_memory().available // (1024 * 1024)
        # Each worker process needs: array + indices + Python runtime (~200 MB overhead)
        per_worker_overhead = 200
        est_total = latency_buf_mb + (n_phys - 1) * (bw_buf_mb + per_worker_overhead) + 1024
        if est_total > avail_mb * 0.70:
            old_bw = bw_buf_mb
            usable = int(avail_mb * 0.70) - latency_buf_mb - 1024
            bw_buf_mb = max(256, usable // max(1, n_phys - 1) - per_worker_overhead)
            bw_coverage = (l3_total_mb / bw_buf_mb * 100) if bw_buf_mb > 0 else 0
            print(f"  RAM safety: reduced BW buffer {old_bw} -> {bw_buf_mb} MB/thread")
            print(f"              ({avail_mb} MB available, est. {est_total} MB needed)")

    print(f"\n  CPU      : {cpu_info['model']}")
    print(f"  Vendor   : {vendor}")
    print(f"  Physical : {n_phys} cores detected")
    print(f"  CCX/CCD  : {len(topo['ccd_groups'])} group(s) — {[len(g) for g in topo['ccd_groups']]} cores each")
    print(f"  L3 Cache : {l3_total_mb} MB total" + (" (incl. V-Cache)" if cfg.get("v_cache_mb", 0) > 0 else ""))
    print(f"  +-- Buffer Auto-Sizing (target: L3 = {cache_coverage_pct:.0f}% of working set) --+")
    lat_tag = "auto" if auto_lat else "manual"
    bw_tag = "auto" if auto_bw else "manual"
    print(f"  |  Latency buffer : {latency_buf_mb:>6} MB  ({lat_tag})  L3 covers {lat_coverage:>5.1f}%  |")
    print(f"  |  BW buffer/thr  : {bw_buf_mb:>6} MB  ({bw_tag})  L3 covers {bw_coverage:>5.1f}%  |")
    print(f"  +------------------------------------------------------------+")
    # Estimate total RAM
    est_ram_gb = (latency_buf_mb + (n_phys - 1) * (bw_buf_mb + 200) + 1024) / 1024
    print(f"  Measure window : {measure_seconds:.1f}s per step (+ {warmup_seconds:.1f}s warmup)")
    print(f"  Est. RAM usage : ~{est_ram_gb:.1f} GB total"
          + (f" ({psutil.virtual_memory().available // (1024**3)} GB available)" if HAS_PSUTIL else ""))

    if n_phys < 2:
        print("  ERROR: Need at least 2 physical cores for loaded latency test.")
        return {"error": "need >= 2 physical cores"}

    # ── Determine test scenarios ──
    latency_core = phys_cores[0]
    bw_cores = phys_cores[1:]  # all other physical cores

    # Determine which CCD the latency core is on
    lat_ccd_idx = 0
    for idx, group in enumerate(topo["ccd_groups"]):
        if latency_core in group:
            lat_ccd_idx = idx
            break

    # Categorize bandwidth cores by same-CCX, same-CCD, other-CCD
    same_ccx_cores = []
    other_ccd_cores = []
    for c in bw_cores:
        in_lat_ccd = any(c in g for i, g in enumerate(topo["ccd_groups"]) if i == lat_ccd_idx)
        if in_lat_ccd:
            same_ccx_cores.append(c)
        else:
            other_ccd_cores.append(c)

    # Build progressive core loading order:
    # 1. Same CCX/CCD first (maximum contention at XI/IFOP level)
    # 2. Other CCD (contention at memory controller level)
    ordered_bw_cores = same_ccx_cores + other_ccd_cores

    max_bw_threads = min(len(ordered_bw_cores), n_phys - 1)
    if max_bw_threads < 1:
        print("  ERROR: Not enough cores for bandwidth threads.")
        return {"error": "insufficient cores"}

    print(f"\n  Latency core    : core {latency_core} (CCD {lat_ccd_idx})")
    print(f"  Same-CCD BW     : {len(same_ccx_cores)} cores — {same_ccx_cores}")
    print(f"  Other-CCD BW    : {len(other_ccd_cores)} cores — {other_ccd_cores}")
    print(f"  Loading order   : {ordered_bw_cores}")
    print(f"  Max BW threads  : {max_bw_threads}")

    # ── Build latency chase buffer ──
    print(f"\n  Building {latency_buf_mb} MB pointer-chase buffer...")
    rng = np.random.default_rng(rng_seed)
    lat_buf_bytes = latency_buf_mb * 1024 * 1024
    # Page size of the latency buffer (6.97.1). 2 MB pages keep page walks out
    # of the measurement; 4 KB pages add one per hop (to DRAM under load).
    page_mode = page_mode if page_mode in ("2m", "4k") else "2m"
    lat_pages: Dict[str, Any] = {"requested": page_mode, "used": None,
                                 "fallback_reason": None, "fallback_hint": None}
    lat_buf = None
    if page_mode == "2m":
        try:
            lat_buf, lat_n_nodes = build_random_chase(
                lat_buf_bytes, 64, rng=rng, alloc=make_page_allocator("2m"))
            if sys.platform.startswith("linux"):
                _cov = hugepage_coverage_pct(lat_buf)
                if _cov is None or _cov < LINUX_2M_MIN_COVERAGE_PCT:
                    raise LargePageUnavailable(
                        f"only {_cov if _cov is not None else '?'}% of the buffer got 2 MB pages",
                        "Memory is fragmented. Run soon after boot, or: "
                        "echo 1 | sudo tee /proc/sys/vm/compact_memory")
            lat_pages["used"] = "2m"
        except LargePageUnavailable as e:
            lat_buf = None
            gc.collect()
            lat_pages.update(fallback_reason=e.reason, fallback_hint=e.hint)
            print("\n  " + "!" * 68)
            print("  !! 2 MB pages unavailable -- falling back to 4 KB pages.")
            print(f"  !! Reason: {e.reason}")
            if e.hint:
                print(f"  !! Fix   : {e.hint}")
            print("  !! On 4 KB pages each loaded hop also pays a page walk that goes")
            print("  !! to DRAM under load, so loaded latency reads roughly 2x higher")
            print("  !! than a 2 MB-page tool such as MLC.")
            print("  " + "!" * 68 + "\n")
            rng = np.random.default_rng(rng_seed)          # same chain as a clean 4 KB run
    if lat_buf is None:
        lat_buf, lat_n_nodes = build_random_chase(
            lat_buf_bytes, 64, rng=rng, alloc=make_page_allocator("4k"))
        lat_pages["used"] = "4k"
    _pg_txt = "2 MB" if lat_pages["used"] == "2m" else "4 KB"
    print(f"  Latency buffer pages: {_pg_txt}"
          + ("" if lat_pages["used"] == page_mode else "  (fallback; 2 MB requested)"))

    # ── Bandwidth calibration (verify workers generate real traffic) ──
    print("  Calibrating: verifying bandwidth worker generates real DRAM traffic...")
    cal_worker = _BandwidthWorkerMP(array_mb=bw_buf_mb, core_id=phys_cores[1] if len(phys_cores) > 1 else phys_cores[0])
    cal_worker.start()
    if not cal_worker.wait_ready():
        cal_worker.stop()
        cal_worker.join(timeout=5.0)
        print("  ERROR: the calibration bandwidth worker did not start; test aborted.")
        print(f"         ({bw_buf_mb} MB per worker requested -- try a smaller BW buffer.)")
        return {"error": "bandwidth worker failed to start"}
    time.sleep(2.0)  # let it run for 2 seconds
    p0 = cal_worker.snapshot_passes()
    t_cal0 = time.perf_counter()
    time.sleep(3.0)  # measure for 3 seconds
    p1 = cal_worker.snapshot_passes()
    t_cal1 = time.perf_counter()
    cal_worker.stop()
    cal_worker.join()
    cal_passes = p1 - p0
    cal_elapsed = t_cal1 - t_cal0
    cal_bw = (cal_passes * cal_worker.bytes_per_pass / cal_elapsed) / 1e9 if cal_elapsed > 0 else 0
    print(f"  Calibration: {cal_passes} passes in {cal_elapsed:.1f}s = {cal_bw:.1f} GB/s per worker (Numba JIT)")
    if cal_bw < 1.0:
        print(f"  WARNING: Single-worker BW is low ({cal_bw:.1f} GB/s). Results may understate contention.")
    else:
        print(f"  OK: Worker generating {cal_bw:.1f} GB/s of sequential streaming DRAM traffic.")
    del cal_worker

    # ── Baseline measurement (no load) ──
    print("  Measuring baseline (unloaded) latency...")
    try:
        proc = psutil.Process(os.getpid())
        proc.cpu_affinity([latency_core])
    except Exception as e:
        print(f"  Warning: Could not pin to core {latency_core}: {e}")

    # Run context (6.96): page backing of the latency buffer and the clock of
    # the (pinned) latency core, recorded in the JSON.
    lat_buf_thp = hugepage_coverage_pct(lat_buf)
    lat_clock = measure_clock_ghz()
    lat_cpu_now = current_cpu()
    print(f"  Latency core clock: {lat_clock} GHz (cpu {lat_cpu_now})"
          f"  |  latency buffer huge-page coverage: "
          f"{'n/a' if lat_buf_thp is None else f'{lat_buf_thp:.0f}%'}")

    # Warmup
    _chase_kernel(lat_buf, lat_n_nodes, max(1, 500_000 // lat_n_nodes))

    # Calibrate for measurement window
    cal_traversals = calibrate_chase(lat_buf, lat_n_nodes, measure_seconds)

    results_data = []

    def _measure_latency(label: str, n_bw_threads: int,
                         active_workers: List[_BandwidthWorkerMP]) -> Dict:
        """Measure latency from the chase thread with current BW load."""
        # Let bandwidth workers stabilize their access patterns
        time.sleep(warmup_seconds)

        # gc BEFORE snapshotting to avoid gc during measurement
        gc.collect()

        # Snapshot pass counters (lock-free reads from RawValue)
        pass_starts = [w.snapshot_passes() for w in active_workers]
        t_wall_start = time.perf_counter()

        # Run latency measurement — this is the core pointer chase
        t_lat_start = time.perf_counter()
        _chase_kernel(lat_buf, lat_n_nodes, cal_traversals)
        t_lat_end = time.perf_counter()

        # Snapshot counters again
        t_wall_end = time.perf_counter()
        pass_ends = [w.snapshot_passes() for w in active_workers]

        # Calculate latency
        elapsed_lat = t_lat_end - t_lat_start
        accesses = lat_n_nodes * cal_traversals
        ns_per_hop = (elapsed_lat / accesses) * 1e9

        # Calculate aggregate bandwidth from pass deltas
        elapsed_wall = t_wall_end - t_wall_start
        total_bw_bytes = 0
        for w, ps, pe in zip(active_workers, pass_starts, pass_ends):
            delta_passes = pe - ps
            total_bw_bytes += delta_passes * w.bytes_per_pass
        agg_bw_gbs = (total_bw_bytes / elapsed_wall) / 1e9 if elapsed_wall > 0.001 else 0

        step = {
            "label": label,
            "n_bw_threads": n_bw_threads,
            "latency_ns": round(ns_per_hop, 2),
            "aggregate_bw_gbs": round(agg_bw_gbs, 2),
            "elapsed_s": round(elapsed_lat, 3),
            "accesses": accesses,
            "pass_deltas": [int(pe - ps) for ps, pe in zip(pass_starts, pass_ends)],
        }
        return step

    # ── Step 0: Baseline (no BW threads) ──
    print(f"\n  {'Step':<6} {'BW Thr':>7} {'Latency (ns)':>14} {'Agg BW (GB/s)':>15} {'Label'}")
    print("  " + "-" * 65)

    baseline = _measure_latency("baseline (unloaded)", 0, [])
    results_data.append(baseline)
    print(f"  {0:<6} {0:>7} {baseline['latency_ns']:>14.2f} {0:>15.2f} {baseline['label']}")

    # ── Progressive loading ──
    active_workers: List[_BandwidthWorkerMP] = []

    try:
        for step_idx, bw_core in enumerate(ordered_bw_cores):
            # Determine label
            in_same_ccd = bw_core in same_ccx_cores
            if in_same_ccd:
                label = f"same-CCD core {bw_core}"
            else:
                # Find which CCD
                bw_ccd = "?"
                for idx, g in enumerate(topo["ccd_groups"]):
                    if bw_core in g:
                        bw_ccd = str(idx)
                        break
                label = f"other-CCD({bw_ccd}) core {bw_core}"

            # Spawn bandwidth worker
            worker = _BandwidthWorkerMP(array_mb=bw_buf_mb, core_id=bw_core)
            try:
                worker.start()
                if not worker.wait_ready():
                    worker.stop()
                    worker.join(timeout=5.0)
                    print(f"  WARNING: Worker on core {bw_core} did not start (likely out of memory).")
                    print(f"           Stopping test at {len(active_workers)} BW threads.")
                    break
            except Exception as e:
                print(f"  WARNING: Failed to spawn worker on core {bw_core}: {e}")
                print(f"           Stopping test at {len(active_workers)} BW threads.")
                break

            active_workers.append(worker)

            # Re-pin main thread (latency) to latency core
            try:
                proc.cpu_affinity([latency_core])
            except Exception:
                pass

            n_active = len(active_workers)
            step_result = _measure_latency(label, n_active, active_workers)
            results_data.append(step_result)

            lat_str = f"{step_result['latency_ns']:>14.2f}"
            bw_str = f"{step_result['aggregate_bw_gbs']:>15.2f}"
            print(f"  {step_idx + 1:<6} {n_active:>7} {lat_str} {bw_str} +{label}")

    except KeyboardInterrupt:
        print("\n  Interrupted — stopping bandwidth threads...")
    finally:
        # Stop all workers
        for w in active_workers:
            w.stop()
        for w in active_workers:
            w.join(timeout=5.0)

    # Reset affinity
    try:
        proc.cpu_affinity([])
    except Exception:
        pass

    # ── Analysis ──
    baseline_lat = results_data[0]["latency_ns"] if results_data else 0
    worst_lat = max(r["latency_ns"] for r in results_data) if results_data else 0
    max_bw = max(r["aggregate_bw_gbs"] for r in results_data) if results_data else 0

    print("\n" + "=" * 72)
    print("  LOADED LATENCY ANALYSIS")
    print("=" * 72)
    print(f"  L3 cache / latency buffer    : {l3_total_mb} MB / {latency_buf_mb} MB"
          f"  ({lat_coverage:.1f}% coverage)")
    print(f"  Baseline latency (unloaded)  : {baseline_lat:.2f} ns")
    print(f"  Worst-case latency (loaded)  : {worst_lat:.2f} ns")
    if baseline_lat > 0:
        degradation = worst_lat / baseline_lat
        print(f"  Degradation factor           : {degradation:.1f}x")
    print(f"  Peak aggregate bandwidth     : {max_bw:.2f} GB/s")

    # Detect compounding: find where latency jumps most
    if len(results_data) >= 3:
        max_jump = 0
        max_jump_step = 0
        for i in range(1, len(results_data)):
            jump = results_data[i]["latency_ns"] - results_data[i-1]["latency_ns"]
            if jump > max_jump:
                max_jump = jump
                max_jump_step = i
        if max_jump > 0:
            print(f"  Largest latency spike        : +{max_jump:.1f} ns at step {max_jump_step}"
                  f" ({results_data[max_jump_step]['label']})")

    # Detect CCD boundary effect
    if len(topo["ccd_groups"]) > 1 and same_ccx_cores and other_ccd_cores:
        last_same_idx = len(same_ccx_cores)  # index in results_data (add 1 for baseline)
        if last_same_idx < len(results_data) and last_same_idx + 1 < len(results_data):
            lat_before_cross = results_data[last_same_idx]["latency_ns"]
            lat_after_cross = results_data[last_same_idx + 1]["latency_ns"]
            cross_ccd_effect = lat_after_cross - lat_before_cross
            print(f"\n  Cross-CCD effect             : {cross_ccd_effect:+.1f} ns"
                  f" (transition from same-CCD to other-CCD bandwidth load)")

    # ── Chiplet penalty assessment ──
    print(f"\n  Interpretation:")
    if baseline_lat > 0:
        ratio = worst_lat / baseline_lat
        if ratio > 5:
            print("    SEVERE chiplet/interconnect penalty detected.")
            print("    XI queue starvation and IFOP contention are compounding.")
            print("    This matches Zen 4 behavior described by Chips & Cheese.")
        elif ratio > 3:
            print("    MODERATE interconnect penalty detected.")
            print("    Queue contention is visible under heavy bandwidth load.")
        elif ratio > 2:
            print("    MILD penalty. Memory subsystem handles load reasonably well.")
            print("    Zen 5 or well-configured monolithic designs show this pattern.")
        else:
            print("    MINIMAL penalty. Memory subsystem is well-balanced under load.")
            print("    Typical of monolithic designs or VCache-equipped chips")
            print("    with low L3 miss rates (e.g., 7800X3D in gaming workloads).")
    print("=" * 72)

    # ── Build final results dict ──
    final = {
        "test": "loaded_latency",
        "cpu_model": cpu_info["model"],
        "gen_key": cpu_info.get("gen_key"),
        "freq_ghz": freq_ghz,
        "topology": {
            "vendor": vendor,
            "n_physical": n_phys,
            "ccd_groups": topo["ccd_groups"],
            "latency_core": latency_core,
            "bw_core_order": ordered_bw_cores,
        },
        "config": {
            "latency_buf_mb": latency_buf_mb,
            "bw_buf_mb_per_thread": bw_buf_mb,
            "measure_seconds": measure_seconds,
            "cache_coverage_pct": cache_coverage_pct,
            "l3_total_mb": l3_total_mb,
            "latency_cache_coverage_pct": round(lat_coverage, 2),
            "bw_cache_coverage_pct": round(bw_coverage, 2),
            "latency_pages": lat_pages,
        },
        "steps": results_data,
        "summary": {
            "baseline_ns": baseline_lat,
            "worst_ns": worst_lat,
            "degradation_x": round(worst_lat / baseline_lat, 2) if baseline_lat > 0 else 0,
            "peak_bw_gbs": max_bw,
        },
        "run_context": {
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "numba": numba.__version__ if HAS_NUMBA else None},
            "pinning": {"mode": "pinned (latency thread)", "latency_core": latency_core,
                        "cpu_seen_at_baseline": lat_cpu_now,
                        "bandwidth_worker_cores": ordered_bw_cores},
            "smt": detect_smt_state(),
            "pages": dict(memory_page_context(),
                          latency_buf_hugepage_coverage_pct=lat_buf_thp,
                          latency_buf_pages_requested=lat_pages["requested"],
                          latency_buf_pages_used=lat_pages["used"],
                          bandwidth_buffers=("np.ones in each worker process (NumPy's own "
                                             "MADV_HUGEPAGE >= 4 MiB on Linux); not measured")),
            "clock": {"detected_ghz": freq_ghz,
                      "measured_ghz_latency_core": lat_clock,
                      "method": (f"dependent rotate+add chain, {CLOCK_PROBE_CYCLES_PER_ITER} "
                                 "cycles/iteration, best of 3, measured on the pinned "
                                 "latency core before the baseline"),
                      "approximate": False},
        },
    }

    # ── Save JSON ──
    if output_dir is None:
        output_dir = os.path.expanduser("~")
        for d in [os.path.join(output_dir, "Documents"), output_dir, os.getcwd()]:
            if os.path.isdir(d):
                output_dir = d
                break
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"loaded_latency_{ts}.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)
        print(f"\n  Results saved: {os.path.abspath(json_path)}")
    except Exception as e:
        print(f"  Warning: Could not save JSON: {e}")

    # ── Generate HTML report ──
    html_path = os.path.join(output_dir, f"loaded_latency_{ts}.html")
    try:
        generate_loaded_latency_html(final, html_path)
    except Exception as e:
        print(f"  Warning: Could not save HTML: {e}")

    # ── Plot if matplotlib available ──
    if HAS_MATPLOTLIB and len(results_data) >= 2:
        _plot_loaded_latency(final, output_dir, ts)

    del lat_buf
    gc.collect()
    return final


def _plot_loaded_latency(data: Dict, output_dir: str, ts: str) -> None:
    """Generate a dual-axis plot: latency and bandwidth vs thread count."""
    steps = data["steps"]
    n_threads = [s["n_bw_threads"] for s in steps]
    latencies = [s["latency_ns"] for s in steps]
    bandwidths = [s["aggregate_bw_gbs"] for s in steps]
    labels = [s["label"] for s in steps]

    fig, ax1 = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0a0a0f")
    ax1.set_facecolor("#0a0a0f")

    # Latency (left axis)
    color_lat = "#ff4444"
    ax1.plot(n_threads, latencies, "o-", color=color_lat, linewidth=2.2,
             markersize=7, label="Latency (ns)", zorder=3)
    ax1.set_xlabel("Bandwidth Threads", color="#ccc", fontsize=11)
    ax1.set_ylabel("Latency (ns)", color=color_lat, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_lat, colors="#888")
    ax1.tick_params(axis="x", colors="#888")

    # Shade baseline
    baseline = latencies[0] if latencies else 0
    ax1.axhline(baseline, color=color_lat, ls="--", lw=0.8, alpha=0.4,
                label=f"Baseline: {baseline:.1f} ns")

    # Mark CCD boundary if applicable
    topo = data.get("topology", {})
    ccd_groups = topo.get("ccd_groups", [])
    if len(ccd_groups) > 1:
        same_ccd_count = len([c for c in topo.get("bw_core_order", [])
                              if any(c in g for i, g in enumerate(ccd_groups)
                                     if i == 0)])
        if 0 < same_ccd_count < len(n_threads) - 1:
            ax1.axvline(same_ccd_count, color="#ffaa00", ls=":", lw=1.5, alpha=0.7)
            ax1.text(same_ccd_count + 0.1, max(latencies) * 0.9,
                     "← CCD boundary", color="#ffaa00", fontsize=8, alpha=0.8)

    # Bandwidth (right axis)
    ax2 = ax1.twinx()
    color_bw = "#44aaff"
    ax2.plot(n_threads, bandwidths, "s--", color=color_bw, linewidth=1.8,
             markersize=6, label="Bandwidth (GB/s)", zorder=2)
    ax2.set_ylabel("Aggregate Bandwidth (GB/s)", color=color_bw, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_bw, colors="#888")

    # Title and legend
    cpu_model = data.get("cpu_model", "Unknown")
    freq = data.get("freq_ghz")
    freq_str = f" @ {freq:.2f} GHz" if freq else ""
    summary = data.get("summary", {})
    deg = summary.get("degradation_x", 0)

    ax1.set_title(
        f"Loaded Latency — {cpu_model}{freq_str}\n"
        f"Baseline: {summary.get('baseline_ns', 0):.1f} ns → "
        f"Worst: {summary.get('worst_ns', 0):.1f} ns "
        f"({deg:.1f}x degradation)  |  Peak BW: {summary.get('peak_bw_gbs', 0):.1f} GB/s",
        color="#ddd", fontsize=10, pad=12)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
               facecolor="#1a1a2e", labelcolor="#ccc", edgecolor="#444",
               loc="upper left")

    ax1.grid(True, alpha=0.15, color="#555")
    ax1.spines[:].set_color("#333")
    ax2.spines[:].set_color("#333")

    # Annotate each point with its label (rotated for readability)
    for i, (nt, lat, lbl) in enumerate(zip(n_threads, latencies, labels)):
        if i > 0 and i % 2 == 0:  # skip some to avoid clutter
            ax1.annotate(lbl.replace("same-CCD ", "").replace("other-CCD", "xCCD"),
                         (nt, lat), textcoords="offset points", xytext=(5, 8),
                         fontsize=6, color="#aaa", rotation=30, alpha=0.7)

    plot_path = os.path.join(output_dir, f"loaded_latency_{ts}.png")
    try:
        plt.savefig(plot_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Plot saved : {os.path.abspath(plot_path)}")
    except Exception as e:
        print(f"  Warning: Could not save plot: {e}")
    print("  Displaying plot (close window to continue)...")
    plt.show()


def generate_loaded_latency_html(data: Dict, html_path: str) -> None:
    """Generate a self-contained interactive HTML report for loaded latency results."""
    steps = data.get("steps", [])
    summary = data.get("summary", {})
    config = data.get("config", {})
    topology = data.get("topology", {})
    cpu_model = data.get("cpu_model", "Unknown CPU")
    freq_ghz = data.get("freq_ghz")
    freq_str = f"{freq_ghz:.2f} GHz" if freq_ghz else "N/A"

    baseline = summary.get("baseline_ns", 0)
    worst = summary.get("worst_ns", 0)
    degradation = summary.get("degradation_x", 0)
    peak_bw = summary.get("peak_bw_gbs", 0)

    # Severity assessment
    if degradation > 5:
        severity = "SEVERE"
        severity_color = "#ff2222"
        severity_desc = ("XI queue starvation and IFOP contention are compounding catastrophically. "
                         "Latency-sensitive threads experience order-of-magnitude penalties when "
                         "other cores generate sustained memory traffic. This matches Zen 4 behavior "
                         "documented by Chips &amp; Cheese.")
    elif degradation > 3:
        severity = "MODERATE"
        severity_color = "#ffaa00"
        severity_desc = ("Queue contention is clearly visible under heavy bandwidth load. "
                         "The interconnect cannot fully service all cores simultaneously without "
                         "significant latency penalties to contending threads.")
    elif degradation > 2:
        severity = "MILD"
        severity_color = "#44aaff"
        severity_desc = ("The memory subsystem handles load reasonably well. Some degradation "
                         "is expected and the curve is relatively smooth. Typical of Zen 5 or "
                         "well-configured systems with adequate interconnect headroom.")
    else:
        severity = "MINIMAL"
        severity_color = "#44ff88"
        severity_desc = ("Memory subsystem is well-balanced under load. Degradation is smooth "
                         "and moderate. Typical of monolithic designs or workloads that remain "
                         "mostly within cache.")

    # Build step table rows
    step_rows = ""
    prev_lat = 0
    for i, s in enumerate(steps):
        lat = s.get("latency_ns", 0)
        bw = s.get("aggregate_bw_gbs", 0)
        delta = lat - prev_lat if i > 0 else 0
        label = s.get("label", "")
        # Color-code latency: green < 150, yellow < 300, orange < 500, red >= 500
        if lat < 150:
            lat_color = "#44ff88"
        elif lat < 300:
            lat_color = "#ffaa00"
        elif lat < 500:
            lat_color = "#ff8844"
        else:
            lat_color = "#ff2222"
        delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}" if delta < 0 else "—"
        delta_color = "#ff4444" if delta > 50 else ("#ffaa00" if delta > 20 else "#888")
        step_rows += f"""<tr>
            <td>{i}</td>
            <td>{s.get('n_bw_threads', 0)}</td>
            <td style="color:{lat_color};font-weight:bold">{lat:.1f} ns</td>
            <td style="color:{delta_color}">{delta_str} ns</td>
            <td>{bw:.1f} GB/s</td>
            <td style="color:#888">{label}</td>
        </tr>"""
        prev_lat = lat

    # Detect the "knee" — where latency acceleration is worst
    max_jump = 0
    knee_step = 0
    for i in range(1, len(steps)):
        jump = steps[i]["latency_ns"] - steps[i-1]["latency_ns"]
        if jump > max_jump:
            max_jump = jump
            knee_step = i

    # Chart data as JSON
    chart_json = json.dumps({
        "threads": [s["n_bw_threads"] for s in steps],
        "latencies": [s["latency_ns"] for s in steps],
        "bandwidths": [s["aggregate_bw_gbs"] for s in steps],
        "labels": [s.get("label", "") for s in steps],
    })

    # Topology info
    n_phys = topology.get("n_physical", "?")
    vendor = topology.get("vendor", "?")
    ccd_groups = topology.get("ccd_groups", [])
    lat_core = topology.get("latency_core", 0)

    # Config info
    lat_buf = config.get("latency_buf_mb", "?")
    bw_buf = config.get("bw_buf_mb_per_thread", "?")
    l3_mb = config.get("l3_total_mb", "?")
    lat_cov = config.get("latency_cache_coverage_pct", "?")
    bw_cov = config.get("bw_cache_coverage_pct", "?")
    cov_target = config.get("cache_coverage_pct", 5.0)
    # Latency-buffer page size (6.97.1); files from older versions have none
    _lp = config.get("latency_pages") or {}
    _lp_used = {"2m": "2 MB", "4k": "4 KB"}.get(_lp.get("used"), "not recorded")
    _lp_req = {"2m": "2 MB", "4k": "4 KB"}.get(_lp.get("requested"))
    pages_line = (f"Latency buffer pages: {_lp_used}"
                  + (f" (fallback; {_lp_req} requested)" if _lp_req and _lp.get("used") != _lp.get("requested") else ""))
    pages_banner = ""
    if _lp.get("fallback_reason"):
        pages_banner = (
            "<div class='knee-callout'><div class='title'>2 MB pages unavailable &mdash; this run used 4 KB pages</div>"
            f"<div class='desc'>Reason: {_lp.get('fallback_reason')}"
            + (f"<br>Fix: {_lp.get('fallback_hint')}" if _lp.get('fallback_hint') else "")
            + "<br>On 4 KB pages each loaded hop also pays a page walk that goes to DRAM under "
              "load, so loaded latency reads roughly 2&times; higher than a 2 MB-page tool such as MLC.</div></div>")

    # Generational comparison context
    gen_comparison = ""
    if degradation > 0:
        gen_comparison = f"""
        <div class="ll-section">
            <h3>Generational Context</h3>
            <table>
                <tr><th>Architecture</th><th>Era</th><th>Worst-Case Loaded Latency</th><th>Degradation</th></tr>
                <tr><td>Intel Comet Lake (10900K)</td><td>2020</td><td style="color:#44ff88">~234 ns (all 10 cores)</td><td>~2.1x</td></tr>
                <tr><td>AMD Zen 2 (3950X)</td><td>2019</td><td style="color:#44aaff">~285 ns (all CCD cores)</td><td>~4.0x</td></tr>
                <tr><td style="color:#fff;font-weight:bold">This CPU</td><td style="color:#fff">—</td>
                    <td style="color:{severity_color};font-weight:bold">{worst:.0f} ns</td>
                    <td style="color:{severity_color};font-weight:bold">{degradation:.1f}x</td></tr>
            </table>
            <p style="color:#888;font-size:0.82em;margin-top:8px">
                Reference data from Chips &amp; Cheese "Pushing AMD's Infinity Fabric to its Limits" (Nov 2024).
                Zen 2 and Comet Lake tested under comparable all-core loaded latency methodology.
            </p>
        </div>"""

    chartjs_tag = chartjs_script_tag()   # inline Chart.js: report works offline (6.96)
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Loaded Latency Report — {cpu_model}</title>
{chartjs_tag}
<style>
:root {{ --bg:#0a0a0f; --card:#12121e; --border:#2a2a3e; --text:#d0d0e0;
         --accent:#ff4444; --blue:#44aaff; --orange:#ffaa00; --green:#44ff88; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
        line-height:1.6; padding:20px; max-width:1400px; margin:0 auto; }}
h1 {{ color:#fff; font-size:1.8em; margin-bottom:4px; }}
h2 {{ color:#ccc; font-size:1.2em; margin:24px 0 12px; border-bottom:1px solid var(--border); padding-bottom:6px; }}
.header {{ background:linear-gradient(135deg,#1a1a2e,#16213e); border-radius:12px;
           padding:24px 32px; margin-bottom:24px; border:1px solid var(--border); position:relative; }}
.header-sub {{ color:#888; font-size:0.9em; }}
.severity-badge {{ display:inline-block; background:{severity_color}; color:#fff; font-size:1.1em;
                   font-weight:bold; padding:8px 20px; border-radius:20px; float:right; margin-top:-5px; }}
.degradation-big {{ font-size:2.4em; font-weight:bold; color:{severity_color}; }}
.stats-row {{ display:flex; gap:16px; flex-wrap:wrap; margin:16px 0; }}
.stat-box {{ background:var(--card); border:1px solid var(--border); border-radius:8px;
             padding:12px 20px; min-width:140px; flex:1; }}
.stat-box .val {{ font-size:1.4em; font-weight:bold; color:#fff; }}
.stat-box .lbl {{ font-size:0.8em; color:#888; }}
.chart-container {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
                    padding:20px; margin:16px 0; }}
canvas {{ max-height:500px; }}
table {{ width:100%; border-collapse:collapse; font-size:0.85em; margin:12px 0; }}
th {{ background:#1a1a2e; color:#aaa; padding:8px 12px; text-align:left; border-bottom:2px solid var(--border); }}
td {{ padding:6px 12px; border-bottom:1px solid #1a1a2e; }}
tr:hover td {{ background:#16213e; }}
.ll-section {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
               padding:24px 28px; margin:16px 0; }}
.ll-section h3 {{ color:#fff; font-size:1.1em; margin:0 0 14px; padding-bottom:8px;
                  border-bottom:1px solid var(--border); }}
.ll-section p {{ color:var(--text); font-size:0.88em; line-height:1.75; margin:0 0 12px; }}
.config-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:16px 0; }}
@media(max-width:800px) {{ .config-grid {{ grid-template-columns:1fr; }} .stats-row {{ flex-direction:column; }} }}
.config-item {{ background:#0d0d18; border:1px solid var(--border); border-radius:8px; padding:14px 18px; }}
.config-item .label {{ color:#888; font-size:0.78em; text-transform:uppercase; letter-spacing:0.05em; }}
.config-item .value {{ color:#fff; font-size:1.05em; font-weight:600; margin-top:2px; }}
.knee-callout {{ background:linear-gradient(135deg,#2a1a0a,#1a1a2e); border:1px solid #5a3a1a;
                 border-radius:10px; padding:18px 24px; margin:16px 0; }}
.knee-callout .title {{ color:var(--orange); font-weight:bold; font-size:0.95em; }}
.knee-callout .desc {{ color:#bbb; font-size:0.85em; margin-top:4px; }}
.footer {{ text-align:center; color:#555; font-size:0.75em; margin-top:32px; }}
</style>
</head>
<body>

<div class="header">
    <div class="severity-badge">{severity}</div>
    <h1>Loaded Latency Report</h1>
    <div class="header-sub">{cpu_model} @ {freq_str}</div>
    <div class="header-sub">{vendor} · {n_phys} physical cores · {len(ccd_groups)} CCD(s)</div>
    <div class="header-sub">Methodology: Chips &amp; Cheese loaded latency (progressive core loading)</div>
    <div class="header-sub">{pages_line}</div>
    <div style="margin-top:12px">
        <span class="degradation-big">{degradation:.1f}x</span>
        <span style="color:#888;font-size:0.9em;margin-left:8px">degradation under full load</span>
    </div>
</div>

<div class="stats-row">
    <div class="stat-box"><div class="val">{baseline:.1f} ns</div><div class="lbl">Baseline (unloaded)</div></div>
    <div class="stat-box"><div class="val" style="color:{severity_color}">{worst:.1f} ns</div><div class="lbl">Worst Case (loaded)</div></div>
    <div class="stat-box"><div class="val">{peak_bw:.1f} GB/s</div><div class="lbl">Peak Aggregate BW</div></div>
    <div class="stat-box"><div class="val">{degradation:.1f}x</div><div class="lbl">Degradation Factor</div></div>
</div>
{pages_banner}

<h2>Latency &amp; Bandwidth vs Core Loading</h2>
<div class="chart-container"><canvas id="loadedChart"></canvas></div>

{"" if knee_step == 0 else f'''
<div class="knee-callout">
    <div class="title">⚠ Saturation Knee Detected at Step {knee_step} — +{max_jump:.0f} ns spike</div>
    <div class="desc">The largest single-step latency jump occurred when adding bandwidth thread #{knee_step}
    ({steps[knee_step]["label"] if knee_step < len(steps) else "?"}). This indicates the point where
    interconnect queue capacity or IFOP bandwidth was exceeded, triggering the compounding delay cascade.</div>
</div>
'''}

<div class="ll-section">
    <h3>Severity Assessment: {severity}</h3>
    <p>{severity_desc}</p>
</div>

{gen_comparison}

<h2>Step-by-Step Results</h2>
<div style="overflow-x:auto">
<table>
    <tr>
        <th>Step</th><th>BW Threads</th><th>Latency</th>
        <th>Delta vs Prev</th><th>Aggregate BW</th><th>Label</th>
    </tr>
    {step_rows}
</table>
</div>

<h2>Test Configuration</h2>
<div class="config-grid">
    <div class="config-item">
        <div class="label">Latency Buffer</div>
        <div class="value">{lat_buf} MB <span style="color:#888;font-size:0.8em">(L3 covers {lat_cov}%)</span></div>
    </div>
    <div class="config-item">
        <div class="label">BW Buffer / Thread</div>
        <div class="value">{bw_buf} MB <span style="color:#888;font-size:0.8em">(L3 covers {bw_cov}%)</span></div>
    </div>
    <div class="config-item">
        <div class="label">L3 Cache Total</div>
        <div class="value">{l3_mb} MB</div>
    </div>
    <div class="config-item">
        <div class="label">Latency Buffer Pages</div>
        <div class="value">{_lp_used}{(f' <span style="color:#ffaa00;font-size:0.8em">(fallback; {_lp_req} requested)</span>' if _lp_req and _lp.get("used") != _lp.get("requested") else "")}</div>
    </div>
    <div class="config-item">
        <div class="label">Coverage Target</div>
        <div class="value">{cov_target}%</div>
    </div>
    <div class="config-item">
        <div class="label">Measurement Window</div>
        <div class="value">{config.get('measure_seconds', '?')}s per step</div>
    </div>
    <div class="config-item">
        <div class="label">Latency Core</div>
        <div class="value">Core {lat_core}</div>
    </div>
    <div class="config-item">
        <div class="label">Topology</div>
        <div class="value">{len(ccd_groups)} CCD(s) — {[len(g) for g in ccd_groups]} cores each</div>
    </div>
    <div class="config-item">
        <div class="label">BW Worker Order</div>
        <div class="value">{topology.get('bw_core_order', '?')}</div>
    </div>
</div>

<h2>Methodology</h2>
<div class="ll-section">
    <h3>How This Test Works</h3>
    <p>A <strong>latency thread</strong> is pinned to a single core, running a random pointer-chase through a
    buffer large enough that only {lat_cov}% fits in L3 cache. Every hop is a serialized dependent load that
    must complete before the next can begin — this measures true memory access latency with no opportunity
    for the CPU to hide or overlap the wait.</p>
    <p><strong>Bandwidth workers</strong> are spawned one at a time on other physical cores, each running a
    Numba JIT-compiled sequential streaming kernel through a {bw_buf} MB array. Sequential access enables
    the hardware prefetcher, allowing each core to generate 30-35 GB/s of sustained DRAM traffic — the
    maximum the memory subsystem can serve per core.</p>
    <p>As workers are added, the latency thread's requests must compete with bandwidth traffic for
    <strong>XI queue entries</strong> (finite slots tracking outstanding L3 misses),
    <strong>IFOP bandwidth</strong> (the 32 bytes/cycle link between CCD and IO die), and
    <strong>memory controller scheduling</strong>. Delays at each level compound — a request delayed at the XI
    arrives late to the IFOP, which arrives late to DRAM. This compounding effect is what produces the
    nonlinear latency explosion visible in the chart above.</p>
    <p style="color:#888;font-size:0.82em">Inspired by "Pushing AMD's Infinity Fabric to its Limits" by
    clamchowder at Chips &amp; Cheese (November 2024). Buffer sizes are auto-calculated so L3 covers
    {cov_target}% of the latency working set, enabling consistent cross-platform comparison.</p>
</div>

<div class="footer">Generated by MemLat Pro v{VERSION} | {datetime.now().strftime('%Y-%m-%d %H:%M')} | Loaded Latency Test</div>

<script>
const D = {chart_json};
const ctx = document.getElementById('loadedChart').getContext('2d');

new Chart(ctx, {{
    type: 'line',
    data: {{
        labels: D.threads.map(t => t + ' threads'),
        datasets: [
            {{
                label: 'Latency (ns)',
                data: D.latencies,
                borderColor: '#ff4444',
                backgroundColor: 'rgba(255,68,68,0.1)',
                borderWidth: 2.5,
                pointRadius: 6,
                pointBackgroundColor: D.latencies.map(l =>
                    l < 150 ? '#44ff88' : l < 300 ? '#ffaa00' : l < 500 ? '#ff8844' : '#ff2222'),
                pointBorderColor: '#fff',
                pointBorderWidth: 1,
                fill: true,
                tension: 0.2,
                yAxisID: 'y',
            }},
            {{
                label: 'Bandwidth (GB/s)',
                data: D.bandwidths,
                borderColor: '#44aaff',
                backgroundColor: 'rgba(68,170,255,0.05)',
                borderWidth: 2,
                pointRadius: 4,
                pointBackgroundColor: '#44aaff',
                borderDash: [6, 3],
                fill: false,
                tension: 0.2,
                yAxisID: 'y1',
            }}
        ]
    }},
    options: {{
        responsive: true,
        interaction: {{ mode: 'index', intersect: false }},
        plugins: {{
            tooltip: {{
                callbacks: {{
                    afterBody: function(items) {{
                        const idx = items[0].dataIndex;
                        return 'Core: ' + D.labels[idx];
                    }}
                }}
            }},
            legend: {{ labels: {{ color: '#aaa' }} }}
        }},
        scales: {{
            x: {{
                ticks: {{ color: '#888' }},
                grid: {{ color: 'rgba(255,255,255,0.05)' }}
            }},
            y: {{
                type: 'linear',
                position: 'left',
                title: {{ display: true, text: 'Latency (ns)', color: '#ff4444' }},
                ticks: {{ color: '#ff4444' }},
                grid: {{ color: 'rgba(255,68,68,0.08)' }},
                min: 0,
            }},
            y1: {{
                type: 'linear',
                position: 'right',
                title: {{ display: true, text: 'Bandwidth (GB/s)', color: '#44aaff' }},
                ticks: {{ color: '#44aaff' }},
                grid: {{ drawOnChartArea: false }},
                min: 0,
            }}
        }}
    }}
}});
</script>
</body></html>"""

    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  HTML saved : {os.path.abspath(html_path)}")
    except Exception as e:
        print(f"  Warning: Could not save HTML: {e}")


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
    # (6.98) every boundary gets a label; a 5th one used to have none and
    # crashed analyze() before the JSON was written.
    for i, b in enumerate(boundaries):
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
 
    # Writeback overhead: extra ns/hop of the dirty-writeback chase over the
    # plain random chase in the L3 range. Lower is better: best ~0 ns, worst ~10 ns.
    wb = summary.get("writeback_overhead_ns")
    wb_score = _score_ns(wb, 0.0, 10.0)
    # 0 is a legitimate worst-case score; only missing data is neutral (50).
    # (Before 6.96 this read `_score_ns(...) or 50`, which turned every
    #  overhead of ~9.9 ns or more into a neutral 50 instead of 0.)
    scores["Writeback Overhead"] = wb_score if wb_score is not None else 50

    # 4 KB-page random access (6.97): ~2 GB random chase on 4 KB pages, page
    # walks included. Graded separately and NOT part of Overall (RAM Latency
    # already covers DRAM). Best ~60 ns, worst ~130 ns.
    r4 = _score_ns(summary.get("ram_4k_ns"), 60.0, 130.0)
    if r4 is not None:
        scores["4K Random Access"] = r4

    # Overall — weighted average (L3 and RAM matter most for real workloads)
    weights = {
        "L1 Latency": 1.0, "L2 Latency": 1.5,
        "L3 Latency": 2.5, "RAM Latency": 2.5,
        "Prefetcher": 1.5, "Writeback Overhead": 1.0,
    }
    total_w = sum(weights.values())
    overall = sum(scores[k] * w for k, w in weights.items() if k in scores) / total_w
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
        "random_chase", "stride64_chase", "stride256_chase", "dirty_writeback",
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
        tlb_test: bool = True,
        pin_core_type: Optional[str] = None,
        rng_seed: int = DEFAULT_RNG_SEED,
        rfo: bool = False,
    ):
        self.fast = fast
        self.quick = quick
        self.bandwidth = bandwidth
        self.rfo_enabled = rfo            # cross-core RFO test (6.96)
        self.rfo_results: Optional[Dict] = None
        self.tlb_test = tlb_test
        self.rng = np.random.default_rng(rng_seed)
        self.rng_seed = rng_seed
 
        self.cpu_info = detect_cpu()
        self.cfg = get_cache_config(self.cpu_info)
        self.freq_ghz = self.cpu_info.get("freq_ghz")
 
        if quick and max_size_mb > QUICK_MAX_MB:
            max_size_mb = QUICK_MAX_MB
 
        self._setup_affinity(pin_core_type)
 
        self.page_modes: Optional[Dict] = None     # 6.97
        mem_limit_bytes: Optional[int] = None
        if HAS_PSUTIL:
            avail_mb = psutil.virtual_memory().available // (1024 ** 2)
            safe_mb = int(avail_mb * 0.45)
            mem_limit_bytes = safe_mb * 1024 * 1024
            if max_size_mb > safe_mb:
                print(f"  Auto-limiting max size to {safe_mb} MB (45% of available RAM)")
                max_size_mb = safe_mb

        self.mem_limit_bytes = mem_limit_bytes
        self.max_bytes = max_size_mb * 1024 * 1024
        self.sizes = generate_test_sizes(self.cfg, self.max_bytes)
        # RAM plateau points (6.96): random-chase-only sizes beyond the size cap,
        # added when fewer than RAM_PLATEAU_MIN_POINTS grid sizes reach 8x L3
        # (e.g. 1 GB + 2 GB in Quick mode on a 96 MB L3). Same memory limit.
        self.plateau_sizes, self.plateau_notes = ram_plateau_sizes(
            self.cfg, self.sizes, mem_limit_bytes)
 
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

        # ── Run context recorded in the JSON (6.96) ──
        self.smt = detect_smt_state()
        self.mem_ctx = memory_page_context()
        self.start_cpu = current_cpu()
        self.start_affinity = current_affinity()
        self.start_clock_ghz = measure_clock_ghz()
        clk_str = (f"{self.start_clock_ghz:.2f} GHz" if self.start_clock_ghz else "n/a")
        if self.pin_mode == "unpinned":
            clk_str += f" on cpu {self.start_cpu} (unpinned: approximate)"
        else:
            clk_str += f" on cpu {self.pinned_cpu}"
        smt_on = self.smt.get("smt_active")
        smt_str = "on" if smt_on else ("off" if smt_on is False else "unknown")
        thp = self.mem_ctx.get("thp_enabled")

        print(f"\n  CPU   : {self.cpu_info['model']}")
        print(f"  Gen   : {gen}  |  Hybrid: {hybrid}")
        print(f"  Freq  : {freq_str} detected  |  measured {clk_str}")
        print(f"  Pin   : {self.pin_mode}  |  SMT: {smt_str}"
              + (f"  |  THP: {thp}" if thp else f"  |  Pages: {self.mem_ctx['base_page_kb']} KB"))
        print(f"  L1/L2 : {self.cfg.get('l1_p_kb')} KB / {self.cfg.get('l2_p_kb')} KB (P-core)")
        if hybrid and self.cfg.get("l1_e_kb"):
            print(f"  L1/L2E: {self.cfg['l1_e_kb']} KB / {self.cfg['l2_e_kb']} KB (E-core cluster)")
        print(f"  L3    : {self.cfg.get('l3_mb')} MB")
        _src = self.cfg.get("cache_source", "table")
        _tc = self.cfg.get("table_cache")
        print(f"  Cache : sizes from {_src}"
              + (f"  (table had {_tc['l1_p_kb']} KB / {_tc['l2_p_kb']} KB / {_tc['l3_mb']} MB)" if _tc else ""))
        print(f"  Sizes : {len(self.sizes)} test points up to {max_size_mb} MB")
        if self.plateau_sizes:
            print(f"  RAM   : + plateau points "
                  f"{', '.join(format_size(s) for s in self.plateau_sizes)} (random chase only)")
        for note in self.plateau_notes:
            print(f"  Note  : {note}")
        print(f"  Mode  : {mode_label}")
        print(f"  TLB   : {'ON' if tlb_test else 'OFF'}")
        print(f"  Seed  : {rng_seed}")
        notes = self.cfg.get("notes")
        if notes:
            print(f"  Note  : {notes}")
        print()
 
    def _setup_affinity(self, pin_core_type: Optional[str]) -> None:
        # Recorded in the JSON (6.96). Default is unpinned, as before.
        self.pin_mode = "unpinned"
        self.pinned_cpu: Optional[int] = None
        # (6.98) affinity before any pinning: the bandwidth triad runs on this
        self.unpinned_affinity: Optional[List[int]] = current_affinity()
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
                self.pin_mode = f"pinned ({pin_core_type}-core)"
                self.pinned_cpu = target[0]
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
    def _measure_size(self, size_bytes: int, plateau_only: bool = False) -> Optional[Dict]:
        size_mb = size_bytes / 1024 / 1024
        size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{size_bytes/1024:.1f} KB"
        result: Dict = {"size_bytes": size_bytes, "size_str": size_str}
        if plateau_only:
            # RAM plateau point (6.96): only the random chase is measured here
            result["ram_plateau_only"] = True
        # Run context for this size (6.96): which CPU we start on and its clock.
        result["cpu"] = current_cpu()
        result["measured_clock_ghz"] = measure_clock_ghz()

        # ── Core chase patterns (random, stride64, stride256) ──
        core_patterns = [
            ("random_chase", 64),
            ("stride64_chase", 64),
            ("stride256_chase", 256),
        ]
        if plateau_only:
            core_patterns = core_patterns[:1]
        for pattern, stride_bytes in core_patterns:
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
                result[pattern]["hugepage_coverage_pct"] = hugepage_coverage_pct(buf)
                del buf
                gc.collect()
            except Exception as e:
                result[pattern] = {"error": str(e)}

        if plateau_only:
            return result

        # ── Dirty writeback (formerly "write_rfo"; measures writeback, not RFO) ──
        try:
            buf, n_nodes = build_dirty_writeback(size_bytes, rng=self.rng)
            target_sec, iters = get_timing_budget(size_bytes, self.cfg,
                                                  self.fast, self.quick)
            traversals = calibrate_dirty_writeback(buf, n_nodes, target_sec)
            samples_ns = []
            for _ in range(iters):
                elapsed = timed_dirty_writeback(buf, n_nodes, traversals)
                accesses = n_nodes * traversals
                samples_ns.append(elapsed / accesses * 1e9)
                time.sleep(0.05)
            result["dirty_writeback"] = percentile_stats(samples_ns)
            result["dirty_writeback"]["total_traversal_ms"] = (
                result["dirty_writeback"]["median"] * n_nodes / 1_000_000
            )
            result["dirty_writeback"]["hugepage_coverage_pct"] = hugepage_coverage_pct(buf)
            del buf
            gc.collect()
        except Exception as e:
            result["dirty_writeback"] = {"error": str(e)}
 
        # ── TLB stress patterns ──
        if self.tlb_test:
            for pat_name, page_sz, page_mode in [("tlb_4k_chase", 4096, "4k"),
                                                  ("tlb_2m_chase", 2 * 1024 * 1024, "2m")]:
                if size_bytes < page_sz * 8:
                    continue  # too small for this stride
                try:
                    buf, n_nodes, page_info = self._build_tlb_buffer(size_bytes, page_sz, page_mode)
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
                    result[pat_name]["hugepage_coverage_pct"] = hugepage_coverage_pct(buf)
                    result[pat_name].update(page_info)
                    del buf
                    gc.collect()
                except LargePageUnavailable as e:
                    # 2 MB pattern without 2 MB pages would not be a 2 MB test: skip it
                    result[pat_name] = {"error": f"2 MB pages unavailable: {e.reason}",
                                        "skipped": True}
                except Exception as e:
                    result[pat_name] = {"error": str(e)}
 
        # ── Bandwidth ──
        if self.bandwidth:
            if bandwidth_fits(size_bytes):
                result["bandwidth_gbs"] = measure_bandwidth(size_bytes, self.unpinned_affinity)
            else:
                result["bandwidth_gbs"] = None
                result["bandwidth_note"] = (f"skipped: needs {BANDWIDTH_ARRAYS} x "
                                            f"{format_size(size_bytes)} of free RAM")
 
        return result
 
    def _build_tlb_buffer(self, size_bytes: int, page_sz: int,
                          page_mode: str) -> Tuple[np.ndarray, int, Dict]:
        """(6.98) TLB-pattern buffer on the page size the pattern is named after:
        tlb_4k_chase on forced 4 KB pages (it used to be THP-backed on Linux from
        4 MB up), tlb_2m_chase on 2 MB pages. Raises LargePageUnavailable when
        2 MB pages cannot be had; the caller records that and skips the pattern."""
        forced = True
        try:
            alloc = make_page_allocator(page_mode)
        except LargePageUnavailable:
            if page_mode == "2m":
                raise
            alloc, forced = alloc_i64, False        # no page-size control on this OS
        buf, n_nodes = build_tlb_chase(size_bytes, page_sz, rng=self.rng, alloc=alloc)
        if page_mode == "2m" and sys.platform.startswith("linux"):
            cov = hugepage_coverage_pct(buf)
            if cov is None or cov < LINUX_2M_MIN_COVERAGE_PCT:
                del buf
                gc.collect()
                raise LargePageUnavailable(
                    f"only {cov if cov is not None else '?'}% of the buffer got 2 MB pages")
        return buf, n_nodes, {"page_kb": 2048 if page_mode == "2m" else mmap.PAGESIZE // 1024,
                              "page_size_forced": forced,
                              "node_layout": "random line per page"}

    def _save_checkpoint(self, results: List[Dict]) -> None:
        """(6.98) Write the raw results to the JSON after every size, so an
        interruption or an error later on cannot lose the sweep. The file is
        marked "partial"; analyze() replaces it with the full payload. Compare
        mode reads a partial file like any other (it recomputes the summary)."""
        try:
            tmp = self.json_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"meta": self._build_meta(results), "partial": True,
                           "results": results}, f, indent=2)
            os.replace(tmp, self.json_path)
        except Exception as e:
            if not getattr(self, "_checkpoint_warned", False):
                self._checkpoint_warned = True
                self._log(f"\n  Warning: could not checkpoint results -- {e}")

    # ── Full run ──────────────────────────────────────────────────────────────
    def run(self) -> List[Dict]:
        self._open_log()
        freq_str = f"{self.freq_ghz:.2f} GHz" if self.freq_ghz else "unknown"
        self._log("=" * 80)
        self._log(f"  MemLat Pro v{VERSION} -- Full Run")
        self._log(f"  Timestamp : {datetime.now().isoformat()}")
        self._log(f"  Platform  : {platform.platform()}")
        self._log(f"  Processor : {self.cpu_info['model']}")
        self._log(f"  CPU Freq  : {freq_str} (detected; used for ns->cycle conversion)")
        self._log(f"  Clock     : {self.start_clock_ghz} GHz measured at start "
                  f"({self.pin_mode}; per-size values in JSON)")
        self._log(f"  Pinning   : {self.pin_mode}  affinity={self.start_affinity}")
        self._log(f"  SMT       : {self.smt}")
        self._log(f"  Pages     : {self.mem_ctx}")
        self._log(f"  Numba     : JIT {numba.__version__}")
        self._log("=" * 80)
        self._log(
            f"{'Size':<12} {'RAND med':>10} {'S64 med':>10} "
            f"{'S256 med':>10} {'DWB med':>10} {'CV%':>6}")
        self._log("-" * 80)
 
        results: List[Dict] = []
        # Grid sizes, then any RAM plateau points (random chase only)
        plan = [(s, False) for s in self.sizes] + [(s, True) for s in self.plateau_sizes]
        total = len(plan)
        for i, (size, plateau_only) in enumerate(plan):
            pct = (i + 1) / total * 100
            bar_w = 30
            filled = int(pct / 100 * bar_w)
            bar = "#" * filled + "-" * (bar_w - filled)
            size_str = format_size(size) + (" (RAM plateau)" if plateau_only else "")
            sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {size_str:<26}")
            sys.stdout.flush()

            r = self._measure_size(size, plateau_only=plateau_only)
            if r is None:
                continue
            results.append(r)
            self._save_checkpoint(results)

            def _med(pat: str) -> str:
                v = r.get(pat, {})
                if not v:
                    return "         -"
                if "error" in v:
                    return "  ERR"
                return f"{v.get('median', 0):10.1f}"
 
            cv = r.get("random_chase", {}).get("cv_pct", 0)
            self._log(
                f"\n{r['size_str']:<12} {_med('random_chase')} {_med('stride64_chase')} "
                f"{_med('stride256_chase')} {_med('dirty_writeback')} {cv:5.1f}%",
                console=False)
 
        sys.stdout.write("\r" + " " * 70 + "\r")
        sys.stdout.flush()
        print("  Measurement complete.")
        return results

    # ── Page-size modes (6.97) ───────────────────────────────────────────────
    def _page_mode_size(self, results: List[Dict]) -> Optional[int]:
        """Working set for the 4 KB vs 2 MB runs: the smallest RAM-plateau size
        (>= 8x L3), else the first power of two above it, within the memory limit."""
        lo = summary_windows(self.cfg)["RAM"][0]
        cands = sorted(r["size_bytes"] for r in results
                       if r.get("size_bytes", 0) >= lo and _stat_ok(r.get("random_chase")))
        size = cands[0] if cands else 1 << max(0, (lo - 1).bit_length())
        if self.mem_limit_bytes is not None and size > self.mem_limit_bytes:
            return None
        return size

    def _measure_random_paged(self, size_bytes: int, mode: str) -> Dict:
        alloc = make_page_allocator(mode)           # may raise LargePageUnavailable
        buf, n_nodes = build_random_chase(size_bytes, 64, rng=self.rng, alloc=alloc)
        out: Dict[str, Any] = {"available": True, "mode": mode}
        cov = hugepage_coverage_pct(buf) if sys.platform.startswith("linux") else None
        if sys.platform.startswith("linux"):
            out["hugepage_coverage_pct"] = cov
            if mode == "2m" and (cov is None or cov < LINUX_2M_MIN_COVERAGE_PCT):
                del buf
                gc.collect()
                raise LargePageUnavailable(
                    f"only {cov if cov is not None else '?'}% of the buffer got 2 MB pages",
                    "Memory is fragmented. Run soon after boot, or: echo 1 | sudo tee "
                    "/proc/sys/vm/compact_memory")
        out["page_kb"] = 2048 if mode == "2m" else 4
        target_sec, iters = get_timing_budget(size_bytes, self.cfg, self.fast, self.quick)
        traversals = calibrate_chase(buf, n_nodes, target_sec)
        samples: List[float] = []
        for _ in range(iters):
            samples.append(timed_chase(buf, n_nodes, traversals) / (n_nodes * traversals) * 1e9)
            time.sleep(0.05)
        out["stats"] = percentile_stats(samples)
        out["median_ns"] = out["stats"]["median"]
        del buf
        gc.collect()
        return out

    def measure_page_modes(self, results: List[Dict]) -> Dict:
        """Random chase at one RAM-plateau size on forced 4 KB pages and on 2 MB
        pages, plus the ~256 MB fallback point taken from the sweep."""
        pm: Dict[str, Any] = {"test": "page_modes"}
        # Fallback headline: the sweep point closest to 256 MB that is beyond L3
        l3 = int(self.cfg.get("l3_mb", 16) * 1024 * 1024)
        pts = [r for r in results if r.get("size_bytes", 0) > l3 and _stat_ok(r.get("random_chase"))]
        if pts:
            best = min(pts, key=lambda r: abs(r["size_bytes"] - PAGE_MODE_FALLBACK_BYTES))
            pm["fallback"] = {"size_bytes": best["size_bytes"],
                              "median_ns": best["random_chase"]["median"]}
        size = self._page_mode_size(results)
        if size is None:
            pm["skipped"] = "RAM plateau size exceeds the memory safety limit"
            self._log(f"\n  Page-size modes skipped: {pm['skipped']}")
            self.page_modes = pm
            return pm
        pm["size_bytes"] = size
        self._log(f"\n  Page-size modes: random chase at {format_size(size)} on 4 KB and 2 MB pages...")
        for mode in ("4k", "2m"):
            try:
                pm[mode] = self._measure_random_paged(size, mode)
                extra = (f", THP coverage {pm[mode]['hugepage_coverage_pct']}%"
                         if pm[mode].get("hugepage_coverage_pct") is not None else "")
                self._log(f"    {mode.upper():>3} pages: {pm[mode]['median_ns']:6.1f} ns{extra}")
            except LargePageUnavailable as e:
                pm[mode] = {"available": False, "mode": mode, "reason": e.reason, "hint": e.hint}
                self._log(f"    {mode.upper():>3} pages: unavailable -- {e.reason}")
                if e.hint:
                    self._log(f"              fix: {e.hint}")
            except Exception as e:
                pm[mode] = {"available": False, "mode": mode, "reason": f"error: {e}", "hint": ""}
                self._log(f"    {mode.upper():>3} pages: error -- {e}")
        self.page_modes = pm
        return pm

    # ── Cross-core RFO (6.96) ───────────────────────────────────────────────
    def measure_cross_core_rfo(self) -> Optional[Dict]:
        """Real read-for-ownership cost between two cores (Section 9A)."""
        self._log("\n  Measuring cross-core RFO (ownership transfer)...")
        try:
            self.rfo_results = run_cross_core_rfo(self.cfg)
        except Exception as e:
            self.rfo_results = {"test": "cross_core_rfo", "skipped": f"error: {e}"}
        for line in format_rfo_report(self.rfo_results):
            self._log(line)
        return self.rfo_results

    # ── Run metadata (shared by JSON and HTML) ────────────────────────────────
    def _build_meta(self, results: List[Dict]) -> Dict:
        """meta block for the JSON/HTML. 6.96 adds run_context: pinning, SMT,
        page size + per-pattern huge-page coverage, and the measured clock."""
        clocks = [r["measured_clock_ghz"] for r in results if r.get("measured_clock_ghz")]
        cpus = sorted({r["cpu"] for r in results if r.get("cpu") is not None})
        coverage: Dict[str, Dict[str, float]] = {}
        for pat in self.ALL_PATTERNS:
            vals = [r[pat]["hugepage_coverage_pct"] for r in results
                    if r.get("size_bytes", 0) >= (2 << 20) and isinstance(r.get(pat), dict)
                    and r[pat].get("hugepage_coverage_pct") is not None]
            if vals:
                coverage[pat] = {"min": min(vals), "median": float(np.median(vals)),
                                 "max": max(vals), "sizes": len(vals)}
        return {
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
            "tlb_patterns": ("6.98 layout: one node per page on a random cache line; "
                             "tlb_4k_chase on forced 4 KB pages, tlb_2m_chase on 2 MB pages. "
                             "Not comparable with TLB columns from before 6.98."),
            "run_context": {
                "versions": {"python": platform.python_version(),
                             "numpy": np.__version__,
                             "numba": numba.__version__ if HAS_NUMBA else None},
                "pinning": {
                    "mode": self.pin_mode,
                    "pinned_cpu": self.pinned_cpu,
                    "affinity_at_start": self.start_affinity,
                    "cpu_at_start": self.start_cpu,
                    "cpus_seen": cpus,   # CPU at the start of each size (per-size: results[i].cpu)
                },
                "smt": self.smt,
                "pages": dict(self.mem_ctx,
                              hugepage_coverage_pct_by_pattern=coverage,
                              per_buffer="results[i][pattern].hugepage_coverage_pct"),
                "clock": {
                    "detected_ghz": self.freq_ghz,
                    "detected_note": ("OS report at startup (Linux/psutil: average current "
                                      "clock of all CPUs; Windows: usually nominal). "
                                      "Still used for the ns->cycle conversions."),
                    "measured_ghz_at_start": self.start_clock_ghz,
                    "measured_ghz_median": float(np.median(clocks)) if clocks else None,
                    "measured_ghz_min": min(clocks) if clocks else None,
                    "measured_ghz_max": max(clocks) if clocks else None,
                    "per_size": "results[i].measured_clock_ghz",
                    "method": (f"dependent rotate+add chain, {CLOCK_PROBE_CYCLES_PER_ITER} "
                               "cycles/iteration, best of 3, before each size"),
                    "approximate": self.pin_mode == "unpinned",
                },
            },
        }

    # ── Build summary dict ────────────────────────────────────────────────────
    def build_summary(self, results: List[Dict]) -> Dict:
        """Compute summary stats used by scoring, diagram, and report.
        6.96: plateau windows, see compute_summary() / summary_windows()."""
        small = sweep_uses_small_pages(sys.platform, self.mem_ctx)
        s = compute_summary(results, self.cfg, small)
        if self.page_modes is not None:
            s = apply_ram_headline(s, self.page_modes, self.cfg)
        return s

    def cycle_clock_ghz(self, results: List[Dict]) -> Tuple[Optional[float], str]:
        """Clock for ns->cycle conversion: median measured clock, else OS value."""
        clocks = [r["measured_clock_ghz"] for r in results if r.get("measured_clock_ghz")]
        if clocks:
            return float(np.median(clocks)), "measured"
        return self.freq_ghz, "OS-reported"
 
    # ── Analysis ──────────────────────────────────────────────────────────────
    def analyze(self, results: List[Dict]) -> Tuple[Dict, List[Dict], Dict]:
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

        # (6.98) Save the JSON before anything is printed: a formatting error
        # below can no longer cost the run. The message is logged at the end.
        json_err: Optional[str] = None
        try:
            payload = {
                "meta": self._build_meta(results),
                "summary": summary,
                "scores": scores,
                "boundaries": boundaries,
                "results": results,
            }
            if self.page_modes is not None:
                payload["page_modes"] = self.page_modes
            if self.rfo_results is not None:
                payload["cross_core_rfo"] = self.rfo_results
            tmp = self.json_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self.json_path)
        except Exception as e:
            json_err = str(e)
 
        clk, clk_src = self.cycle_clock_ghz(results)
        ns_per_cycle = (1.0 / clk) if (clk and clk > 0.1) else (1.0 / 3.0)
        freq_label = f"{clk:.2f} GHz {clk_src}" if clk else "3.00 GHz (est.)"
 
        self._log("\n" + "=" * 80)
        self._log(f"  ANALYSIS SUMMARY -- MemLat Pro v{VERSION}")
        self._log("=" * 80)
 
        # ── Hierarchy diagram ──
        self._log(draw_hierarchy(summary, self.cfg, clk))
 
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
        if summary.get("ram_median_ns") is None:
            ram_lo = summary_windows(self.cfg)["RAM"][0]
            self._log(f"    {'Main Memory':<20} not reached (needs a working set >= "
                      f"{format_size(ram_lo)}, {RAM_PLATEAU_FACTOR}x L3)")
        else:
            self._log(_fmt("Main Memory", summary.get("ram_median_ns"), self.cfg.get("expected_ram_ns")))
            head = summary.get("ram_headline")
            if head:
                self._log(f"    {'':<20} ({head.get('label')})")
        if summary.get("ram_4k_ns") is not None:
            self._log(_fmt("RAM, 4 KB pages", summary["ram_4k_ns"]))
        if summary.get("page_walk_ns") is not None:
            self._log(f"    {'Page-walk cost':<20} {summary['page_walk_ns']:+7.1f} ns  (4 KB minus 2 MB pages)")
        lp = (summary.get("ram_headline") or {}).get("large_pages") or {}
        if lp.get("available") is False and lp.get("hint"):
            self._log(f"    Large pages unavailable: {lp.get('reason')}")
            self._log(f"    To enable: {lp.get('hint')}")

        # ── Which sizes fed each number (6.96 plateau windows) ──
        self._log("\n  Summary windows (median of random chase over these sizes):")
        for level, w in summary.get("windows", {}).items():
            lo, hi = w["lo_bytes"], w["hi_bytes"]
            if hi is None:
                rng_s = f">= {format_size(lo)}"
            elif lo <= 1:
                rng_s = f"<= {format_size(hi)}"
            else:
                rng_s = f"{format_size(lo)} - {format_size(hi)}"
            used = ", ".join(w["sizes"]) if w["sizes"] else "none"
            self._log(f"    {level:<4} {rng_s:<22} {used}")

        # ── TLB results ──
        if summary.get("tlb_4k_ram_ns") or summary.get("tlb_2m_ram_ns"):
            self._log(f"\n  TLB patterns (RAM window; one node per page, random line in the page):")
            if summary.get("tlb_4k_ram_ns"):
                self._log(f"    One node per 4 KB page, 4 KB pages: {summary['tlb_4k_ram_ns']:.1f} ns")
            if summary.get("tlb_2m_ram_ns"):
                self._log(f"    One node per 2 MB page, 2 MB pages: {summary['tlb_2m_ram_ns']:.1f} ns")
            if summary.get("page_walk_ns") is not None:
                self._log(f"    For the cost of a page walk see 'Page-walk cost' above (4 KB minus 2 MB pages).")
 
        # ── Prefetcher + dirty writeback ──
        pf = summary.get("prefetcher_benefit_ns")
        if pf is not None:
            self._log(f"\n  Prefetcher benefit (L2 region): {pf:.1f} ns")
        wb = summary.get("writeback_overhead_ns")
        if wb is not None:
            self._log(f"  L3 dirty-writeback overhead:    {wb:+.1f} ns vs read")
 
        # ── Auto-detected boundaries ──
        if boundaries:
            self._log("\n  Auto-Detected Cache Boundaries:")
            for b in boundaries:
                self._log(f"    {b.get('label', 'unknown'):<12} at ~{b['size_str']:<10} "
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

        # ── JSON (written above, before the analysis was printed) ──
        if json_err is None:
            self._log(f"\n  JSON saved : {os.path.abspath(self.json_path)}")
        else:
            self._log(f"  Warning: Could not save JSON -- {json_err}")
 
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
               f"{'Stride256':>10} {'DirtyWB':>10} "
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
                    f"{_v('stride256_chase')} {_v('dirty_writeback')} "
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
            "dirty_writeback": ("#cc44ff", "Dirty writeback"),
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
 
        # Dirty-writeback overhead subplot
        rand_med = _get("random_chase", "median")
        write_med = _get("dirty_writeback", "median")
        delta_x, delta_y = [], []
        for s, rm, wm in zip(sizes_mb, rand_med, write_med):
            if rm and wm:
                delta_x.append(s)
                delta_y.append(wm - rm)
        if delta_x:
            ax_write.bar(delta_x, delta_y,
                         width=[s * 0.3 for s in delta_x],
                         color="#cc44ff", alpha=0.7)
        ax_write.set_ylabel("Writeback overhead (ns)", color="#ccc", fontsize=8)
        ax_write.set_xlabel("Working Set (MB)", color="#ccc", fontsize=8)
        ax_write.set_title("Dirty-Writeback Overhead vs Read", color="#ddd", fontsize=9)
 
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
                     "dirty_writeback", "tlb_4k_chase", "tlb_2m_chase"]:
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
    # Run context line (6.96)
    rc = meta.get("run_context", {}) or {}
    _clk = (rc.get("clock") or {}).get("measured_ghz_median")
    _pin = (rc.get("pinning") or {}).get("mode", "unknown")
    _smt = (rc.get("smt") or {}).get("smt_active")
    _pg = rc.get("pages") or {}
    ctx_bits = [f"Pinning: {_pin}",
                "SMT: " + ("on" if _smt else ("off" if _smt is False else "?")),
                (f"THP: {_pg['thp_enabled']}" if _pg.get("thp_enabled")
                 else f"Pages: {_pg.get('base_page_kb', '?')} KB"),
                (f"Measured clock: {_clk:.2f} GHz"
                 + (" (approx., unpinned)" if (rc.get('clock') or {}).get('approximate') else "")
                 if _clk else "Measured clock: n/a")]
    ctx_line = " | ".join(ctx_bits) if rc else "Run context not recorded (pre-6.96 file)"
    # Which sizes fed the headline numbers (6.96 plateau windows)
    _wins = summary.get("windows") or {}
    _wparts = []
    for _lvl, _w in _wins.items():
        if _w.get("hi_bytes") is None:
            _rng = f"&ge; {format_size(_w['lo_bytes'])}"
        elif _w["lo_bytes"] <= 1:
            _rng = f"&le; {format_size(_w['hi_bytes'])}"
        else:
            _rng = f"{format_size(_w['lo_bytes'])}&ndash;{format_size(_w['hi_bytes'])}"
        _used = ", ".join(_w.get("sizes") or []) or "not reached"
        _wparts.append(f"<strong>{_lvl}</strong> {_rng} <span style='color:#666'>({_used})</span>")
    windows_html = ("<p style='color:#888;font-size:0.8em;margin:-4px 0 12px'>"
                    "Headline numbers are the median random-chase latency over each level's "
                    "plateau sizes: " + " &middot; ".join(_wparts) + "</p>") if _wparts else ""

    # RAM headline + page-size section (6.97)
    _head = summary.get("ram_headline") or {}
    _lp = _head.get("large_pages") or {}
    _ram_sub = _head.get("label") or ""
    def _nsv(x):
        return f"{x:.1f} ns" if isinstance(x, (int, float)) else "n/a"
    _r4 = summary.get("ram_4k_ns")
    _r4_score = scores.get("4K Random Access")
    _r4_grade = (f" &middot; Grade {_score_grade(_r4_score)} ({_r4_score}/100)"
                 if isinstance(_r4_score, int) else "")
    _pw = summary.get("page_walk_ns")
    _r2 = summary.get("ram_median_ns") if _head.get("source") == "large_pages" else None
    pages_html = ""
    if _head:
        _lp_note = ""
        if _lp.get("available") is False:
            _lp_note = (f"<div class='lp-note'><strong>Large pages unavailable:</strong> "
                        f"{_lp.get('reason') or 'unknown'}"
                        + (f"<br><span style='color:#aaa'>To enable: {_lp.get('hint')}</span>"
                           if _lp.get('hint') else "") + "</div>")
        pages_html = f"""
<h2>RAM Access &amp; Page Walks</h2>
<p style="color:#888;font-size:0.85em;margin:-8px 0 12px">
  The same fully random chase run twice at one large working set. On <strong>2 MB pages</strong>
  almost every hop hits the TLB, so it measures the DRAM path itself (comparable to MLC / AIDA64;
  this is the graded RAM headline). On <strong>4 KB pages</strong> almost every hop misses the TLB
  and waits for a page-table walk first &mdash; what large-footprint software (databases, big hash
  tables, asset streaming) actually pays. The difference is the page-walk cost.
</p>
<div class="stats-row">
  <div class="stat-box"><div class="val">{_nsv(_r2)}</div><div class="lbl">DRAM latency, 2 MB pages (graded as RAM Latency)</div></div>
  <div class="stat-box"><div class="val">{_nsv(_r4)}</div><div class="lbl">Random access, 4 KB pages{_r4_grade} &mdash; not in Overall</div></div>
  <div class="stat-box"><div class="val">{(f"{_pw:+.1f} ns" if isinstance(_pw, (int, float)) else "n/a")}</div><div class="lbl">Page-walk cost (4 KB &minus; 2 MB), not graded</div></div>
</div>
{_lp_note}"""

    # Cross-core RFO section (6.96)
    rfo = payload.get("cross_core_rfo")
    rfo_html = ""
    if rfo and rfo.get("skipped"):
        rfo_html = ("<h2>Cross-Core RFO (ownership transfer)</h2>"
                    f"<p style='color:#888;font-size:0.85em'>Skipped: {rfo['skipped']}</p>")
    elif rfo:
        _labels = [("local_read", "A's own lines, plain load"),
                   ("local_rmw", "A's own lines, locked RMW (lock xadd)"),
                   ("remote_read", "B's dirty lines, plain load"),
                   ("remote_rmw", "B's dirty lines, locked RMW (lock xadd)")]
        _blocks = []
        for _p in rfo.get("pairs", []):
            _rows = ""
            for _k, _lab in _labels:
                _s = (_p.get("variants") or {}).get(_k)
                if _s:
                    _rows += (f"<tr><td>{_lab}</td><td>{_s['median']:.1f}</td>"
                              f"<td>{_s['p5']:.1f}</td><td>{_s['p95']:.1f}</td><td>{_s['max']:.1f}</td></tr>")
                else:
                    _rows += f"<tr><td>{_lab}</td><td colspan='4'>n/a</td></tr>"
            _d = _p.get("derived") or {}
            def _fx(x):
                return f"{x:+.1f} ns" if isinstance(x, (int, float)) else "n/a"
            _sl3 = _p.get("shares_l3")
            _rel = _p.get("label", "") + ("" if _sl3 is None else
                                          (" (shared L3)" if _sl3 else " (separate L3)"))
            _errs = "".join(f"<div style='color:#ffaa00'>! {e}</div>" for e in _p.get("errors", []))
            _blocks.append(f"""
<div class="chart-container">
  <div style="color:#ccc;margin-bottom:6px">Core A = {_p['core_a']} takes lines that core B = {_p['core_b']} just wrote &mdash; {_rel}</div>
  <table>
    <tr><th>Variant</th><th>Median ns/hop</th><th>p5</th><th>p95</th><th>max</th></tr>
    {_rows}
  </table>
  <div class="stats-row">
    <div class="stat-box"><div class="val">{_fx(_d.get('c2c_read_ns'))}</div><div class="lbl">Cache-to-cache read (remote load &minus; local load)</div></div>
    <div class="stat-box"><div class="val">{_fx(_d.get('rfo_transfer_ns'))}</div><div class="lbl">RFO / ownership transfer (remote RMW &minus; local RMW)</div></div>
    <div class="stat-box"><div class="val">{_fx(_d.get('ownership_premium_ns'))}</div><div class="lbl">Ownership premium (RFO &minus; c2c read)</div></div>
  </div>
  {_errs}
</div>""")
        rfo_html = (
            "<h2>Cross-Core RFO (ownership transfer)</h2>"
            "<p style='color:#888;font-size:0.85em;margin:-8px 0 12px'>"
            f"Core B writes a random chain of {rfo.get('nodes')} lines "
            f"({rfo.get('node_stride_bytes')} B apart, {rfo.get('footprint_kb')} KB) so every line is "
            "Modified in B's cache; core A then walks it once, either with plain loads or with "
            "<code>lock xadd</code>, which cannot complete until A owns the line. Baselines walk lines "
            f"A dirtied itself. {rfo.get('rounds')} rounds per variant, TSC-timed. "
            "This is the real read-for-ownership cost that a single-core write test cannot see.</p>"
            + "".join(_blocks))
 
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
            <div class="score-label">{cat}{"<br><span style='color:#777'>not in Overall</span>" if cat == "4K Random Access" else ""}</div>
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
                   "dirty_writeback", "tlb_4k_chase", "tlb_2m_chase"]:
            raw_rows += _td(r, p)
        raw_rows += "</tr>"

    chartjs_tag = chartjs_script_tag()   # inline Chart.js: report works offline (6.96)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MemLat Pro Report -- {meta.get('cpu_model','CPU')}</title>
{chartjs_tag}
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
.stat-box .sub {{ font-size:0.72em; color:#777; margin-top:2px; line-height:1.3; }}
.lp-note {{ background:#2a1a0a; border:1px solid #5a3a1a; border-radius:8px; padding:10px 16px;
            font-size:0.82em; color:#ddd; margin:4px 0 12px; }}
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
    <div class="header-sub">{ctx_line}</div>
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
    <div class="stat-box"><div class="val">{f"{summary['ram_median_ns']:.1f} ns" if isinstance(summary.get('ram_median_ns'), (int,float)) else 'N/A'}</div><div class="lbl">RAM</div>{f'<div class="sub">{_ram_sub}</div>' if _ram_sub else ''}</div>
</div>
{windows_html}
{pages_html}
 
<h2>Score Card</h2>
<div class="scores-grid">{score_cards_html}</div>
 
<h2>Latency Curves (all patterns)</h2>
<div class="chart-container"><canvas id="latChart"></canvas></div>
 
<h2>Dirty-Writeback Overhead vs Read</h2>
<p style="color:#888;font-size:0.85em;margin:-8px 0 12px">
  Extra ns per hop when every visited line is also written (dirtied), versus the plain random chase.
  This is the cost of writing dirty lines back on eviction. It is not RFO: the line is already
  owned by the time the store executes.
</p>
<div class="chart-container"><canvas id="wbChart"></canvas></div>
{rfo_html}
 
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
<tr><th>Size</th><th>Random</th><th>Stride64</th><th>Stride256</th><th>DirtyWB</th><th>TLB-4K</th><th>TLB-2M</th></tr>
{raw_rows}
</table>
</div>
 
<div class="footer">Generated by MemLat Pro v{VERSION} | {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
 
</div><!-- /tab-results -->
 
<div id="tab-whylat" class="tab-panel">
 
<a class="wlm-video" href="https://www.youtube.com/watch?v=5qjSGEOEaXo" target="_blank" rel="noopener">
  <div class="wlm-video-icon">▶</div>
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
    {{key:'dirty_writeback', label:'Dirty writeback', color:'#cc44ff'}},
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
 
// Dirty-writeback overhead chart
const wbLabels = cd.filter(d => d.random_chase && d.dirty_writeback).map(d => d.size_str);
const wbData = cd.filter(d => d.random_chase && d.dirty_writeback).map(d => d.dirty_writeback - d.random_chase);
new Chart(document.getElementById('wbChart'), {{
    type: 'bar',
    data: {{ labels: wbLabels, datasets: [{{ label: 'Dirty-writeback overhead (ns)', data: wbData,
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
    {{key:'dirty_writeback_total_ms', label:'Dirty writeback', color:'#cc44ff'}},
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
 
 
def _normalize_legacy_keys(data: Dict) -> Dict:
    """
    Rename keys written by versions before 6.96 so old result files compare
    cleanly with new ones:
      results[*]["write_rfo"]           -> ["dirty_writeback"]
      summary["rfo_overhead_ns"]        -> ["writeback_overhead_ns"]
      scores["Write Overhead"]          -> ["Writeback Overhead"]
    The measurement itself is unchanged; only the name was wrong.
    """
    for r in data.get("results", []) or []:
        if isinstance(r, dict) and "write_rfo" in r and "dirty_writeback" not in r:
            r["dirty_writeback"] = r.pop("write_rfo")
    summ = data.get("summary")
    if isinstance(summ, dict) and "rfo_overhead_ns" in summ and "writeback_overhead_ns" not in summ:
        summ["writeback_overhead_ns"] = summ.pop("rfo_overhead_ns")
    sc = data.get("scores")
    if isinstance(sc, dict) and "Write Overhead" in sc and "Writeback Overhead" not in sc:
        # rebuild to keep the category order
        data["scores"] = {("Writeback Overhead" if k == "Write Overhead" else k): v
                          for k, v in sc.items()}
    return data


def _build_summary_from_results(data: Dict) -> Dict:
    """
    Summary used by compare mode. When the file carries raw results and its
    cache config (every version that writes "results"), the summary is
    recomputed with the current plateau windows (compute_summary), so both
    runs are summarised the same way whichever version produced them.
    Files without raw results fall back to their stored summary.
    """
    results = data.get("results", [])
    cfg = data.get("meta", {}).get("cache_config", {})
    if results and cfg:
        meta = data.get("meta", {})
        small = sweep_uses_small_pages(meta.get("platform"),
                                       (meta.get("run_context") or {}).get("pages"))
        s = compute_summary(results, cfg, small)
        if data.get("page_modes"):
            s = apply_ram_headline(s, data["page_modes"], cfg)
        s["_recomputed"] = True
        return s
    summary = dict(data.get("summary", {}))
    summary.setdefault("prefetcher_benefit_ns", None)
    summary.setdefault("writeback_overhead_ns", None)
    summary["_recomputed"] = False
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
    a = _normalize_legacy_keys(a)
    b = _normalize_legacy_keys(b)
 
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
        ("Writeback Ovh (ns)", "writeback_overhead_ns"),
    ]
    # Cross-core RFO (6.96 files): first core pair's derived numbers
    def _rfo_val(d: Dict, key: str) -> Optional[float]:
        pairs = (d.get("cross_core_rfo") or {}).get("pairs") or []
        return (pairs[0].get("derived") or {}).get(key) if pairs else None
    for _lbl, _key in (("C2C read (ns)", "c2c_read_ns"), ("RFO transfer (ns)", "rfo_transfer_ns")):
        sa[_key], sb[_key] = _rfo_val(a, _key), _rfo_val(b, _key)
        if sa[_key] is not None or sb[_key] is not None:
            metrics.append((_lbl, _key))
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

    def _ver(d: Dict) -> Tuple[int, ...]:
        try:
            return tuple(int(x) for x in re.findall(r"\d+", str((d.get("meta") or {}).get("version", "")))[:2])
        except Exception:
            return ()

    def _has_tlb(d: Dict) -> bool:
        return any(_stat_ok(r.get("tlb_4k_chase")) or _stat_ok(r.get("tlb_2m_chase"))
                   for r in d.get("results", []) or [] if isinstance(r, dict))
    _old_tlb = [lbl for lbl, d in (("A", a), ("B", b)) if _has_tlb(d) and _ver(d) < (6, 98)]
    if _old_tlb:
        _who = (f"Run {_old_tlb[0]} predates 6.98. Its" if len(_old_tlb) == 1
                else "Runs A and B predate 6.98. Their")
        print(f"\n  Note: {_who} TLB-4K / TLB-2M columns")
        print("  measured cache-set conflicts (every node at page offset 0), not TLB cost,")
        print("  and are not comparable with 6.98+ TLB columns. Everything above is unaffected.")
    for lbl, d in (("A", a), ("B", b)):
        if d.get("partial"):
            print(f"\n  Note: Run {lbl} is a partial checkpoint (the sweep did not finish).")

    if sa.get("_recomputed") or sb.get("_recomputed"):
        print("\n  Summaries recomputed from raw results with the 6.96 plateau windows")
        print(f"  (RAM = sizes >= {RAM_PLATEAU_FACTOR}x L3). N/A = that run never reached the window,")
        print("  e.g. RAM in a pre-6.96 Quick run whose sizes stopped at 256 MB.")

    # Score comparison: recomputed from the recomputed summaries when possible,
    # otherwise the scores stored in the file (v1 files have none)
    def _scores_for(d: Dict, s: Dict) -> Dict:
        if s.get("_recomputed"):
            return compute_scores(s, d.get("meta", {}).get("cache_config", {}))
        return d.get("scores", {})
    sca, scb = _scores_for(a, sa), _scores_for(b, sb)
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
    _pause("\nPress Enter to exit...")
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 16 — Dependency Check & Numba Warmup
# ══════════════════════════════════════════════════════════════════════════════
def _print_numba_refusal(missing: List[str]) -> None:
    """Explain why measurements will not run without Numba and how to fix it."""
    script = os.path.basename(sys.argv[0]) or "this script"
    pkgs = " ".join(dict.fromkeys(["numpy", "numba"] + missing + ["psutil", "matplotlib"]))
    print("""
  ==================================================================
   CANNOT RUN MEASUREMENTS: Numba is not installed (or too old).
  ==================================================================
   Without Numba every pointer-chase hop runs in the Python
   interpreter, which adds roughly 50-100 ns per access. That swamps
   the 1-15 ns cache latencies being measured, so the results would
   be wrong, not just slow. MemLat Pro therefore refuses to run.

   Install the dependencies, then run the script again:""")
    print(f"       pip install {pkgs}")
    print("""
   If your Linux distribution blocks system-wide pip installs
   ("externally-managed-environment"), use a virtual environment:""")
    print("       python3 -m venv ~/.venvs/memlat")
    print(f"       ~/.venvs/memlat/bin/pip install {pkgs}")
    print(f"       ~/.venvs/memlat/bin/python {script}")
    print("""
   Compare mode (menu [4] or --compare a.json b.json) does not need
   Numba and still works.
  ==================================================================""")


def check_dependencies() -> bool:
    import importlib
    # numba is required for every measurement mode: there is no Python
    # fallback (see Section 7).
    REQUIRED = [("numpy", "numpy", "1.20"), ("numba", "numba", "0.55")]
    OPTIONAL = [
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
        except Exception:  # ImportError, or a broken install raising anything else
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
        if "numba" in missing_req or not HAS_NUMBA:
            _print_numba_refusal(missing_req)
        else:
            print("  Run:  pip install " + " ".join(missing_req))
    elif not HAS_NUMBA:
        # numba imports here but failed at module load (e.g. NumPy too new
        # for this numba build) -- same outcome: refuse to measure.
        missing_req.append("numba")
        _print_numba_refusal(missing_req)
    if missing_opt:
        print("  Optional missing: " + ", ".join(missing_opt))
    if not missing_req and not missing_opt:
        print("\n  All dependencies satisfied.\n")
    return len(missing_req) == 0


def _warmup_numba() -> bool:
    """Compile the kernels up front. Returns False if compilation fails."""
    if not HAS_NUMBA:
        _print_numba_refusal([])
        return False
    print("  Compiling Numba kernels (first run only)...")
    try:
        tb = np.zeros(128, dtype=np.int64)
        tb[0] = 8; tb[8] = 0
        _chase_kernel(tb, 2, 1)
        # Dirty-writeback kernel has the same signature as chase: (buf, n_nodes, traversals)
        wb = np.zeros(128, dtype=np.int64)
        wb[0] = 8; wb[8] = 0
        _dirty_writeback_kernel(wb, 2, 1)
        a = np.ones(64, dtype=np.float64)
        b = np.ones(64, dtype=np.float64)
        c = np.ones(64, dtype=np.float64)
        _stream_triad(a, b, c, 0.5, 1)
        del tb, wb, a, b, c
        print("  Numba JIT: ready")
        return True
    except Exception as e:
        print(f"\n  Numba kernel compilation failed: {e}")
        print("  There is no Python fallback (it would produce wrong numbers).")
        print("  Reinstall/upgrade numba (pip install -U numba) and try again.")
        return False
 
 
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
    print("    [3]  Custom  -- Choose max size, toggle bandwidth/TLB")
    print("    [4]  Compare -- Diff two previous JSON result files")
    print("    [5]  Extreme -- Loaded latency stress (Chips & Cheese method)")
    print("                    Exposes chiplet XI queue / IFOP / IMC contention")
    print("    [6]  Exit")
    print()
    while True:
        choice = input("  Enter choice [1-6]: ").strip()
        if choice in ("1", "2", "3", "4", "5", "6"):
            break
        print("  Invalid choice.")
 
    if choice == "6":
        print("\n  Goodbye.")
        sys.exit(0)
 
    if choice == "4":
        pa = input("  Path to Run A JSON: ").strip().strip('"')
        pb = input("  Path to Run B JSON: ").strip().strip('"')
        return {"mode": "compare", "compare_a": pa, "compare_b": pb}

    if choice == "5":
        print("\n  +--------------------------------------------------------------+")
        print("  |  LOADED LATENCY TEST -- Extreme Interconnect Stress          |")
        print("  |                                                              |")
        print("  |  Pins a latency pointer chase to one core, then spawns       |")
        print("  |  bandwidth-hungry threads on other cores progressively.      |")
        print("  |  Measures how DRAM latency degrades as XI queues, IFOP       |")
        print("  |  links, and the memory controller become saturated.          |")
        print("  |                                                              |")
        print("  |  Requires: psutil    Runtime: ~3-8 min    RAM: ~8-20 GB      |")
        print("  +--------------------------------------------------------------+")
        print()
        print("  Buffer sizing: Auto mode sizes buffers so L3 covers only")
        print("  ~5% of the latency working set (10% for BW workers),")
        print("  ensuring consistent fabric stress across all CPUs.")
        print()
        print("    [a]  Auto   -- 5% coverage, recommended (default)")
        print("    [c]  Custom coverage -- set your own target %")
        print("    [m]  Manual -- specify buffer sizes in MB")
        print()
        sizing = input("  Buffer mode [a/c/m]: ").strip().lower()

        lat_mb = None  # None = auto
        bw_mb = None
        coverage = 5.0

        if sizing == "c":
            cov_str = input("  Cache coverage target % [5.0]: ").strip()
            if cov_str:
                try:
                    coverage = float(cov_str)
                    if coverage <= 0 or coverage > 50:
                        print("  Invalid -- using 5.0%.")
                        coverage = 5.0
                except ValueError:
                    print("  Invalid -- using 5.0%.")
        elif sizing == "m":
            lat_mb_str = input("  Latency buffer size in MB: ").strip()
            if lat_mb_str:
                try:
                    lat_mb = int(lat_mb_str)
                    if lat_mb < 1:
                        raise ValueError
                except ValueError:
                    print("  Invalid -- using auto.")
                    lat_mb = None
            bw_mb_str = input("  BW buffer per thread in MB: ").strip()
            if bw_mb_str:
                try:
                    bw_mb = int(bw_mb_str)
                    if bw_mb < 1:
                        raise ValueError
                except ValueError:
                    print("  Invalid -- using auto.")
                    bw_mb = None
        else:
            pass  # auto: coverage = 5.0, lat_mb = None, bw_mb = None

        print()
        print("  Latency buffer pages:")
        print("    [2]  2 MB pages -- DRAM latency without page walks, comparable to MLC (default)")
        print("    [4]  4 KB pages -- adds a page walk per hop (goes to DRAM under load)")
        pg = input("  Pages [2/4]: ").strip()
        ll_pages = "4k" if pg in ("4", "4k", "4K") else "2m"

        dur_str = input("  Measurement window per step in seconds [5.0]: ").strip()
        dur = 5.0
        if dur_str:
            try:
                dur = float(dur_str)
                if not (dur > 0):
                    raise ValueError
            except ValueError:
                print("  Invalid -- using 5.0 seconds.")
                dur = 5.0
        out = input("  Output directory [auto]: ").strip()
        return {
            "mode": "loaded_latency",
            "latency_buf_mb": lat_mb,
            "bw_buf_mb": bw_mb,
            "cache_coverage_pct": coverage,
            "measure_seconds": dur,
            "page_mode": ll_pages,
            "output": out if out else None,
        }
 
    params: Dict = {
        "mode": "quick" if choice == "1" else ("full" if choice == "2" else "custom"),
        "max_size_mb": QUICK_MAX_MB if choice == "1" else 1024,
        "fast": False,
        "quick": choice == "1",
        "bandwidth": False,
        "rfo": False,
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
                if params["max_size_mb"] < 1:
                    raise ValueError
            except ValueError:
                print("  Invalid -- using 1024 MB.")
                params["max_size_mb"] = 1024
        bw = input("  Measure STREAM bandwidth? [y/N]: ").strip().lower()
        params["bandwidth"] = bw in ("y", "yes")
        rf = input("  Measure cross-core RFO (ownership transfer, ~5 s)? [Y/n]: ").strip().lower()
        params["rfo"] = rf not in ("n", "no")
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
        rf = input("  Also measure cross-core RFO (ownership transfer, ~5 s)? [Y/n]: ").strip().lower()
        params["rfo"] = rf not in ("n", "no")
 
    return params
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Section 17B — Embedded Chart.js 4.4.0 (self-contained HTML reports)  [6.96]
#  The HTML reports used to load Chart.js from cdn.jsdelivr.net, so offline,
#  behind a firewall, or if that URL ever changed, their charts came up blank.
#  A copy now ships inside this script (zlib + base64, ~90 KB) and is inlined
#  into every report (~205 KB per HTML file). Chart.js is MIT-licensed; the
#  notice below is reproduced in each report next to the inlined script.
# ══════════════════════════════════════════════════════════════════════════════
CHARTJS_VERSION = "4.4.0"
CHARTJS_SHA256 = "321e3a3fa98da4aaa957d10be57cbb514de0989eed8f9d726b5d05902cd01904"
CHARTJS_CDN_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"
CHARTJS_LICENSE = """The MIT License (MIT)

Copyright (c) 2014-2022 Chart.js Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
_CHARTJS_ZB64 = """
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
3xDT//r//H/sAFJQ
"""
_CHARTJS_TAG_CACHE: Optional[str] = None


def chartjs_script_tag() -> str:
    """Inline <script> carrying Chart.js (verified by SHA-256), preceded by its
    MIT notice. Falls back to the CDN tag only if the embedded copy is damaged."""
    global _CHARTJS_TAG_CACHE
    if _CHARTJS_TAG_CACHE is None:
        try:
            raw = zlib.decompress(base64.b64decode("".join(_CHARTJS_ZB64.split())))
            if hashlib.sha256(raw).hexdigest() != CHARTJS_SHA256:
                raise ValueError("SHA-256 mismatch")
            js = raw.decode("utf-8").replace("</script", "<\\/script")
            _CHARTJS_TAG_CACHE = (f"<!-- Chart.js {CHARTJS_VERSION}, inlined so this report works offline.\n"
                                  f"{CHARTJS_LICENSE}\n-->\n<script>{js}</script>")
        except Exception as e:
            print(f"  Warning: embedded Chart.js unusable ({e}); this report will load it from the CDN")
            _CHARTJS_TAG_CACHE = f'<script src="{CHARTJS_CDN_URL}"></script>'
    return _CHARTJS_TAG_CACHE



# ══════════════════════════════════════════════════════════════════════════════
#  Section 18 — Entry Point
# ══════════════════════════════════════════════════════════════════════════════
def _refuse_and_exit() -> None:
    """Stop before measuring (missing/broken Numba). Pause so a console
    window opened by double-click stays up long enough to read why."""
    _pause()
    sys.exit(1)


def _run_loaded_latency_guarded(**kwargs) -> None:
    """(6.98) A failure inside the loaded latency test is reported, not left
    to close the console window with an unread traceback."""
    try:
        run_loaded_latency_test(**kwargs)
    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
    except Exception as e:
        print(f"\n  Loaded latency test failed: {e}")
        traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description=f"MemLat Pro v{VERSION}")
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--bandwidth", action="store_true")
    parser.add_argument("--rfo", action="store_true",
                        help="Also run the cross-core RFO (ownership transfer) test")
    parser.add_argument("--no-tlb", action="store_true", help="Skip TLB stress tests")
    parser.add_argument("--ll-pages", choices=["2m", "4k"], default="2m",
                        help="Loaded latency test: latency buffer pages (default 2m)")
    parser.add_argument("--no-page-modes", action="store_true",
                        help="Skip the 4 KB vs 2 MB page RAM test (RAM headline falls back to plateau)")
    parser.add_argument("--core", choices=["P", "E"], default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_RNG_SEED)
    parser.add_argument("--no-menu", action="store_true")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report")
    parser.add_argument("--loaded-latency", action="store_true",
                        help="Run loaded latency stress test (Chips & Cheese method)")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                        help="Compare two result files")
    args = parser.parse_args()
 
    # ── Comparison mode ──
    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
        return
 
    # ── Menu vs CLI ──
    # ── Loaded latency CLI mode ──
    if hasattr(args, 'loaded_latency') and args.loaded_latency:
        if not check_dependencies() or not _warmup_numba():
            _refuse_and_exit()
        _run_loaded_latency_guarded(
            cache_coverage_pct=5.0,
            output_dir=args.output,
            page_mode=args.ll_pages,
        )
        _pause()
        return

    has_mode_flag = args.fast or args.quick or args.max_size is not None
    use_menu = not args.no_menu and not has_mode_flag
 
    if use_menu:
        try:
            params = show_menu()
        except EOFError:
            print("\n\n  No interactive input available for the menu. Use --no-menu together with")
            print("  --quick / --fast / --max-size, or --loaded-latency, or --compare A.json B.json.")
            sys.exit(1)
        if params.get("mode") == "compare":
            compare_runs(params["compare_a"], params["compare_b"])
            return
        if params.get("mode") == "loaded_latency":
            if not check_dependencies() or not _warmup_numba():
                _refuse_and_exit()
            _run_loaded_latency_guarded(
                latency_buf_mb=params.get("latency_buf_mb"),  # None = auto
                bw_buf_mb=params.get("bw_buf_mb"),            # None = auto
                cache_coverage_pct=params.get("cache_coverage_pct", 5.0),
                measure_seconds=params.get("measure_seconds", 5.0),
                output_dir=params.get("output"),
                page_mode=params.get("page_mode", "2m"),
            )
            _pause()
            return
        run_fast = params.get("fast", False)
        run_quick = params.get("quick", False)
        max_size_mb = params.get("max_size_mb", 1024)
        bandwidth = params.get("bandwidth", False)
        rfo = params.get("rfo", False)
        tlb = params.get("tlb", True)
        core = params.get("core")
        seed = params.get("seed", DEFAULT_RNG_SEED)
        output = params.get("output")
        no_html = False
        page_modes = params.get("page_modes", True)
    else:
        print(f"\n{'=' * 72}")
        print(f"  MemLat Pro v{VERSION}")
        print(f"{'=' * 72}")
        run_fast = args.fast
        run_quick = args.quick
        max_size_mb = args.max_size if args.max_size is not None else 1024
        bandwidth = args.bandwidth
        rfo = args.rfo
        tlb = not args.no_tlb
        core = args.core
        seed = args.seed
        output = args.output
        no_html = args.no_html
        page_modes = not args.no_page_modes
 
    if not check_dependencies() or not _warmup_numba():
        _refuse_and_exit()

    tester = None
    try:
        tester = MemLatPro(
            max_size_mb=max_size_mb,
            output_dir=output,
            fast=run_fast,
            quick=run_quick,
            bandwidth=bandwidth,
            tlb_test=tlb,
            pin_core_type=core,
            rng_seed=seed,
            rfo=rfo,
        )
        t_start = time.perf_counter()
        results = tester.run()
        elapsed = time.perf_counter() - t_start

        if page_modes:
            tester.measure_page_modes(results)
        if rfo:
            tester.measure_cross_core_rfo()
        summary, boundaries, scores = tester.analyze(results)
        tester.export_csv(results)
        tester.print_raw_data(results)
 
        # HTML report
        if not no_html:
            payload = {
                "meta": tester._build_meta(results),
                "summary": summary,
                "scores": scores,
                "boundaries": boundaries,
                "results": results,
            }
            if tester.page_modes is not None:
                payload["page_modes"] = tester.page_modes
            if tester.rfo_results is not None:
                payload["cross_core_rfo"] = tester.rfo_results
            generate_html_report(payload, tester.html_path)
 
        tester.plot(results)
 
        print(f"\n  Total runtime : {elapsed/60:.1f} min")
        print(f"  Output folder : {os.path.abspath(tester.output_dir)}")
        print(f"  Files: .json  .csv  .html  .png  .txt")
        print("=" * 72)
    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
        if tester is not None and os.path.isfile(tester.json_path):
            print(f"  Results measured so far: {os.path.abspath(tester.json_path)}")
    except Exception as e:
        print(f"\n  Fatal error: {e}")
        traceback.print_exc()
        if tester is not None and os.path.isfile(tester.json_path):
            print(f"\n  Results measured so far: {os.path.abspath(tester.json_path)}")
    finally:
        if tester is not None:
            tester._close_log()
 
    _pause()
 
 
if __name__ == "__main__":
    main()