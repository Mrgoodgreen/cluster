#!/usr/bin/env python3
"""Auto XY-tiling for large TLS clouds (RAM preflight + merge by core).

Used by ``classify_tls_all_in_one`` when a monolithic run would exceed
available memory, or when ``--force-tile`` is set.

Speed path (label-preserving algorithm, same mono stages per leaf):
- classify leaves in-memory (no per-tile LAS rewrite)
- optional parallel leaf workers (thread pool; arrays are independent copies)
- anti-waste adaptive split: skip splits that barely shrink the load window
"""

from __future__ import annotations

import gc
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np

from classify_tls import CLS_DEFAULT, log


# Empirical peak working set for all-in-one + RF (arrays, grids, HDBSCAN temps).
BYTES_PER_POINT_PEAK = 110
# Soft cap even on big RAM machines (building grids blow up on dense TLS).
DEFAULT_MAX_POINTS_MONO = 70_000_000
DEFAULT_TILE_M = 0.0  # 0 = auto from RAM + density
DEFAULT_OVERLAP_M = 15.0  # > half car length + peel pad
DEFAULT_MAX_TILE_POINTS = 0  # 0 = auto from MemAvailable
DEFAULT_MIN_TILE_M = 12.0
# Refuse a split unless the heavier child load drops by at least this fraction.
SPLIT_MIN_LOAD_REDUCTION = 0.18
RAM_SAFETY = 0.52  # use at most ~52% of MemAvailable for peak estimate
# After parent XYZ is resident, leaf peak may use this fraction of MemAvailable.
LEAF_RAM_FRACTION = 0.42
LOAD_OVERLAP_FACTOR = 1.6  # load window vs core point count (typical)
MIN_AUTO_TILE_POINTS = 4_000_000
MAX_AUTO_TILE_POINTS = 22_000_000
DEFAULT_TILE_WORKERS = 0  # 0 = auto


@dataclass(frozen=True)
class TileRect:
    """Core [x0,x1) × [y0,y1); overlap expands the load window."""

    x0: float
    x1: float
    y0: float
    y1: float


@dataclass(frozen=True)
class TileGeometry:
    """Resolved tile knobs (after RAM/density auto)."""

    tile_m: float
    max_tile_points: int
    overlap_m: float
    min_tile_m: float
    workers: int
    reason: str


def _cgroup_available_bytes() -> int | None:
    """Remaining hard-limit budget, including ancestors, cgroup v1 and v2.

    Resolve the process membership against mountinfo: do not assume that the
    process is in the mount root. An ancestor can be tighter than its child.
    """
    budgets = []
    try:
        memberships = Path('/proc/self/cgroup').read_text().splitlines()
        mounts = Path('/proc/self/mountinfo').read_text().splitlines()
        for membership in memberships:
            _, controllers, member = membership.split(':', 2)
            version = 2 if not controllers else 1
            if version == 1 and 'memory' not in controllers.split(','):
                continue
            for mount in mounts:
                before, after = mount.split(' - ', 1)
                fields, fs = before.split(), after.split()
                if fs[0] != ('cgroup2' if version == 2 else 'cgroup'):
                    continue
                if version == 1 and 'memory' not in fs[2].split(','):
                    continue
                unescape = lambda s: s.replace('\\040', ' ').replace('\\011', '\t').replace('\\134', '\\')
                mount_root, mount_path = unescape(fields[3]), Path(unescape(fields[4]))
                # Cgroup namespaces may report '/' with a non-root mount root.
                if member == '/':
                    current = mount_path
                else:
                    try:
                        relative = Path(member).relative_to(mount_root)
                    except ValueError:
                        continue
                    if '..' in relative.parts:
                        continue
                    current = mount_path / relative
                while True:
                    limit_name, usage_name = (('memory.max', 'memory.current') if version == 2
                                              else ('memory.limit_in_bytes', 'memory.usage_in_bytes'))
                    try:
                        limit_text = (current / limit_name).read_text().strip()
                        if limit_text != 'max':
                            limit = int(limit_text)
                            if 0 <= limit < (1 << 60):  # v1 unlimited sentinel
                                used = int((current / usage_name).read_text().strip())
                                budgets.append(max(0, limit - used))
                    except (OSError, ValueError):
                        pass
                    if current == mount_path:
                        break
                    current = current.parent
    except (OSError, ValueError, IndexError):
        pass
    return min(budgets) if budgets else None


def available_ram_bytes() -> int | None:
    """Available host RAM constrained by the process's cgroup hard limits."""
    host = None
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        for line in meminfo.splitlines():
            if line.startswith("MemAvailable:"):
                host = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    if host is None:
        try:
            import psutil  # type: ignore
            host = int(psutil.virtual_memory().available)
        except Exception:
            pass
    cgroup = _cgroup_available_bytes()
    candidates = [n for n in (host, cgroup) if n is not None]
    return min(candidates) if candidates else None


def estimate_peak_bytes(n_points: int, with_rf: bool = False) -> int:
    b = int(n_points) * BYTES_PER_POINT_PEAK
    if with_rf:
        b = int(b * 1.35)
    return b


def should_use_tiles(
    n_points: int,
    *,
    force_tile: bool = False,
    force_mono: bool = False,
    max_points_mono: int = 0,
    with_rf: bool = False,
) -> tuple[bool, str]:
    """Decide whether to tile before starting the heavy pipeline.

    ``max_points_mono <= 0`` → derive from MemAvailable (capped at 70M).
    """
    if force_mono:
        return False, "forced monolithic (--no-tile)"
    if force_tile:
        return True, "forced tiling (--force-tile)"

    mono_cap = (
        int(max_points_mono)
        if max_points_mono and max_points_mono > 0
        else suggest_max_points_mono(with_rf=with_rf)
    )
    if n_points > mono_cap:
        return True, f"n_points={n_points:,} > max_points_mono={mono_cap:,}"

    avail = available_ram_bytes()
    peak = estimate_peak_bytes(n_points, with_rf=with_rf)
    if avail is None:
        # Unknown RAM: still tile very large clouds
        if n_points > 50_000_000:
            return True, f"n_points={n_points:,} (RAM unknown; conservative tile)"
        return False, f"n_points={n_points:,} (RAM unknown; try mono)"

    budget = int(avail * RAM_SAFETY)
    if peak > budget:
        return (
            True,
            f"est. peak {peak / 1e9:.1f} GB > budget {budget / 1e9:.1f} GB "
            f"(MemAvailable {avail / 1e9:.1f} GB)",
        )
    return (
        False,
        f"est. peak {peak / 1e9:.1f} GB within budget {budget / 1e9:.1f} GB "
        f"(n={n_points:,}; mono_cap={mono_cap:,})",
    )


def suggest_tile_workers(
    n_points: int,
    max_tile_points: int,
    *,
    with_rf: bool = False,
    requested: int = 0,
    max_workers_cap: int = 4,
) -> int:
    """Pick parallel leaf workers from RAM budget (default auto).

    Call after the full cloud is resident: MemAvailable already reflects the
    parent XYZ/cls footprint, so do not subtract ``parent`` again.
    """
    if requested and requested > 0:
        return max(1, min(int(requested), max_workers_cap))

    avail = available_ram_bytes()
    cpu = os.cpu_count() or 2
    # Dense TLS often exceeds max_tile_points when overlap-dominated.
    load_guess = int(max_tile_points * LOAD_OVERLAP_FACTOR)
    per_worker = estimate_peak_bytes(load_guess, with_rf=with_rf)
    if avail is None:
        return 1 if cpu < 4 else 2
    # Leave OS / fragmentation headroom; parent already counted in MemAvailable.
    free_for_workers = max(0, int(avail * 0.55))
    by_ram = max(1, free_for_workers // max(per_worker, 1))
    by_cpu = max(1, min(max_workers_cap, cpu // 4))
    n_auto = int(max(1, min(max_workers_cap, by_ram, by_cpu)))
    # On ~24–32 GB hosts with huge clouds, keep at most 2 concurrent leaves.
    if n_points >= 80_000_000 and avail < 40_000_000_000:
        n_auto = min(n_auto, 2)
    # RF feature matrices are heavy — serialize leaves on mid-RAM hosts.
    if with_rf and avail < 40_000_000_000:
        n_auto = 1
    return n_auto


def suggest_max_points_mono(*, with_rf: bool = False) -> int:
    """Dynamic mono cap from MemAvailable (never above DEFAULT_MAX_POINTS_MONO)."""
    avail = available_ram_bytes()
    if avail is None:
        return DEFAULT_MAX_POINTS_MONO
    bpp = BYTES_PER_POINT_PEAK * (1.35 if with_rf else 1.0)
    by_ram = int((avail * RAM_SAFETY) / bpp)
    return int(max(8_000_000, min(DEFAULT_MAX_POINTS_MONO, by_ram)))


def suggest_tile_geometry(
    n_points: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    *,
    with_rf: bool = False,
    tile_m: float = 0.0,
    max_tile_points: int = 0,
    overlap_m: float = DEFAULT_OVERLAP_M,
    min_tile_m: float = DEFAULT_MIN_TILE_M,
    tile_workers: int = 0,
) -> TileGeometry:
    """Derive tile_m / max_tile_points / workers from RAM + XY density.

    ``tile_m <= 0`` or ``max_tile_points <= 0`` means auto. Explicit positive
    values are kept (still clamped to sane bounds).
    """
    avail = available_ram_bytes()
    bpp = BYTES_PER_POINT_PEAK * (1.35 if with_rf else 1.0)
    overlap_m = float(max(5.0, overlap_m))
    min_tile_m = float(max(6.0, min_tile_m))

    if max_tile_points and max_tile_points > 0:
        mtp = int(max_tile_points)
        mtp_src = "cli"
    elif avail is None:
        mtp = 12_000_000
        mtp_src = "default(no-RAM)"
    else:
        # After parent is resident: leaf peak budget from remaining MemAvailable.
        leaf_budget = avail * LEAF_RAM_FRACTION
        mtp = int(leaf_budget / (bpp * LOAD_OVERLAP_FACTOR))
        mtp = int(max(MIN_AUTO_TILE_POINTS, min(MAX_AUTO_TILE_POINTS, mtp)))
        # Extra caution on small hosts
        if avail < 20_000_000_000:
            mtp = min(mtp, 8_000_000 if with_rf else 10_000_000)
        elif avail < 32_000_000_000:
            mtp = min(mtp, 8_000_000 if with_rf else 12_000_000)
        elif avail < 48_000_000_000 and with_rf:
            mtp = min(mtp, 12_000_000)
        mtp_src = f"RAM({avail / 1e9:.1f}GB)"

    span_x = max(1e-3, float(xmax - xmin))
    span_y = max(1e-3, float(ymax - ymin))
    area = span_x * span_y
    density = float(n_points) / area  # pts/m²

    if tile_m and tile_m > 0:
        tm = float(tile_m)
        tm_src = "cli"
    elif density <= 1e-6:
        tm = 80.0
        tm_src = "default(empty)"
    else:
        # Load window side ≈ sqrt(mtp / density); core = that − 2·overlap.
        load_side = (mtp / density) ** 0.5
        tm = load_side - 2.0 * overlap_m
        tm = float(max(min_tile_m, min(120.0, tm)))
        # If density is extreme, prefer smaller cores so splits can still help.
        if density > 2500:
            tm = min(tm, 50.0)
        if density > 4000:
            tm = min(tm, 35.0)
        tm_src = f"density({density:.0f}/m²)"

    # Ensure min_tile_m does not exceed tile_m
    min_tm = float(min(min_tile_m, tm))
    workers = suggest_tile_workers(
        n_points,
        mtp,
        with_rf=with_rf,
        requested=tile_workers,
    )
    reason = (
        f"max_tile_pts={mtp:,} ({mtp_src}); tile_m={tm:g} m ({tm_src}); "
        f"overlap={overlap_m:g} m; workers={workers}"
    )
    return TileGeometry(
        tile_m=tm,
        max_tile_points=mtp,
        overlap_m=overlap_m,
        min_tile_m=min_tm,
        workers=workers,
        reason=reason,
    )


def build_tile_grid(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    tile_m: float,
    overlap_m: float,
) -> list[TileRect]:
    """Axis-aligned core tiles covering the AABB."""
    if xmax <= xmin or ymax <= ymin:
        return [TileRect(xmin, xmax + 1e-3, ymin, ymax + 1e-3)]
    tiles: list[TileRect] = []
    x0 = xmin
    while x0 < xmax:
        x1 = min(x0 + tile_m, xmax)
        y0 = ymin
        while y0 < ymax:
            y1 = min(y0 + tile_m, ymax)
            tiles.append(TileRect(x0, x1, y0, y1))
            if y1 >= ymax:
                break
            y0 = y1
        if x1 >= xmax:
            break
        x0 = x1
    # Ensure last edges include max exactly when span is exact multiple
    _ = overlap_m  # reserved for load window, not core grid
    return tiles


def _mask_window(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    xmax: float,
    ymax: float,
    inclusive_max: bool,
) -> np.ndarray:
    """Half-open window; last row/col that touches cloud max is inclusive."""
    if inclusive_max and x1 >= xmax - 1e-6:
        x_ok = (x >= x0) & (x <= xmax + 1e-9)
    else:
        x_ok = (x >= x0) & (x < x1)
    if inclusive_max and y1 >= ymax - 1e-6:
        y_ok = (y >= y0) & (y <= ymax + 1e-9)
    else:
        y_ok = (y >= y0) & (y < y1)
    return x_ok & y_ok


def _load_count(
    x: np.ndarray,
    y: np.ndarray,
    rect: TileRect,
    overlap_m: float,
    *,
    xmax: float,
    ymax: float,
) -> int:
    load_x0, load_x1 = rect.x0 - overlap_m, rect.x1 + overlap_m
    load_y0, load_y1 = rect.y0 - overlap_m, rect.y1 + overlap_m
    return int(
        _mask_window(
            x, y, load_x0, load_x1, load_y0, load_y1,
            xmax=xmax, ymax=ymax, inclusive_max=True,
        ).sum()
    )


def run_pipeline_tiled(
    input_path: Path,
    output_path: Path,
    *,
    run_mono=None,
    run_arrays=None,
    source_class: int = 1,
    cars_target_class: int = 91,
    noise_target_class: int = 7,
    fast: bool = False,
    reset_all: bool = False,
    reset_to: int = CLS_DEFAULT,
    no_subsample: bool = False,
    rf_refiner: Path | None = None,
    tile_m: float = DEFAULT_TILE_M,
    overlap_m: float = DEFAULT_OVERLAP_M,
    max_tile_points: int = DEFAULT_MAX_TILE_POINTS,
    min_tile_m: float = DEFAULT_MIN_TILE_M,
    tile_workers: int = DEFAULT_TILE_WORKERS,
    hdbscan_backend: str | None = None,
    work_dir: Path | None = None,
) -> None:
    """Classify by overlapping XY tiles; keep core labels; write one LAS.

    Prefers ``run_arrays`` (in-memory). ``run_mono`` kept for API compatibility
    but is unused when ``run_arrays`` is provided.
    """
    import time

    if run_arrays is None:
        # Lazy import keeps tiling usable from workers that only pass arrays.
        from classify_tls_all_in_one import _run_pipeline_arrays as run_arrays

    _ = run_mono  # legacy path-based hook (LAS I/O removed)
    _ = work_dir  # no per-tile temp LAS anymore

    t0 = time.time()
    with laspy.open(str(input_path)) as reader:
        parent_bytes = reader.header.point_count * (reader.header.point_format.size + 36)
    available = available_ram_bytes()
    if available is not None and parent_bytes > available * 0.80:
        raise MemoryError('TILE_PARENT_EXCEEDS_RAM: parent LAS and coordinate arrays cannot safely fit')
    log(f"[TILE] Reading {input_path} for tiled classify...")
    las = laspy.read(str(input_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    cls_in = np.asarray(las.classification, dtype=np.uint8)
    n = len(cls_in)
    inten = np.asarray(las.intensity, dtype=np.float64) if hasattr(las, "intensity") else None

    if reset_all:
        cls_seed = np.full(n, np.uint8(reset_to), dtype=np.uint8)
        src = int(reset_to)
    else:
        cls_seed = cls_in.copy()
        src = int(source_class)

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    geom = suggest_tile_geometry(
        n,
        xmin,
        xmax,
        ymin,
        ymax,
        with_rf=rf_refiner is not None,
        tile_m=tile_m,
        max_tile_points=max_tile_points,
        overlap_m=overlap_m,
        min_tile_m=min_tile_m,
        tile_workers=tile_workers,
    )
    tile_m = geom.tile_m
    max_tile_points = geom.max_tile_points
    overlap_m = geom.overlap_m
    min_tile_m = geom.min_tile_m
    workers = geom.workers
    tiles = build_tile_grid(xmin, xmax, ymin, ymax, tile_m, overlap_m)
    log(f"[TILE] auto geometry: {geom.reason}")
    log(
        f"[TILE] {n:,} pts → {len(tiles)} core tiles "
        f"(tile={tile_m:g} m, overlap={overlap_m:g} m, max_tile_pts={max_tile_points:,}, "
        f"workers={workers})"
    )

    cls_out = cls_seed.copy()
    assigned = np.zeros(n, dtype=bool)
    log_lock = threading.Lock()
    merge_lock = threading.Lock()

    def _indices_in_window(
        idx_cand: np.ndarray | None,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
    ) -> np.ndarray:
        """Mask full cloud or a candidate index subset (no full-cloud rescans)."""
        if idx_cand is None:
            m = _mask_window(
                x, y, x0, x1, y0, y1,
                xmax=xmax, ymax=ymax, inclusive_max=True,
            )
            return np.where(m)[0]
        if len(idx_cand) == 0:
            return idx_cand
        m = _mask_window(
            x[idx_cand], y[idx_cand], x0, x1, y0, y1,
            xmax=xmax, ymax=ymax, inclusive_max=True,
        )
        return idx_cand[m]

    def _collect_leaves(
        rect: TileRect,
        depth: int = 0,
        idx_cand: np.ndarray | None = None,
    ) -> list[tuple[str, TileRect, int]]:
        load_x0, load_x1 = rect.x0 - overlap_m, rect.x1 + overlap_m
        load_y0, load_y1 = rect.y0 - overlap_m, rect.y1 + overlap_m
        idx = _indices_in_window(idx_cand, load_x0, load_x1, load_y0, load_y1)
        n_load = int(len(idx))
        if n_load == 0:
            return []

        span_x = rect.x1 - rect.x0
        span_y = rect.y1 - rect.y0
        hard_cap = int(max_tile_points * 1.05)
        if n_load > max_tile_points and max(span_x, span_y) > min_tile_m * 1.05:
            if span_x >= span_y:
                mid = 0.5 * (rect.x0 + rect.x1)
                left = TileRect(rect.x0, mid, rect.y0, rect.y1)
                right = TileRect(mid, rect.x1, rect.y0, rect.y1)
            else:
                mid = 0.5 * (rect.y0 + rect.y1)
                left = TileRect(rect.x0, rect.x1, rect.y0, mid)
                right = TileRect(rect.x0, rect.x1, mid, rect.y1)
            # Child load windows ⊆ parent load → count from idx only.
            n_left = len(
                _indices_in_window(
                    idx,
                    left.x0 - overlap_m, left.x1 + overlap_m,
                    left.y0 - overlap_m, left.y1 + overlap_m,
                )
            )
            n_right = len(
                _indices_in_window(
                    idx,
                    right.x0 - overlap_m, right.x1 + overlap_m,
                    right.y0 - overlap_m, right.y1 + overlap_m,
                )
            )
            heavier = max(n_left, n_right)
            beneficial = heavier <= n_load * (1.0 - SPLIT_MIN_LOAD_REDUCTION)
            if beneficial or n_load > hard_cap:
                out: list[tuple[str, TileRect, int]] = []
                out.extend(_collect_leaves(left, depth + 1, idx))
                out.extend(_collect_leaves(right, depth + 1, idx))
                return out
            with log_lock:
                log(
                    f"[TILE] skip-split d{depth} load={n_load:,} "
                    f"(children max={heavier:,}; overlap-dominated)"
                )

        in_core = _mask_window(
            x[idx], y[idx], rect.x0, rect.x1, rect.y0, rect.y1,
            xmax=xmax, ymax=ymax, inclusive_max=True,
        )
        if not in_core.any():
            return []
        tag = f"d{depth}_{rect.x0:.0f}_{rect.y0:.0f}"
        return [(tag, rect, depth)]

    leaves: list[tuple[str, TileRect, int]] = []
    for rect in tiles:
        leaves.extend(_collect_leaves(rect))
    log(f"[TILE] {len(leaves)} leaf jobs (in-memory classify)")

    def _run_leaf(tag: str, rect: TileRect, depth: int) -> tuple[str, np.ndarray, np.ndarray]:
        # Prefer full overlap; shrink only if load still exceeds a hard RAM cap.
        target_load = int(max_tile_points * 1.25)
        eff_overlap = float(overlap_m)
        idx = np.array([], dtype=np.int64)
        for _ in range(8):
            load_x0, load_x1 = rect.x0 - eff_overlap, rect.x1 + eff_overlap
            load_y0, load_y1 = rect.y0 - eff_overlap, rect.y1 + eff_overlap
            in_load = _mask_window(
                x, y, load_x0, load_x1, load_y0, load_y1,
                xmax=xmax, ymax=ymax, inclusive_max=True,
            )
            idx = np.where(in_load)[0]
            if len(idx) <= target_load or eff_overlap <= 5.0:
                break
            eff_overlap = max(5.0, eff_overlap * 0.75)

        in_core = _mask_window(
            x[idx], y[idx], rect.x0, rect.x1, rect.y0, rect.y1,
            xmax=xmax, ymax=ymax, inclusive_max=True,
        )
        with log_lock:
            ov_note = f" overlap={eff_overlap:g}" if eff_overlap < overlap_m - 1e-6 else ""
            log(
                f"[TILE] {tag}: load={len(idx):,} core={int(in_core.sum()):,} "
                f"bbox=[{rect.x0:.1f},{rect.x1:.1f}]×[{rect.y0:.1f},{rect.y1:.1f}]{ov_note}"
            )
        tx = np.ascontiguousarray(x[idx])
        ty = np.ascontiguousarray(y[idx])
        tz = np.ascontiguousarray(z[idx])
        tcls = np.ascontiguousarray(cls_seed[idx])
        tint = np.ascontiguousarray(inten[idx]) if inten is not None else None
        tcls = run_arrays(
            tx, ty, tz, tcls,
            intensity=tint,
            source_class=src,
            cars_target_class=cars_target_class,
            noise_target_class=noise_target_class,
            fast=fast,
            no_subsample=no_subsample,
            rf_refiner=rf_refiner,
            hdbscan_backend=hdbscan_backend,
        )
        core_idx = idx[in_core]
        core_cls = np.ascontiguousarray(tcls[in_core])
        del tx, ty, tz, tcls, tint, idx
        gc.collect()
        return tag, core_idx, core_cls

    def _merge(tag: str, core_idx: np.ndarray, core_cls: np.ndarray) -> None:
        with merge_lock:
            cls_out[core_idx] = core_cls
            assigned[core_idx] = True
        with log_lock:
            log(f"[TILE] merged {tag}: {len(core_idx):,} core pts")

    if workers <= 1 or len(leaves) <= 1:
        for tag, rect, depth in leaves:
            _merge(*_run_leaf(tag, rect, depth))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_run_leaf, tag, rect, depth): tag
                for tag, rect, depth in leaves
            }
            for fut in as_completed(futs):
                tag = futs[fut]
                try:
                    _merge(*fut.result())
                except Exception:
                    with log_lock:
                        log(f"[TILE] FAILED {tag}")
                    raise

    n_miss = int((~assigned).sum())
    if n_miss:
        # Fallback: keep seed / default for any uncovered edge points
        log(f"[TILE] WARNING: {n_miss:,} points not covered by any core; leaving seed class")
        cls_out[~assigned] = cls_seed[~assigned]

    las.classification = cls_out
    log(f"[TILE] Writing merged {output_path} ...")
    from classify_tls_all_in_one import write_las_atomic

    write_las_atomic(las, output_path)
    log(
        f"[TILE] Done in {(time.time() - t0) / 60:.1f} min "
        f"({len(tiles)} top-level, {len(leaves)} leaves, workers={workers})."
    )
