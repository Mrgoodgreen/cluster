"""Guarded tiled cuML backend for the independent GPU 0.0.1 fork."""

from __future__ import annotations

import gc
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import classify_tls as tls


@dataclass
class GPUConfig:
    tile_m: float = 30.0
    overlap_m: float = 7.0
    max_tile_points: int = 2_000_000
    min_tile_m: float = 15.0
    cpu_safety_pass: bool = True
    gpu_device: int = 0
    build_algo: str = "brute_force"
    nn_descent_threshold: int = 400_000
    voxel_m: float = 0.08


CFG = GPUConfig()


@dataclass
class GPUStats:
    gpu_tiles: int = 0
    cpu_fallback_tiles: int = 0
    split_tiles: int = 0
    gpu_points: int = 0
    car_clusters: int = 0


STATS = GPUStats()


def configure(
    *,
    tile_m: float = 30.0,
    overlap_m: float = 7.0,
    max_tile_points: int = 2_000_000,
    min_tile_m: float = 15.0,
    cpu_safety_pass: bool = True,
    gpu_device: int = 0,
    build_algo: str = "brute_force",
    nn_descent_threshold: int = 400_000,
    voxel_m: float = 0.08,
) -> None:
    CFG.tile_m = tile_m
    CFG.overlap_m = overlap_m
    CFG.max_tile_points = max_tile_points
    CFG.min_tile_m = min_tile_m
    CFG.cpu_safety_pass = cpu_safety_pass
    CFG.gpu_device = gpu_device
    CFG.build_algo = build_algo
    CFG.nn_descent_threshold = nn_descent_threshold
    CFG.voxel_m = voxel_m


def check_gpu() -> str:
    """Validate CUDA/cuML and return a short device description."""
    import cupy as cp
    import cuml

    cp.cuda.Device(CFG.gpu_device).use()
    props = cp.cuda.runtime.getDeviceProperties(CFG.gpu_device)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    total = int(props["totalGlobalMem"]) / (1024**3)
    return f"{name}; VRAM={total:.1f} GiB; cuML={cuml.__version__}; CuPy={cp.__version__}"


def _free_gpu() -> None:
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    gc.collect()


def _gpu_labels(
    pts: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    epsilon: float,
) -> np.ndarray:
    """Run brute-force cuML HDBSCAN on locally centered coordinates."""
    import cupy as cp
    from cuml.cluster import HDBSCAN as cuHDBSCAN

    cp.cuda.Device(CFG.gpu_device).use()
    # Critical for Moscow coordinates: float32 must only see local deltas.
    centered64 = pts - pts.mean(axis=0, dtype=np.float64)
    inv: np.ndarray | None = None
    gpu_min_cluster = min_cluster_size
    gpu_min_samples = min_samples
    gpu_epsilon = epsilon
    if CFG.voxel_m > 0 and len(pts) >= 80_000:
        cents, inv, _ = tls.voxel_centroids(
            centered64[:, 0], centered64[:, 1], centered64[:, 2], CFG.voxel_m
        )
        tls.log(
            f"      GPU voxel: {len(pts):,} pts -> {len(cents):,} centroids "
            f"(cell={CFG.voxel_m:g} m)"
        )
        gpu_points = cents
        gpu_min_cluster = max(5, min_cluster_size // 4)
        gpu_min_samples = max(3, min_samples // 2)
        gpu_epsilon = max(epsilon, CFG.voxel_m * 2)
    else:
        gpu_points = centered64

    centered = gpu_points.astype(np.float32, copy=False)
    d_pts = cp.asarray(centered)
    selected_algo = (
        "nn_descent"
        if CFG.build_algo == "adaptive" and len(gpu_points) >= CFG.nn_descent_threshold
        else ("brute_force" if CFG.build_algo == "adaptive" else CFG.build_algo)
    )
    model = cuHDBSCAN(
        min_cluster_size=gpu_min_cluster,
        min_samples=gpu_min_samples,
        cluster_selection_epsilon=gpu_epsilon,
        metric="euclidean",
        build_algo=selected_algo,
        output_type="cupy",
    )
    labels = model.fit_predict(d_pts)
    result = cp.asnumpy(labels).astype(np.int32, copy=False)
    if inv is not None:
        result = result[inv]
    del labels, model, d_pts, centered
    _free_gpu()
    return result


def _is_oom(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "cudaerrormemoryallocation" in text or "bad_alloc" in text


def _is_splitworthy_gpu_error(exc: Exception) -> bool:
    """OOM and known transient CUDA failures that often clear after a smaller tile."""
    if _is_oom(exc):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return "cudaerrorinvalidvalue" in text or "invalid argument" in text


def _cpu_fallback_labels(
    pts: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    epsilon: float,
) -> np.ndarray:
    """CPU HDBSCAN matching the GPU voxel path — never cluster millions of raw TLS pts."""
    voxel = CFG.voxel_m if CFG.voxel_m > 0 else 0.08
    if len(pts) >= 80_000:
        cents, inv, _ = tls.voxel_centroids(
            pts[:, 0], pts[:, 1], pts[:, 2], voxel
        )
        tls.log(
            f"      CPU fallback voxel: {len(pts):,} pts -> {len(cents):,} centroids "
            f"(cell={voxel:g} m)"
        )
        labels_c = tls._cpu_hdbscan_labels(
            cents,
            max(5, min_cluster_size // 4),
            max(3, min_samples // 2),
            max(epsilon, voxel * 2),
        )
        return labels_c[inv]
    return tls._cpu_hdbscan_labels(pts, min_cluster_size, min_samples, epsilon)


def _rect_clusters(
    idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rect: tuple[float, float, float, float],
    *,
    min_cluster_size: int,
    min_samples: int,
    epsilon: float,
    depth: int = 0,
) -> Iterator[np.ndarray]:
    """Yield original point indices for clusters owned by a non-overlap core rectangle."""
    x0, x1, y0, y1 = rect
    px = x[idx]
    py = y[idx]
    expanded = (
        (px >= x0 - CFG.overlap_m)
        & (px < x1 + CFG.overlap_m)
        & (py >= y0 - CFG.overlap_m)
        & (py < y1 + CFG.overlap_m)
    )
    tidx = idx[expanded]
    if len(tidx) < min_cluster_size:
        return

    width = x1 - x0
    height = y1 - y0
    can_split = max(width, height) > CFG.min_tile_m
    if len(tidx) > CFG.max_tile_points and can_split:
        STATS.split_tiles += 1
        if width >= height:
            mid = (x0 + x1) / 2.0
            yield from _rect_clusters(
                tidx, x, y, z, (x0, mid, y0, y1),
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                epsilon=epsilon,
                depth=depth + 1,
            )
            yield from _rect_clusters(
                tidx, x, y, z, (mid, x1, y0, y1),
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                epsilon=epsilon,
                depth=depth + 1,
            )
        else:
            mid = (y0 + y1) / 2.0
            yield from _rect_clusters(
                tidx, x, y, z, (x0, x1, y0, mid),
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                epsilon=epsilon,
                depth=depth + 1,
            )
            yield from _rect_clusters(
                tidx, x, y, z, (x0, x1, mid, y1),
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                epsilon=epsilon,
                depth=depth + 1,
            )
        return

    pts = np.column_stack([x[tidx], y[tidx], z[tidx]])
    failure = None
    try:
        started = time.perf_counter()
        selected_algo = (
            "nn_descent"
            if CFG.build_algo == "adaptive" and len(tidx) >= CFG.nn_descent_threshold
            else ("brute_force" if CFG.build_algo == "adaptive" else CFG.build_algo)
        )
        tls.log(
            f"    GPU tile: {len(tidx):,} pts; core={width:.1f}x{height:.1f} m; "
            f"algo={selected_algo}"
        )
        labels = _gpu_labels(pts, min_cluster_size, min_samples, epsilon)
        tls.log(f"    GPU tile done in {time.perf_counter() - started:.1f}s")
        STATS.gpu_tiles += 1
        STATS.gpu_points += len(tidx)
    except Exception as exc:
        failure = (str(exc), _is_splitworthy_gpu_error(exc), 'OOM' if _is_oom(exc) else 'CUDA')
    # Release failed cuML arrays held by the exception traceback before retrying.
    if failure is not None:
        error_text, splitworthy, reason = failure
        _free_gpu()
        if splitworthy and can_split:
            STATS.split_tiles += 1
            tls.log(
                f"    GPU {reason} on {len(tidx):,} pts; splitting tile "
                f"{width:.1f}x{height:.1f} m"
            )
            if width >= height:
                mid = (x0 + x1) / 2.0
                parts = ((x0, mid, y0, y1), (mid, x1, y0, y1))
            else:
                mid = (y0 + y1) / 2.0
                parts = ((x0, x1, y0, mid), (x0, x1, mid, y1))
            for part in parts:
                yield from _rect_clusters(
                    tidx, x, y, z, part,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    epsilon=epsilon,
                    depth=depth + 1,
                )
            return

        tls.log(f"    GPU tile failed; CPU fallback ({len(tidx):,} pts): {error_text}")
        labels = _cpu_fallback_labels(pts, min_cluster_size, min_samples, epsilon)
        STATS.cpu_fallback_tiles += 1

    for lab in np.unique(labels[labels >= 0]):
        local = labels == lab
        cidx = tidx[local]
        # Every cluster is emitted by exactly one core tile, despite the overlap.
        owner_x = float(np.median(x[cidx]))
        owner_y = float(np.median(y[cidx]))
        if x0 <= owner_x < x1 and y0 <= owner_y < y1:
            yield cidx


def iter_gpu_clusters(
    idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
    epsilon: float,
) -> Iterator[np.ndarray]:
    if len(idx) < min_cluster_size:
        return

    xmin = math.floor(float(x[idx].min()) / CFG.tile_m) * CFG.tile_m
    xmax = math.ceil(float(x[idx].max()) / CFG.tile_m) * CFG.tile_m
    ymin = math.floor(float(y[idx].min()) / CFG.tile_m) * CFG.tile_m
    ymax = math.ceil(float(y[idx].max()) / CFG.tile_m) * CFG.tile_m
    nx = max(1, int(round((xmax - xmin) / CFG.tile_m)))
    ny = max(1, int(round((ymax - ymin) / CFG.tile_m)))
    tls.log(
        f"  guarded GPU: {len(idx):,} candidates; {nx}x{ny} cores; "
        f"tile={CFG.tile_m:g} m overlap={CFG.overlap_m:g} m "
        f"cap={CFG.max_tile_points:,}"
    )

    for ix in range(nx):
        x0 = xmin + ix * CFG.tile_m
        x1 = x0 + CFG.tile_m
        for iy in range(ny):
            y0 = ymin + iy * CFG.tile_m
            y1 = y0 + CFG.tile_m
            yield from _rect_clusters(
                idx, x, y, z, (x0, x1, y0, y1),
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                epsilon=epsilon,
            )


def extract_cars_guarded(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    source_classes: tuple[int, ...],
    car_target_class: int,
    min_height: float = 0.10,
    max_height: float | None = None,
    max_points: int | None = None,
    fast: bool = False,
    voxel: float = 0.08,
    backend: str = tls.HDBSCAN_BACKEND_CPU,
    stage: str = "E",
    strict: bool = False,
) -> list[tls.CarBox]:
    """GPU proposals, CPU float64 shape validation, optional final CPU safety pass."""
    del max_points, fast, voxel, backend
    cand = np.zeros(len(cls), dtype=bool)
    for value in source_classes:
        cand |= cls == value
    h = z - ground_z
    elevated = cand & (h > min_height)
    if max_height is not None:
        elevated &= h <= max_height
    idx = np.where(elevated)[0]
    if len(idx) < tls.CFG.car_min_cluster:
        tls.log(f"  no elevated candidates in classes {source_classes}")
        return []

    boxes: list[tls.CarBox] = []
    core_points = 0
    for cidx in iter_gpu_clusters(
        idx, x, y, z,
        min_cluster_size=tls.CFG.car_min_cluster,
        min_samples=tls.CFG.car_min_samples,
        epsilon=tls.CFG.car_epsilon,
    ):
        p = np.column_stack([x[cidx], y[cidx], z[cidx]])
        hc = h[cidx]
        # Critical guard: final decision remains the CPU 1.0.3 float64 heuristic.
        if not tls.is_car_cluster(p, hc, len(cidx), strict=strict):
            continue
        xmin, ymin, _ = p.min(axis=0)
        xmax, ymax, zmax = p.max(axis=0)
        cls[cidx] = car_target_class
        core_points += len(cidx)
        STATS.car_clusters += 1
        boxes.append((float(xmin), float(xmax), float(ymin), float(ymax), float(zmax)))

    tls.log(
        f"  guarded GPU cars -> class {car_target_class}: "
        f"{core_points:,} core; boxes={len(boxes)}"
        + (" [strict]" if strict else "")
    )

    # One residual CPU pass at the last stage. It catches GPU misses while E/I stay fast.
    if CFG.cpu_safety_pass and stage in {"J", "J_bld"}:
        tls.log(f"  [{stage}] CPU residual safety pass...")
        safety = tls.extract_cars_from_classes(
            x, y, z, cls, ground_z,
            source_classes=source_classes,
            car_target_class=car_target_class,
            min_height=min_height,
            max_height=max_height,
            max_points=tls.CFG.car_max_points,
            fast=False,
            voxel=tls.CFG.fast_voxel,
            backend=tls.HDBSCAN_BACKEND_CPU,
            stage=f"{stage}_safety",
            strict=strict,
        )
        boxes.extend(safety)
    return boxes


def cluster_objects_guarded(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    min_cluster_size: int = tls.CFG.obj_min_cluster,
    min_samples: int = tls.CFG.obj_min_samples,
    cluster_selection_epsilon: float = tls.CFG.obj_epsilon,
    max_points: int | None = None,
    car_target_class: int = tls.CLS_CAR,
    ped_target_class: int = tls.CLS_PED,
    ghost_target_class: int = tls.CLS_GHOST,
    fast: bool = False,
    voxel: float = tls.CFG.fast_voxel,
    backend: str = tls.HDBSCAN_BACKEND_CPU,
) -> list[tls.CarBox]:
    """Tiled GPU object proposals with the original CPU geometry rules."""
    del max_points, fast, voxel, backend
    idx = np.where(cls == tls.CLS_MED_VEG)[0]
    if len(idx) < min_cluster_size:
        tls.log("  Not enough points in class 4 for clustering.")
        return []

    cars_points = 0
    ped_points = 0
    ghost_points = 0
    car_boxes: list[tls.CarBox] = []

    for cidx in iter_gpu_clusters(
        idx, x, y, z,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        epsilon=cluster_selection_epsilon,
    ):
        p = np.column_stack([x[cidx], y[cidx], z[cidx]])
        xmin, ymin, zmin = p.min(axis=0)
        xmax, ymax, zmax = p.max(axis=0)
        dx, dy = xmax - xmin, ymax - ymin
        length = max(dx, dy)
        width = min(dx, dy)
        height_range = zmax - zmin
        count = len(cidx)
        h_cluster = p[:, 2] - ground_z[cidx]
        h_med = float(np.median(h_cluster))
        density = count / max(length * width, 0.01)

        kind: str | None = None
        if tls.is_car_cluster(p, h_cluster, count):
            kind = "car"
        elif tls.is_ghost_multipath_cluster(length, width, height_range, density, count):
            kind = "ghost"
        elif (
            0.25 <= width <= 1.0
            and 0.20 <= length <= 1.40
            and height_range <= 2.2
            and 1.0 <= h_med <= 2.1
            and count >= tls.CFG.ped_cluster_min_count
        ):
            kind = "ped"

        if kind == "car":
            cls[cidx] = car_target_class
            cars_points += count
            STATS.car_clusters += 1
            car_boxes.append((float(xmin), float(xmax), float(ymin), float(ymax), float(zmax)))
        elif kind == "ghost":
            cls[cidx] = ghost_target_class
            ghost_points += count
        elif kind == "ped":
            cls[cidx] = ped_target_class
            ped_points += count

    footprint_points = tls.peel_car_volume(
        x, y, z, cls, ground_z, car_boxes, car_target_class,
        peel_classes=tls.car_volume_peel_classes(),
        xy_pad=tls.CFG.car_xy_pad,
        min_h=tls.CFG.car_min_h,
        z_top_pad=tls.CFG.car_z_top_pad,
        dilate_cells=tls.CFG.car_peel_dilate,
        exclude_near_building_m=tls.CFG.peel_exclude_near_building_m,
    )
    tls.log(
        f"  guarded GPU objects: cars={cars_points:,} core + {footprint_points:,} volume; "
        f"peds={ped_points:,}; ghosts={ghost_points:,}"
    )
    return car_boxes


def log_stats() -> None:
    tls.log(
        "\nGPU 0.0.1 backend stats:"
        f"\n  GPU tiles: {STATS.gpu_tiles:,}"
        f"\n  split tiles: {STATS.split_tiles:,}"
        f"\n  CPU fallback tiles: {STATS.cpu_fallback_tiles:,}"
        f"\n  GPU tile-points (with overlap): {STATS.gpu_points:,}"
        f"\n  accepted car clusters: {STATS.car_clusters:,}"
    )
