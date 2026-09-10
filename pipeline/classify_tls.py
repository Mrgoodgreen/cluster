#!/usr/bin/env python3
"""
TLS street point cloud classification pipeline.
Processes source class (default: 11) while preserving buildings (class 6).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import CSF
import hdbscan
import laspy
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_opening,
    grey_dilation,
    grey_erosion,
    grey_opening,
    label as nd_label,
)
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# ASPRS / project classes
CLS_DEFAULT = 1
CLS_GROUND = 2
CLS_LOW_VEG = 3
CLS_MED_VEG = 4
CLS_HIGH_VEG = 5
CLS_BUILDING = 6
CLS_LOW_NOISE = 7
CLS_GHOST = 8
CLS_DIRTY = 11
CLS_CAR = 66
CLS_PED = 67
# TerraScan / etalon Vehicle class (cars, vans)
CLS_VEHICLE = 9

PRESERVE_CLASSES = {CLS_BUILDING}


@dataclass(frozen=True)
class PipelineConfig:
    """Validated defaults (Moscow TLS dirty-1-4 / v8r9)."""

    # ground
    ground_cell: float = 1.0
    ground_opening: int = 7
    ground_tol_above: float = 0.06
    ground_tol_below: float = 0.04
    elevated_peel_above: float = 0.12
    # residual junk left in ground (wheels, people, multipath over asphalt)
    ground_noise_cell: float = 0.30
    ground_noise_max_above: float = 0.040
    ground_noise_max_below: float = 0.06
    ground_sparse_cell: float = 0.12
    ground_sparse_min: int = 6
    # compact elevated islands in ground (wheel prints, bumper tips, litter)
    island_cell: float = 0.15
    island_min_h: float = 0.030
    island_max_h: float = 0.85
    island_min_count: int = 10
    island_max_length: float = 4.00
    island_max_width: float = 1.60
    # low car parts left in ground next to class 7 (wheels / rocker / bumper tips)
    car_residual_xy: float = 0.90
    car_residual_min_h: float = 0.03
    car_residual_max_h: float = 0.90
    car_residual_seed_h: float = 0.08
    # height layers (etalon low veg mostly <= 0.28 m)
    curb_max: float = 0.30
    obj_max: float = 2.20
    # object HDBSCAN (class 4)
    obj_min_cluster: int = 30
    obj_min_samples: int = 6
    obj_epsilon: float = 0.40
    obj_max_points: int = 1_500_000
    # car extract HDBSCAN
    car_min_cluster: int = 25
    car_min_samples: int = 6
    car_epsilon: float = 0.45
    car_max_points: int = 2_000_000
    car_min_height: float = 0.10
    # car volume peel
    car_xy_pad: float = 0.60
    car_min_h: float = 0.05
    car_z_top_pad: float = 0.40
    car_peel_dilate: int = 2
    # conservative car/noise: keep ground + facades; manual wheel cleanup OK
    car_peel_include_ground: bool = False
    car_peel_include_building: bool = False
    peel_near_car_ground: bool = False
    # Etalon-tuned: cars often trapped under early building/wall labels
    extract_cars_from_building: bool = True
    peel_ground_islands: bool = True
    peel_ground_noise_finalize: bool = True
    peel_elevated_junk_from_ground: bool = True
    ground_sparse_cleanup: bool = False
    demote_orphan_building_to_noise: bool = False
    # asphalt restore — DTM height cap prevents car roofs / floating junk → ground
    asphalt_restore_max_h: float = 0.07
    asphalt_restore_xy_radius: float = 1.25
    asphalt_restore_z_tol: float = 0.06
    ground_elevated_junk_min_h: float = 0.35
    ground_elevated_junk_max_h: float = 4.0
    ground_peel_protect_near_building_h: float = 2.80
    # curb restore (class 3 band; RS10 thinned)
    curb_restore_cell: float = 0.15
    curb_restore_min_count: int = 4
    curb_restore_min_length: float = 0.7
    curb_restore_max_width: float = 0.42
    curb_restore_min_h: float = 0.06
    curb_restore_max_h: float = 0.50
    # pedestrians absorbed into building near facades
    ped_demote_cell: float = 0.22
    ped_demote_min_count: int = 10
    ped_demote_max_count: int = 2500
    ped_demote_min_h: float = 0.85
    ped_demote_max_h: float = 2.35
    ped_demote_max_width: float = 0.55
    ped_demote_max_length: float = 0.95
    ped_demote_max_aspect: float = 2.4
    ped_demote_min_span: float = 0.75
    ped_cluster_min_count: int = 22
    # sparse cleanup
    sparse_veg_cell: float = 0.12
    sparse_veg_min: int = 6
    sparse_bld_cell: float = 0.20
    sparse_bld_min: int = 12
    # building detection (full-cloud / no prelabeled class 6)
    # Etalon: trees were swallowed (top_h/span like facades) — reject porous canopy cells
    building_cell: float = 1.25
    building_min_top_h: float = 3.8
    building_min_span: float = 2.8
    building_min_count: int = 60
    building_assign_min_h: float = 1.5
    building_base_assign_min_h: float = 0.45
    building_dilate: int = 1
    building_reject_veg: bool = True
    building_veg_min_top_h: float = 4.0
    building_veg_xy_iso_min: float = 0.35
    building_veg_fill_min: float = 0.55
    # walls / fences (RS10 mobile, ~50% thinned)
    # Etalon: car bodies matched wall top_h band — require thinner / taller posts
    wall_cell: float = 0.50
    wall_min_top_h: float = 0.90
    wall_max_top_h: float = 3.4
    wall_min_span: float = 0.55
    wall_min_count: int = 18
    wall_max_count: int = 220
    wall_assign_min_h: float = 0.20
    peel_exclude_near_building_m: float = 0.45
    car_max_vertical_aspect: float = 2.2
    # fast mode
    fast_voxel: float = 0.08
    fast_tile_m: float = 60.0


CFG = PipelineConfig()


def car_volume_peel_classes() -> tuple[int, ...]:
    """Classes eligible for car-box volume peel (default: vegetation only)."""
    classes: list[int] = [CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG]
    if CFG.car_peel_include_ground:
        classes.append(CLS_GROUND)
    if CFG.car_peel_include_building:
        classes.append(CLS_BUILDING)
    return tuple(classes)


# HDBSCAN backends: cpu = validated v8r9 baseline; cuml = full GPU; hybrid = GPU on D only.
HDBSCAN_BACKEND_CPU = "cpu"
HDBSCAN_BACKEND_CUML = "cuml"
HDBSCAN_BACKEND_HYBRID = "hybrid"
# Car-critical stages always use CPU in hybrid mode (E/I2 extract cars from veg/ground).
HDBSCAN_CPU_STAGES = frozenset({"E", "I2"})


_LOG_LOCK = __import__("threading").Lock()


def log(msg: str) -> None:
    with _LOG_LOCK:
        print(msg, flush=True)


def log_stage_timings(timings: dict[str, float]) -> None:
    if not timings:
        return
    total = sum(timings.values())
    log("\nStage timings:")
    for name, seconds in timings.items():
        share = (seconds / total * 100.0) if total > 0 else 0.0
        log(f"  {name:>10}: {seconds:6.1f}s  ({share:5.1f}%)")
    log(f"  {'TOTAL':>10}: {total:6.1f}s")


def class_histogram(cls: np.ndarray, title: str = "Classes") -> None:
    u, c = np.unique(cls, return_counts=True)
    log(f"\n{title}:")
    for a, b in zip(u, c):
        log(f"  class {int(a):3d}: {b:>12,}")


def grid_surface_z(x: np.ndarray, y: np.ndarray, z: np.ndarray, cell: float) -> np.ndarray:
    """Local surface estimate: 20th percentile Z per cell, smoothed."""
    xmin, ymin = x.min(), y.min()
    ix = np.floor((x - xmin) / cell).astype(np.int64)
    iy = np.floor((y - ymin) / cell).astype(np.int64)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    lin = ix * ny + iy

    order = np.argsort(lin, kind="mergesort")
    lin_s = lin[order]
    z_s = z[order]

    uniq, starts, counts = np.unique(lin_s, return_index=True, return_counts=True)
    p20 = np.empty(len(uniq), dtype=np.float64)
    for i, (s, c) in enumerate(zip(starts, counts)):
        chunk = z_s[s : s + c]
        p20[i] = np.percentile(chunk, 20)

    surf_flat = np.empty(uniq.max() + 1, dtype=np.float64)
    surf_flat[uniq] = p20
    surf = surf_flat[lin]

    grid = np.full((nx, ny), np.nan, dtype=np.float64)
    for u, v in zip(uniq, p20):
        i, j = divmod(int(u), ny)
        grid[i, j] = v
    fill = np.nanmedian(grid[np.isfinite(grid)])
    grid[~np.isfinite(grid)] = fill
    grid = grey_opening(grid, size=3)
    return np.minimum(surf, grid[ix, iy])


def radius_low_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    dz: float = 0.20,
) -> np.ndarray:
    """Points clearly below local minimum envelope (true multipath / buried noise).

    Uses per-cell min-Z, not p20: with buildings in the cloud a percentile
    surface rises onto the facade and wrongly marks road points as low.
    """
    surf = grid_min_z(x, y, z, 0.5)
    return z < (surf - dz)


def _grid_z_stats(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, cell: float, stat: str
) -> tuple[np.ndarray, float, float, int, int]:
    """Build per-cell Z statistic grid. Returns (values_per_point, xmin, ymin, nx, ny)."""
    xmin, ymin = x.min(), y.min()
    ix = np.floor((x - xmin) / cell).astype(np.int64)
    iy = np.floor((y - ymin) / cell).astype(np.int64)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    lin = ix * ny + iy

    order = np.argsort(lin, kind="mergesort")
    lin_s = lin[order]
    z_s = z[order]
    uniq, starts, counts = np.unique(lin_s, return_index=True, return_counts=True)

    vals = np.empty(len(uniq), dtype=np.float64)
    for i, (s, c) in enumerate(zip(starts, counts)):
        chunk = z_s[s : s + c]
        if stat == "min":
            vals[i] = chunk.min()
        elif stat == "p10":
            vals[i] = np.percentile(chunk, 10)
        else:
            vals[i] = np.percentile(chunk, 20)

    flat = np.empty(uniq.max() + 1, dtype=np.float64)
    flat[uniq] = vals
    return flat[lin], xmin, ymin, nx, ny


def grid_min_z(x: np.ndarray, y: np.ndarray, z: np.ndarray, cell: float) -> np.ndarray:
    """Per-point minimum Z from 2D grid."""
    vals, _, _, _, _ = _grid_z_stats(x, y, z, cell, "min")
    return vals


def robust_road_dtm(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell: float = 1.0,
    opening_size: int = 7,
) -> np.ndarray:
    """Road DTM that ignores car-sized elevated plateaus.

    Uses per-cell min-Z, then morphological opening so occluded parking
    stalls (where the lowest return is a car body) are pulled down to
    the surrounding true road level.
    """
    xmin, ymin = float(x.min()), float(y.min())
    ix = np.floor((x - xmin) / cell).astype(np.int64)
    iy = np.floor((y - ymin) / cell).astype(np.int64)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    lin = ix * ny + iy

    order = np.argsort(lin, kind="mergesort")
    lin_s = lin[order]
    z_s = z[order]
    uniq, starts = np.unique(lin_s, return_index=True)
    mins = np.minimum.reduceat(z_s, starts)

    grid = np.full((nx, ny), np.nan, dtype=np.float64)
    for u, v in zip(uniq, mins):
        i, j = divmod(int(u), ny)
        grid[i, j] = v

    finite = np.isfinite(grid)
    if not finite.any():
        return z.copy()
    fill = float(np.nanmedian(grid[finite]))
    grid[~finite] = fill
    # opening removes bright (high) bumps smaller than the structuring element
    grid = grey_opening(grid, size=max(3, opening_size))
    return grid[ix, iy]


def classify_ground_robust(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell: float = 1.0,
    opening_size: int = 7,
    tol_above: float = 0.06,
    tol_below: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ground_mask, road_dtm_per_point)."""
    dtm = robust_road_dtm(x, y, z, cell=cell, opening_size=opening_size)
    ground = (z >= (dtm - tol_below)) & (z <= (dtm + tol_above))
    return ground, dtm


def peel_elevated_from_ground(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    cell: float = 1.0,
    opening_size: int = 7,
    max_above: float = 0.12,
) -> int:
    """Reclassify class-2 points that sit above a robust road DTM (car roofs etc.)."""
    gmask = cls == CLS_GROUND
    if not gmask.any():
        return 0
    # Build DTM from ALL points so true road under/near cars is visible where scanned,
    # then opening removes car bumps from the surface model.
    dtm = robust_road_dtm(x, y, z, cell=cell, opening_size=opening_size)
    elevated = gmask & (z > (dtm + max_above))
    n = int(elevated.sum())
    if n:
        cls[elevated] = CLS_DIRTY  # back to work for height layers
    return n


def ground_dtm(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ground_mask: np.ndarray,
    cell: float = 0.5,
) -> np.ndarray:
    """Road DTM at every point: min-Z grid from ground-class points only."""
    fallback = grid_min_z(x, y, z, cell)
    if not ground_mask.any():
        return fallback

    gx, gy, gz = x[ground_mask], y[ground_mask], z[ground_mask]
    xmin = float(min(gx.min(), x.min()))
    ymin = float(min(gy.min(), y.min()))
    nx = int(np.floor((max(gx.max(), x.max()) - xmin) / cell)) + 1
    ny = int(np.floor((max(gy.max(), y.max()) - ymin) / cell)) + 1
    ix_g = np.floor((gx - xmin) / cell).astype(np.int64)
    iy_g = np.floor((gy - ymin) / cell).astype(np.int64)
    lin_g = ix_g * ny + iy_g

    order = np.argsort(lin_g, kind="mergesort")
    lin_s = lin_g[order]
    z_s = gz[order]
    uniq, starts = np.unique(lin_s, return_index=True)
    mins = np.minimum.reduceat(z_s, starts)

    grid_flat = np.full(nx * ny, np.nan, dtype=np.float64)
    grid_flat[uniq] = mins

    ix = np.clip(np.floor((x - xmin) / cell).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.floor((y - ymin) / cell).astype(np.int64), 0, ny - 1)
    lin = ix * ny + iy
    dtm = grid_flat[lin]
    missing = ~np.isfinite(dtm)
    dtm = dtm.copy()
    dtm[missing] = fallback[missing]
    return dtm


def is_car_cluster(
    p: np.ndarray,
    h_cluster: np.ndarray,
    count: int,
    strict: bool = False,
) -> bool:
    """Heuristic for cars and vans (TLS street scenes).

    ``strict=True`` (etalon-tuned): used when extracting from class 6 so facade
    chunks / street furniture are not accepted as vehicles.
    """
    xmin, ymin, zmin = p.min(axis=0)
    xmax, ymax, zmax = p.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin
    length = max(dx, dy)
    width = min(dx, dy)
    height_range = zmax - zmin
    h_p90 = float(np.percentile(h_cluster, 90))
    h_p50 = float(np.percentile(h_cluster, 50))
    density = count / max(length * width, 0.01)
    aspect = width / max(length, 0.01)
    vert_aspect = height_range / max(width, 0.05)
    if vert_aspect > CFG.car_max_vertical_aspect:
        return False
    if strict:
        # Etalon vehicle blobs: L~2.5–5 m, W~1.75–3.5 m, h mostly 0.35–2.2
        # Reject facade slabs: too many points, or planar (small λ_min / λ_max).
        if count > 120_000:
            return False
        if not (
            2.0 <= length <= 8.5
            and 1.10 <= width <= 3.90
            and 0.55 <= height_range <= 2.60
            and 0.45 <= h_p90 <= 2.40
            and 0.30 <= h_p50 <= 1.80
            and count >= 80
            and density >= 12
            and aspect <= 0.88
            and aspect >= 0.28
        ):
            return False
        # 3D planarity: building walls are flat sheets; cars are volumetric.
        c = p - p.mean(axis=0)
        cov = (c.T @ c) / max(count, 1)
        evals = np.linalg.eigvalsh(cov)
        flatness = float(evals[0] / max(evals[2], 1e-9))
        if flatness < 0.035:
            return False
        # Cars fill XY bbox; thin facade ribs leave most of the box empty.
        fill = count / max(length * width * max(height_range, 0.3), 0.01)
        if fill < 6.0:
            return False
        return True
    return (
        1.2 <= length <= 12.0
        and 0.40 <= width <= 5.0
        and 0.20 <= height_range <= 4.0
        and 0.12 <= h_p90 <= 2.50  # reject tree canopy blobs
        and count >= 20
        and density >= 6
        and aspect <= 0.92
    )


def extract_cars_by_xy_blobs(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    source_classes: tuple[int, ...],
    car_target_class: int,
    min_h: float = 0.35,
    max_h: float = 2.30,
    cell: float = 0.25,
    min_length: float = 2.0,
    max_length: float = 8.0,
    min_width: float = 1.15,
    max_width: float = 3.90,
    min_count: int = 120,
    max_count: int = 120_000,
    min_h_p50: float = 0.40,
    max_h_p90: float = 2.35,
    protect_near_building_m: float = 0.0,
    reject_under_tall: bool = False,
    tall_h: float = 3.5,
) -> list[CarBox]:
    """Fallback car extract via XY connected components (catches HDBSCAN misses).

    Etalon: many vehicle points remain in class 4 as fragmented clusters; grid
    blobs recover car-sized footprints in the object height band.

    If ``reject_under_tall``, drop candidate cells that also contain points
    with h >= tall_h (facade columns); cars have empty air above them.
    """
    h = z - ground_z
    cand = np.zeros(len(cls), dtype=bool)
    for c in source_classes:
        cand |= cls == c
    cand &= (h >= min_h) & (h <= max_h)
    if protect_near_building_m > 0:
        cand &= ~_mask_near_building_xy(x, y, cls, protect_near_building_m)
    if reject_under_tall:
        # XY cells that have any tall point (any class) — facade / tree trunk
        tall = h >= tall_h
        if tall.any():
            tcell = 0.40
            xmin0, ymin0 = float(x.min()), float(y.min())
            tix = np.floor((x - xmin0) / tcell).astype(np.int64)
            tiy = np.floor((y - ymin0) / tcell).astype(np.int64)
            nx_t, ny_t = int(tix.max()) + 1, int(tiy.max()) + 1
            tall_occ = np.zeros((nx_t, ny_t), dtype=bool)
            tall_occ[tix[tall], tiy[tall]] = True
            from scipy.ndimage import binary_dilation
            tall_occ = binary_dilation(tall_occ, iterations=1)
            under = tall_occ[tix, tiy]
            cand &= ~under
    if int(cand.sum()) < min_count:
        return []

    idx = np.where(cand)[0]
    px, py, pz = x[idx], y[idx], z[idx]
    ph = h[idx]
    xmin, ymin = float(px.min()), float(py.min())
    ix = np.floor((px - xmin) / cell).astype(np.int64)
    iy = np.floor((py - ymin) / cell).astype(np.int64)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    if nx * ny > 80_000_000:
        return []

    occ = np.zeros((nx, ny), dtype=np.uint8)
    occ[ix, iy] = 1
    # break thin bridges between parked cars
    occ = binary_opening(occ, structure=np.ones((2, 2), dtype=bool))
    labeled, nlab = nd_label(occ)
    if nlab == 0:
        return []

    lab = labeled[ix, iy]
    car_boxes: list[CarBox] = []
    moved = 0
    for lid in range(1, nlab + 1):
        sel = lab == lid
        cnt = int(sel.sum())
        if cnt < min_count or cnt > max_count:
            continue
        xs, ys = px[sel], py[sel]
        dx = float(xs.max() - xs.min())
        dy = float(ys.max() - ys.min())
        length, width = max(dx, dy), min(dx, dy)
        if not (min_length <= length <= max_length and min_width <= width <= max_width):
            continue
        aspect = width / max(length, 0.01)
        if aspect < 0.28 or aspect > 0.90:
            continue
        hs = ph[sel]
        h_p50 = float(np.percentile(hs, 50))
        h_p90 = float(np.percentile(hs, 90))
        if not (min_h_p50 <= h_p50 <= 1.90 and h_p90 <= max_h_p90):
            continue
        # volumetric fill vs facade sheet
        zspan = float(pz[sel].max() - pz[sel].min())
        if zspan < 0.45 or zspan > 2.70:
            continue
        density = cnt / max(length * width, 0.01)
        if density < 10:
            continue
        cls[idx[sel]] = car_target_class
        moved += cnt
        car_boxes.append(
            (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()), float(pz[sel].max()))
        )

    if moved:
        log(f"  XY-blob cars -> {car_target_class}: {moved:,} pts, boxes={len(car_boxes)}")
    return car_boxes


def is_ghost_multipath_cluster(
    length: float,
    width: float,
    height_range: float,
    density: float,
    count: int,
) -> bool:
    """Long thin low-density cluster — road multipath ghost, not a fence rail."""
    if count >= 900:
        return False
    if not (length > 3.2 and width < 1.8 and height_range < 2.6 and density < 140):
        return False
    # vertical fence post / facade rib — not a flat ghost streak
    if height_range / max(width, 0.05) > CFG.car_max_vertical_aspect:
        return False
    # horizontal fence rail with real height — not surface multipath
    if height_range >= 0.45 and height_range / max(length, 0.05) > 0.22:
        return False
    # paved curb lip: long, very thin, low — not road multipath
    if length >= 1.2 and width <= 0.42 and 0.06 <= height_range <= 0.55:
        return False
    return True


def voxel_centroids(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centroid_xyz [M,3], voxel_key per point [N], unique_keys [M])."""
    ix = np.floor(x / cell).astype(np.int64)
    iy = np.floor(y / cell).astype(np.int64)
    iz = np.floor(z / cell).astype(np.int64)
    # stable 1D key
    key = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791
    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    uniq, starts, counts = np.unique(key_s, return_index=True, return_counts=True)
    cx = np.add.reduceat(x[order], starts) / counts
    cy = np.add.reduceat(y[order], starts) / counts
    cz = np.add.reduceat(z[order], starts) / counts
    cents = np.column_stack([cx, cy, cz])
    inv = np.empty(len(key), dtype=np.int64)
    inv[order] = np.repeat(np.arange(len(uniq)), counts)
    return cents, inv, uniq


def _cpu_hdbscan_labels(
    pts: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    epsilon: float,
) -> np.ndarray:
    """Validated CPU hdbscan path (v8r9 baseline). Do not change without regression test."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        core_dist_n_jobs=1,
    )
    return clusterer.fit_predict(pts).astype(np.int32)


def _cuml_hdbscan_labels(
    pts: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    epsilon: float,
) -> np.ndarray:
    """cuML HDBSCAN; center coords first — float32 loses sub-meter detail on large absolutes."""
    from cuml.cluster import HDBSCAN as cuHDBSCAN  # type: ignore

    centered = (pts - pts.mean(axis=0)).astype(np.float32, copy=False)
    cu_model = cuHDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
    )
    labels = cu_model.fit_predict(centered)
    if hasattr(labels, "get"):
        labels = labels.get()
    return np.asarray(labels, dtype=np.int32)


def _resolve_hdbscan_gpu(backend: str, stage: str) -> bool:
    if backend == HDBSCAN_BACKEND_CPU:
        return False
    if backend == HDBSCAN_BACKEND_CUML:
        return True
    if backend == HDBSCAN_BACKEND_HYBRID:
        return stage not in HDBSCAN_CPU_STAGES
    return False


def hdbscan_labels_fast(
    pts: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    epsilon: float,
    fast: bool = False,
    voxel: float = 0.08,
    backend: str = HDBSCAN_BACKEND_CPU,
    stage: str = "",
) -> np.ndarray:
    """HDBSCAN; in fast mode cluster voxel centroids then map labels back."""
    use_gpu = _resolve_hdbscan_gpu(backend, stage)
    n = len(pts)
    if n < min_cluster_size:
        return np.full(n, -1, dtype=np.int32)

    if not fast or n < 80_000:
        if use_gpu:
            try:
                return _cuml_hdbscan_labels(pts, min_cluster_size, min_samples, epsilon)
            except Exception as e:
                log(f"  [GPU/{stage or '?'}] cuML HDBSCAN fail, fallback to CPU: {e}")

        return _cpu_hdbscan_labels(pts, min_cluster_size, min_samples, epsilon)

    cents, inv, _ = voxel_centroids(pts[:, 0], pts[:, 1], pts[:, 2], voxel)
    log(f"    fast HDBSCAN: {n:,} pts -> {len(cents):,} voxels (cell={voxel} m)")
    # scale min_cluster for voxels (~points per voxel ~ few)
    v_min = max(5, min_cluster_size // 4)
    v_samp = max(3, min_samples // 2)
    eps_v = max(epsilon, voxel * 2)
    if use_gpu:
        try:
            return _cuml_hdbscan_labels(cents, v_min, v_samp, eps_v)[inv]
        except Exception as e:
            log(f"  [GPU/{stage or '?'}] cuML HDBSCAN fast-path fail, fallback to CPU: {e}")

    vlab = _cpu_hdbscan_labels(cents, v_min, v_samp, eps_v)
    return vlab[inv].astype(np.int32)


def _mask_near_building_xy(
    x: np.ndarray,
    y: np.ndarray,
    cls: np.ndarray,
    radius: float,
    building_class: int = CLS_BUILDING,
) -> np.ndarray:
    """True where point is within XY radius of any building point."""
    if radius <= 0:
        return np.zeros(len(cls), dtype=bool)
    bidx = np.where(cls == building_class)[0]
    if len(bidx) == 0:
        return np.zeros(len(cls), dtype=bool)
    if len(bidx) > 2_000_000:
        rng = np.random.default_rng(42)
        bidx = rng.choice(bidx, 2_000_000, replace=False)
    tree = cKDTree(np.column_stack([x[bidx], y[bidx]]))
    near = np.zeros(len(cls), dtype=bool)
    batch = 500_000
    for start in range(0, len(cls), batch):
        stop = min(start + batch, len(cls))
        chunk = np.arange(start, stop)
        dists, _ = tree.query(
            np.column_stack([x[chunk], y[chunk]]),
            k=1,
            distance_upper_bound=radius,
        )
        near[chunk] = np.isfinite(dists)
    return near


def peel_car_volume(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    boxes: list[CarBox],
    target_class: int,
    peel_classes: tuple[int, ...] = (CLS_GROUND, CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
    xy_pad: float = 0.45,
    min_h: float = 0.06,
    z_top_pad: float = 0.40,
    cell: float = 0.25,
    dilate_cells: int = 1,
    exclude_near_building_m: float = 0.0,
) -> int:
    """Move elevated points inside car boxes to target (vectorized grid)."""
    if not boxes:
        return 0
    peel_m = _peel_classes_mask(cls, peel_classes)
    h = z - ground_z
    elevated = (h >= min_h) & peel_m & (cls != target_class)
    if not elevated.any():
        return 0

    occ, zmax_g, gxmin, gymin = build_footprint_grid(
        boxes, x, y, cell=cell, xy_pad=xy_pad, dilate_cells=dilate_cells,
    )
    # override zmax from boxes into grid already done in build_footprint_grid
    ix = np.floor((x - gxmin) / cell).astype(np.int64)
    iy = np.floor((y - gymin) / cell).astype(np.int64)
    nx, ny = occ.shape
    in_grid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & elevated
    m = np.zeros(len(cls), dtype=bool)
    valid = in_grid
    if valid.any():
        under = occ[ix[valid], iy[valid]]
        z_ok = z[valid] <= (zmax_g[ix[valid], iy[valid]] + z_top_pad)
        m[valid] = under & z_ok
    if exclude_near_building_m > 0 and m.any():
        m &= ~_mask_near_building_xy(x, y, cls, exclude_near_building_m)
    n = int(m.sum())
    if n:
        cls[m] = target_class
    return n


def peel_low_car_residuals_from_ground(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    seed_class: int,
    target_class: int | None = None,
    boxes: list[CarBox] | None = None,
    cell: float = 0.25,
    xy_radius: float = 0.75,
    min_h: float = 0.04,
    max_h: float = 0.90,
    seed_min_h: float = 0.10,
    seed_max_h: float = 3.50,
) -> int:
    """Peel low ground bumps next to cars (wheels / bumpers / rockers).

    Prefer ``boxes`` (HDBSCAN car footprints) when available — seeding from all
    class-7 noise is too broad on dirty TLS streets. Flat asphalt (h < min_h) stays.
    """
    if target_class is None:
        target_class = seed_class
    h = z - ground_z
    cand = (cls == CLS_GROUND) & (h >= min_h) & (h <= max_h)
    if not cand.any():
        return 0

    if boxes:
        occ, _zmax, xmin, ymin = build_footprint_grid(
            boxes, x, y, cell=cell, xy_pad=xy_radius, dilate_cells=max(1, int(round(xy_radius / cell))),
        )
        nx, ny = occ.shape
        cidx = np.where(cand)[0]
        cix = np.floor((x[cidx] - xmin) / cell).astype(np.int64)
        ciy = np.floor((y[cidx] - ymin) / cell).astype(np.int64)
        inb = (cix >= 0) & (cix < nx) & (ciy >= 0) & (ciy < ny)
        hit = np.zeros(len(cidx), dtype=bool)
        if inb.any():
            hit[inb] = occ[cix[inb], ciy[inb]]
        n = int(hit.sum())
        if n:
            cls[cidx[hit]] = target_class
        return n

    seed = (cls == seed_class) & (h >= seed_min_h) & (h <= seed_max_h)
    if not seed.any():
        return 0

    sx, sy = x[seed], y[seed]
    xmin = float(min(sx.min(), x[cand].min())) - xy_radius
    ymin = float(min(sy.min(), y[cand].min())) - xy_radius
    xmax = float(max(sx.max(), x[cand].max())) + xy_radius
    ymax = float(max(sy.max(), y[cand].max())) + xy_radius
    nx = max(1, int(np.ceil((xmax - xmin) / cell)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell)))
    if nx * ny > 80_000_000:
        cell = max(cell, 0.40)
        nx = max(1, int(np.ceil((xmax - xmin) / cell)))
        ny = max(1, int(np.ceil((ymax - ymin) / cell)))

    occ = np.zeros((nx, ny), dtype=bool)
    six = np.floor((sx - xmin) / cell).astype(np.int64)
    siy = np.floor((sy - ymin) / cell).astype(np.int64)
    ok = (six >= 0) & (six < nx) & (siy >= 0) & (siy < ny)
    occ[six[ok], siy[ok]] = True

    dilate = max(1, int(round(xy_radius / cell)))
    if dilate > 0:
        occ = grey_dilation(occ.astype(np.uint8), size=2 * dilate + 1).astype(bool)

    cidx = np.where(cand)[0]
    cix = np.floor((x[cidx] - xmin) / cell).astype(np.int64)
    ciy = np.floor((y[cidx] - ymin) / cell).astype(np.int64)
    inb = (cix >= 0) & (cix < nx) & (ciy >= 0) & (ciy < ny)
    hit = np.zeros(len(cidx), dtype=bool)
    if inb.any():
        hit[inb] = occ[cix[inb], ciy[inb]]
    n = int(hit.sum())
    if n:
        cls[cidx[hit]] = target_class
    return n


CarBox = tuple[float, float, float, float, float]  # xmin, xmax, ymin, ymax, zmax


def _peel_classes_mask(cls: np.ndarray, peel_classes: tuple[int, ...]) -> np.ndarray:
    m = np.zeros(len(cls), dtype=bool)
    for c in peel_classes:
        m |= cls == c
    return m


def build_footprint_grid(
    boxes: list[CarBox],
    x: np.ndarray,
    y: np.ndarray,
    cell: float = 0.25,
    xy_pad: float = 0.85,
    dilate_cells: int = 2,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Rasterize car bounding boxes to occupancy + per-cell roof Z."""
    if not boxes:
        return np.zeros((1, 1), dtype=bool), np.zeros((1, 1), dtype=np.float64), 0.0, 0.0

    xmin = min(b[0] for b in boxes) - xy_pad
    xmax = max(b[1] for b in boxes) + xy_pad
    ymin = min(b[2] for b in boxes) - xy_pad
    ymax = max(b[3] for b in boxes) + xy_pad

    nx = max(1, int(np.ceil((xmax - xmin) / cell)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell)))
    occ = np.zeros((nx, ny), dtype=bool)
    zmax_g = np.zeros((nx, ny), dtype=np.float64)

    for bxmin, bxmax, bymin, bymax, bzmax in boxes:
        i0 = max(0, int(np.floor((bxmin - xy_pad - xmin) / cell)))
        i1 = min(nx, int(np.ceil((bxmax + xy_pad - xmin) / cell)))
        j0 = max(0, int(np.floor((bymin - xy_pad - ymin) / cell)))
        j1 = min(ny, int(np.ceil((bymax + xy_pad - ymin) / cell)))
        occ[i0:i1, j0:j1] = True
        zmax_g[i0:i1, j0:j1] = np.maximum(zmax_g[i0:i1, j0:j1], bzmax)

    if dilate_cells > 0:
        occ = grey_erosion(~occ, size=2 * dilate_cells + 1) == 0
        zmax_g = grey_dilation(zmax_g, size=2 * dilate_cells + 1)

    return occ, zmax_g, xmin, ymin


def peel_by_footprint_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    occ: np.ndarray,
    zmax_g: np.ndarray,
    gxmin: float,
    gymin: float,
    cell: float,
    target_class: int,
    peel_classes: tuple[int, ...] = (CLS_GROUND, CLS_LOW_VEG, CLS_MED_VEG),
    z_top_pad: float = 0.35,
) -> int:
    """Single O(n) pass: peel road/curb points under rasterized car footprints."""
    peel_m = _peel_classes_mask(cls, peel_classes)
    if not peel_m.any() or not occ.any():
        return 0

    ix = np.floor((x - gxmin) / cell).astype(np.int64)
    iy = np.floor((y - gymin) / cell).astype(np.int64)
    nx, ny = occ.shape
    in_grid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    m = np.zeros(len(cls), dtype=bool)
    valid = peel_m & in_grid
    if valid.any():
        under = occ[ix[valid], iy[valid]]
        z_ok = z[valid] <= (zmax_g[ix[valid], iy[valid]] + z_top_pad)
        m[valid] = under & z_ok

    n = int(m.sum())
    if n:
        cls[m] = target_class
    return n


def peel_near_car_seeds(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    car_class: int,
    target_class: int,
    peel_classes: tuple[int, ...] = (CLS_GROUND, CLS_LOW_VEG),
    xy_radius: float = 0.55,
    z_pad: float = 0.20,
    max_car_h: float = 0.55,
    max_peel_h: float = 0.12,
) -> int:
    """Peel road-surface points within XY radius of low car body/wheel points."""
    car_m = cls == car_class
    if not car_m.any():
        return 0

    car_h = z[car_m] - ground_z[car_m]
    low_car = car_h <= max_car_h
    if not low_car.any():
        return 0

    car_xy = np.column_stack([x[car_m][low_car], y[car_m][low_car]])
    car_z = z[car_m][low_car]
    tree = cKDTree(car_xy)

    peel_m = _peel_classes_mask(cls, peel_classes)
    peel_idx = np.where(peel_m)[0]
    if len(peel_idx) == 0:
        return 0

    peel_h = z[peel_idx] - ground_z[peel_idx]
    surface = peel_h <= max_peel_h
    peel_idx = peel_idx[surface]
    if len(peel_idx) == 0:
        return 0

    pts_xy = np.column_stack([x[peel_idx], y[peel_idx]])
    dists, nn = tree.query(pts_xy, k=1, distance_upper_bound=xy_radius)
    valid = np.isfinite(dists)
    if not valid.any():
        return 0

    hits = peel_idx[valid]
    z_ok = z[hits] <= (car_z[nn[valid]] + z_pad)
    n = int(z_ok.sum())
    if n:
        cls[hits[z_ok]] = target_class
    return n


def recover_landscape_from_noise(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    noise_class: int = CLS_LOW_NOISE,
    cell: float = 0.15,
    min_count: int = 8,
    min_h: float = 0.10,
    max_h: float = 2.50,
) -> int:
    """Return dense elevated 'noise' (planters/flowerbeds) to height layers.

    True low-noise is sparse or below ground; cars are caught later by clustering.
    """
    noise = cls == noise_class
    if not noise.any():
        return 0
    h = z - ground_z
    cand = noise & (h >= min_h) & (h <= max_h)
    idx = np.where(cand)[0]
    if len(idx) < min_count:
        return 0

    px, py, pz = x[idx], y[idx], z[idx]
    ph = h[idx]
    ix = np.floor(px / cell).astype(np.int64)
    iy = np.floor(py / cell).astype(np.int64)
    iz = np.floor(pz / cell).astype(np.int64)
    key = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791
    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    uniq, starts, counts = np.unique(key_s, return_index=True, return_counts=True)

    # flat near-ground slabs in noise are paving mislabeled, not flowerbeds
    xy_cell = 1.0
    xymin_x, xymin_y = float(px.min()), float(py.min())
    ixy = np.floor((px - xymin_x) / xy_cell).astype(np.int64)
    iyy = np.floor((py - xymin_y) / xy_cell).astype(np.int64)
    nyy = int(iyy.max()) + 1
    xylin = ixy * nyy + iyy
    xy_order = np.argsort(xylin, kind="mergesort")
    xylin_s = xylin[xy_order]
    xy_uniq, xy_starts, xy_counts = np.unique(xylin_s, return_index=True, return_counts=True)
    flat_xy_grid = np.zeros(int(xy_uniq.max()) + 1, dtype=bool)
    for u, s, c in zip(xy_uniq, xy_starts, xy_counts):
        if c < 15:
            continue
        sl = slice(s, s + c)
        o = xy_order[sl]
        if float(pz[o].max() - pz[o].min()) <= 0.14 and float(ph[o].mean()) <= 0.30:
            flat_xy_grid[int(u)] = True

    cell_count = np.repeat(counts, counts)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    dense = cell_count[inv] >= min_count
    not_flat = ~flat_xy_grid[np.clip(xylin, 0, len(flat_xy_grid) - 1)]
    dense &= not_flat

    take = idx[dense]
    if len(take) == 0:
        return 0

    ht = h[take]
    m = ht < 0.35
    cls[take[m]] = CLS_LOW_VEG
    m = (ht >= 0.35) & (ht < 2.20)
    cls[take[m]] = CLS_MED_VEG
    m = ht >= 2.20
    cls[take[m]] = CLS_HIGH_VEG
    return len(take)


def relayer_by_height(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    source_classes: tuple[int, ...] = (CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
    curb_max: float = 0.35,
    obj_max: float = 2.20,
    ground_tol: float = 0.08,
) -> dict[str, int]:
    """Re-assign veg points by height above robust DTM (fixes paving in high veg etc.)."""
    mask = np.zeros(len(cls), dtype=bool)
    for c in source_classes:
        mask |= cls == c
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return {"ground": 0, "low": 0, "med": 0, "high": 0}

    h = z[idx] - ground_z[idx]
    counts = {"ground": 0, "low": 0, "med": 0, "high": 0}

    m = h <= ground_tol
    cls[idx[m]] = CLS_GROUND
    counts["ground"] = int(m.sum())

    m = (h > ground_tol) & (h < curb_max)
    cls[idx[m]] = CLS_LOW_VEG
    counts["low"] = int(m.sum())

    m = (h >= curb_max) & (h < obj_max)
    cls[idx[m]] = CLS_MED_VEG
    counts["med"] = int(m.sum())

    m = h >= obj_max
    cls[idx[m]] = CLS_HIGH_VEG
    counts["high"] = int(m.sum())
    return counts


def detect_buildings(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    cell: float = CFG.building_cell,
    min_top_h: float = CFG.building_min_top_h,
    min_span: float = CFG.building_min_span,
    min_count: int = CFG.building_min_count,
    assign_min_h: float = CFG.building_assign_min_h,
    dilate: int = CFG.building_dilate,
    building_class: int = CLS_BUILDING,
    protect_classes: tuple[int, ...] = (CLS_VEHICLE,),
    veg_class: int = CLS_HIGH_VEG,
) -> int:
    """Mark vertical dense XY columns as buildings (for unlabeled full-cloud runs).

    TLS facades fill tall XY cells. Tree canopies often match top_h/span but are
    more isotropic in XY and fill height bins more uniformly — those go to veg.
    Protected classes (e.g. vehicles) are never overwritten.
    """
    h = z - ground_z
    protect = np.zeros(len(cls), dtype=bool)
    for c in protect_classes:
        protect |= cls == c
    # candidates: not ground / protected, elevated enough to matter
    cand = (cls != CLS_GROUND) & (~protect) & (h >= 0.3)
    if not cand.any():
        return 0

    xmin, ymin = float(x.min()), float(y.min())
    ix = np.floor((x - xmin) / cell).astype(np.int64)
    iy = np.floor((y - ymin) / cell).astype(np.int64)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    lin = ix * ny + iy

    cidx = np.where(cand)[0]
    clin = lin[cidx]
    cx = x[cidx]
    cy = y[cidx]
    cz = z[cidx]
    ch = h[cidx]

    order = np.argsort(clin, kind="mergesort")
    clin_s = clin[order]
    cx_s = cx[order]
    cy_s = cy[order]
    cz_s = cz[order]
    ch_s = ch[order]
    uniq, starts, counts = np.unique(clin_s, return_index=True, return_counts=True)

    build_cells = np.zeros(nx * ny, dtype=bool)
    veg_cells = np.zeros(nx * ny, dtype=bool)
    reject_veg = CFG.building_reject_veg
    for u, s, c in zip(uniq, starts, counts):
        if c < min_count:
            continue
        chunk_z = cz_s[s : s + c]
        chunk_h = ch_s[s : s + c]
        top_h = float(np.percentile(chunk_h, 90))
        span = float(np.percentile(chunk_z, 90) - np.percentile(chunk_z, 10))
        if top_h < min_top_h or span < min_span:
            continue
        if reject_veg and top_h >= CFG.building_veg_min_top_h:
            # XY anisotropy: facade = elongated; canopy = more circular
            px = cx_s[s : s + c]
            py = cy_s[s : s + c]
            px = px - px.mean()
            py = py - py.mean()
            cov_xx = float(np.dot(px, px) / max(c, 1))
            cov_yy = float(np.dot(py, py) / max(c, 1))
            cov_xy = float(np.dot(px, py) / max(c, 1))
            # eigenvalues of 2x2 covariance
            tr = cov_xx + cov_yy
            det = cov_xx * cov_yy - cov_xy * cov_xy
            disc = max(tr * tr * 0.25 - det, 0.0)
            l1 = 0.5 * tr + np.sqrt(disc)
            l2 = 0.5 * tr - np.sqrt(disc)
            iso = float(l2 / max(l1, 1e-9))
            # vertical fill: fraction of 0.5 m bins occupied
            hmin = float(chunk_h.min())
            nb = max(int(np.ceil((top_h - max(hmin, 0.5)) / 0.5)), 1)
            bins = np.floor((chunk_h - max(hmin, 0.5)) / 0.5).astype(np.int64)
            bins = bins[(bins >= 0) & (bins < nb)]
            if len(bins):
                occ = np.bincount(bins, minlength=nb)
                fill = float((occ >= 3).mean())
            else:
                fill = 0.0
            if iso >= CFG.building_veg_xy_iso_min and fill >= CFG.building_veg_fill_min:
                veg_cells[int(u)] = True
                continue
        build_cells[int(u)] = True

    if dilate > 0 and build_cells.any():
        grid = build_cells.reshape(nx, ny)
        grid = grey_dilation(grid.astype(np.uint8), size=2 * dilate + 1).astype(bool)
        # do not dilate into veg canopy cells
        if veg_cells.any():
            grid &= ~veg_cells.reshape(nx, ny)
        build_cells = grid.ravel()

    n = 0
    if veg_cells.any():
        in_veg = veg_cells[lin]
        mv = in_veg & (h >= 1.0) & (cls != CLS_GROUND) & (~protect)
        n_veg = int(mv.sum())
        if n_veg:
            cls[mv] = veg_class
            n += n_veg

    if not build_cells.any():
        return n

    in_col = build_cells[lin]
    # elevated points in building columns → building (never overwrite protected)
    m = in_col & (h >= assign_min_h) & (cls != CLS_GROUND) & (~protect)
    nb = int(m.sum())
    if nb:
        cls[m] = building_class
        n += nb
    return n


def detect_walls_and_fences(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    cell: float = CFG.wall_cell,
    min_top_h: float = CFG.wall_min_top_h,
    max_top_h: float = CFG.wall_max_top_h,
    min_span: float = CFG.wall_min_span,
    min_count: int = CFG.wall_min_count,
    assign_min_h: float = CFG.wall_assign_min_h,
    building_class: int = CLS_BUILDING,
    protect_classes: tuple[int, ...] = (CLS_VEHICLE,),
    max_count: int = CFG.wall_max_count,
) -> int:
    """Mark low vertical columns as building (fences, parapets, facade ribs).

    Complements ``detect_buildings`` which needs top_h >= ~3.8 m. Tuned for
    thinned RS10 mobile scans where fence posts are sparse but vertically coherent.
    Dense car-body cells (high point count in the wall band) are skipped.
    """
    h = z - ground_z
    protect = np.zeros(len(cls), dtype=bool)
    for c in protect_classes:
        protect |= cls == c
    cand = (cls != CLS_GROUND) & (cls != building_class) & (~protect) & (h >= assign_min_h)
    if not cand.any():
        return 0

    xmin, ymin = float(x.min()), float(y.min())
    ix = np.floor((x - xmin) / cell).astype(np.int64)
    iy = np.floor((y - ymin) / cell).astype(np.int64)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    lin = ix * ny + iy

    cidx = np.where(cand)[0]
    clin = lin[cidx]
    cz = z[cidx]
    ch = h[cidx]

    order = np.argsort(clin, kind="mergesort")
    clin_s = clin[order]
    cz_s = cz[order]
    ch_s = ch[order]
    uniq, starts, counts = np.unique(clin_s, return_index=True, return_counts=True)

    wall_cells = np.zeros(nx * ny, dtype=bool)
    for u, s, c in zip(uniq, starts, counts):
        if c < min_count or c > max_count:
            continue
        chunk_z = cz_s[s : s + c]
        chunk_h = ch_s[s : s + c]
        top_h = float(np.percentile(chunk_h, 90))
        span = float(np.percentile(chunk_z, 90) - np.percentile(chunk_z, 10))
        if min_top_h <= top_h <= max_top_h and span >= min_span:
            wall_cells[int(u)] = True

    if not wall_cells.any():
        return 0

    in_col = wall_cells[lin]
    m = (
        in_col
        & (h >= assign_min_h)
        & (cls != CLS_GROUND)
        & (cls != building_class)
        & (~protect)
    )
    n = int(m.sum())
    if n:
        cls[m] = building_class
    return n


def attach_facade_band_near_buildings(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    source_classes: tuple[int, ...] = (CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG, CLS_LOW_NOISE),
    building_class: int = CLS_BUILDING,
    band_min: float = 0.35,
    band_max: float = 2.80,
    xy_radius: float = 0.22,
) -> int:
    """Attach facade skirts / fence bases in the car-height band near class 6.

    Fills the blind zone skipped by ``attach_near_buildings_xy`` (curb..obj_max).
    """
    bmask = cls == building_class
    smask = np.zeros(len(cls), dtype=bool)
    for c in source_classes:
        smask |= cls == c
    h = z - ground_z
    smask &= (h >= band_min) & (h <= band_max)
    if not bmask.any() or not smask.any():
        return 0

    bidx = np.where(bmask)[0]
    sidx = np.where(smask)[0]
    if len(bidx) > 3_000_000:
        rng = np.random.default_rng(42)
        bidx = rng.choice(bidx, 3_000_000, replace=False)

    tree = cKDTree(np.column_stack([x[bidx], y[bidx]]))
    batch = 400_000
    moved = 0
    for start in range(0, len(sidx), batch):
        chunk = sidx[start : start + batch]
        dists, _ = tree.query(
            np.column_stack([x[chunk], y[chunk]]),
            k=1,
            distance_upper_bound=xy_radius,
        )
        valid = np.isfinite(dists)
        if valid.any():
            cls[chunk[valid]] = building_class
            moved += int(valid.sum())
    return moved


def restore_curbs_from_noise(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    noise_class: int = CLS_LOW_NOISE,
    curb_class: int = CLS_LOW_VEG,
    cell: float = CFG.curb_restore_cell,
    min_count: int = CFG.curb_restore_min_count,
    min_length: float = CFG.curb_restore_min_length,
    max_width: float = CFG.curb_restore_max_width,
    min_h: float = CFG.curb_restore_min_h,
    max_h: float = CFG.curb_restore_max_h,
) -> int:
    """Recover paved curb lips wrongly sent to noise.

    Curbs live in the low-veg height band (class 3) but sparse cleanup on thinned
    mobile TLS often demotes elongated near-ground structure to class 7.
    """
    h = z - ground_z
    cand = (cls == noise_class) & (h >= min_h) & (h <= max_h)
    cidx = np.where(cand)[0]
    if len(cidx) < min_count:
        return 0

    px, py = x[cidx], y[cidx]
    xmin, ymin = float(px.min()), float(py.min())
    ix = np.floor((px - xmin) / cell).astype(np.int64)
    iy = np.floor((py - ymin) / cell).astype(np.int64)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1

    occ = np.zeros((nx, ny), dtype=np.uint8)
    occ[ix, iy] = 1
    occ = grey_dilation(occ, size=3)
    labeled, nlab = nd_label(occ)
    if nlab == 0:
        return 0

    point_lab = labeled[ix, iy]
    keep = np.zeros(len(cidx), dtype=bool)
    for lid in range(1, nlab + 1):
        cy, cx = np.where(labeled == lid)
        dx = float(cx.max() - cx.min()) * cell
        dy = float(cy.max() - cy.min()) * cell
        length, width = max(dx, dy), min(dx, dy)
        if length < min_length or width > max_width:
            continue
        if length / max(width, cell * 0.5) < 3.0:
            continue
        pt_mask = point_lab == lid
        if int(pt_mask.sum()) < min_count:
            continue
        keep |= pt_mask

    n = int(keep.sum())
    if n:
        cls[cidx[keep]] = curb_class
    return n


def demote_pedestrians_from_building(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    target_class: int = CLS_LOW_NOISE,
    building_class: int = CLS_BUILDING,
    cell: float = CFG.ped_demote_cell,
    min_count: int = CFG.ped_demote_min_count,
    max_count: int = CFG.ped_demote_max_count,
    min_h: float = CFG.ped_demote_min_h,
    max_h: float = CFG.ped_demote_max_h,
    max_width: float = CFG.ped_demote_max_width,
    max_length: float = CFG.ped_demote_max_length,
    max_aspect: float = CFG.ped_demote_max_aspect,
    min_span: float = CFG.ped_demote_min_span,
) -> int:
    """Move compact human-sized blobs out of building (class 6) near facades.

    Facade attach / restore often swallows standing people into class 6.
    Keeps wall ribbons (elongated) and tall columns.
    Etalon: people → Low Point (Noise) class 7.
    """
    h = z - ground_z
    cand = (cls == building_class) & (h >= min_h) & (h <= max_h)
    cidx = np.where(cand)[0]
    if len(cidx) < min_count:
        return 0

    px, py, pz = x[cidx], y[cidx], z[cidx]
    ph = h[cidx]
    xmin, ymin = float(px.min()), float(py.min())
    ix = np.floor((px - xmin) / cell).astype(np.int64)
    iy = np.floor((py - ymin) / cell).astype(np.int64)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1

    occ = np.zeros((nx, ny), dtype=np.uint8)
    occ[ix, iy] = 1
    labeled, nlab = nd_label(occ)
    if nlab == 0:
        return 0

    point_lab = labeled[ix, iy]
    keep = np.zeros(len(cidx), dtype=bool)
    for lid in range(1, nlab + 1):
        pt_mask = point_lab == lid
        cnt = int(pt_mask.sum())
        if cnt < min_count or cnt > max_count:
            continue
        xs, ys, zs = px[pt_mask], py[pt_mask], pz[pt_mask]
        dx = float(xs.max() - xs.min())
        dy = float(ys.max() - ys.min())
        length, width = max(dx, dy), min(dx, dy)
        if width > max_width or length > max_length:
            continue
        if length / max(width, 0.04) > max_aspect:
            continue
        z_span = float(zs.max() - zs.min())
        if z_span < min_span or z_span > max_h + 0.15:
            continue
        mean_h = float(ph[pt_mask].mean())
        if mean_h < min_h or mean_h > max_h:
            continue
        keep |= pt_mask

    n = int(keep.sum())
    if n:
        cls[cidx[keep]] = target_class
    return n


def attach_near_buildings_xy(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray | None = None,
    source_classes: tuple[int, ...] = (CLS_MED_VEG, CLS_LOW_VEG),
    building_class: int = CLS_BUILDING,
    xy_radius: float = 0.40,
    curb_max: float = 0.40,
    obj_max: float = 2.20,
) -> int:
    """Attach veg points within XY radius of a building point (facade skirts).

    If ground_z is given, skip the car-height band (curb_max..obj_max) so
    parked cars next to facades are not swallowed into class 6.
    """
    bmask = cls == building_class
    smask = np.zeros(len(cls), dtype=bool)
    for c in source_classes:
        smask |= cls == c
    if ground_z is not None:
        h = z - ground_z
        smask &= (h < curb_max) | (h >= obj_max)
    if not bmask.any() or not smask.any():
        return 0

    bidx = np.where(bmask)[0]
    sidx = np.where(smask)[0]
    if len(bidx) > 3_000_000:
        rng = np.random.default_rng(42)
        bidx = rng.choice(bidx, 3_000_000, replace=False)

    tree = cKDTree(np.column_stack([x[bidx], y[bidx]]))
    batch = 400_000
    moved = 0
    for start in range(0, len(sidx), batch):
        chunk = sidx[start : start + batch]
        dists, _ = tree.query(
            np.column_stack([x[chunk], y[chunk]]),
            k=1,
            distance_upper_bound=xy_radius,
        )
        valid = np.isfinite(dists)
        if valid.any():
            cls[chunk[valid]] = building_class
            moved += int(valid.sum())
    return moved


def restore_flat_paving_xy(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    source_classes: tuple[int, ...] = (CLS_LOW_VEG, CLS_MED_VEG),
    ground_class: int = CLS_GROUND,
    cell: float = 1.0,
    max_h: float = 0.38,
    max_z_span: float = 0.14,
    min_count: int = 20,
    min_near_frac: float = 0.82,
) -> int:
    """Restore whole XY tiles of flat paving/sidewalk wrongly labeled as vegetation.

    Sidewalk slabs appear as green squares: many points in a 1 m tile, all near
    DTM and with tiny Z span. Cars (taller, more Z span) stay in veg for later peel.
    """
    src = np.zeros(len(cls), dtype=bool)
    for c in source_classes:
        src |= cls == c
    if not src.any():
        return 0

    idx = np.where(src)[0]
    h = z[idx] - ground_z[idx]
    xmin, ymin = float(x[idx].min()), float(y[idx].min())
    ix = np.floor((x[idx] - xmin) / cell).astype(np.int64)
    iy = np.floor((y[idx] - ymin) / cell).astype(np.int64)
    ny = int(iy.max()) + 1
    lin = ix * ny + iy

    order = np.argsort(lin, kind="mergesort")
    lin_s = lin[order]
    h_s = h[order]
    z_s = z[idx[order]]
    uniq, starts, counts = np.unique(lin_s, return_index=True, return_counts=True)

    pave_cell = np.zeros(len(uniq), dtype=bool)
    for i, (s, c) in enumerate(zip(starts, counts)):
        if c < min_count:
            continue
        ch = h_s[s : s + c]
        cz = z_s[s : s + c]
        near = (ch >= -0.05) & (ch <= max_h)
        if near.mean() < min_near_frac:
            continue
        if float(cz.max() - cz.min()) > max_z_span:
            continue
        pave_cell[i] = True

    if not pave_cell.any():
        return 0

    # map lin -> flat flag
    cell_flat = np.zeros(int(uniq.max()) + 1, dtype=bool)
    cell_flat[uniq[pave_cell]] = True

    all_lin = np.floor((x - xmin) / cell).astype(np.int64) * ny + np.floor((y - ymin) / cell).astype(np.int64)
    all_h = z - ground_z
    m = src & (all_h <= max_h + 0.05) & cell_flat[np.clip(all_lin, 0, len(cell_flat) - 1)]
    # only cells that exist in uniq
    valid_lin = np.isin(all_lin, uniq[pave_cell])
    m &= valid_lin
    n = int(m.sum())
    if n:
        cls[m] = ground_class
    return n


def restore_building_columns_from_noise(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    noise_class: int = CLS_LOW_NOISE,
    building_class: int = CLS_BUILDING,
    cell: float = 0.75,
    min_h: float = 0.40,
    max_h: float = 25.0,
    dilate: int = 1,
    tall_top_h: float = 3.50,
    car_band_max: float = 2.80,
) -> int:
    """Recover facade/window points in noise that share tall building XY columns.

    Only XY cells that still look like real facade columns (tall_top_h) are used.
    Inside those cells, points of any height >= min_h are restored. In merely
    dilated neighbor cells, only points above car_band_max are restored so
    parked cars beside the facade stay in noise.
    """
    bmask = cls == building_class
    if not bmask.any():
        return 0

    h = z - ground_z
    nidx = np.where((cls == noise_class) & (h >= min_h) & (h <= max_h))[0]
    if len(nidx) == 0:
        return 0

    bidx = np.where(bmask)[0]
    xmin, ymin = float(x[bidx].min()), float(y[bidx].min())
    ix_b = np.floor((x[bidx] - xmin) / cell).astype(np.int64)
    iy_b = np.floor((y[bidx] - ymin) / cell).astype(np.int64)
    nx = int(ix_b.max()) + 1
    ny = int(iy_b.max()) + 1
    lin_b = ix_b * ny + iy_b

    # per-cell max height of remaining building
    cell_max_h = np.full(nx * ny, -1.0, dtype=np.float64)
    bh = h[bidx]
    order = np.argsort(lin_b, kind="mergesort")
    lin_s = lin_b[order]
    h_s = bh[order]
    uniq, starts, counts = np.unique(lin_s, return_index=True, return_counts=True)
    for u, s, c in zip(uniq, starts, counts):
        cell_max_h[int(u)] = float(h_s[s : s + c].max())

    tall = cell_max_h >= tall_top_h
    if not tall.any():
        return 0

    tall_core = tall.copy()
    if dilate > 0:
        grid = tall.reshape(nx, ny)
        grid = grey_dilation(grid.astype(np.uint8), size=2 * dilate + 1).astype(bool)
        tall = grid.ravel()

    ix_n = np.floor((x[nidx] - xmin) / cell).astype(np.int64)
    iy_n = np.floor((y[nidx] - ymin) / cell).astype(np.int64)
    valid = (ix_n >= 0) & (ix_n < nx) & (iy_n >= 0) & (iy_n < ny)
    if not valid.any():
        return 0

    lin_n = np.zeros(len(nidx), dtype=np.int64)
    lin_n[valid] = ix_n[valid] * ny + iy_n[valid]
    hn = h[nidx]
    ok = np.zeros(len(nidx), dtype=bool)
    # core tall cells: restore full height band (facade + windows + roof)
    in_core = valid & tall_core[lin_n]
    ok[in_core] = True
    # dilated ring only: above car roofs
    in_ring = valid & tall[lin_n] & (~tall_core[lin_n])
    ok[in_ring] = hn[in_ring] >= car_band_max

    take = nidx[ok]
    if len(take):
        cls[take] = building_class
    return int(len(take))


def demote_orphan_building(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    target_class: int = CLS_LOW_NOISE,
    ground_class: int = CLS_GROUND,
    building_class: int = CLS_BUILDING,
    cell: float = 1.25,
    min_top_h: float = 3.20,
    min_span: float = 2.20,
    roof_top_h: float = 5.0,
    min_count: int = 25,
    keep_dilate: int = 2,
    demote_elevated_to_noise: bool = True,
) -> tuple[int, int]:
    """Move isolated short building blobs (cars, street noise) out of class 6.

    Keeps:
    - tall facade columns (top_h / span),
    - high roofs even if locally flat (roof_top_h),
    - any building cell adjacent to a kept core (keep_dilate).

    Only isolated short blobs far from real buildings are demoted.
    """
    bmask = cls == building_class
    if not bmask.any():
        return 0, 0

    h = z - ground_z
    bidx = np.where(bmask)[0]
    xmin, ymin = float(x[bidx].min()), float(y[bidx].min())
    ix = np.floor((x[bidx] - xmin) / cell).astype(np.int64)
    iy = np.floor((y[bidx] - ymin) / cell).astype(np.int64)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    lin = ix * ny + iy

    order = np.argsort(lin, kind="mergesort")
    lin_s = lin[order]
    h_s = h[bidx[order]]
    z_s = z[bidx[order]]
    uniq, starts, counts = np.unique(lin_s, return_index=True, return_counts=True)

    keep = np.zeros(nx * ny, dtype=bool)
    for u, s, c in zip(uniq, starts, counts):
        chunk_h = h_s[s : s + c]
        chunk_z = z_s[s : s + c]
        top_h = float(np.percentile(chunk_h, 90))
        span = float(np.percentile(chunk_z, 90) - np.percentile(chunk_z, 10))
        # facade column OR elevated roof slab
        if (c >= min_count and top_h >= min_top_h and span >= min_span) or top_h >= roof_top_h:
            keep[int(u)] = True

    if not keep.any():
        # nothing looks like a building — demote all building (rare)
        flat = (h[bidx] <= 0.28) & (h[bidx] >= -0.08)
        n_ground = 0
        n_noise = 0
        if flat.any():
            cls[bidx[flat]] = ground_class
            n_ground = int(flat.sum())
        elev = ~flat
        if elev.any() and demote_elevated_to_noise:
            cls[bidx[elev]] = target_class
            n_noise = int(elev.sum())
        return n_ground, n_noise

    if keep_dilate > 0:
        grid = keep.reshape(nx, ny)
        grid = grey_dilation(grid.astype(np.uint8), size=2 * keep_dilate + 1).astype(bool)
        keep = grid.ravel()

    # demote building points whose cell is NOT kept
    in_keep = keep[lin]
    take = bidx[~in_keep]
    if len(take) == 0:
        return 0, 0

    flat = (h[take] <= 0.28) & (h[take] >= -0.08)
    n_ground = 0
    n_noise = 0
    if flat.any():
        cls[take[flat]] = ground_class
        n_ground = int(flat.sum())
    elev = ~flat
    if elev.any() and demote_elevated_to_noise:
        cls[take[elev]] = target_class
        n_noise = int(elev.sum())
    return n_ground, n_noise


def restore_building_from_noise(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    noise_class: int = CLS_LOW_NOISE,
    building_class: int = CLS_BUILDING,
    xy_radius: float = 0.35,
    min_h: float = 0.20,
    max_h: float = 8.0,
) -> int:
    """Recover facade-base points that were pushed into noise near buildings.

    Keeps near-ground paving in class 2 while restoring elevated points hugging
    existing building facades back into class 6.
    """
    bidx = np.where(cls == building_class)[0]
    nidx = np.where(cls == noise_class)[0]
    if len(bidx) == 0 or len(nidx) == 0:
        return 0

    h = z[nidx] - ground_z[nidx]
    take = (h >= min_h) & (h <= max_h)
    nidx = nidx[take]
    if len(nidx) == 0:
        return 0

    if len(bidx) > 3_000_000:
        rng = np.random.default_rng(42)
        bidx = rng.choice(bidx, 3_000_000, replace=False)

    tree = cKDTree(np.column_stack([x[bidx], y[bidx]]))
    batch = 400_000
    moved = 0
    for start in range(0, len(nidx), batch):
        chunk = nidx[start : start + batch]
        dists, _ = tree.query(
            np.column_stack([x[chunk], y[chunk]]),
            k=1,
            distance_upper_bound=xy_radius,
        )
        valid = np.isfinite(dists)
        if valid.any():
            cls[chunk[valid]] = building_class
            moved += int(valid.sum())
    return moved


def restore_building_skin(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    source_classes: tuple[int, ...] = (CLS_LOW_VEG, CLS_MED_VEG, CLS_LOW_NOISE),
    building_class: int = CLS_BUILDING,
    xy_radius: float = 0.14,
    min_h: float = 0.18,
    max_h: float = 8.0,
) -> int:
    """Promote points hugging facade skin back into building.

    Uses a very small XY radius so we recover facade bases/wall ribbons without
    swallowing parked cars or larger sidewalk patches.
    """
    bidx = np.where(cls == building_class)[0]
    if len(bidx) == 0:
        return 0

    src = np.zeros(len(cls), dtype=bool)
    for c in source_classes:
        src |= cls == c
    h = z - ground_z
    src &= (h >= min_h) & (h <= max_h)
    sidx = np.where(src)[0]
    if len(sidx) == 0:
        return 0

    if len(bidx) > 3_000_000:
        rng = np.random.default_rng(42)
        bidx = rng.choice(bidx, 3_000_000, replace=False)

    tree = cKDTree(np.column_stack([x[bidx], y[bidx]]))
    moved = 0
    batch = 400_000
    for start in range(0, len(sidx), batch):
        chunk = sidx[start : start + batch]
        dists, _ = tree.query(
            np.column_stack([x[chunk], y[chunk]]),
            k=1,
            distance_upper_bound=xy_radius,
        )
        valid = np.isfinite(dists)
        if valid.any():
            cls[chunk[valid]] = building_class
            moved += int(valid.sum())
    return moved


def restore_near_dtm(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    source_classes: tuple[int, ...] = (CLS_LOW_VEG, CLS_MED_VEG),
    ground_class: int = CLS_GROUND,
    max_h: float = 0.14,
    flat_span: float = 0.08,
    cell: float = 0.50,
) -> int:
    """Restore flat sidewalk/road patches sitting slightly above DTM back to ground.

    Targets square/rectangular paving wrongly put into low/med veg: near DTM
    and locally flat (small Z span in XY cell).
    """
    src = np.zeros(len(cls), dtype=bool)
    for c in source_classes:
        src |= cls == c
    h = z - ground_z
    cand = src & (h >= -0.04) & (h <= max_h)
    if not cand.any():
        return 0

    idx = np.where(cand)[0]
    xmin, ymin = float(x[idx].min()), float(y[idx].min())
    ix = np.floor((x[idx] - xmin) / cell).astype(np.int64)
    iy = np.floor((y[idx] - ymin) / cell).astype(np.int64)
    ny = int(iy.max()) + 1
    lin = ix * ny + iy

    order = np.argsort(lin, kind="mergesort")
    lin_s = lin[order]
    z_s = z[idx[order]]
    uniq, starts, counts = np.unique(lin_s, return_index=True, return_counts=True)

    flat_cell = np.zeros(len(uniq), dtype=bool)
    for i, (s, c) in enumerate(zip(starts, counts)):
        if c < 8:
            continue
        chunk = z_s[s : s + c]
        if float(chunk.max() - chunk.min()) <= flat_span:
            flat_cell[i] = True

    if not flat_cell.any():
        return 0

    # map each candidate point to whether its cell is flat
    cell_of = np.searchsorted(uniq, lin)
    # searchsorted assumes sorted uniq which it is; exact match
    ok = np.zeros(len(idx), dtype=bool)
    exact = uniq[np.clip(cell_of, 0, len(uniq) - 1)] == lin
    cell_of = np.where(exact, cell_of, -1)
    valid = cell_of >= 0
    ok[valid] = flat_cell[cell_of[valid]]
    take = idx[ok]
    if len(take) == 0:
        return 0
    cls[take] = ground_class
    return len(take)


def remove_sparse_airborne(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    source_classes: tuple[int, ...] = (CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG, CLS_BUILDING),
    target_class: int = CLS_LOW_NOISE,
    cell: float = 0.15,
    min_count: int = 6,
    ground_z: np.ndarray | None = None,
    min_h: float | None = None,
    max_h: float | None = None,
) -> int:
    """Move isolated / floating points in veg+building to noise (voxel density).

    Dense facades and real vegetation fill voxels; multipath ghosts do not.
    """
    mask = np.zeros(len(cls), dtype=bool)
    for c in source_classes:
        mask |= cls == c
    if ground_z is not None and (min_h is not None or max_h is not None):
        h = z - ground_z
        if min_h is not None:
            mask &= h >= min_h
        if max_h is not None:
            mask &= h <= max_h
    idx = np.where(mask)[0]
    if len(idx) < min_count:
        return 0

    log(
        f"  voxel sparse cleanup on {len(idx):,} pts in {source_classes} "
        f"(cell={cell} m, min_count={min_count})..."
    )
    px, py, pz = x[idx], y[idx], z[idx]
    ix = np.floor(px / cell).astype(np.int64)
    iy = np.floor(py / cell).astype(np.int64)
    iz = np.floor(pz / cell).astype(np.int64)
    # pack to 1D key
    key = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791

    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    uniq, starts, counts = np.unique(key_s, return_index=True, return_counts=True)
    # map each point -> its cell count
    cell_count = np.repeat(counts, counts)
    # reorder back
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    per_point = cell_count[inv]

    sparse = per_point < min_count
    n = int(sparse.sum())
    if n:
        cls[idx[sparse]] = target_class
    return n


def restore_asphalt_from_noise(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    source_classes: tuple[int, ...] = (CLS_LOW_NOISE, CLS_LOW_VEG, CLS_MED_VEG),
    ground_class: int = CLS_GROUND,
    xy_radius: float = 1.25,
    z_tol: float = 0.06,
    max_noise_points: int = 8_000_000,
    ground_z: np.ndarray | None = None,
    max_h_above_dtm: float | None = None,
) -> int:
    """Return flat road points mislabeled as noise/veg back to ground.

    A point is restored if it is coplanar with nearby ground (same Z within
    z_tol) and, when ground_z is given, within max_h_above_dtm of the DTM.
    Car bodies / true vegetation sit higher and are left alone.
    """
    ground_m = cls == ground_class
    src = np.zeros(len(cls), dtype=bool)
    for c in source_classes:
        src |= cls == c
    if not ground_m.any() or not src.any():
        return 0

    gidx = np.where(ground_m)[0]
    nidx = np.where(src)[0]

    if len(gidx) > 3_000_000:
        rng = np.random.default_rng(42)
        gidx = rng.choice(gidx, 3_000_000, replace=False)

    tree = cKDTree(np.column_stack([x[gidx], y[gidx]]))

    batch = 500_000
    restored = 0
    for start in range(0, len(nidx), batch):
        chunk = nidx[start : start + batch]
        dists, nn = tree.query(
            np.column_stack([x[chunk], y[chunk]]),
            k=1,
            distance_upper_bound=xy_radius,
        )
        valid = np.isfinite(dists)
        if not valid.any():
            continue
        hits = chunk[valid]
        gz = z[gidx[nn[valid]]]
        coplanar = np.abs(z[hits] - gz) <= z_tol
        if ground_z is not None and max_h_above_dtm is not None:
            coplanar &= (z[hits] - ground_z[hits]) <= max_h_above_dtm
        take = hits[coplanar]
        if len(take):
            cls[take] = ground_class
            restored += len(take)
    return restored


def peel_car_footprint(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmax: float,
    target_class: int,
    peel_classes: tuple[int, ...] = (CLS_LOW_VEG, CLS_MED_VEG),
    xy_pad: float = 0.40,
    z_top_pad: float = 0.25,
    min_surface_h: float = 0.05,
    max_surface_h: float = 0.55,
) -> int:
    """Peel car remnant points inside XY footprint — never flat asphalt.

    Only points clearly above the road DTM (wheels / lower body) are moved.
    Road surface (h ≈ 0) stays in ground.
    """
    in_box = (
        (x >= xmin - xy_pad)
        & (x <= xmax + xy_pad)
        & (y >= ymin - xy_pad)
        & (y <= ymax + xy_pad)
        & (z <= zmax + z_top_pad)
    )
    peel_m = _peel_classes_mask(cls, peel_classes)
    h = z - ground_z
    remnant = (h >= min_surface_h) & (h <= max_surface_h)
    m = in_box & peel_m & remnant
    n = int(m.sum())
    if n:
        cls[m] = target_class
    return n


def peel_noise_from_ground(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    target_class: int = CLS_LOW_NOISE,
    max_above: float = 0.08,
    max_below: float = 0.06,
    sparse_cell: float = 0.20,
    sparse_min_count: int = 4,
    sparse_enabled: bool | None = None,
    protect_near_building_m: float = 0.0,
    protect_near_building_h: float = 2.80,
) -> tuple[int, int]:
    """Remove off-surface and (optionally) sparse elevated points from ground class."""
    gmask = cls == CLS_GROUND
    if not gmask.any():
        return 0, 0

    h = z - ground_z
    curb_on_ground = gmask & (h >= 0.08) & (h <= CFG.curb_max + 0.06)
    if curb_on_ground.any():
        cls[curb_on_ground] = CLS_LOW_VEG
    off = gmask & ((h > max_above) | (h < -max_below)) & ~curb_on_ground
    if protect_near_building_m > 0 and off.any():
        near = _mask_near_building_xy(x, y, cls, protect_near_building_m)
        off &= ~(near & (h < protect_near_building_h))
    n_off = int(off.sum())
    if n_off:
        cls[off] = target_class

    if sparse_enabled is None:
        sparse_enabled = CFG.ground_sparse_cleanup
    n_sparse = 0
    if sparse_enabled and sparse_min_count > 0:
        # RS10-thinned TLS: only sparse-clean slightly elevated ground, not flat asphalt.
        n_sparse = remove_sparse_airborne(
            x, y, z, cls,
            source_classes=(CLS_GROUND,),
            target_class=target_class,
            cell=sparse_cell,
            min_count=sparse_min_count,
            ground_z=ground_z,
            min_h=0.05,
        )
        if protect_near_building_m > 0 and n_sparse:
            near = _mask_near_building_xy(x, y, cls, protect_near_building_m)
            undo = (cls == target_class) & near & (h < protect_near_building_h)
            if undo.any():
                cls[undo] = CLS_GROUND
                n_sparse = max(0, n_sparse - int(undo.sum()))
    return n_off, n_sparse


def peel_elevated_junk_from_ground_class(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    target_class: int = CLS_LOW_NOISE,
    min_h: float | None = None,
    max_h: float | None = None,
    protect_near_building_m: float = 0.0,
    protect_near_building_h: float = 0.45,
) -> int:
    """Move tall leftovers still labeled ground (car chunks, people, floaters) to noise."""
    min_h = CFG.ground_elevated_junk_min_h if min_h is None else min_h
    max_h = CFG.ground_elevated_junk_max_h if max_h is None else max_h
    h = z - ground_z
    m = (cls == CLS_GROUND) & (h > min_h) & (h < max_h)
    if protect_near_building_m > 0 and m.any():
        near = _mask_near_building_xy(x, y, cls, protect_near_building_m)
        m &= ~(near & (h < protect_near_building_h))
    n = int(m.sum())
    if n:
        cls[m] = target_class
    return n


def peel_elevated_islands_from_ground(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    target_class: int = CLS_LOW_NOISE,
    cell: float = 0.18,
    min_h: float = 0.025,
    max_h: float = 0.85,
    min_count: int = 12,
    max_length: float = 3.50,
    max_width: float = 1.40,
    dtm_cell: float = 0.30,
    protect_near_building_m: float = 0.0,
    protect_near_building_h: float = 2.80,
    ground_z: np.ndarray | None = None,
) -> int:
    """Remove compact elevated islands still labeled ground (wheels, litter, bumper tips).

    Builds a fine local min-Z from ground, clusters slightly elevated points, and
    peels blobs that are small in XY (not long curbs / continuous sidewalk lips).
    """
    gmask = cls == CLS_GROUND
    if int(gmask.sum()) < min_count:
        return 0

    local_z = ground_dtm(x, y, z, gmask, cell=dtm_cell)
    h = z - local_z
    cand = gmask & (h >= min_h) & (h <= max_h)
    n_cand = int(cand.sum())
    if n_cand < min_count:
        return 0

    px, py, pz = x[cand], y[cand], z[cand]
    ph = h[cand]
    xmin, ymin = float(px.min()), float(py.min())
    ix = np.floor((px - xmin) / cell).astype(np.int64)
    iy = np.floor((py - ymin) / cell).astype(np.int64)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    if nx * ny > 120_000_000:
        cell = max(cell, 0.25)
        ix = np.floor((px - xmin) / cell).astype(np.int64)
        iy = np.floor((py - ymin) / cell).astype(np.int64)
        nx, ny = int(ix.max()) + 1, int(iy.max()) + 1

    occ = np.zeros((nx, ny), dtype=bool)
    occ[ix, iy] = True
    # break thin bridges to asphalt undulation so wheel prints become separate blobs
    occ = binary_opening(occ, structure=np.ones((3, 3), dtype=bool))
    labeled, nlab = nd_label(occ.astype(np.uint8))
    if nlab == 0:
        return 0

    # map points to opened labels (points in eroded-away cells keep 0 → skipped)
    lab = labeled[ix, iy]
    cidx = np.where(cand)[0]
    move = np.zeros(len(cls), dtype=bool)
    peeled = 0
    for lid in range(1, nlab + 1):
        sel = lab == lid
        cnt = int(sel.sum())
        if cnt < min_count:
            continue
        xs, ys = px[sel], py[sel]
        dx = float(xs.max() - xs.min())
        dy = float(ys.max() - ys.min())
        length, width = max(dx, dy), min(dx, dy)
        # long curb / continuous lip — keep
        if length > max_length or width > max_width:
            continue
        mean_h = float(ph[sel].mean())
        max_hh = float(ph[sel].max())
        # nearly flat sheet — keep
        if mean_h < min_h + 0.005 and max_hh < min_h + 0.025:
            continue
        # reject huge dense patches (plaza / ramp chunk)
        if cnt > 30_000 and length > 2.5:
            continue
        move[cidx[sel]] = True
        peeled += cnt

    if peeled and protect_near_building_m > 0:
        ref_z = ground_z if ground_z is not None else local_z
        hh = z - ref_z
        near = _mask_near_building_xy(x, y, cls, protect_near_building_m)
        move &= ~(near & (hh < protect_near_building_h))

    if move.any():
        cls[move] = target_class
        peeled = int(move.sum())
    else:
        peeled = 0
    return peeled


def demote_false_vehicles(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    vehicle_class: int = CLS_VEHICLE,
    max_h_to_ground: float = 0.18,
) -> int:
    """Etalon cleanup: near-DTM points wrongly labeled vehicle → ground."""
    h = z - ground_z
    to_g = (cls == vehicle_class) & (h < max_h_to_ground)
    n_g = int(to_g.sum())
    if n_g:
        cls[to_g] = CLS_GROUND
    return n_g


def restore_low_object_band_to_ground(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    source_classes: tuple[int, ...] = (CLS_LOW_VEG, CLS_MED_VEG, CLS_VEHICLE),
    max_h: float = 0.20,
    flat_span: float = 0.08,
    cell: float = 0.50,
) -> int:
    """Pull near-DTM flat leftovers (often asphalt in class 3/4/91) back to ground.

    Etalon ground p95 ~0.20–0.28; geometric layers often park that band in med_veg.
    """
    h = z - ground_z
    cand = np.zeros(len(cls), dtype=bool)
    for c in source_classes:
        cand |= cls == c
    cand &= (h >= -0.05) & (h <= max_h)
    if not cand.any():
        return 0
    idx = np.where(cand)[0]
    xmin, ymin = float(x[idx].min()), float(y[idx].min())
    ix = np.floor((x[idx] - xmin) / cell).astype(np.int64)
    iy = np.floor((y[idx] - ymin) / cell).astype(np.int64)
    ny = int(iy.max()) + 1
    lin = ix * ny + iy
    order = np.argsort(lin, kind="mergesort")
    lin_s = lin[order]
    z_s = z[idx[order]]
    uniq, starts, counts = np.unique(lin_s, return_index=True, return_counts=True)
    z_min = np.minimum.reduceat(z_s, starts)
    z_max = np.maximum.reduceat(z_s, starts)
    good = (counts >= 6) & ((z_max - z_min) <= flat_span)
    if not good.any():
        return 0
    good_keys = uniq[good]
    move = np.isin(lin, good_keys)
    n = int(move.sum())
    if n:
        cls[idx[move]] = CLS_GROUND
    return n


def extract_cars_from_classes(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    source_classes: tuple[int, ...],
    car_target_class: int,
    min_height: float = 0.10,
    max_height: float | None = None,
    max_points: int | None = 2_000_000,
    fast: bool = False,
    voxel: float = 0.08,
    backend: str = HDBSCAN_BACKEND_CPU,
    stage: str = "E",
    strict: bool = False,
) -> list[CarBox]:
    """Pull car-shaped clusters out of ground/veg/building classes."""
    cand = np.zeros(len(cls), dtype=bool)
    for c in source_classes:
        cand |= cls == c
    h = z - ground_z
    elevated = cand & (h > min_height)
    if max_height is not None:
        elevated &= h <= max_height
    idx = np.where(elevated)[0]
    if len(idx) < 30:
        log(f"  no elevated candidates in classes {source_classes}")
        return []

    if max_points is not None and len(idx) > max_points:
        log(f"  subsampling elevated {len(idx):,} -> {max_points:,} for car scan")
        rng = np.random.default_rng(42)
        idx = rng.choice(idx, max_points, replace=False)
    elif max_points is None:
        log(f"  no-subsample: using all {len(idx):,} elevated pts for car scan")

    pts = np.column_stack([x[idx], y[idx], z[idx]])
    log(f"  scanning {len(idx):,} elevated pts in classes {source_classes} for cars...")
    labels = hdbscan_labels_fast(
        pts,
        min_cluster_size=CFG.car_min_cluster,
        min_samples=CFG.car_min_samples,
        epsilon=CFG.car_epsilon,
        fast=fast,
        voxel=voxel,
        backend=backend,
        stage=stage,
    )

    removed = 0
    car_boxes: list[CarBox] = []
    for lab in np.unique(labels[labels >= 0]):
        cidx = idx[labels == lab]
        p = np.column_stack([x[cidx], y[cidx], z[cidx]])
        hc = h[cidx]
        if is_car_cluster(p, hc, len(cidx), strict=strict):
            xmin, ymin, _ = p.min(axis=0)
            xmax, ymax, zmax = p.max(axis=0)
            removed += len(cidx)
            cls[cidx] = car_target_class
            car_boxes.append((float(xmin), float(xmax), float(ymin), float(ymax), float(zmax)))

    log(
        f"  cars removed -> class {car_target_class}: {removed:,} core; "
        f"boxes={len(car_boxes)}"
        + (" [strict]" if strict else "")
    )
    return car_boxes


def cpu_finalize_cars_in_veg(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    car_target_class: int,
    fast: bool = False,
    voxel: float = 0.08,
) -> int:
    """Extra CPU-only car sweep on vegetation after GPU clustering (safety net)."""
    boxes = extract_cars_from_classes(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
        car_target_class=car_target_class,
        min_height=CFG.car_min_height,
        max_points=CFG.car_max_points,
        fast=fast,
        voxel=voxel,
        backend=HDBSCAN_BACKEND_CPU,
        stage="I2",
    )
    return peel_car_volume(
        x, y, z, cls, ground_z, boxes, car_target_class,
        peel_classes=(CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
        xy_pad=CFG.car_xy_pad,
        min_h=CFG.car_min_h,
        z_top_pad=CFG.car_z_top_pad,
    )


def statistical_outlier(
    xyz: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    batch_size: int = 200_000,
) -> np.ndarray:
    """Return boolean mask of outlier points."""
    n = len(xyz)
    if n < nb_neighbors + 1:
        return np.zeros(n, dtype=bool)

    sample_n = min(n, max(500_000, n // 20))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n, sample_n, replace=False)
    sample = xyz[sample_idx]

    nn = NearestNeighbors(n_neighbors=nb_neighbors, algorithm="kd_tree", n_jobs=-1)
    nn.fit(sample)

    mean_dists = np.zeros(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        dists, _ = nn.kneighbors(xyz[start:end], return_distance=True)
        mean_dists[start:end] = dists[:, 1:].mean(axis=1)

    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    thresh = global_mean + std_ratio * global_std
    return mean_dists > thresh


def csf_ground(xyz: np.ndarray, cloth_resolution: float = 0.8, class_threshold: float = 0.35) -> np.ndarray:
    """CSF ground mask."""
    csf = CSF.CSF()
    csf.setPointCloud(xyz.astype(np.float64))
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = 3
    csf.params.time_step = 0.65
    csf.params.class_threshold = class_threshold
    csf.params.interations = 500
    ground_idx = CSF.VecInt()
    off_idx = CSF.VecInt()
    csf.do_filtering(ground_idx, off_idx, False)
    ground = np.zeros(len(xyz), dtype=bool)
    ground[np.array(ground_idx, dtype=np.int64)] = True
    return ground


def height_above_ground(z: np.ndarray, ground_z: np.ndarray) -> np.ndarray:
    return z - ground_z


def assign_height_layers(
    cls: np.ndarray,
    idx: np.ndarray,
    h: np.ndarray,
    curb_max: float = 0.35,
    obj_max: float = 2.20,
) -> None:
    m = (h >= 0.0) & (h < curb_max)
    cls[idx[m]] = CLS_LOW_VEG

    m = (h >= curb_max) & (h < obj_max)
    cls[idx[m]] = CLS_MED_VEG

    m = h >= obj_max
    cls[idx[m]] = CLS_HIGH_VEG


def cluster_objects(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    min_cluster_size: int = CFG.obj_min_cluster,
    min_samples: int = CFG.obj_min_samples,
    cluster_selection_epsilon: float = CFG.obj_epsilon,
    max_points: int | None = CFG.obj_max_points,
    car_target_class: int = CLS_CAR,
    ped_target_class: int = CLS_PED,
    ghost_target_class: int = CLS_GHOST,
    fast: bool = False,
    voxel: float = CFG.fast_voxel,
    backend: str = HDBSCAN_BACKEND_CPU,
) -> list[CarBox]:
    """Classify cars / pedestrians / ghosts on medium-vegetation layer."""
    med = cls == CLS_MED_VEG
    idx = np.where(med)[0]
    if len(idx) < min_cluster_size:
        log("  Not enough points in class 4 for clustering.")
        return []

    if max_points is not None and len(idx) > max_points:
        log(f"  Class 4 has {len(idx):,} points — subsampling to {max_points:,} for HDBSCAN.")
        rng = np.random.default_rng(42)
        idx = rng.choice(idx, max_points, replace=False)
    elif max_points is None:
        log(f"  no-subsample: HDBSCAN on all {len(idx):,} class-4 points")

    pts = np.column_stack([x[idx], y[idx], z[idx]])
    log(f"  HDBSCAN on {len(idx):,} points (class 4)...")
    labels = hdbscan_labels_fast(
        pts,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        epsilon=cluster_selection_epsilon,
        fast=fast,
        voxel=voxel,
        backend=backend,
        stage="D",
    )

    cars_points = 0
    footprint_points = 0
    ped_points = 0
    ghost_points = 0
    car_boxes: list[CarBox] = []

    for lab in tqdm(np.unique(labels[labels >= 0]), desc="  clusters"):
        cidx = idx[labels == lab]
        p = pts[labels == lab]
        xmin, ymin, zmin = p.min(axis=0)
        xmax, ymax, zmax = p.max(axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        length = max(dx, dy)
        width = min(dx, dy)
        height_range = zmax - zmin
        count = len(cidx)
        h_cluster = p[:, 2] - ground_z[cidx]
        h_med = float(np.median(h_cluster))
        density = count / max(length * width, 0.01)

        target = None
        if is_car_cluster(p, h_cluster, count):
            target = car_target_class
        elif is_ghost_multipath_cluster(length, width, height_range, density, count):
            target = ghost_target_class
        elif (
            0.25 <= width <= 1.0
            and 0.20 <= length <= 1.40
            and height_range <= 2.2
            and 1.0 <= h_med <= 2.1
            and count >= CFG.ped_cluster_min_count
        ):
            target = ped_target_class

        if target is not None:
            cls[cidx] = target
            if target == car_target_class:
                cars_points += count
                car_boxes.append((float(xmin), float(xmax), float(ymin), float(ymax), float(zmax)))
            elif target == ped_target_class:
                ped_points += count
            elif target == ghost_target_class:
                ghost_points += count

    # one vectorized peel for all cars (elevated veg remnants only)
    if car_boxes:
        footprint_points = peel_car_volume(
            x, y, z, cls, ground_z, car_boxes, car_target_class,
            peel_classes=(CLS_MED_VEG,),  # not low_veg — curb/asphalt bleed
            xy_pad=min(CFG.car_xy_pad, 0.40),
            min_h=max(CFG.car_min_h, 0.18),
            z_top_pad=0.30,
            dilate_cells=1,
            exclude_near_building_m=CFG.peel_exclude_near_building_m,
        )

    log(
        f"  cluster assign: cars-> {car_target_class} = {cars_points:,} core + "
        f"{footprint_points:,} volume; "
        f"peds-> {ped_target_class} = {ped_points:,}; "
        f"ghosts-> {ghost_target_class} = {ghost_points:,}"
    )
    return car_boxes


def run_pipeline(
    input_path: Path,
    output_path: Path,
    source_class: int = CLS_DIRTY,
    use_csf: bool = True,
    skip_clustering: bool = False,
    cars_target_class: int = CLS_LOW_NOISE,
    fast: bool = False,
    hdbscan_backend: str = HDBSCAN_BACKEND_CUML,
    skip_low_noise: bool = False,
) -> None:
    t0 = time.time()
    timings: dict[str, float] = {}

    def stage_start() -> float:
        return time.perf_counter()

    def stage_end(name: str, started: float) -> None:
        timings[name] = time.perf_counter() - started

    if fast:
        log("FAST mode: voxel-HDBSCAN + vectorized car peel")
    if hdbscan_backend != HDBSCAN_BACKEND_CPU:
        log(f"HDBSCAN backend: {hdbscan_backend} (CPU baseline = {HDBSCAN_BACKEND_CPU})")
    log(f"Reading {input_path} ...")
    las = laspy.read(str(input_path))
    n = len(las.points)
    log(f"  points: {n:,}")

    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)
    cls = np.array(las.classification, dtype=np.uint8)

    class_histogram(cls, "Input")

    work = cls == source_class
    log(f"\nProcessing source class {source_class}: {work.sum():,} points")
    if work.sum() == 0:
        log(f"No points in source class {source_class}. Exit.")
        sys.exit(1)

    widx = np.where(work)[0]
    wx, wy, wz = x[widx], y[widx], z[widx]
    wxyz = np.column_stack([wx, wy, wz])

    # --- A: noise (near-ground only — full-cloud facades must not enter SOR) ---
    ts = stage_start()
    if skip_low_noise:
        log("\n[A] Low-point / statistical noise skipped (--skip-low-noise)")
    else:
        log("\n[A] Low points (grid)...")
        low = radius_low_points(wx, wy, wz)
        cls[widx[low]] = CLS_LOW_NOISE
        work[widx[low]] = False
        log(f"  low noise: {low.sum():,}")

        widx = np.where(work)[0]
        wx, wy, wz = x[widx], y[widx], z[widx]
        wxyz = np.column_stack([wx, wy, wz])

        log("[A] Statistical outliers (near-ground only)...")
        zmin = grid_min_z(wx, wy, wz, 1.0)
        near_ground = wz < (zmin + 1.2)
        out = np.zeros(len(wz), dtype=bool)
        if near_ground.sum() > 100:
            ng = np.where(near_ground)[0]
            out_ng = statistical_outlier(
                np.column_stack([wx[ng], wy[ng], wz[ng]]),
                nb_neighbors=20,
                std_ratio=2.2,
            )
            out[ng[out_ng]] = True
        cls[widx[out]] = CLS_LOW_NOISE
        work[widx[out]] = False
        log(f"  statistical noise: {out.sum():,} (near-ground candidates: {int(near_ground.sum()):,})")
    stage_end("A_noise", ts)

    widx = np.where(work)[0]
    wx, wy, wz = x[widx], y[widx], z[widx]
    wxyz = np.column_stack([wx, wy, wz])

    # --- B: ground ---
    ts = stage_start()
    log("\n[B] Ground classification...")
    if use_csf and len(wxyz) > 0:
        log(f"  CSF on {len(wxyz):,} points (may take several minutes)...")
        gmask_local = csf_ground(wxyz, cloth_resolution=0.8, class_threshold=0.35)
        gidx = widx[gmask_local]
        cls[gidx] = CLS_GROUND
        work[gidx] = False
        log(f"  CSF ground: {gmask_local.sum():,}")
    else:
        # Robust road DTM: large cells + morphological opening so car roofs
        # (occluding the road in TLS) do not become the local ground surface.
        ground_local, _ = classify_ground_robust(
            wx, wy, wz, cell=1.0, opening_size=7, tol_above=0.06, tol_below=0.04
        )
        gidx = widx[ground_local]
        cls[gidx] = CLS_GROUND
        work[gidx] = False
        log(f"  robust ground: {ground_local.sum():,}")

    ground_mask = cls == CLS_GROUND
    log(f"  total ground points (before peel): {ground_mask.sum():,}")

    # CSF over-classifies TLS — peel only newly assigned ground points
    if use_csf and len(gidx) > 0:
        log("  Refining CSF ground (peel objects >0.40 m above local surface)...")
        surf_g = grid_surface_z(x[gidx], y[gidx], z[gidx], 0.5)
        peel = z[gidx] > (surf_g + 0.40)
        peeled_idx = gidx[peel]
        cls[peeled_idx] = CLS_DIRTY
        work[peeled_idx] = True
        log(f"  peeled from ground: {len(peeled_idx):,}")

    # Peel car roofs / elevated plateaus that snuck into class 2
    # (includes preexisting class-2 from the input LAS).
    log("  Peeling elevated plateaus from class 2 (cars on occluded road)...")
    n_peel = peel_elevated_from_ground(
        x, y, z, cls, cell=1.0, opening_size=7, max_above=0.12
    )
    log(f"  elevated peel from ground: {n_peel:,}")
    # Peeled points are class 11 (DIRTY) — put them back into work for height layers
    if n_peel:
        work[cls == CLS_DIRTY] = True
        work[cls == CLS_BUILDING] = False
        work[cls == CLS_GROUND] = False
        work[cls == CLS_LOW_NOISE] = False

    ground_mask = cls == CLS_GROUND
    log(f"  total ground points: {ground_mask.sum():,}")
    # DTM from robust model — not from polluted class-2 mins
    ground_z = robust_road_dtm(x, y, z, cell=1.0, opening_size=7)

    # below surface cleanup on remaining work
    rem = np.where(work)[0]
    if len(rem):
        below = z[rem] < (ground_z[rem] - 0.12)
        cls[rem[below]] = CLS_LOW_NOISE
        work[rem[below]] = False
        log(f"  below ground noise: {below.sum():,}")
    stage_end("B_ground", ts)

    # --- B2: buildings from vertical columns (needed when class 6 not prelabeled) ---
    ts = stage_start()
    log("\n[B2] Detect buildings (dense vertical XY columns)...")
    n_build = detect_buildings(x, y, z, cls, ground_z)
    if n_build:
        work[cls == CLS_BUILDING] = False
    log(f"  building points: {n_build:,} (total class 6: {(cls == CLS_BUILDING).sum():,})")
    stage_end("B2_buildings", ts)

    # --- C: height layers ---
    ts = stage_start()
    log("\n[C] Height layers...")
    rem = np.where(work)[0]
    h = z[rem] - ground_z[rem]
    assign_height_layers(cls, rem, h, curb_max=0.35, obj_max=2.20)
    work[rem] = False
    log(f"  assigned layers on {len(rem):,} points")
    stage_end("C_layers", ts)

    # --- D: clustering on object layer ---
    car_boxes: list[CarBox] = []
    if not skip_clustering:
        ts = stage_start()
        log("\n[D] Object clustering (class 4)...")
        car_boxes = cluster_objects(
            x,
            y,
            z,
            cls,
            ground_z,
            car_target_class=cars_target_class,
            ped_target_class=CLS_GHOST,
            fast=fast,
            voxel=CFG.fast_voxel,
            backend=hdbscan_backend,
        )
        stage_end("D_cluster4", ts)
    else:
        log("\n[D] Clustering skipped.")

    # --- E: cars from ground + all vegetation layers ---
    if not skip_clustering:
        ts = stage_start()
        log("\n[E] Extract cars from ground/veg (classes 2-5)...")
        e_boxes = extract_cars_from_classes(
            x, y, z, cls, ground_z,
            source_classes=(CLS_GROUND, CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
            car_target_class=cars_target_class,
            min_height=CFG.car_min_height,
            max_points=CFG.car_max_points,
            fast=fast,
            voxel=CFG.fast_voxel,
            backend=hdbscan_backend,
            stage="E",
        )
        car_boxes += e_boxes
        n_vol = peel_car_volume(
            x, y, z, cls, ground_z, car_boxes, cars_target_class,
            peel_classes=(CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
            xy_pad=CFG.car_xy_pad,
            min_h=CFG.car_min_h,
            z_top_pad=CFG.car_z_top_pad,
        )
        log(f"  car volume peel (veg only): {n_vol:,}")
        stage_end("E_cars", ts)

    # --- F: restore flat asphalt / sidewalk ---
    ts = stage_start()
    log("\n[F] Restore flat road/sidewalk from noise / low veg...")
    n_rest = restore_asphalt_from_noise(
        x, y, z, cls,
        source_classes=(cars_target_class, CLS_LOW_VEG),
        ground_class=CLS_GROUND,
        xy_radius=1.50,
        z_tol=0.08,
    )
    n_dtm = restore_near_dtm(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG),
        max_h=0.16,
        flat_span=0.08,
        cell=0.50,
    )
    n_pave = restore_flat_paving_xy(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG),
        cell=1.0,
        max_h=0.38,
        max_z_span=0.14,
        min_count=20,
    )
    log(f"  restored asphalt -> class 2: {n_rest:,}; flat-near-DTM: {n_dtm:,}; flat tiles: {n_pave:,}")
    stage_end("F_restore", ts)

    # --- G: re-layer veg by robust height (paving out of high veg, etc.) ---
    ts = stage_start()
    log("\n[G] Re-layer vegetation by height above robust DTM...")
    layer_counts = relayer_by_height(x, y, z, cls, ground_z)
    log(f"  relayer: {layer_counts}")
    # catch facade remnants still in high veg after full-cloud height layering
    n_build2 = detect_buildings(x, y, z, cls, ground_z)
    log(f"  building re-detect: {n_build2:,} (total class 6: {(cls == CLS_BUILDING).sum():,})")
    n_dtm2 = restore_near_dtm(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG),
        max_h=0.16,
        flat_span=0.08,
        cell=0.50,
    )
    n_pave2 = restore_flat_paving_xy(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG),
        cell=1.0,
        max_h=0.38,
    )
    log(f"  flat sidewalk restore after relayer: {n_dtm2:,}; flat tiles: {n_pave2:,}")
    stage_end("G_relayer", ts)

    # --- H: facade skirts in veg -> building (skip car-height band) ---
    ts = stage_start()
    log("\n[H] Attach near-building veg to building class...")
    n_facade = attach_near_buildings_xy(
        x, y, z, cls,
        ground_z=ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
        building_class=CLS_BUILDING,
        xy_radius=0.50,
        curb_max=0.40,
        obj_max=2.20,
    )
    log(f"  attached to building: {n_facade:,}")
    stage_end("H_facades", ts)

    # --- H2: cars again after facade attach (catch cars left in veg) ---
    if not skip_clustering:
        ts = stage_start()
        log("\n[H2] Re-extract cars from veg (post-facade)...")
        boxes_h2 = extract_cars_from_classes(
            x, y, z, cls, ground_z,
            source_classes=(CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
            car_target_class=cars_target_class,
            min_height=CFG.car_min_height,
            max_points=CFG.car_max_points,
            fast=fast,
            voxel=CFG.fast_voxel,
            backend=hdbscan_backend,
            stage="E",
        )
        n_vol2 = peel_car_volume(
            x, y, z, cls, ground_z, boxes_h2, cars_target_class,
            peel_classes=(CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG, CLS_BUILDING),
            xy_pad=CFG.car_xy_pad,
            min_h=CFG.car_min_h,
            z_top_pad=CFG.car_z_top_pad,
        )
        log(f"  H2 car volume peel: {n_vol2:,}")
        stage_end("H2_cars", ts)

    # --- I: floating trash in vegetation + building ---
    ts = stage_start()
    log("\n[I] Remove sparse airborne points from veg/building...")
    n_veg = remove_sparse_airborne(
        x, y, z, cls,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
        target_class=cars_target_class,
        cell=0.12,
        min_count=6,
    )
    n_bld = remove_sparse_airborne(
        x, y, z, cls,
        source_classes=(CLS_BUILDING,),
        target_class=cars_target_class,
        cell=0.25,
        min_count=16,
    )
    log(f"  sparse veg: {n_veg:,}; sparse building: {n_bld:,}")
    stage_end("I_sparse", ts)

    # --- I2: flowerbeds wrongly marked as low-points ---
    ts = stage_start()
    log("\n[I2] Recover dense elevated landscape from noise...")
    n_land = recover_landscape_from_noise(x, y, z, cls, ground_z, noise_class=cars_target_class)
    log(f"  recovered landscape -> veg: {n_land:,}")
    if n_land and not skip_clustering:
        boxes_l = extract_cars_from_classes(
            x, y, z, cls, ground_z,
            source_classes=(CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
            car_target_class=cars_target_class,
            min_height=CFG.car_min_height,
            max_points=CFG.car_max_points,
            fast=fast,
            voxel=CFG.fast_voxel,
            backend=hdbscan_backend,
            stage="I2",
        )
        peel_car_volume(
            x, y, z, cls, ground_z, boxes_l, cars_target_class,
            peel_classes=(CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
            xy_pad=CFG.car_xy_pad,
            min_h=CFG.car_min_h,
            z_top_pad=CFG.car_z_top_pad,
        )
    n_dtm3 = restore_near_dtm(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG),
        max_h=0.16,
        flat_span=0.08,
        cell=0.50,
    )
    n_pave3 = restore_flat_paving_xy(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG),
        cell=1.0,
        max_h=0.38,
    )
    log(f"  final flat sidewalk restore: {n_dtm3:,}; flat tiles: {n_pave3:,}")
    stage_end("I2_recover", ts)

    # --- L: final cars in veg + last paving pass ---
    if not skip_clustering:
        ts = stage_start()
        log("\n[L] Final cars from low/med veg + flat tile cleanup...")
        boxes_l = extract_cars_from_classes(
            x, y, z, cls, ground_z,
            source_classes=(CLS_LOW_VEG, CLS_MED_VEG),
            car_target_class=cars_target_class,
            min_height=0.08,
            max_points=CFG.car_max_points,
            fast=fast,
            voxel=CFG.fast_voxel,
            backend=hdbscan_backend,
            stage="E",
        )
        n_lvol = peel_car_volume(
            x, y, z, cls, ground_z, boxes_l, cars_target_class,
            peel_classes=(CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
            xy_pad=CFG.car_xy_pad,
            min_h=CFG.car_min_h,
            z_top_pad=CFG.car_z_top_pad,
        )
        n_pave4 = restore_flat_paving_xy(
            x, y, z, cls, ground_z,
            source_classes=(CLS_LOW_VEG, CLS_MED_VEG),
            cell=1.0,
            max_h=0.38,
        )
        log(f"  L car peel: {n_lvol:,}; flat tiles: {n_pave4:,}")
        stage_end("L_final", ts)

    # --- J: noise that leaked into ground ---
    ts = stage_start()
    log("\n[J] Peel noise from ground class...")
    n_off, n_gs = peel_noise_from_ground(
        x, y, z, cls, ground_z,
        target_class=cars_target_class,
        max_above=0.15,
        max_below=0.08,
        sparse_cell=0.20,
        sparse_min_count=4,
    )
    log(f"  off-DTM: {n_off:,}; sparse ground: {n_gs:,}")
    stage_end("J_groundfix", ts)

    # --- J2: recover sidewalk/facade base from class 7 after ground peel ---
    ts = stage_start()
    log("\n[J2] Recover paving/facade base from low-noise...")
    n_pave5 = restore_flat_paving_xy(
        x, y, z, cls, ground_z,
        source_classes=(cars_target_class,),
        cell=1.0,
        max_h=0.22,
        max_z_span=0.10,
        min_count=20,
        min_near_frac=0.90,
    )
    n_dtm5 = restore_near_dtm(
        x, y, z, cls, ground_z,
        source_classes=(cars_target_class,),
        max_h=0.12,
        flat_span=0.06,
        cell=0.50,
    )
    n_bld5 = restore_building_from_noise(
        x, y, z, cls, ground_z,
        noise_class=cars_target_class,
        xy_radius=0.35,
        min_h=0.20,
        max_h=6.0,
    )
    log(f"  J2 restore: flat tiles {n_pave5:,}; near-DTM {n_dtm5:,}; facade base {n_bld5:,}")
    stage_end("J2_restore", ts)

    # --- J3: last restore from veg/noise after every car/noise pass ---
    ts = stage_start()
    log("\n[J3] Final restore from vegetation/noise...")
    n_pave6 = restore_flat_paving_xy(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, cars_target_class),
        cell=1.0,
        max_h=0.26,
        max_z_span=0.10,
        min_count=18,
        min_near_frac=0.90,
    )
    n_dtm6 = restore_near_dtm(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, cars_target_class),
        max_h=0.12,
        flat_span=0.06,
        cell=0.50,
    )
    n_skin6 = restore_building_skin(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, cars_target_class),
        xy_radius=0.14,
        min_h=0.18,
        max_h=6.0,
    )
    log(f"  J3 restore: flat tiles {n_pave6:,}; near-DTM {n_dtm6:,}; facade skin {n_skin6:,}")
    stage_end("J3_restore", ts)

    # --- K: CPU safety sweep after full-GPU clustering (cuml only) ---
    if hdbscan_backend == HDBSCAN_BACKEND_CUML and not skip_clustering:
        ts = stage_start()
        log("\n[K] CPU finalize: cars in vegetation (post-GPU safety pass)...")
        n_fin = cpu_finalize_cars_in_veg(
            x, y, z, cls, ground_z, cars_target_class,
            fast=fast, voxel=CFG.fast_voxel,
        )
        log(f"  CPU finalize peel: {n_fin:,}")
        stage_end("K_cpu_finalize", ts)

    class_histogram(cls, "Output")
    log_stage_timings(timings)

    log(f"\nWriting {output_path} ...")
    las.classification = cls
    las.write(str(output_path))
    log(f"Done in {(time.time() - t0) / 60:.1f} min.")


def main() -> None:
    p = argparse.ArgumentParser(description="TLS street point cloud classifier")
    p.add_argument("input", type=Path, help="Input LAS/LAZ")
    p.add_argument("-o", "--output", type=Path, help="Output LAS/LAZ")
    p.add_argument("--source-class", type=int, default=CLS_DIRTY, help="Class to process (default 11)")
    p.add_argument("--no-csf", action="store_true", help="Use grid ground instead of CSF")
    p.add_argument("--skip-clustering", action="store_true", help="Skip HDBSCAN object step")
    p.add_argument(
        "--cars-to",
        type=int,
        default=CLS_LOW_NOISE,
        help=f"Output class id for detected cars (default: {CLS_LOW_NOISE} => remove cars to noise).",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Speed up clustering: voxel-downsample before HDBSCAN (~3-10x on steps D/E).",
    )
    p.add_argument(
        "--hdbscan-backend",
        choices=[HDBSCAN_BACKEND_CPU, HDBSCAN_BACKEND_CUML, HDBSCAN_BACKEND_HYBRID],
        default=HDBSCAN_BACKEND_CUML,
        help=(
            "HDBSCAN backend: cuml (GPU-first, default; CPU on failure), "
            "cpu (v8r9 baseline), hybrid (GPU on D; CPU on E/I2 + finalize). "
            "Env: HDBSCAN_BACKEND"
        ),
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        help=f"Alias for --hdbscan-backend {HDBSCAN_BACKEND_CUML}.",
    )
    p.add_argument(
        "--skip-low-noise",
        action="store_true",
        help="Skip low-point grid + statistical outlier steps (class 7 not used for noise).",
    )
    args = p.parse_args()

    backend = HDBSCAN_BACKEND_CUML if args.gpu else args.hdbscan_backend
    if args.gpu and args.hdbscan_backend != HDBSCAN_BACKEND_CPU:
        p.error("Use either --gpu or --hdbscan-backend, not both.")

    out = args.output or args.input.with_name(args.input.stem + "_classified.las")
    run_pipeline(
        args.input,
        out,
        source_class=args.source_class,
        use_csf=not args.no_csf,
        skip_clustering=args.skip_clustering,
        cars_target_class=args.cars_to,
        fast=args.fast,
        hdbscan_backend=backend,
        skip_low_noise=args.skip_low_noise,
    )


if __name__ == "__main__":
    main()
