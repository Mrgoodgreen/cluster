#!/usr/bin/env python3
"""RAM-aware tiled RF post-pass for an already geometry-classified LAS."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import laspy
import numpy as np

from classify_tls import log
from classify_tls_all_in_one import write_las_atomic
from classify_tls_tiling import (
    TileRect, _mask_window, available_ram_bytes, build_tile_grid, suggest_tile_geometry,
)
from etalon_rf_refiner import refine_classification

RF_PEAK_BYTES_PER_POINT = 400  # features, sorting, probability arrays, temporary copies


def rf_point_budget(requested: int, model_bytes: int) -> int:
    """A user cap can tighten, never bypass, the currently available RAM budget."""
    available = available_ram_bytes()
    if available is None:
        return requested if requested > 0 else 5_000_000
    reserve = model_bytes * 3 + 256 * 1024**2
    budget = int((available * 0.60 - reserve) / RF_PEAK_BYTES_PER_POINT)
    if budget < 10_000:
        raise MemoryError('RF_RAM_BUDGET_EXHAUSTED: free more RAM or increase the container limit')
    return min(requested, budget) if requested > 0 else budget


def refine_window_adaptive(x, y, z, cls, intensity, rect, overlap, xmax, ymax,
                           max_points, model_path, *, depth=0):
    """Yield only core labels; preserve overlap and split on actual load or OOM.

    A spatial split cannot solve an overlap neighbourhood that alone exceeds the
    memory budget. Fail explicitly instead of reducing context or dropping points.
    """
    idx = np.flatnonzero(_mask_window(
        x, y, rect.x0-overlap, rect.x1+overlap, rect.y0-overlap, rect.y1+overlap,
        xmax=xmax, ymax=ymax, inclusive_max=True,
    ))
    if not len(idx):
        return
    core = _mask_window(x[idx], y[idx], rect.x0, rect.x1, rect.y0, rect.y1,
                        xmax=xmax, ymax=ymax, inclusive_max=True)
    if not core.any():
        return
    cap = rf_point_budget(max_points, model_path.stat().st_size)
    split = len(idx) > cap
    if not split:
        failed = False
        try:
            result, _ = refine_classification(
                x[idx], y[idx], z[idx], cls[idx].copy(),
                intensity[idx] if intensity is not None else None, model_path,
            )
        except MemoryError:
            failed = True
        # Outside the except block: no traceback keeps failed feature arrays alive.
        if not failed:
            yield idx[core], result[core]
            return
        gc.collect()
        split = True
        log(f'[RF-TILE] MemoryError; splitting window with {len(idx):,} points')
    if split:
        width, height = rect.x1-rect.x0, rect.y1-rect.y0
        if depth >= 24 or max(width, height) <= 1.0:
            raise MemoryError(
                f'RF_CONTEXT_EXCEEDS_BUDGET: load={len(idx):,}, cap={cap:,}, '
                f'overlap={overlap:g}m. Original geometry LAS is preserved. '
                'Increase available RAM or the explicit RF_MAX_TILE_POINTS limit '
                'to preserve this neighbourhood.'
            )
        if width >= height:
            mid = (rect.x0+rect.x1)/2
            parts = (TileRect(rect.x0,mid,rect.y0,rect.y1), TileRect(mid,rect.x1,rect.y0,rect.y1))
        else:
            mid = (rect.y0+rect.y1)/2
            parts = (TileRect(rect.x0,rect.x1,rect.y0,mid), TileRect(rect.x0,rect.x1,mid,rect.y1))
        log(f'[RF-TILE] split depth={depth} load={len(idx):,} cap={cap:,}')
        del idx, core
        for part in parts:
            yield from refine_window_adaptive(x,y,z,cls,intensity,part,overlap,xmax,ymax,
                                               max_points,model_path,depth=depth+1)


def run_rf_tiled(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    *,
    max_tile_points: int = 0,
    tile_m: float = 0.0,
    overlap_m: float = 8.0,
    context_mode: str = 'windows',
) -> None:
    """Apply RF in overlapping XY windows and merge classifications by core."""
    started = time.time()
    if max_tile_points < 0 or tile_m < 0 or overlap_m < 0:
        raise ValueError('RF tile settings must be non-negative')
    if context_mode == 'scene':
        from rf_scene_context import run_rf_scene
        return run_rf_scene(input_path,output_path,model_path,max_batch_points=max_tile_points)
    if context_mode != 'windows':
        raise ValueError('Unknown RF context mode')
    with laspy.open(str(input_path)) as reader:
        parent_bytes = reader.header.point_count * (reader.header.point_format.size + 36)
    available = available_ram_bytes()
    if available is not None and parent_bytes > available * 0.80:
        raise MemoryError('RF_PARENT_EXCEEDS_RAM: the parent cloud cannot safely fit in available RAM')
    log(f"[RF-TILE] Reading {input_path} ...")
    las = laspy.read(str(input_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    cls = np.asarray(las.classification, dtype=np.uint8)
    intensity = (
        np.asarray(las.intensity, dtype=np.float64)
        if hasattr(las, "intensity")
        else None
    )

    n_points = len(cls)
    if not n_points:
        write_las_atomic(las, output_path)
        return
    max_tile_points = rf_point_budget(max_tile_points, model_path.stat().st_size)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    geometry = suggest_tile_geometry(
        n_points,
        xmin,
        xmax,
        ymin,
        ymax,
        with_rf=True,
        tile_m=tile_m,
        max_tile_points=max_tile_points,
        overlap_m=overlap_m,
        tile_workers=1,
    )
    tiles = build_tile_grid(
        xmin,
        xmax,
        ymin,
        ymax,
        geometry.tile_m,
        geometry.overlap_m,
    )
    log(f"[RF-TILE] {geometry.reason}")
    log(f"[RF-TILE] {n_points:,} pts -> {len(tiles)} cores")

    cls_out = cls.copy()
    assigned = np.zeros(n_points, dtype=bool)
    overrides_total = 0

    for tile_index, rect in enumerate(tiles, start=1):
        in_load = _mask_window(
            x,
            y,
            rect.x0 - geometry.overlap_m,
            rect.x1 + geometry.overlap_m,
            rect.y0 - geometry.overlap_m,
            rect.y1 + geometry.overlap_m,
            xmax=xmax,
            ymax=ymax,
            inclusive_max=True,
        )
        idx = np.where(in_load)[0]
        if len(idx) == 0:
            continue
        in_core = _mask_window(
            x[idx],
            y[idx],
            rect.x0,
            rect.x1,
            rect.y0,
            rect.y1,
            xmax=xmax,
            ymax=ymax,
            inclusive_max=True,
        )
        if not in_core.any():
            continue

        log(
            f"[RF-TILE] {tile_index}/{len(tiles)} "
            f"load={len(idx):,} core={int(in_core.sum()):,}"
        )
        del idx, in_load, in_core
        for core_idx, core_cls in refine_window_adaptive(
            x,y,z,cls,intensity,rect,geometry.overlap_m,xmax,ymax,max_tile_points,model_path,
        ):
            if assigned[core_idx].any():
                raise RuntimeError('RF_CORE_OVERLAP: point assigned more than once')
            overrides_total += int(np.count_nonzero(cls[core_idx] != core_cls))
            cls_out[core_idx] = core_cls
            assigned[core_idx] = True
            del core_idx, core_cls
        gc.collect()

    n_missing = int((~assigned).sum())
    if n_missing:
        raise RuntimeError(f'RF_UNCOVERED_POINTS: {n_missing:,}; refusing an incomplete result')

    las.classification = cls_out
    log(f"[RF-TILE] overrides~={overrides_total:,}; writing {output_path}")
    write_las_atomic(las, output_path)
    log(f"[RF-TILE] Done in {(time.time() - started) / 60:.1f} min")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--rf-refiner", type=Path, required=True)
    parser.add_argument("--max-tile-points", type=int, default=0, help='0 = derive from available RAM')
    parser.add_argument("--tile-m", type=float, default=0.0)
    parser.add_argument("--overlap-m", type=float, default=8.0)
    parser.add_argument('--context-mode', choices=('windows','scene'), default='windows')
    args = parser.parse_args()
    run_rf_tiled(
        args.input,
        args.output,
        args.rf_refiner,
        max_tile_points=args.max_tile_points,
        tile_m=args.tile_m,
        overlap_m=args.overlap_m,
        context_mode=args.context_mode,
    )


if __name__ == "__main__":
    main()
