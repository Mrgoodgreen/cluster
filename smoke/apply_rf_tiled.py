#!/usr/bin/env python3
"""Apply etalon RF refiner to a classified LAS via RAM-aware XY tiles."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import laspy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from classify_tls import log  # noqa: E402
from classify_tls_all_in_one import write_las_atomic  # noqa: E402
from classify_tls_tiling import (  # noqa: E402
    build_tile_grid,
    suggest_tile_geometry,
    _mask_window,
)
from etalon_rf_refiner import refine_classification  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Tiled RF refine on already-classified LAS")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--rf-refiner", type=Path, required=True)
    p.add_argument("--max-tile-points", type=int, default=5_000_000)
    p.add_argument("--tile-m", type=float, default=0.0)
    p.add_argument("--overlap-m", type=float, default=8.0)
    args = p.parse_args()

    t0 = time.time()
    log(f"[RF-TILE] Reading {args.input} ...")
    las = laspy.read(str(args.input))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    cls = np.asarray(las.classification, dtype=np.uint8)
    inten = np.asarray(las.intensity, dtype=np.float64) if hasattr(las, "intensity") else None
    n = len(cls)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    geom = suggest_tile_geometry(
        n, xmin, xmax, ymin, ymax,
        with_rf=True,
        tile_m=args.tile_m,
        max_tile_points=args.max_tile_points,
        overlap_m=args.overlap_m,
        tile_workers=1,
    )
    tiles = build_tile_grid(xmin, xmax, ymin, ymax, geom.tile_m, geom.overlap_m)
    log(f"[RF-TILE] {geom.reason}")
    log(f"[RF-TILE] {n:,} pts → {len(tiles)} cores")

    cls_out = cls.copy()
    assigned = np.zeros(n, dtype=bool)
    n_ov_total = 0

    for i, rect in enumerate(tiles):
        load_x0, load_x1 = rect.x0 - geom.overlap_m, rect.x1 + geom.overlap_m
        load_y0, load_y1 = rect.y0 - geom.overlap_m, rect.y1 + geom.overlap_m
        in_load = _mask_window(
            x, y, load_x0, load_x1, load_y0, load_y1,
            xmax=xmax, ymax=ymax, inclusive_max=True,
        )
        idx = np.where(in_load)[0]
        if len(idx) == 0:
            continue
        in_core = _mask_window(
            x[idx], y[idx], rect.x0, rect.x1, rect.y0, rect.y1,
            xmax=xmax, ymax=ymax, inclusive_max=True,
        )
        if not in_core.any():
            continue
        log(f"[RF-TILE] {i+1}/{len(tiles)} load={len(idx):,} core={int(in_core.sum()):,}")
        tcls = cls[idx].copy()
        tint = inten[idx] if inten is not None else None
        tcls, n_ov = refine_classification(
            x[idx], y[idx], z[idx], tcls, tint, args.rf_refiner
        )
        n_ov_total += n_ov
        core_idx = idx[in_core]
        cls_out[core_idx] = tcls[in_core]
        assigned[core_idx] = True
        del tcls, tint, idx
        gc.collect()

    miss = int((~assigned).sum())
    if miss:
        log(f"[RF-TILE] WARNING: {miss:,} uncovered; keep input class")
        cls_out[~assigned] = cls[~assigned]

    las.classification = cls_out
    log(f"[RF-TILE] overrides≈{n_ov_total:,}; writing {args.output}")
    write_las_atomic(las, args.output)
    log(f"[RF-TILE] Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
