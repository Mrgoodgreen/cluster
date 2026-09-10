"""Apply tightened ground-cleanup + island peel to classified LAS (fast QA)."""
from __future__ import annotations

import sys
from pathlib import Path

import laspy
import numpy as np
from scipy.ndimage import label as nd_label

ROOT = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(ROOT))

from classify_tls import (  # noqa: E402
    CFG,
    CLS_BUILDING,
    CLS_GROUND,
    CLS_HIGH_VEG,
    CLS_LOW_VEG,
    CLS_MED_VEG,
    ground_dtm,
    peel_car_volume,
    peel_elevated_islands_from_ground,
    peel_low_car_residuals_from_ground,
    peel_noise_from_ground,
    restore_asphalt_from_noise,
    restore_flat_paving_xy,
    restore_near_dtm,
    robust_road_dtm,
)


def car_boxes_from_class7(
    x, y, z, cls, ground_z, cars_class=7, cell=0.5, min_h=0.20, min_count=80
):
    m = (cls == cars_class) & ((z - ground_z) >= min_h) & ((z - ground_z) <= 3.5)
    if int(m.sum()) < min_count:
        return []
    px, py, pz = x[m], y[m], z[m]
    xmin, ymin = float(px.min()), float(py.min())
    ix = np.floor((px - xmin) / cell).astype(np.int64)
    iy = np.floor((py - ymin) / cell).astype(np.int64)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    occ = np.zeros((nx, ny), dtype=np.uint8)
    occ[ix, iy] = 1
    labeled, nlab = nd_label(occ)
    if nlab == 0:
        return []
    lab = labeled[ix, iy]
    boxes = []
    for lid in range(1, nlab + 1):
        sel = lab == lid
        if int(sel.sum()) < min_count:
            continue
        xs, ys, zs = px[sel], py[sel], pz[sel]
        dx, dy = float(xs.max() - xs.min()), float(ys.max() - ys.min())
        length, width = max(dx, dy), min(dx, dy)
        if not (1.2 <= length <= 12.0 and 0.40 <= width <= 5.0):
            continue
        if width / max(length, 1e-6) > 0.92:
            continue
        boxes.append((float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()), float(zs.max())))
    return boxes


def main() -> None:
    src = Path(r"c:\Users\roman\Documents\Cursor\3dit1_00811cr\3dit1_00811cr1_1.las")
    out = src.with_name(src.stem + "_groundclean3.las")
    print("open", src)
    las = laspy.read(str(src))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    cls = np.asarray(las.classification, dtype=np.uint8).copy()
    cars = 7

    g0 = int((cls == CLS_GROUND).sum())
    print(f"before ground={g0:,} class7={int((cls == cars).sum()):,}")

    ground_z = robust_road_dtm(x, y, z, cell=CFG.ground_cell, opening_size=CFG.ground_opening)
    boxes = car_boxes_from_class7(x, y, z, cls, ground_z)
    print(f"car boxes: {len(boxes)}")

    n_vol = peel_car_volume(
        x, y, z, cls, ground_z, boxes, cars,
        peel_classes=(CLS_GROUND, CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG, CLS_BUILDING),
        xy_pad=CFG.car_xy_pad, min_h=CFG.car_min_h, z_top_pad=CFG.car_z_top_pad,
        dilate_cells=CFG.car_peel_dilate,
    )
    n_near = peel_low_car_residuals_from_ground(
        x, y, z, cls, ground_z, seed_class=cars, boxes=boxes,
        xy_radius=CFG.car_residual_xy, min_h=CFG.car_residual_min_h,
        max_h=CFG.car_residual_max_h, seed_min_h=CFG.car_residual_seed_h,
    )
    print(f"car peel / near-car: {n_vol:,} / {n_near:,}")

    local_z = ground_dtm(x, y, z, cls == CLS_GROUND, cell=CFG.ground_noise_cell)
    n_off, n_gs = peel_noise_from_ground(
        x, y, z, cls, local_z, target_class=cars,
        max_above=CFG.ground_noise_max_above, max_below=CFG.ground_noise_max_below,
        sparse_cell=CFG.ground_sparse_cell, sparse_min_count=CFG.ground_sparse_min,
    )
    h = z - ground_z
    elev = (cls == CLS_GROUND) & (h > 0.15) & (h < 4.0)
    n_elev = int(elev.sum())
    cls[elev] = cars
    n_isl = peel_elevated_islands_from_ground(
        x, y, z, cls, target_class=cars,
        cell=CFG.island_cell, min_h=CFG.island_min_h, max_h=CFG.island_max_h,
        min_count=CFG.island_min_count, max_length=CFG.island_max_length,
        max_width=CFG.island_max_width, dtm_cell=CFG.ground_noise_cell,
    )
    print(f"off/sparse/elev/islands: {n_off:,}/{n_gs:,}/{n_elev:,}/{n_isl:,}")

    n_rest = restore_asphalt_from_noise(
        x, y, z, cls, source_classes=(cars, CLS_LOW_VEG), ground_class=CLS_GROUND,
        xy_radius=1.5, z_tol=0.040,
    )
    n_dtm = restore_near_dtm(
        x, y, z, cls, ground_z, source_classes=(cars, CLS_LOW_VEG),
        max_h=0.20, flat_span=0.10, cell=0.40,
    )
    n_pave = restore_flat_paving_xy(
        x, y, z, cls, ground_z, source_classes=(cars, CLS_LOW_VEG),
        cell=1.0, max_h=0.22, max_z_span=0.10, min_count=14, min_near_frac=0.85,
    )
    print(f"restore: {n_rest:,}/{n_dtm:,}/{n_pave:,}")

    n_isl2 = peel_elevated_islands_from_ground(
        x, y, z, cls, target_class=cars,
        cell=CFG.island_cell, min_h=CFG.island_min_h, max_h=CFG.island_max_h,
        min_count=CFG.island_min_count, max_length=CFG.island_max_length,
        max_width=CFG.island_max_width, dtm_cell=CFG.ground_noise_cell,
    )
    local2 = ground_dtm(x, y, z, cls == CLS_GROUND, cell=CFG.ground_noise_cell)
    n_off2, n_gs2 = peel_noise_from_ground(
        x, y, z, cls, local2, target_class=cars,
        max_above=CFG.ground_noise_max_above, max_below=CFG.ground_noise_max_below,
        sparse_cell=CFG.ground_sparse_cell, sparse_min_count=CFG.ground_sparse_min,
    )
    print(f"final islands/off/sparse: {n_isl2:,}/{n_off2:,}/{n_gs2:,}")

    g1 = int((cls == CLS_GROUND).sum())
    print(f"after ground={g1:,} class7={int((cls == cars).sum()):,} delta={g1 - g0:+,}")
    hg = (z - ground_dtm(x, y, z, cls == CLS_GROUND, cell=0.30))[cls == CLS_GROUND]
    for t in (0.025, 0.04, 0.055, 0.08, 0.10, 0.15, 0.20):
        print(f"  remaining ground h>{t:.3f}: {int((hg > t).sum()):,}")

    las.classification = cls
    print("write", out)
    las.write(str(out))
    print("done")


if __name__ == "__main__":
    main()
