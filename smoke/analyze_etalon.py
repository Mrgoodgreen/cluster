#!/usr/bin/env python3
"""Analyze etalon LAS class geometry and compare auto pipeline on a crop."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import laspy
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "_etalon_tune"
OUT_DIR.mkdir(exist_ok=True)

from classify_tls import (  # noqa: E402
    CLS_BUILDING,
    CLS_GROUND,
    CLS_HIGH_VEG,
    CLS_LOW_NOISE,
    CLS_LOW_VEG,
    CLS_MED_VEG,
    CLS_VEHICLE,
    robust_road_dtm,
)

NAMES = {
    2: "ground",
    3: "low_veg",
    4: "med_veg",
    5: "high_veg",
    6: "building",
    7: "noise",
    8: "ghost",
    91: "vehicle",
}


def hist(cls: np.ndarray) -> dict[int, int]:
    u, c = np.unique(cls, return_counts=True)
    return {int(a): int(b) for a, b in zip(u, c)}


def pct(a: np.ndarray, qs=(5, 25, 50, 75, 90, 95)) -> dict[str, float]:
    if len(a) == 0:
        return {f"p{q}": float("nan") for q in qs}
    vals = np.percentile(a, qs)
    return {f"p{q}": float(v) for q, v in zip(qs, vals)}


def analyze_file(path: Path, max_pts: int = 8_000_000, seed: int = 42) -> dict:
    print(f"\n=== analyze {path.name} ===", flush=True)
    t0 = time.perf_counter()
    las = laspy.read(str(path))
    n = len(las.points)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    cls = np.asarray(las.classification, dtype=np.uint8)
    intensity = np.asarray(las.intensity, dtype=np.float64) if hasattr(las, "intensity") else None
    print(f"  loaded {n:,} in {time.perf_counter()-t0:.1f}s", flush=True)

    rng = np.random.default_rng(seed)
    if n > max_pts:
        idx = rng.choice(n, max_pts, replace=False)
        x, y, z, cls = x[idx], y[idx], z[idx], cls[idx]
        if intensity is not None:
            intensity = intensity[idx]
        print(f"  subsampled to {len(cls):,}", flush=True)

    # DTM from etalon ground
    gmask = cls == CLS_GROUND
    if gmask.sum() < 1000:
        raise RuntimeError("too few ground points in etalon subsample")
    # approximate DTM via cell mins of ground
    cell = 1.0
    xmin, ymin = float(x.min()), float(y.min())
    ix = np.floor((x - xmin) / cell).astype(np.int64)
    iy = np.floor((y - ymin) / cell).astype(np.int64)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    grid = np.full(nx * ny, np.inf, dtype=np.float64)
    gix, giy, gz = ix[gmask], iy[gmask], z[gmask]
    lin = gix * ny + giy
    order = np.argsort(lin)
    lin_s = lin[order]
    gz_s = gz[order]
    uniq, starts = np.unique(lin_s, return_index=True)
    mins = np.minimum.reduceat(gz_s, starts)
    grid[uniq] = mins
    # fill missing with global ground median
    fill = float(np.median(gz))
    grid[~np.isfinite(grid)] = fill
    # light dilation of mins via neighbor min (simple)
    g2 = grid.reshape(nx, ny)
    pad = np.pad(g2, 1, mode="edge")
    neigh = np.stack(
        [
            pad[0:-2, 0:-2],
            pad[0:-2, 1:-1],
            pad[0:-2, 2:],
            pad[1:-1, 0:-2],
            pad[1:-1, 1:-1],
            pad[1:-1, 2:],
            pad[2:, 0:-2],
            pad[2:, 1:-1],
            pad[2:, 2:],
        ],
        axis=0,
    )
    g2 = neigh.min(axis=0)
    ground_z = g2.ravel()[ix * ny + iy]
    h = z - ground_z

    report: dict = {"file": path.name, "n_full": n, "n_used": int(len(cls)), "hist": hist(cls)}
    per_class = {}
    for c in sorted(report["hist"].keys()):
        m = cls == c
        hc = h[m]
        info = {
            "count": int(m.sum()),
            "h": pct(hc),
            "h_mean": float(hc.mean()) if m.any() else None,
            "h_std": float(hc.std()) if m.any() else None,
        }
        if intensity is not None:
            ic = intensity[m]
            info["intensity"] = pct(ic)
        # local vertical span in 0.5 m cells for building/vehicle/noise
        if c in (CLS_BUILDING, CLS_VEHICLE, CLS_LOW_NOISE, CLS_HIGH_VEG):
            cell2 = 0.5
            ix2 = np.floor((x[m] - xmin) / cell2).astype(np.int64)
            iy2 = np.floor((y[m] - ymin) / cell2).astype(np.int64)
            lin2 = ix2 * (int(iy2.max()) + 1) + iy2
            order2 = np.argsort(lin2)
            lin2s = lin2[order2]
            zs = z[m][order2]
            hs = h[m][order2]
            u2, st2, ct2 = np.unique(lin2s, return_index=True, return_counts=True)
            spans = []
            tops = []
            for s, ctn in zip(st2, ct2):
                if ctn < 8:
                    continue
                chunk = zs[s : s + ctn]
                spans.append(float(chunk.max() - chunk.min()))
                tops.append(float(np.percentile(hs[s : s + ctn], 90)))
            if spans:
                info["cell_span"] = pct(np.asarray(spans))
                info["cell_top_h"] = pct(np.asarray(tops))
                info["cell_n"] = len(spans)
        per_class[NAMES.get(c, str(c))] = info
    report["per_class"] = per_class

    # vehicle blob sizes (connected components on vehicle XY)
    vm = cls == CLS_VEHICLE
    if vm.sum() > 50:
        cellv = 0.25
        vx, vy = x[vm], y[vm]
        vix = np.floor((vx - float(vx.min())) / cellv).astype(np.int64)
        viy = np.floor((vy - float(vy.min())) / cellv).astype(np.int64)
        from scipy.ndimage import label as nd_label

        occ = np.zeros((int(vix.max()) + 1, int(viy.max()) + 1), dtype=np.uint8)
        occ[vix, viy] = 1
        lab, nlab = nd_label(occ)
        sizes = []
        for lid in range(1, nlab + 1):
            yy, xx = np.where(lab == lid)
            if len(xx) < 8:
                continue
            L = max(xx.max() - xx.min(), yy.max() - yy.min()) * cellv
            W = min(xx.max() - xx.min(), yy.max() - yy.min()) * cellv
            sizes.append((L, W, int(len(xx))))
        if sizes:
            arr = np.asarray(sizes)
            report["vehicle_blobs"] = {
                "n": len(sizes),
                "length": pct(arr[:, 0]),
                "width": pct(arr[:, 1]),
                "cells": pct(arr[:, 2]),
            }

    # noise height band vs vehicle
    for c, key in ((CLS_LOW_NOISE, "noise"), (CLS_VEHICLE, "vehicle"), (CLS_BUILDING, "building")):
        m = cls == c
        if not m.any():
            continue
        report.setdefault("band_share", {})[key] = {
            "h<0.35": float((h[m] < 0.35).mean()),
            "0.35-2.2": float(((h[m] >= 0.35) & (h[m] < 2.2)).mean()),
            "2.2-4": float(((h[m] >= 2.2) & (h[m] < 4)).mean()),
            "h>=4": float((h[m] >= 4).mean()),
        }

    return report


def crop_bbox(x, y, z, cls, cx, cy, half: float):
    m = (np.abs(x - cx) <= half) & (np.abs(y - cy) <= half)
    return x[m], y[m], z[m], cls[m], m


def write_las(path: Path, x, y, z, cls, template: laspy.LasData):
    header = laspy.LasHeader(point_format=template.header.point_format, version=template.header.version)
    header.offsets = template.header.offsets
    header.scales = template.header.scales
    out = laspy.LasData(header)
    out.x = x
    out.y = y
    out.z = z
    out.classification = cls.astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(path))


def confusion(gt: np.ndarray, pred: np.ndarray, labels=(2, 3, 4, 5, 6, 7, 91)) -> dict:
    mat = {}
    for a in labels:
        row = {}
        ga = gt == a
        n = int(ga.sum())
        if n == 0:
            continue
        for b in labels:
            row[str(b)] = int(((pred == b) & ga).sum())
        row["other"] = int(ga.sum() - sum(row.values()))
        row["recall"] = float(row.get(str(a), 0) / n) if n else 0.0
        mat[str(a)] = row
    # overall accuracy on labeled classes
    mask = np.isin(gt, labels)
    acc = float((pred[mask] == gt[mask]).mean()) if mask.any() else 0.0
    return {"overall_acc": acc, "per_class": mat}


def main():
    etalon = Path("/mnt/c/Users/roman/Downloads/11/Etalon.las")
    template = Path("/mnt/c/Users/roman/Downloads/11/3dit_01208sm2_template.las")
    out_dir = OUT_DIR

    reports = {}
    for p in (etalon, template):
        reports[p.name] = analyze_file(p, max_pts=6_000_000)
        with open(out_dir / f"report_{p.stem}.json", "w", encoding="utf-8") as f:
            json.dump(reports[p.name], f, indent=2)
        print(json.dumps(reports[p.name]["band_share"], indent=2), flush=True)
        if "vehicle_blobs" in reports[p.name]:
            print("vehicle_blobs", json.dumps(reports[p.name]["vehicle_blobs"], indent=2), flush=True)

    # prepare crop from Etalon for pipeline comparison
    print("\n=== prepare Etalon crop ===", flush=True)
    las = laspy.read(str(etalon))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    cls = np.asarray(las.classification, dtype=np.uint8)
    # pick crop centered on dense vehicle+building area
    vm = cls == CLS_VEHICLE
    if vm.sum() < 1000:
        cx, cy = float(np.median(x)), float(np.median(y))
    else:
        cx, cy = float(np.median(x[vm])), float(np.median(y[vm]))
    half = 40.0
    xc, yc, zc, cc, m = crop_bbox(x, y, z, cls, cx, cy, half)
    print(f"  crop center=({cx:.1f},{cy:.1f}) half={half} n={len(cc):,}", flush=True)
    print("  crop hist", hist(cc), flush=True)
    crop_path = out_dir / "etalon_crop80m.las"
    gt_path = out_dir / "etalon_crop80m_gt.npy"
    write_las(crop_path, xc, yc, zc, cc, las)
    np.save(gt_path, cc)
    meta = {"cx": cx, "cy": cy, "half": half, "n": int(len(cc)), "hist": hist(cc)}
    (out_dir / "etalon_crop80m_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  wrote {crop_path}", flush=True)


if __name__ == "__main__":
    main()
