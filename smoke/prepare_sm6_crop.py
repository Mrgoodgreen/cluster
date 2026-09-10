#!/usr/bin/env python3
"""Prepare sm6 template crop (GT + geo auto) for RF refiner retrain."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import laspy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "_etalon_tune"
OUT.mkdir(exist_ok=True)

SRC = Path("/mnt/c/Users/roman/Downloads/3dit_01208sm6_template/3dit_01208sm6_template.las")
HALF = 14.0  # ~28 m window — dense urban TLS (~3k pts/m²)


def hist(cls: np.ndarray) -> dict[int, int]:
    return {int(k): int(v) for k, v in sorted(Counter(cls.tolist()).items())}


def write_las(path: Path, x, y, z, cls, inten, template: laspy.LasData) -> None:
    header = laspy.LasHeader(
        point_format=template.header.point_format,
        version=template.header.version,
    )
    header.offsets = template.header.offsets
    header.scales = template.header.scales
    out = laspy.LasData(header)
    out.x = x
    out.y = y
    out.z = z
    out.classification = cls.astype(np.uint8)
    if inten is not None and hasattr(out, "intensity"):
        try:
            out.intensity = inten
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(path))


def main() -> None:
    print(f"Reading {SRC} ...", flush=True)
    las = laspy.read(str(SRC))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    cls = np.asarray(las.classification, dtype=np.uint8)
    inten = np.asarray(las.intensity, dtype=np.float64) if hasattr(las, "intensity") else None
    print(f"  n={len(cls):,} hist={hist(cls)}", flush=True)

    vm = cls == 91
    if int(vm.sum()) < 5000:
        bm = cls == 6
        cx = float(np.median(x[bm])) if bm.any() else float(np.median(x))
        cy = float(np.median(y[bm])) if bm.any() else float(np.median(y))
    else:
        cx = float(np.median(x[vm]))
        cy = float(np.median(y[vm]))

    m = (np.abs(x - cx) <= HALF) & (np.abs(y - cy) <= HALF)
    xc, yc, zc, cc = x[m], y[m], z[m], cls[m]
    ic = inten[m] if inten is not None else None
    print(f"  crop center=({cx:.1f},{cy:.1f}) half={HALF} n={len(cc):,}", flush=True)
    print(f"  crop hist={hist(cc)}", flush=True)

    crop_gt = OUT / "sm6_crop28m_gt.las"
    gt_npy = OUT / "sm6_crop28m_gt.npy"
    write_las(crop_gt, xc, yc, zc, cc, ic, las)
    np.save(gt_npy, cc.astype(np.uint8))
    meta = {
        "source": str(SRC),
        "cx": cx,
        "cy": cy,
        "half": HALF,
        "n": int(len(cc)),
        "hist": hist(cc),
        "classes_note": {
            "2": "Ground",
            "3": "Low Vegetation (grass)",
            "4": "Medium Vegetation (empty in this scene)",
            "5": "High Vegetation (trees)",
            "6": "Building (buildings, fences, lamps, bins, signs)",
            "7": "Low Point / Noise (noise, wires, movable)",
            "91": "Vehicle (cars)",
        },
    }
    (OUT / "sm6_crop28m_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {crop_gt}", flush=True)
    print(f"wrote {gt_npy}", flush=True)


if __name__ == "__main__":
    main()
