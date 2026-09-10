#!/usr/bin/env python3
"""Etalon-trained RandomForest post-pass for TLS class refinement.

Train with:  python cluster/smoke/train_etalon_refiner.py train
Apply via:   classify_tls_all_in_one.py ... --rf-refiner _etalon_tune/etalon_rf_refiner.pkl
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

LABELS = np.array([2, 3, 4, 5, 6, 7, 9], dtype=np.uint8)


def dtm_from_ground(x, y, z, gmask, cell=1.0):
    xmin, ymin = float(x.min()), float(y.min())
    ix = np.floor((x - xmin) / cell).astype(np.int64)
    iy = np.floor((y - ymin) / cell).astype(np.int64)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    grid = np.full(nx * ny, np.inf, dtype=np.float64)
    gix, giy, gz = ix[gmask], iy[gmask], z[gmask]
    if len(gz) < 100:
        return np.full(len(x), float(np.median(z)))
    lin = gix * ny + giy
    order = np.argsort(lin)
    lin_s, gz_s = lin[order], gz[order]
    uniq, starts = np.unique(lin_s, return_index=True)
    grid[uniq] = np.minimum.reduceat(gz_s, starts)
    fill = float(np.median(gz))
    grid[~np.isfinite(grid)] = fill
    g2 = grid.reshape(nx, ny)
    pad = np.pad(g2, 1, mode="edge")
    neigh = np.stack(
        [
            pad[0:-2, 0:-2], pad[0:-2, 1:-1], pad[0:-2, 2:],
            pad[1:-1, 0:-2], pad[1:-1, 1:-1], pad[1:-1, 2:],
            pad[2:, 0:-2], pad[2:, 1:-1], pad[2:, 2:],
        ],
        axis=0,
    )
    return neigh.min(axis=0).ravel()[ix * ny + iy]


def build_features(x, y, z, intensity, pred, ground_z):
    h = (z - ground_z).astype(np.float64)
    n = len(x)
    cell = 0.35
    xmin, ymin, zmin = float(x.min()), float(y.min()), float(z.min())
    ix = np.floor((x - xmin) / cell).astype(np.int64)
    iy = np.floor((y - ymin) / cell).astype(np.int64)
    iz = np.floor((z - zmin) / cell).astype(np.int64)
    key = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791
    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    uniq, starts, counts = np.unique(key_s, return_index=True, return_counts=True)
    z_s, h_s = z[order], h[order]
    z_min = np.minimum.reduceat(z_s, starts)
    z_max = np.maximum.reduceat(z_s, starts)
    h_mean = np.add.reduceat(h_s, starts) / counts
    span = z_max - z_min
    inv = np.empty(len(key), dtype=np.int64)
    inv[order] = np.repeat(np.arange(len(uniq)), counts)
    v_count = counts[inv].astype(np.float64)
    v_span = span[inv]
    v_hmean = h_mean[inv]

    tall = (pred == 6) & (h >= 3.5)
    if tall.any():
        tidx = np.where(tall)[0]
        if len(tidx) > 250_000:
            tidx = np.random.default_rng(0).choice(tidx, 250_000, replace=False)
        bt = cKDTree(np.column_stack([x[tidx], y[tidx]]))
        d_bldg, _ = bt.query(np.column_stack([x, y]), k=1, workers=-1)
    else:
        d_bldg = np.full(n, 50.0)

    inten = intensity if intensity is not None else np.zeros(n, dtype=np.float64)
    if inten.max() > 0:
        pos = inten > 0
        scale = float(np.percentile(inten[pos], 95)) if pos.any() else 1.0
        inten = np.clip(inten / max(scale, 1e-6), 0, 2.0)

    return np.column_stack(
        [
            h,
            np.log1p(v_count),
            v_span,
            v_hmean,
            inten,
            np.clip(d_bldg, 0, 20),
            pred.astype(np.float64),
            (h >= 0.30).astype(np.float64),
            (h >= 2.20).astype(np.float64),
            (h >= 4.00).astype(np.float64),
        ]
    )


def apply_refiner(cls, X, model, conf_min=0.60, amb_classes=(3, 4, 5, 6, 7, 9)):
    proba = model.predict_proba(X)
    classes = model.classes_
    best = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    rf = classes[best].astype(np.uint8)
    amb = np.isin(cls, amb_classes)
    ground_fix = (cls == 2) & (rf != 2) & (conf >= 0.75) & np.isin(rf, [3, 9, 6])
    override = ((amb & (conf >= conf_min)) | ground_fix) & (rf != cls)
    out = cls.copy()
    out[override] = rf[override]
    return out, int(override.sum())


def prediction_features(x, y, z, cls, intensity):
    """Features available at inference; ground truth must never enter this path."""
    ground_z = dtm_from_ground(x, y, z, cls == 2)
    return build_features(x, y, z, intensity, cls, ground_z)


def refine_classification(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    intensity: np.ndarray | None,
    model_path: Path,
    conf_min: float | None = None,
) -> tuple[np.ndarray, int]:
    with open(model_path, "rb") as f:
        blob = pickle.load(f)
    model = blob["model"]
    if conf_min is None:
        conf_min = float(blob.get("conf_min", 0.60))
    X = prediction_features(x, y, z, cls, intensity)
    return apply_refiner(cls, X, model, conf_min=conf_min)
