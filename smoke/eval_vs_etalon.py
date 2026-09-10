#!/usr/bin/env python3
"""Confusion matrix: auto LAS vs GT npy (same point order)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import laspy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "_etalon_tune"

LABELS = (2, 3, 4, 5, 6, 7, 91)
NAMES = {2: "ground", 3: "low_veg", 4: "med_veg", 5: "high_veg", 6: "building", 7: "noise", 91: "vehicle"}


def confusion(gt: np.ndarray, pred: np.ndarray, labels=LABELS) -> dict:
    mat = {}
    for a in labels:
        ga = gt == a
        n = int(ga.sum())
        if n == 0:
            continue
        row = {str(b): int(((pred == b) & ga).sum()) for b in labels}
        row["other"] = int(n - sum(row.values()))
        row["n"] = n
        row["recall"] = float(row.get(str(a), 0) / n)
        mat[NAMES.get(a, str(a))] = row
    mask = np.isin(gt, labels)
    acc = float((pred[mask] == gt[mask]).mean()) if mask.any() else 0.0
    prec = {}
    for a in labels:
        pa = pred == a
        n = int(pa.sum())
        if n == 0:
            continue
        prec[NAMES.get(a, str(a))] = float(((gt == a) & pa).sum() / n)
    return {"overall_acc": acc, "precision": prec, "recall_rows": mat}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--auto", type=Path, required=True)
    p.add_argument("-o", "--out", type=Path, default=None)
    args = p.parse_args()
    gt = np.load(args.gt)
    pred = np.asarray(laspy.read(str(args.auto)).classification, dtype=np.uint8)
    if len(pred) != len(gt):
        raise SystemExit(f"length mismatch gt={len(gt)} pred={len(pred)}")
    rep = confusion(gt, pred)
    confs = []
    for name, row in rep["recall_rows"].items():
        true_id = {v: k for k, v in NAMES.items()}[name]
        for b, cnt in row.items():
            if b in ("n", "recall", "other"):
                continue
            if int(b) == true_id or cnt == 0:
                continue
            confs.append((cnt, name, NAMES.get(int(b), b), cnt / row["n"]))
    confs.sort(reverse=True)
    rep["top_confusions"] = [
        {"n": n, "gt": a, "pred": b, "frac_of_gt": float(f)} for n, a, b, f in confs[:25]
    ]
    out = args.out or OUT / f"confusion_{args.auto.stem}.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({"file": args.auto.name, "overall_acc": rep["overall_acc"], "precision": rep["precision"]}, indent=2))
    print("\nRecall:")
    for k, v in rep["recall_rows"].items():
        print(f"  {k:10s} recall={v['recall']:.3f}  n={v['n']:,}")
    print("\nTop confusions:")
    for c in rep["top_confusions"][:12]:
        print(f"  {c['gt']:10s} -> {c['pred']:10s}  {c['n']:,}  ({c['frac_of_gt']*100:.1f}%)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
