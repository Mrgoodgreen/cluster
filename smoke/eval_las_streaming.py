#!/usr/bin/env python3
"""Streaming confusion matrix for two same-order LAS/LAZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import laspy
import numpy as np

LABELS = (2, 3, 4, 5, 6, 7, 91)
NAMES = {
    2: "ground",
    3: "low_veg",
    4: "med_veg",
    5: "high_veg",
    6: "building",
    7: "noise",
    91: "vehicle",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-las", type=Path, required=True)
    parser.add_argument("--auto-las", type=Path, required=True)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    args = parser.parse_args()

    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    other = np.zeros(len(LABELS), dtype=np.int64)
    pred_total = np.zeros(len(LABELS), dtype=np.int64)
    total_valid = 0
    total_correct = 0

    with laspy.open(str(args.gt_las)) as gt_reader, laspy.open(
        str(args.auto_las)
    ) as auto_reader:
        if gt_reader.header.point_count != auto_reader.header.point_count:
            raise SystemExit(
                "point-count mismatch: "
                f"gt={gt_reader.header.point_count:,} "
                f"auto={auto_reader.header.point_count:,}"
            )
        gt_iter = gt_reader.chunk_iterator(args.chunk_size)
        auto_iter = auto_reader.chunk_iterator(args.chunk_size)
        for chunk_index, (gt_points, auto_points) in enumerate(
            zip(gt_iter, auto_iter), start=1
        ):
            gt = np.asarray(gt_points.classification, dtype=np.uint8)
            pred = np.asarray(auto_points.classification, dtype=np.uint8)
            valid = np.isin(gt, LABELS)
            total_valid += int(valid.sum())
            total_correct += int(((gt == pred) & valid).sum())
            for pred_index, pred_label in enumerate(LABELS):
                pred_total[pred_index] += int((pred == pred_label).sum())
            for gt_index, gt_label in enumerate(LABELS):
                row = gt == gt_label
                row_known = 0
                for pred_index, pred_label in enumerate(LABELS):
                    count = int((row & (pred == pred_label)).sum())
                    matrix[gt_index, pred_index] += count
                    row_known += count
                other[gt_index] += int(row.sum()) - row_known
            print(
                f"chunk {chunk_index}: processed={min(chunk_index * args.chunk_size, total_valid):,}",
                flush=True,
            )

    recall = {}
    precision = {}
    for index, label in enumerate(LABELS):
        row_total = int(matrix[index].sum() + other[index])
        recall[NAMES[label]] = (
            float(matrix[index, index] / row_total) if row_total else None
        )
        precision[NAMES[label]] = (
            float(matrix[index, index] / pred_total[index])
            if pred_total[index]
            else None
        )

    report = {
        "gt": str(args.gt_las),
        "auto": str(args.auto_las),
        "n": total_valid,
        "overall_acc": float(total_correct / total_valid) if total_valid else 0.0,
        "precision": precision,
        "recall": recall,
        "matrix": {
            NAMES[gt_label]: {
                **{
                    NAMES[pred_label]: int(matrix[gt_i, pred_i])
                    for pred_i, pred_label in enumerate(LABELS)
                },
                "other": int(other[gt_i]),
            }
            for gt_i, gt_label in enumerate(LABELS)
        },
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n", "overall_acc", "precision", "recall")}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
