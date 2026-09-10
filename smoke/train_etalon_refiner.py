#!/usr/bin/env python3
"""Train etalon RF on one or more (auto, gt) pairs; leave-one-scene-out eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import laspy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = Path(__file__).resolve().parents[1] / 'pipeline'
sys.path.insert(0, str(PIPELINE))
OUT = ROOT / "_etalon_tune"

from etalon_rf_refiner import (  # noqa: E402
    LABELS,
    apply_refiner,
    prediction_features,
)


def load_pair(auto: Path, gt: Path):
    truth = laspy.read(str(gt)) if gt.suffix.lower() in ('.las', '.laz') else None
    g = np.asarray(truth.classification, dtype=np.uint8).copy() if truth is not None else np.load(gt)
    las = laspy.read(str(auto))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    pred = np.asarray(las.classification, dtype=np.uint8)
    inten = np.asarray(las.intensity, dtype=np.float64) if hasattr(las, "intensity") else None
    assert len(g) == len(pred), (auto, len(g), len(pred))
    if truth is not None:
        if not np.array_equal(las.header.scales, truth.header.scales) or not np.array_equal(las.header.offsets, truth.header.offsets):
            raise ValueError('Ground truth coordinate encoding differs')
        for field in ('X', 'Y', 'Z'):
            if not np.array_equal(las[field], truth[field]):
                raise ValueError(f'Ground truth coordinates/order differ: {field}')
    else:
        raise ValueError('Use a ground-truth LAS/LAZ to verify point correspondence, not an unbound NPY')
    return x, y, z, pred, inten, g


def balanced_idx(gt, mask, max_per=50_000, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.where(mask)[0]
    keep = []
    for c in LABELS:
        ic = idx[gt[idx] == c]
        if len(ic) == 0:
            continue
        if len(ic) > max_per:
            ic = rng.choice(ic, max_per, replace=False)
        keep.append(ic)
    return np.concatenate(keep) if keep else np.array([], dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", default=[
        str(OUT / "etalon_crop24m_auto.las") + "," + str(OUT / "etalon_crop24m.las"),
        str(OUT / "template_crop_auto.las") + "," + str(OUT / "template_crop.las"),
    ])
    p.add_argument("--model", type=Path, default=OUT / "etalon_rf_refiner.pkl")
    p.add_argument("--eval-json", type=Path, default=OUT / "rf_refiner_eval.json")
    p.add_argument("--conf", type=float, default=0.58)
    args = p.parse_args()
    if args.model.exists() or args.eval_json.exists():
        p.error('Choose new model/report paths: existing artifacts must not be overwritten')
    if len(args.pairs) < 2:
        p.error('At least two independent scenes are required')
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.eval_json.parent.mkdir(parents=True, exist_ok=True)

    scenes = []
    for spec in args.pairs:
        auto_s, gt_s = spec.split(",")
        auto, gt = Path(auto_s), Path(gt_s)
        print(f"load {auto.name} ...", flush=True)
        x, y, z, pred, inten, g = load_pair(auto, gt)
        X = prediction_features(x, y, z, pred, inten)
        scenes.append({"name": auto.stem, "X": X, "gt": g, "pred": pred, "las": laspy.read(str(auto))})

    # leave-one-scene-out
    loso = {}
    for i, held in enumerate(scenes):
        tr_X, tr_y = [], []
        for j, sc in enumerate(scenes):
            if i == j:
                continue
            m = np.isin(sc["gt"], LABELS) & np.isin(sc["pred"], LABELS)
            idx = balanced_idx(sc["gt"], m, max_per=40_000, seed=10 + j)
            tr_X.append(sc["X"][idx])
            tr_y.append(sc["gt"][idx])
        if not tr_X:
            continue
        clf = RandomForestClassifier(
            n_estimators=140, max_depth=14, min_samples_leaf=15,
            n_jobs=-1, class_weight="balanced_subsample", random_state=42,
        )
        clf.fit(np.vstack(tr_X), np.concatenate(tr_y))
        out, n_ov = apply_refiner(held["pred"], held["X"], clf, conf_min=args.conf)
        mask = np.isin(held["gt"], LABELS)
        loso[held["name"]] = {
            "classification_report_geo": classification_report(held['gt'], held['pred'], labels=LABELS.tolist(), output_dict=True, zero_division=0),
            "classification_report_rf": classification_report(held['gt'], out, labels=LABELS.tolist(), output_dict=True, zero_division=0),
            "acc_geo": float((held["pred"][mask] == held["gt"][mask]).mean()),
            "acc_rf": float((out[mask] == held["gt"][mask]).mean()),
            "n_override": n_ov,
            "vehicle_recall_geo": float((held["pred"][held["gt"] == 91] == 91).mean()) if (held["gt"] == 91).any() else None,
            "vehicle_recall_rf": float((out[held["gt"] == 91] == 91).mean()) if (held["gt"] == 91).any() else None,
        }
        print(f"LOSO {held['name']}: {json.dumps(loso[held['name']])}", flush=True)

    # final model on all scenes
    tr_X, tr_y = [], []
    for j, sc in enumerate(scenes):
        m = np.isin(sc["gt"], LABELS) & np.isin(sc["pred"], LABELS)
        idx = balanced_idx(sc["gt"], m, max_per=55_000, seed=20 + j)
        tr_X.append(sc["X"][idx])
        tr_y.append(sc["gt"][idx])
    clf = RandomForestClassifier(
        n_estimators=160, max_depth=16, min_samples_leaf=12,
        n_jobs=-1, class_weight="balanced_subsample", random_state=42,
    )
    clf.fit(np.vstack(tr_X), np.concatenate(tr_y))
    with open(args.model, "xb") as f:
        pickle.dump({"model": clf, "labels": LABELS.tolist(), "conf_min": args.conf,
                     "metadata": {"feature_source": "automatic_prediction_only", "pairs": args.pairs,
                                  "feature_code_sha256": hashlib.sha256((PIPELINE/'etalon_rf_refiner.py').read_bytes()).hexdigest(),
                                  "validation": "leave_one_scene_out", "seed": 42}}, f)
    print(f"wrote {args.model}")

    full = {}
    for sc in scenes:
        out, n_ov = apply_refiner(sc["pred"], sc["X"], clf, conf_min=args.conf)
        mask = np.isin(sc["gt"], LABELS)
        full[sc["name"]] = {
            "acc_geo": float((sc["pred"][mask] == sc["gt"][mask]).mean()),
            "acc_rf": float((out[mask] == sc["gt"][mask]).mean()),
            "n_override": n_ov,
            "recall": {
                int(c): {
                    "geo": float((sc["pred"][sc["gt"] == c] == c).mean()) if (sc["gt"] == c).any() else None,
                    "rf": float((out[sc["gt"] == c] == c).mean()) if (sc["gt"] == c).any() else None,
                }
                for c in LABELS
            },
        }
        sc["las"].classification = out
        out_path = args.model.parent / f"{sc['name']}_rf.las"
        if out_path.exists():
            raise FileExistsError(out_path)
        sc["las"].write(str(out_path))
        print(f"wrote {out_path}")

    rep = {"loso": loso, "full_retrain": full, "conf": args.conf,
           "feature_source": "automatic_prediction_only", "pairs": args.pairs,
           "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
           "note": "LOSO holds out a whole crop; full_retrain is in-sample, not independent validation"}
    args.eval_json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
