#!/usr/bin/env python3
"""Refresh cluster/pipeline from the parent TerraScan repo (maintainers only).

IT deployments use the files already inside cluster/pipeline — do not require this script.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = Path(__file__).resolve().parent
DST = CLUSTER / "pipeline"
GPU_SRC = ROOT / "GPU_0_0_1"
GPU_DST = DST / "GPU_0_0_1"

FILES_ROOT = [
    "classify_tls.py",
    "classify_tls_all_in_one.py",
    "classify_tls_tiling.py",
    "classify_tls_rf_tiling.py",
    "etalon_rf_refiner.py",
    "rf_scene_context.py",
]
FILES_GPU = [
    "classify_tls_gpu_0_0_1.py",
    "gpu_backend.py",
    "requirements-gpu-core.txt",
    "constraints-tested.txt",
    "check_gpu_0_0_1.py",
    "VERSION.txt",
]


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    GPU_DST.mkdir(parents=True, exist_ok=True)
    for name in FILES_ROOT:
        shutil.copy2(ROOT / name, DST / name)
        print("copied", name)
    for name in FILES_GPU:
        shutil.copy2(GPU_SRC / name, GPU_DST / name)
        print("copied GPU_0_0_1/" + name)
    model_src = ROOT / "_etalon_tune" / "etalon_rf_refiner.pkl"
    # Versioned release weights already packaged in pipeline/models are retained.
    # This legacy model copy must not replace etalon_rf_scene_20260909.pkl.
    if model_src.exists():
        model_dst = DST / "models" / model_src.name
        model_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_src, model_dst)
        print("copied models/" + model_src.name)
    print("Done ->", DST)


if __name__ == "__main__":
    main()
