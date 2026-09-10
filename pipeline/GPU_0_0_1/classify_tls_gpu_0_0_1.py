#!/usr/bin/env python3
"""Independent TLS Classify GPU 0.0.1 fork.

The input classification is always discarded. Every point is reset to
Default/class 1 before Stage A.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import classify_tls_all_in_one as cpu_pipeline
from classify_tls import CLS_DEFAULT, CLS_LOW_NOISE, CLS_VEHICLE, log
from classify_tls_rf_tiling import run_rf_tiled
from classify_tls_tiling import available_ram_bytes
from gpu_backend import (
    check_gpu,
    cluster_objects_guarded,
    configure,
    extract_cars_guarded,
    log_stats,
)

VERSION = "GPU 0.0.1"
GPU_MONO_BYTES_PER_POINT = 245
GPU_PARENT_BYTES_PER_POINT = 65


def _header_point_count(path: Path) -> int:
    try:
        import laspy

        with laspy.open(str(path)) as reader:
            return int(reader.header.point_count)
    except Exception:
        return 0


def _outer_tile_defaults(input_path: Path) -> tuple[int, int]:
    """RAM-derived outer geometry cap; inner HDBSCAN has its own VRAM tiles."""
    available = available_ram_bytes()
    n_points = _header_point_count(input_path)
    if available is None:
        return 50_000_000, 24_000_000

    # Measured on sm6: 29.0M monolithic GPU geo peaks near 7.1 GiB (~245 B/pt).
    mono_cap = int(available * 0.75 / GPU_MONO_BYTES_PER_POINT)

    # Tiled mode retains the parent cloud. Budget a leaf from RAM remaining after
    # parent XYZ/class/intensity/output arrays, with fragmentation headroom.
    parent_bytes = n_points * GPU_PARENT_BYTES_PER_POINT
    leaf_budget = max(0, int(available * 0.78) - parent_bytes)
    leaf_cap = int(leaf_budget / GPU_MONO_BYTES_PER_POINT)
    if mono_cap < 100_000 or leaf_cap < 100_000:
        raise MemoryError(
            'RAM_BUDGET_EXHAUSTED: insufficient available RAM after reserving '
            f'the parent cloud ({parent_bytes / 1024**3:.2f} GiB); '
            f'available={available / 1024**3:.2f} GiB. '
            'Increase the container memory limit or free RAM before retrying.'
        )
    leaf_cap = min(50_000_000, leaf_cap)
    return mono_cap, leaf_cap


def _gpu_available_or_reason() -> tuple[bool, str]:
    """Return GPU availability and diagnostic message."""
    try:
        return True, check_gpu()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _gpu_preflight_with_retries() -> tuple[bool, str]:
    retries = int(os.getenv("GPU_PREFLIGHT_RETRIES", "3"))
    delay_s = float(os.getenv("GPU_PREFLIGHT_RETRY_DELAY_SEC", "20"))
    last_msg = "unknown"
    for attempt in range(1, max(1, retries) + 1):
        ok, msg = _gpu_available_or_reason()
        if ok:
            if attempt > 1:
                log(f"GPU preflight recovered on attempt {attempt}/{retries}")
            return True, msg
        last_msg = msg
        if attempt < retries:
            log(
                f"GPU preflight failed ({attempt}/{retries}): {msg}. "
                f"Retry in {delay_s:.1f}s..."
            )
            time.sleep(delay_s)
    return False, last_msg


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "TLS Classify GPU 0.0.1 (WSL2/cuML guarded hybrid). "
            "Always resets every input point to class 1."
        )
    )
    p.add_argument("input", type=Path, help="Input LAS/LAZ; existing classes are ignored")
    p.add_argument("-o", "--output", type=Path, help="Output LAS/LAZ")
    p.add_argument(
        "--cars-to",
        type=int,
        default=CLS_VEHICLE,
        help="Vehicle class (TerraScan etalon default 91)",
    )
    p.add_argument(
        "--noise-to",
        type=int,
        default=CLS_LOW_NOISE,
        help="Low Point / noise class (default 7)",
    )
    p.add_argument("--tile-m", type=float, default=30.0, help="GPU core tile size in metres")
    p.add_argument(
        "--overlap-m",
        type=float,
        default=7.0,
        help="Tile overlap in metres (default 7; half of 12 m vehicle + padding)",
    )
    p.add_argument(
        "--max-tile-points",
        type=int,
        default=2_000_000,
        help="Split a raw tile above this count (default 2000000; GPU sees voxels)",
    )
    p.add_argument("--min-tile-m", type=float, default=15.0, help="Minimum adaptive tile size")
    p.add_argument("--gpu-device", type=int, default=0, help="CUDA device index")
    p.add_argument(
        "--build-algo",
        choices=("adaptive", "nn_descent", "brute_force"),
        default="brute_force",
        help=(
            "cuML kNN builder after voxelization (default brute_force)"
        ),
    )
    p.add_argument(
        "--nn-threshold",
        type=int,
        default=400_000,
        help="Adaptive mode uses nn_descent at this tile point count (default 400000)",
    )
    p.add_argument(
        "--voxel-m",
        type=float,
        default=0.08,
        help="GPU HDBSCAN voxel size in metres (default 0.08; 0 disables)",
    )
    p.add_argument(
        "--no-cpu-safety",
        action="store_true",
        help="Disable final residual CPU car pass (faster, less conservative)",
    )
    bundled_rf = ROOT / "models" / "etalon_rf_scene_20260909.pkl"
    env_rf = os.getenv("RF_REFINER", "").strip()
    default_rf = Path(env_rf) if env_rf else (bundled_rf if bundled_rf.exists() else None)
    p.add_argument(
        "--rf-refiner",
        type=Path,
        default=default_rf,
        help="Three-etalon RF model; defaults to RF_REFINER or bundled model",
    )
    p.add_argument(
        "--no-rf",
        action="store_true",
        help="Skip the RF post-pass even when a bundled model is available",
    )
    p.add_argument(
        "--keep-geo",
        action="store_true",
        help="Keep the intermediate geometry-only LAS when RF is enabled",
    )
    args = p.parse_args()

    rf_context_mode = os.getenv('RF_CONTEXT_MODE', 'scene').strip()
    if not args.no_rf and args.rf_refiner is not None:
        if rf_context_mode not in ('scene', 'windows'):
            p.error('RF_CONTEXT_MODE must be scene or windows')
        if rf_context_mode == 'scene' and (args.cars_to != 9 or args.noise_to != 7):
            p.error('Scene RF expects CARS_TO=9 and NOISE_TO=7; update the worker environment')

    if args.tile_m <= 0 or args.overlap_m < 0 or args.max_tile_points < 10_000:
        p.error("invalid tile/memory parameters")
    if args.min_tile_m <= 0 or args.min_tile_m > args.tile_m:
        p.error("--min-tile-m must be > 0 and <= --tile-m")

    configure(
        tile_m=args.tile_m,
        overlap_m=args.overlap_m,
        max_tile_points=args.max_tile_points,
        min_tile_m=args.min_tile_m,
        cpu_safety_pass=not args.no_cpu_safety,
        gpu_device=args.gpu_device,
        build_algo=args.build_algo,
        nn_descent_threshold=args.nn_threshold,
        voxel_m=args.voxel_m,
    )

    log(f"{VERSION} — independent fork of CPU 1.0.3")
    gpu_ok, gpu_msg = _gpu_preflight_with_retries()
    if gpu_ok:
        log(f"GPU: {gpu_msg}")
        log(f"cuML build_algo: {args.build_algo}; nn threshold={args.nn_threshold:,}")
        log(f"GPU voxel: {args.voxel_m:g} m (no random subsampling)")
        # Patch only this fork process. CPU 1.0.3 source files and CLI remain unchanged.
        cpu_pipeline.cluster_objects = cluster_objects_guarded
        cpu_pipeline.extract_cars_from_classes = extract_cars_guarded
    else:
        # Key resilience fix for cluster workers: if CUDA temporarily disappears
        # (e.g. driver hiccup / device reset), continue with CPU instead of failing.
        log(f"GPU unavailable; falling back to CPU pipeline: {gpu_msg}")
    log("MANDATORY RESET: all input classifications -> class 1 (Default)")

    output = args.output or args.input.with_name(args.input.stem + "_gpu_0_0_1.las")
    rf_model = None if args.no_rf else args.rf_refiner
    if rf_model is not None and not rf_model.exists():
        p.error(f"RF model does not exist: {rf_model}")
    geo_output = (
        output.with_name(output.stem + ".geo_tmp" + output.suffix)
        if rf_model is not None
        else output
    )
    started = time.perf_counter()
    force_tile = os.getenv("FORCE_TILE", "").strip().lower() in ("1", "true", "yes")
    # GPU_0_0_1 monkey-patches every HDBSCAN call to guarded cuML. Report
    # ``hybrid`` to all-in-one so it does not run the generic post-cuML stage K:
    # this wrapper already performs its validated CPU residual safety pass at J.
    # Running both safety passes doubled work and dominated elapsed time.
    backend = "hybrid" if gpu_ok else "cpu"
    if not gpu_ok:
        backend = os.getenv("HDBSCAN_BACKEND", "cpu").strip().lower() or "cpu"
    auto_mono_cap, auto_leaf_cap = _outer_tile_defaults(args.input)
    outer_tile_m = float(os.getenv("TILE_M", "0"))
    outer_overlap_m = float(os.getenv("TILE_OVERLAP_M", "15"))
    outer_mono_cap = int(os.getenv("MAX_POINTS_MONO", str(auto_mono_cap)))
    outer_leaf_cap = int(os.getenv("MAX_TILE_POINTS", str(auto_leaf_cap)))
    outer_workers = int(os.getenv("TILE_WORKERS", "1"))
    outer_tile_desc = "auto" if outer_tile_m <= 0 else f"{outer_tile_m:g}"
    log(
        "GPU-aware outer tiling: "
        f"mono_cap={outer_mono_cap:,}; leaf_cap={outer_leaf_cap:,}; "
        f"tile_m={outer_tile_desc}; "
        f"workers={outer_workers}"
    )
    cpu_pipeline.run_pipeline_all_in_one(
        args.input,
        geo_output,
        source_class=CLS_DEFAULT,
        cars_target_class=args.cars_to,
        noise_target_class=args.noise_to,
        fast=False,
        reset_all=True,
        reset_to=CLS_DEFAULT,
        no_subsample=gpu_ok,
        auto_tile=True,
        force_tile=force_tile,
        tile_m=outer_tile_m,
        tile_overlap_m=outer_overlap_m,
        max_points_mono=outer_mono_cap,
        max_tile_points=outer_leaf_cap,
        tile_workers=outer_workers,
        hdbscan_backend=backend,
    )
    if gpu_ok:
        log_stats()
    if rf_model is not None:
        log(f"\nThree-etalon RF post-pass: {rf_model}")
        run_rf_tiled(
            geo_output,
            output,
            rf_model,
            max_tile_points=int(os.getenv("RF_MAX_TILE_POINTS", "0")),
            tile_m=float(os.getenv("RF_TILE_M", "0")),
            overlap_m=float(os.getenv("RF_OVERLAP_M", "8")),
            context_mode=rf_context_mode,
        )
        if not args.keep_geo:
            try:
                geo_output.unlink(missing_ok=True)
            except OSError as exc:
                log(f"Could not remove temporary geo LAS: {exc}")
    log(f"{VERSION} total wrapper time: {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
