#!/usr/bin/env python3
"""Dedicated all-in-one TLS pipeline: all useful geometry preserved, removable junk to class 7."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import laspy
import numpy as np

from classify_tls import (
    CFG,
    CLS_BUILDING,
    CLS_DEFAULT,
    CLS_DIRTY,
    CLS_GROUND,
    CLS_HIGH_VEG,
    CLS_LOW_NOISE,
    CLS_LOW_VEG,
    CLS_MED_VEG,
    CLS_VEHICLE,
    HDBSCAN_BACKEND_CPU,
    HDBSCAN_BACKEND_CUML,
    HDBSCAN_BACKEND_HYBRID,
    attach_near_buildings_xy,
    attach_facade_band_near_buildings,
    car_volume_peel_classes,
    class_histogram,
    classify_ground_robust,
    cluster_objects,
    detect_buildings,
    detect_walls_and_fences,
    extract_cars_from_classes,
    extract_cars_by_xy_blobs,
    cpu_finalize_cars_in_veg,
    demote_false_vehicles,
    restore_low_object_band_to_ground,
    log,
    log_stage_timings,
    ground_dtm,
    peel_car_volume,
    peel_elevated_from_ground,
    peel_elevated_islands_from_ground,
    peel_elevated_junk_from_ground_class,
    peel_low_car_residuals_from_ground,
    peel_noise_from_ground,
    recover_landscape_from_noise,
    remove_sparse_airborne,
    restore_asphalt_from_noise,
    restore_building_from_noise,
    restore_building_columns_from_noise,
    restore_building_skin,
    restore_curbs_from_noise,
    demote_pedestrians_from_building,
    demote_orphan_building,
    restore_flat_paving_xy,
    restore_near_dtm,
    robust_road_dtm,
)


def demote_low_building_to_ground(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    ground_z: np.ndarray,
    max_h: float = 0.28,
    flat_span: float = 0.12,
    cell: float = 0.75,
) -> int:
    """Move near-DTM flat patches wrongly labeled as building back to ground.

    Catches asphalt / sidewalk swallowed by building dilation without touching
    real facade bases that sit in taller, non-flat columns.
    """
    cand = (cls == CLS_BUILDING) & ((z - ground_z) <= max_h) & ((z - ground_z) >= -0.08)
    if not cand.any():
        return 0

    idx = np.where(cand)[0]
    xmin, ymin = float(x[idx].min()), float(y[idx].min())
    ix = np.floor((x[idx] - xmin) / cell).astype(np.int64)
    iy = np.floor((y[idx] - ymin) / cell).astype(np.int64)
    ny = int(iy.max()) + 1
    lin = ix * ny + iy

    order = np.argsort(lin, kind="mergesort")
    lin_s = lin[order]
    z_s = z[idx[order]]
    uniq, starts, counts = np.unique(lin_s, return_index=True, return_counts=True)

    flat = np.zeros(len(uniq), dtype=bool)
    for i, (s, c) in enumerate(zip(starts, counts)):
        if c < 10:
            continue
        chunk = z_s[s : s + c]
        if float(chunk.max() - chunk.min()) <= flat_span:
            flat[i] = True
    if not flat.any():
        return 0

    cell_flat = np.zeros(int(uniq.max()) + 1, dtype=bool)
    cell_flat[uniq[flat]] = True
    take = idx[cell_flat[lin]]
    cls[take] = CLS_GROUND
    return int(len(take))


def write_las_atomic(las: laspy.LasData, output_path: Path, *, retries: int = 8) -> None:
    """Write via temp file + replace, with retries for SMB locks / brief disconnects.

    Does not change classification — only how the final LAS hits the share.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.writing")
    last_exc: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            las.write(str(tmp_path))
            os.replace(str(tmp_path), str(output_path))
            return
        except OSError as exc:
            last_exc = exc
            log(
                f"  write retry {attempt}/{retries} for {output_path.name}: "
                f"{type(exc).__name__}: {exc}"
            )
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            time.sleep(min(30.0, 1.5 ** attempt))
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except OSError:
        pass
    if last_exc is not None:
        raise last_exc
    raise OSError(f"Failed to write {output_path}")



def resolve_hdbscan_backend(requested: str | None = None) -> str:
    """GPU-first default; env HDBSCAN_BACKEND overrides; CPU is explicit fallback."""
    raw = (requested or os.getenv("HDBSCAN_BACKEND", "") or HDBSCAN_BACKEND_CUML).strip().lower()
    if raw in (HDBSCAN_BACKEND_CPU, HDBSCAN_BACKEND_CUML, HDBSCAN_BACKEND_HYBRID):
        return raw
    return HDBSCAN_BACKEND_CUML


def run_pipeline_all_in_one(
    input_path: Path,
    output_path: Path,
    source_class: int = CLS_DIRTY,
    cars_target_class: int = CLS_VEHICLE,
    noise_target_class: int = CLS_LOW_NOISE,
    fast: bool = False,
    reset_all: bool = False,
    reset_to: int = CLS_DEFAULT,
    no_subsample: bool = False,
    rf_refiner: Path | None = None,
    auto_tile: bool = True,
    force_tile: bool = False,
    tile_m: float = 0.0,
    tile_overlap_m: float = 15.0,
    max_points_mono: int = 0,
    max_tile_points: int = 0,
    tile_workers: int = 0,
    min_tile_m: float = 12.0,
    hdbscan_backend: str | None = None,
) -> None:
    """Classify a LAS. Large clouds auto-fallback to overlapping XY tiles.

    ``tile_m`` / ``max_tile_points`` / ``max_points_mono`` of 0 mean auto from RAM.
    HDBSCAN defaults to cuML GPU; CPU only on failure or ``--hdbscan-backend cpu``.
    """
    hdbscan_backend = resolve_hdbscan_backend(hdbscan_backend)
    from classify_tls_tiling import should_use_tiles, run_pipeline_tiled

    # Cheap header peek for preflight (full read happens inside mono/tiled).
    try:
        with laspy.open(str(input_path)) as reader:
            n_hdr = int(reader.header.point_count)
    except Exception:
        n_hdr = -1

    if force_tile:
        use_tiles, reason = True, "forced tiling (--force-tile)"
    elif auto_tile:
        use_tiles, reason = should_use_tiles(
            max(n_hdr, 0),
            max_points_mono=max_points_mono,
            with_rf=rf_refiner is not None,
        )
    else:
        use_tiles, reason = False, "auto_tile disabled"

    log(f"[PREFLIGHT] points≈{n_hdr:,}; tiling={'YES' if use_tiles else 'NO'} — {reason}")
    if use_tiles:
        run_pipeline_tiled(
            input_path,
            output_path,
            run_arrays=_run_pipeline_arrays,
            source_class=source_class,
            cars_target_class=cars_target_class,
            noise_target_class=noise_target_class,
            fast=fast,
            reset_all=reset_all,
            reset_to=reset_to,
            no_subsample=no_subsample,
            rf_refiner=rf_refiner,
            tile_m=tile_m,
            overlap_m=tile_overlap_m,
            max_tile_points=max_tile_points,
            min_tile_m=min_tile_m,
            tile_workers=tile_workers,
            hdbscan_backend=hdbscan_backend,
        )
        return

    try:
        _run_pipeline_monolithic(
            input_path,
            output_path,
            source_class=source_class,
            cars_target_class=cars_target_class,
            noise_target_class=noise_target_class,
            fast=fast,
            reset_all=reset_all,
            reset_to=reset_to,
            no_subsample=no_subsample,
            rf_refiner=rf_refiner,
            hdbscan_backend=hdbscan_backend,
        )
        return
    except MemoryError:
        if not auto_tile:
            raise
        log("[PREFLIGHT] MemoryError in monolithic run — falling back to tiled mode")
    # Leave the exception scope first: its traceback retains failed mono arrays.
    __import__('gc').collect()
    run_pipeline_tiled(
        input_path, output_path, run_arrays=_run_pipeline_arrays,
        source_class=source_class, cars_target_class=cars_target_class,
        noise_target_class=noise_target_class, fast=fast,
        reset_all=reset_all, reset_to=reset_to, no_subsample=no_subsample,
        rf_refiner=rf_refiner, tile_m=tile_m, overlap_m=tile_overlap_m,
        max_tile_points=max_tile_points, min_tile_m=min_tile_m,
        tile_workers=tile_workers, hdbscan_backend=hdbscan_backend,
    )


def _run_pipeline_monolithic(
    input_path: Path,
    output_path: Path,
    source_class: int = CLS_DIRTY,
    cars_target_class: int = CLS_VEHICLE,
    noise_target_class: int = CLS_LOW_NOISE,
    fast: bool = False,
    reset_all: bool = False,
    reset_to: int = CLS_DEFAULT,
    no_subsample: bool = False,
    rf_refiner: Path | None = None,
    hdbscan_backend: str | None = None,
) -> None:
    t0 = time.time()
    if fast:
        log("FAST mode: voxel-HDBSCAN + vectorized car peel")
    if no_subsample:
        log("NO-SUBSAMPLE mode: HDBSCAN uses all candidate points (high RAM)")

    log(f"Reading {input_path} ...")
    las = laspy.read(str(input_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    cls = np.asarray(las.classification, dtype=np.uint8)
    inten = np.asarray(las.intensity, dtype=np.float64) if hasattr(las, "intensity") else None
    class_histogram(cls, "Input")

    if reset_all:
        log(f"\n[--reset-all] Resetting ALL {len(cls):,} points to class {reset_to}...")
        cls[:] = np.uint8(reset_to)
        source_class = reset_to
        class_histogram(cls, "After reset")

    try:
        cls = _run_pipeline_arrays(
            x, y, z, cls,
            intensity=inten,
            source_class=source_class,
            cars_target_class=cars_target_class,
            noise_target_class=noise_target_class,
            fast=fast,
            no_subsample=no_subsample,
            rf_refiner=rf_refiner,
            hdbscan_backend=hdbscan_backend,
        )
    except RuntimeError as exc:
        if str(exc).startswith("No points in source class"):
            log(f"{exc}. Exit.")
            sys.exit(1)
        raise

    log(f"\nWriting {output_path} ...")
    las.classification = cls
    write_las_atomic(las, output_path)
    log(f"Done in {(time.time() - t0) / 60:.1f} min.")


def _run_pipeline_arrays(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    *,
    intensity: np.ndarray | None = None,
    source_class: int = CLS_DIRTY,
    cars_target_class: int = CLS_VEHICLE,
    noise_target_class: int = CLS_LOW_NOISE,
    fast: bool = False,
    no_subsample: bool = False,
    rf_refiner: Path | None = None,
    hdbscan_backend: str | None = None,
) -> np.ndarray:
    """Classify in-memory arrays; returns classification (mutates cls)."""
    t0 = time.time()
    timings: dict[str, float] = {}
    hdbscan_backend = resolve_hdbscan_backend(hdbscan_backend)

    def stage_start() -> float:
        return time.perf_counter()

    def stage_end(name: str, started: float) -> None:
        timings[name] = time.perf_counter() - started

    # None = no HDBSCAN point-cap (for high-RAM machines)
    car_max_pts: int | None = None if no_subsample else CFG.car_max_points
    obj_max_pts: int | None = None if no_subsample else CFG.obj_max_points
    if hdbscan_backend != HDBSCAN_BACKEND_CPU:
        log(f"HDBSCAN backend: {hdbscan_backend} (CPU = fallback on GPU failure)")

    work = cls == source_class
    log(f"\nProcessing source class {source_class}: {int(work.sum()):,} points")
    if not work.any():
        raise RuntimeError(f"No points in source class {source_class}")

    # A: robust ground on all source points, no early low-point/noise.
    ts = stage_start()
    log("\n[A] Ground from full cloud...")
    idx = np.where(work)[0]
    wx, wy, wz = x[idx], y[idx], z[idx]
    gmask_local, _ = classify_ground_robust(
        wx, wy, wz,
        cell=CFG.ground_cell,
        opening_size=CFG.ground_opening,
        tol_above=CFG.ground_tol_above,
        tol_below=CFG.ground_tol_below,
    )
    gidx = idx[gmask_local]
    cls[gidx] = CLS_GROUND
    work[gidx] = False
    log(f"  robust ground: {len(gidx):,}")
    n_peel = peel_elevated_from_ground(
        x, y, z, cls,
        cell=CFG.ground_cell,
        opening_size=CFG.ground_opening,
        max_above=CFG.elevated_peel_above,
    )
    if n_peel:
        work[cls == source_class] = True
        work[cls == CLS_GROUND] = False
    ground_z = robust_road_dtm(x, y, z, cell=CFG.ground_cell, opening_size=CFG.ground_opening)
    log(f"  elevated peel from ground: {n_peel:,}")
    stage_end("A_ground", ts)

    # B: building bulk early (walls deferred until after cars — etalon cars matched wall band).
    ts = stage_start()
    log("\n[B] Detect building bulk early (walls after cars)...")
    n_build = detect_buildings(
        x, y, z, cls, ground_z,
        cell=CFG.building_cell,
        min_top_h=CFG.building_min_top_h,
        min_span=CFG.building_min_span,
        min_count=CFG.building_min_count,
        assign_min_h=CFG.building_base_assign_min_h,
        dilate=CFG.building_dilate,
        protect_classes=(cars_target_class,),
    )
    n_demote = demote_low_building_to_ground(x, y, z, cls, ground_z, max_h=0.30)
    work[cls == CLS_BUILDING] = False
    work[cls == CLS_GROUND] = False
    work[cls == CLS_HIGH_VEG] = False  # canopy rejected from building
    log(
        f"  building points: {n_build:,} "
        f"(total class 6: {(cls == CLS_BUILDING).sum():,}; "
        f"high veg so far: {(cls == CLS_HIGH_VEG).sum():,}); "
        f"demoted asphalt from 6: {n_demote:,}"
    )
    stage_end("B_building", ts)

    # C: height layers only on the remainder.
    ts = stage_start()
    log("\n[C] Height layers on non-ground/non-building...")
    rem = np.where(work)[0]
    if len(rem):
        h = z[rem] - ground_z[rem]
        m = h <= 0.08
        cls[rem[m]] = CLS_GROUND
        m = (h > 0.08) & (h < CFG.curb_max)
        cls[rem[m]] = CLS_LOW_VEG
        m = (h >= CFG.curb_max) & (h < CFG.obj_max)
        cls[rem[m]] = CLS_MED_VEG
        m = h >= CFG.obj_max
        cls[rem[m]] = CLS_HIGH_VEG
        work[rem] = False
    log(f"  layered points: {len(rem):,}")
    stage_end("C_layers", ts)

    # D: cars → Vehicle; people / ghosts → Low Point (Noise).
    # Facade attach (C2) runs AFTER cars so parked vehicles near walls stay extractable.
    ts = stage_start()
    log(f"\n[D] Objects from class 4 -> vehicle {cars_target_class} / noise {noise_target_class}...")
    car_boxes = cluster_objects(
        x, y, z, cls, ground_z,
        car_target_class=cars_target_class,
        ped_target_class=noise_target_class,
        ghost_target_class=noise_target_class,
        fast=fast,
        voxel=CFG.fast_voxel,
        backend=hdbscan_backend,
        max_points=obj_max_pts,
    )
    stage_end("D_cluster4", ts)

    # E: cars from ground + veg → Vehicle.
    ts = stage_start()
    log(f"\n[E] Cars from ground/veg + low building band -> class {cars_target_class}...")
    e_boxes = extract_cars_from_classes(
        x, y, z, cls, ground_z,
        source_classes=(CLS_GROUND, CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
        car_target_class=cars_target_class,
        min_height=0.08,
        max_height=2.50,
        max_points=car_max_pts,
        fast=fast,
        voxel=CFG.fast_voxel,
        backend=hdbscan_backend,
        stage="E",
    )
    e_bld: list = []
    if CFG.extract_cars_from_building:
        e_bld = extract_cars_from_classes(
            x, y, z, cls, ground_z,
            source_classes=(CLS_BUILDING,),
            car_target_class=cars_target_class,
            min_height=0.45,
            max_height=2.40,
            max_points=car_max_pts,
            fast=fast,
            voxel=CFG.fast_voxel,
            backend=hdbscan_backend,
            stage="E_bld",
            strict=True,
        )
    car_boxes += e_boxes + e_bld
    # Volume peel only from veg/ground-derived boxes — building boxes are already strict cores.
    n_vol = peel_car_volume(
        x, y, z, cls, ground_z, e_boxes, cars_target_class,
        peel_classes=car_volume_peel_classes(),
        xy_pad=CFG.car_xy_pad,
        min_h=CFG.car_min_h,
        z_top_pad=CFG.car_z_top_pad,
        dilate_cells=CFG.car_peel_dilate,
        exclude_near_building_m=CFG.peel_exclude_near_building_m,
    )
    # Tight peel around strict building-car boxes (veg only, small pad).
    n_vol_b = peel_car_volume(
        x, y, z, cls, ground_z, e_bld, cars_target_class,
        peel_classes=(CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
        xy_pad=0.25,
        min_h=0.20,
        z_top_pad=0.25,
        dilate_cells=1,
        exclude_near_building_m=CFG.peel_exclude_near_building_m,
    )
    n_dv = demote_false_vehicles(x, y, z, cls, ground_z, vehicle_class=cars_target_class)
    # XY-blob fallback: recover cars HDBSCAN left in med_veg / low building band.
    # Do NOT exclude near building — etalon cars park against facades.
    blob_boxes = extract_cars_by_xy_blobs(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
        car_target_class=cars_target_class,
        protect_near_building_m=0.0,
    )
    blob_bld = extract_cars_by_xy_blobs(
        x, y, z, cls, ground_z,
        source_classes=(CLS_BUILDING,),
        car_target_class=cars_target_class,
        protect_near_building_m=0.0,
        min_count=150,
        max_count=100_000,
        min_h=0.40,
        max_h=2.25,
        min_width=1.05,
        reject_under_tall=True,
    )
    car_boxes += blob_boxes + blob_bld
    n_vol_blob = peel_car_volume(
        x, y, z, cls, ground_z, blob_boxes + blob_bld, cars_target_class,
        peel_classes=(CLS_MED_VEG,),
        xy_pad=0.30,
        min_h=0.20,
        z_top_pad=0.25,
        dilate_cells=1,
        exclude_near_building_m=CFG.peel_exclude_near_building_m,
    )
    n_low = restore_low_object_band_to_ground(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, cars_target_class),
        max_h=0.20,
    )
    log(
        f"  car volume peel: {n_vol:,}+{n_vol_b:,}+blob{n_vol_blob:,}; "
        f"false vehicle→ground: {n_dv:,}; low-band→ground: {n_low:,}"
    )
    # Walls/fences AFTER cars so vehicle bodies are not labeled as class 6.
    n_walls = detect_walls_and_fences(
        x, y, z, cls, ground_z,
        protect_classes=(cars_target_class,),
    )
    log(f"  walls/fences after cars: {n_walls:,} (class 6 now {(cls == CLS_BUILDING).sum():,})")
    stage_end("E_cars", ts)

    # C2: facade skirts after vehicles locked — avoid pulling cars into class 6.
    ts = stage_start()
    log("\n[C2] Post-car facade / fence attach...")
    n_pre_noise = restore_building_from_noise(
        x, y, z, cls, ground_z,
        noise_class=noise_target_class,
        xy_radius=0.22,
        min_h=CFG.curb_max,
        max_h=2.80,
    )
    n_pre_band = attach_facade_band_near_buildings(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG, noise_target_class),
        band_min=CFG.curb_max,
        band_max=2.80,
        xy_radius=0.22,
    )
    log(f"  post-car facade restore: noise {n_pre_noise:,}; band attach {n_pre_band:,}")
    stage_end("C2_facade", ts)

    # F: aggressively restore flat paving (incl. from building bleed).
    ts = stage_start()
    log("\n[F] Restore paving / asphalt early...")
    n_rest = restore_asphalt_from_noise(
        x, y, z, cls,
        source_classes=(noise_target_class, CLS_LOW_VEG, CLS_MED_VEG),
        ground_class=CLS_GROUND,
        xy_radius=CFG.asphalt_restore_xy_radius,
        z_tol=CFG.asphalt_restore_z_tol,
        ground_z=ground_z,
        max_h_above_dtm=CFG.asphalt_restore_max_h,
    )
    n_dtm = restore_near_dtm(
        x, y, z, cls, ground_z,
        source_classes=(noise_target_class, CLS_LOW_VEG, CLS_MED_VEG),
        max_h=0.22,
        flat_span=0.10,
        cell=0.50,
    )
    n_pave = restore_flat_paving_xy(
        x, y, z, cls, ground_z,
        source_classes=(noise_target_class, CLS_LOW_VEG, CLS_MED_VEG),
        cell=1.0,
        max_h=0.28,
        max_z_span=0.12,
        min_count=14,
        min_near_frac=0.85,
    )
    n_demote_f = demote_low_building_to_ground(x, y, z, cls, ground_z, max_h=0.28)
    log(
        f"  restore paving: asphalt {n_rest:,}; near-DTM {n_dtm:,}; "
        f"flat tiles {n_pave:,}; demote bld {n_demote_f:,}"
    )
    stage_end("F_paving", ts)

    # G: reinforce facades carefully — skip car-height band when restoring from noise.
    ts = stage_start()
    log("\n[G] Reinforce facade skins (no low building re-detect)...")
    n_skin = restore_building_skin(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, noise_target_class),
        xy_radius=0.14,
        min_h=2.80,
        max_h=25.0,
    )
    n_skin_lo = restore_building_skin(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, noise_target_class),
        xy_radius=0.10,
        min_h=0.40,
        max_h=0.85,
    )
    # skip car-height band for loose attach from noise already handled by attach curb/obj gap
    n_facade = attach_near_buildings_xy(
        x, y, z, cls,
        ground_z=ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
        building_class=CLS_BUILDING,
        xy_radius=0.22,
        curb_max=0.40,
        obj_max=2.80,
    )
    n_demote_g = demote_low_building_to_ground(x, y, z, cls, ground_z, max_h=0.28)
    n_orph_g, n_orph_gn = demote_orphan_building(
        x, y, z, cls, ground_z,
        target_class=noise_target_class,
        cell=1.25,
        min_top_h=3.20,
        min_span=2.20,
        roof_top_h=5.0,
        keep_dilate=2,
        demote_elevated_to_noise=CFG.demote_orphan_building_to_noise,
    )
    n_peds_g = demote_pedestrians_from_building(
        x, y, z, cls, ground_z, target_class=noise_target_class,
    )
    log(
        f"  facade skin hi/lo: {n_skin:,}/{n_skin_lo:,}; attach: {n_facade:,}; "
        f"demote flat: {n_demote_g:,}; orphan->g/7: {n_orph_g:,}/{n_orph_gn:,}; "
        f"peds->7: {n_peds_g:,}"
    )
    stage_end("G_building", ts)

    # H: stronger sparse cleanup for veg noise / wires.
    ts = stage_start()
    log("\n[H] Late sparse cleanup to Low Point (Noise)...")
    n_veg = remove_sparse_airborne(
        x, y, z, cls,
        source_classes=(CLS_MED_VEG, CLS_HIGH_VEG),
        target_class=noise_target_class,
        cell=CFG.sparse_veg_cell,
        min_count=CFG.sparse_veg_min,
    )
    n_bld_sparse = remove_sparse_airborne(
        x, y, z, cls,
        source_classes=(CLS_BUILDING,),
        target_class=noise_target_class,
        cell=CFG.sparse_bld_cell,
        min_count=CFG.sparse_bld_min,
        ground_z=ground_z,
        min_h=6.0,
    )
    # second pass: hanging wires / multipath above object band
    veg_hi = (cls == CLS_HIGH_VEG) | (cls == CLS_MED_VEG)
    hanging = veg_hi & ((z - ground_z) > 2.5)
    hang_cls = cls.copy()
    hang_cls[~hanging] = 0
    n_wire = remove_sparse_airborne(
        x, y, z, hang_cls,
        source_classes=(CLS_MED_VEG, CLS_HIGH_VEG),
        target_class=noise_target_class,
        cell=0.15,
        min_count=5,
    )
    cls[hang_cls == noise_target_class] = noise_target_class
    log(f"  sparse veg: {n_veg:,}; sparse building (airborne): {n_bld_sparse:,}; hanging wires: {n_wire:,}")
    # restore tall facade columns (full height in core cells; ring only above cars)
    n_cols_h = restore_building_columns_from_noise(
        x, y, z, cls, ground_z,
        noise_class=noise_target_class,
        cell=0.75,
        min_h=0.40,
        max_h=25.0,
        dilate=1,
        tall_top_h=3.50,
        car_band_max=2.80,
    )
    log(f"  facade columns restored from noise: {n_cols_h:,}")
    n_curbs_h = restore_curbs_from_noise(
        x, y, z, cls, ground_z, noise_class=noise_target_class,
    )
    log(f"  curbs restored from noise: {n_curbs_h:,}")
    stage_end("H_sparse", ts)

    # I: limited landscape recovery, then cars again (incl. building).
    ts = stage_start()
    log("\n[I] Limited landscape recovery + cars from building...")
    n_land = recover_landscape_from_noise(
        x, y, z, cls, ground_z,
        noise_class=noise_target_class,
        cell=0.18,
        min_count=12,
        min_h=0.25,
        max_h=1.80,
    )
    boxes_i = extract_cars_from_classes(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, CLS_HIGH_VEG),
        car_target_class=cars_target_class,
        min_height=0.08,
        max_points=car_max_pts,
        fast=fast,
        voxel=CFG.fast_voxel,
        backend=hdbscan_backend,
        stage="I",
    )
    boxes_i_bld: list = []
    # Building-car extract only in stage E (strict). Re-running on leftovers
    # mostly pulls facade slabs after true cars are gone.
    car_boxes += boxes_i + boxes_i_bld
    n_vol_i = peel_car_volume(
        x, y, z, cls, ground_z, boxes_i, cars_target_class,
        peel_classes=car_volume_peel_classes(),
        xy_pad=CFG.car_xy_pad,
        min_h=CFG.car_min_h,
        z_top_pad=CFG.car_z_top_pad,
        dilate_cells=CFG.car_peel_dilate,
        exclude_near_building_m=CFG.peel_exclude_near_building_m,
    )
    demote_false_vehicles(x, y, z, cls, ground_z, vehicle_class=cars_target_class)
    n_pave_i = restore_flat_paving_xy(
        x, y, z, cls, ground_z,
        source_classes=(noise_target_class, CLS_LOW_VEG, CLS_MED_VEG, CLS_BUILDING),
        cell=1.0,
        max_h=0.28,
        max_z_span=0.12,
        min_count=14,
        min_near_frac=0.85,
    )
    n_cols_i = restore_building_columns_from_noise(
        x, y, z, cls, ground_z,
        noise_class=noise_target_class,
        cell=0.75,
        min_h=0.40,
        max_h=25.0,
        dilate=1,
        tall_top_h=3.50,
        car_band_max=2.80,
    )
    n_orph_i, n_orph_in = demote_orphan_building(
        x, y, z, cls, ground_z,
        target_class=noise_target_class,
        cell=1.25,
        min_top_h=3.20,
        min_span=2.20,
        roof_top_h=5.0,
        keep_dilate=2,
        demote_elevated_to_noise=CFG.demote_orphan_building_to_noise,
    )
    n_peds_i = demote_pedestrians_from_building(
        x, y, z, cls, ground_z, target_class=noise_target_class,
    )
    log(
        f"  recover veg: {n_land:,}; car peel: {n_vol_i:,}; flat tiles: {n_pave_i:,}; "
        f"facade cols: {n_cols_i:,}; orphan->g/7: {n_orph_i:,}/{n_orph_in:,}; "
        f"peds->7: {n_peds_i:,}"
    )
    stage_end("I_recover", ts)

    # J: finalize — facade restore first, ground peel last (RS10: no pre-restore ground sparse).
    ts = stage_start()
    log("\n[J] Final facade restore + ground cleanup...")
    n_off = n_gs = n_elev = n_near = n_isl = 0
    n_near2 = n_isl2 = n_isl3 = n_off3 = n_gs3 = 0
    n_elev_f = 0
    n_rec_skirt = n_rec_band = n_rec_asph = n_rec_curbs = 0

    n_rest_j = restore_asphalt_from_noise(
        x, y, z, cls,
        source_classes=(noise_target_class, CLS_LOW_VEG, CLS_MED_VEG),
        ground_class=CLS_GROUND,
        xy_radius=CFG.asphalt_restore_xy_radius,
        z_tol=CFG.asphalt_restore_z_tol,
        ground_z=ground_z,
        max_h_above_dtm=CFG.asphalt_restore_max_h,
    )
    n_dtm_j = restore_near_dtm(
        x, y, z, cls, ground_z,
        source_classes=(noise_target_class, CLS_LOW_VEG, CLS_MED_VEG),
        max_h=0.18,
        flat_span=0.12,
        cell=0.50,
    )
    n_pave_j = restore_flat_paving_xy(
        x, y, z, cls, ground_z,
        source_classes=(noise_target_class, CLS_LOW_VEG, CLS_MED_VEG),
        cell=1.0,
        max_h=0.28,
        max_z_span=0.12,
        min_count=12,
        min_near_frac=0.82,
    )
    n_demote_j = demote_low_building_to_ground(x, y, z, cls, ground_z, max_h=0.28)
    # restore facade in tall columns (smart); skin skips car-height band
    n_build_j = restore_building_from_noise(
        x, y, z, cls, ground_z,
        noise_class=noise_target_class,
        xy_radius=0.25,
        min_h=2.80,
        max_h=25.0,
    )
    n_build_j_lo = restore_building_from_noise(
        x, y, z, cls, ground_z,
        noise_class=noise_target_class,
        xy_radius=0.12,
        min_h=0.40,
        max_h=0.85,
    )
    n_skin_j = restore_building_skin(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, noise_target_class),
        xy_radius=0.14,
        min_h=2.80,
        max_h=25.0,
    )
    n_skin_j_lo = restore_building_skin(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, noise_target_class),
        xy_radius=0.10,
        min_h=0.40,
        max_h=0.85,
    )
    n_cols_j = restore_building_columns_from_noise(
        x, y, z, cls, ground_z,
        noise_class=noise_target_class,
        cell=0.75,
        min_h=0.40,
        max_h=25.0,
        dilate=1,
        tall_top_h=3.50,
        car_band_max=2.80,
    )
    # final car pass — peel cars out of building
    boxes_j = extract_cars_from_classes(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG),
        car_target_class=cars_target_class,
        min_height=0.10,
        max_points=car_max_pts,
        fast=fast,
        voxel=CFG.fast_voxel,
        backend=hdbscan_backend,
        stage="J",
    )
    boxes_j_bld: list = []
    car_boxes += boxes_j + boxes_j_bld
    n_vol_j = peel_car_volume(
        x, y, z, cls, ground_z, boxes_j, cars_target_class,
        peel_classes=car_volume_peel_classes(),
        xy_pad=CFG.car_xy_pad,
        min_h=CFG.car_min_h,
        z_top_pad=CFG.car_z_top_pad,
        dilate_cells=CFG.car_peel_dilate,
        exclude_near_building_m=CFG.peel_exclude_near_building_m,
    )
    demote_false_vehicles(x, y, z, cls, ground_z, vehicle_class=cars_target_class)
    # Low wheel/bumper tips still sitting in ground beside class 7.
    if CFG.peel_near_car_ground:
        n_near2 = peel_low_car_residuals_from_ground(
            x, y, z, cls, ground_z,
            seed_class=cars_target_class,
            target_class=cars_target_class,
            boxes=car_boxes,
            xy_radius=CFG.car_residual_xy,
            min_h=CFG.car_residual_min_h,
            max_h=CFG.car_residual_max_h,
            seed_min_h=CFG.car_residual_seed_h,
        )
    if CFG.peel_ground_islands:
        n_isl2 = peel_elevated_islands_from_ground(
            x, y, z, cls,
            target_class=noise_target_class,
            cell=CFG.island_cell,
            min_h=CFG.island_min_h,
            max_h=CFG.island_max_h,
            min_count=CFG.island_min_count,
            max_length=CFG.island_max_length,
            max_width=CFG.island_max_width,
            dtm_cell=CFG.ground_noise_cell,
            protect_near_building_m=CFG.peel_exclude_near_building_m,
            protect_near_building_h=CFG.ground_peel_protect_near_building_h,
            ground_z=ground_z,
        )
    # restore facade again AFTER car peel via tall-core columns (cars stay in ring rule)
    n_cols_j2 = restore_building_columns_from_noise(
        x, y, z, cls, ground_z,
        noise_class=noise_target_class,
        cell=0.75,
        min_h=0.40,
        max_h=25.0,
        dilate=1,
        tall_top_h=3.50,
        car_band_max=2.80,
    )
    n_skin_j2 = restore_building_skin(
        x, y, z, cls, ground_z,
        source_classes=(noise_target_class,),
        xy_radius=0.12,
        min_h=2.80,
        max_h=25.0,
    )
    n_orph_j, n_orph_jn = demote_orphan_building(
        x, y, z, cls, ground_z,
        target_class=noise_target_class,
        cell=1.25,
        min_top_h=3.20,
        min_span=2.20,
        roof_top_h=5.0,
        keep_dilate=2,
        demote_elevated_to_noise=CFG.demote_orphan_building_to_noise,
    )
    # one more paving restore after car peel (car peel can leave flat tiles in 7)
    n_pave_j2 = restore_flat_paving_xy(
        x, y, z, cls, ground_z,
        source_classes=(noise_target_class, CLS_LOW_VEG, CLS_MED_VEG),
        cell=1.0,
        max_h=0.28,
        max_z_span=0.12,
        min_count=12,
        min_near_frac=0.82,
    )
    n_rest_j2 = restore_asphalt_from_noise(
        x, y, z, cls,
        source_classes=(noise_target_class, CLS_LOW_VEG),
        ground_class=CLS_GROUND,
        xy_radius=1.0,
        z_tol=CFG.asphalt_restore_z_tol,
        ground_z=ground_z,
        max_h_above_dtm=CFG.asphalt_restore_max_h,
    )
    # Final pass AFTER restores — wheel prints / litter that asphalt restore may put back.
    if CFG.peel_ground_islands:
        n_isl3 = peel_elevated_islands_from_ground(
            x, y, z, cls,
            target_class=noise_target_class,
            cell=CFG.island_cell,
            min_h=CFG.island_min_h,
            max_h=CFG.island_max_h,
            min_count=CFG.island_min_count,
            max_length=CFG.island_max_length,
            max_width=CFG.island_max_width,
            dtm_cell=CFG.ground_noise_cell,
            protect_near_building_m=CFG.peel_exclude_near_building_m,
            protect_near_building_h=CFG.ground_peel_protect_near_building_h,
            ground_z=ground_z,
        )
    if CFG.peel_ground_noise_finalize:
        local_ground_z3 = ground_dtm(x, y, z, cls == CLS_GROUND, cell=CFG.ground_noise_cell)
        n_off3, n_gs3 = peel_noise_from_ground(
            x, y, z, cls, local_ground_z3,
            target_class=noise_target_class,
            max_above=CFG.ground_noise_max_above,
            max_below=CFG.ground_noise_max_below,
            sparse_cell=CFG.ground_sparse_cell,
            sparse_min_count=CFG.ground_sparse_min,
            sparse_enabled=CFG.ground_sparse_cleanup,
            protect_near_building_m=CFG.peel_exclude_near_building_m,
            protect_near_building_h=CFG.ground_peel_protect_near_building_h,
        )
    n_elev3 = 0
    if CFG.peel_elevated_junk_from_ground:
        n_elev3 = peel_elevated_junk_from_ground_class(
            x, y, z, cls, ground_z,
            target_class=noise_target_class,
            protect_near_building_m=CFG.peel_exclude_near_building_m,
            protect_near_building_h=CFG.ground_peel_protect_near_building_h,
        )
    # Recover facade skirts / paving wrongly peeled near buildings (RS10 thinned data).
    n_rec_skirt = restore_building_from_noise(
        x, y, z, cls, ground_z,
        noise_class=noise_target_class,
        xy_radius=0.22,
        min_h=0.12,
        max_h=2.80,
    )
    n_rec_band = attach_facade_band_near_buildings(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, CLS_GROUND, noise_target_class),
        band_min=CFG.curb_max,
        band_max=2.80,
        xy_radius=0.22,
    )
    n_rec_asph = restore_asphalt_from_noise(
        x, y, z, cls,
        source_classes=(noise_target_class,),
        ground_class=CLS_GROUND,
        xy_radius=0.90,
        z_tol=CFG.asphalt_restore_z_tol,
        ground_z=ground_z,
        max_h_above_dtm=CFG.asphalt_restore_max_h,
    )
    n_rec_curbs = restore_curbs_from_noise(
        x, y, z, cls, ground_z, noise_class=noise_target_class,
    )
    n_peds_j = demote_pedestrians_from_building(
        x, y, z, cls, ground_z, target_class=noise_target_class,
    )
    # final low-band asphalt restore (etalon ground often labeled med_veg by layers)
    n_low_j = restore_low_object_band_to_ground(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG, cars_target_class, noise_target_class),
        max_h=0.22,
    )
    # late XY-blob pass on leftovers after facade restores
    late_blobs = extract_cars_by_xy_blobs(
        x, y, z, cls, ground_z,
        source_classes=(CLS_LOW_VEG, CLS_MED_VEG),
        car_target_class=cars_target_class,
    )
    late_blobs += extract_cars_by_xy_blobs(
        x, y, z, cls, ground_z,
        source_classes=(CLS_BUILDING,),
        car_target_class=cars_target_class,
        min_count=150,
        max_count=100_000,
        min_h=0.40,
        max_h=2.25,
        min_width=1.05,
        reject_under_tall=True,
    )
    demote_false_vehicles(x, y, z, cls, ground_z, vehicle_class=cars_target_class)
    log(
        f"  elev junk: {n_elev3:,}; near-car {n_near + n_near2:,}; "
        f"islands {n_isl2 + n_isl3:,}; "
        f"ground re-peel {n_off3:,}/{n_gs3:,}; "
        f"recover skirt/band/asph/curb {n_rec_skirt:,}/{n_rec_band:,}/{n_rec_asph:,}/{n_rec_curbs:,}; "
        f"peds->7 {n_peds_j:,}; "
        f"asphalt {n_rest_j + n_rest_j2:,}; "
        f"near-DTM {n_dtm_j:,}; flat tiles {n_pave_j + n_pave_j2:,}; "
        f"demote bld {n_demote_j:,}; facade {n_build_j + n_build_j_lo:,}; "
        f"skin {n_skin_j + n_skin_j_lo + n_skin_j2:,}; facade cols {n_cols_j + n_cols_j2:,}; "
        f"car peel {n_vol_j:,}; orphan->g/7 {n_orph_j:,}/{n_orph_jn:,}; "
        f"low-band→g {n_low_j:,}; late-blob boxes {len(late_blobs)}"
    )
    stage_end("J_finalize", ts)


    # K: CPU safety sweep after full-GPU clustering (cuml only; hybrid already CPU on E/I2).
    if hdbscan_backend == HDBSCAN_BACKEND_CUML:
        ts = stage_start()
        log("\n[K] CPU finalize: cars in vegetation (post-GPU safety pass)...")
        n_fin = cpu_finalize_cars_in_veg(
            x, y, z, cls, ground_z, cars_target_class,
            fast=fast, voxel=CFG.fast_voxel,
        )
        log(f"  CPU finalize peel: {n_fin:,}")
        stage_end("K_cpu_finalize", ts)

    if rf_refiner is not None:
        ts = stage_start()
        log(f"\n[RF] Etalon RandomForest refiner: {rf_refiner}")
        from etalon_rf_refiner import refine_classification

        cls, n_rf = refine_classification(x, y, z, cls, intensity, Path(rf_refiner))
        log(f"  RF overrides: {n_rf:,}")
        stage_end("RF_refiner", ts)

    class_histogram(cls, "Output")
    log_stage_timings(timings)
    log(f"Done in {(time.time() - t0) / 60:.1f} min.")
    return cls


def main() -> None:
    p = argparse.ArgumentParser(description="TLS all-in-one classifier (single-source-class mode)")
    p.add_argument("input", type=Path, help="Input LAS/LAZ")
    p.add_argument("-o", "--output", type=Path, help="Output LAS/LAZ")
    p.add_argument(
        "--source-class",
        type=int,
        default=1,
        help="Single source class to process (default 1; ignored with --reset-all)",
    )
    p.add_argument(
        "--reset-all",
        action="store_true",
        help="Reset ALL points to one class first, then classify the whole cloud",
    )
    p.add_argument(
        "--reset-to",
        type=int,
        default=CLS_DEFAULT,
        help="Class used by --reset-all (default 1 = Default)",
    )
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
    p.add_argument("--fast", action="store_true", help="Use voxel-HDBSCAN fast path where supported")
    p.add_argument(
        "--hdbscan-backend",
        choices=[HDBSCAN_BACKEND_CPU, HDBSCAN_BACKEND_CUML, HDBSCAN_BACKEND_HYBRID],
        default=None,
        help="HDBSCAN: cuml=GPU-first (default), cpu=baseline, hybrid=GPU except E/I2. Env: HDBSCAN_BACKEND",
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        help=f"Alias for --hdbscan-backend {HDBSCAN_BACKEND_CUML}",
    )
    p.add_argument(
        "--no-subsample",
        action="store_true",
        help="Disable HDBSCAN point-cap subsample (use all candidates; needs high RAM, e.g. 128 GB)",
    )
    p.add_argument(
        "--rf-refiner",
        type=Path,
        default=None,
        help="Optional etalon-trained RF pickle (see etalon_rf_refiner.py / train_etalon_refiner.py)",
    )
    p.add_argument(
        "--force-tile",
        action="store_true",
        help="Always classify via overlapping XY tiles (ignore RAM preflight)",
    )
    p.add_argument(
        "--no-tile",
        action="store_true",
        help="Disable auto tiling (monolithic only; may OOM on huge LAS)",
    )
    p.add_argument(
        "--tile-m",
        type=float,
        default=0.0,
        help="Core tile size in metres (0=auto from RAM + density)",
    )
    p.add_argument(
        "--tile-overlap-m",
        type=float,
        default=15.0,
        help="Tile overlap in metres (cars/facades across borders)",
    )
    p.add_argument(
        "--max-points-mono",
        type=int,
        default=0,
        help="If point count exceeds this, force tiled mode (0=auto from RAM, cap 70M)",
    )
    p.add_argument(
        "--max-tile-points",
        type=int,
        default=0,
        help="Split a tile further when heavier than this (0=auto from MemAvailable)",
    )
    p.add_argument(
        "--tile-workers",
        type=int,
        default=0,
        help="Parallel leaf tiles (0=auto from RAM/CPU, 1=sequential, max 4)",
    )
    p.add_argument(
        "--min-tile-m",
        type=float,
        default=12.0,
        help="Stop adaptive XY splits below this core span (metres)",
    )
    args = p.parse_args()
    if args.force_tile and args.no_tile:
        p.error("Use either --force-tile or --no-tile, not both")
    if args.gpu and args.hdbscan_backend not in (None, HDBSCAN_BACKEND_CUML):
        p.error("Use either --gpu or --hdbscan-backend, not both")
    backend = HDBSCAN_BACKEND_CUML if args.gpu else args.hdbscan_backend
    if args.tile_workers < 0:
        p.error("--tile-workers must be >= 0")
    if args.min_tile_m <= 0:
        p.error("--min-tile-m must be > 0")
    if args.tile_m > 0 and args.min_tile_m > args.tile_m:
        p.error("--min-tile-m must be <= --tile-m when --tile-m is set")

    out = args.output or args.input.with_name(args.input.stem + "_allinone.las")
    run_pipeline_all_in_one(
        args.input,
        out,
        source_class=args.source_class,
        cars_target_class=args.cars_to,
        noise_target_class=args.noise_to,
        fast=args.fast,
        reset_all=args.reset_all,
        reset_to=args.reset_to,
        no_subsample=args.no_subsample,
        rf_refiner=args.rf_refiner,
        auto_tile=not args.no_tile,
        force_tile=args.force_tile,
        tile_m=args.tile_m,
        tile_overlap_m=args.tile_overlap_m,
        max_points_mono=args.max_points_mono,
        max_tile_points=args.max_tile_points,
        tile_workers=args.tile_workers,
        min_tile_m=args.min_tile_m,
        hdbscan_backend=backend,
    )


if __name__ == "__main__":
    main()
