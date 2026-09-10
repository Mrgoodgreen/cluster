#!/usr/bin/env python3
"""GPU 0.0.1 environment and centered-HDBSCAN smoke test."""

from __future__ import annotations

import numpy as np

import gpu_backend


def main() -> None:
    print(gpu_backend.check_gpu())
    rng = np.random.default_rng(42)
    a = rng.normal((0.0, 0.0, 0.0), 0.05, size=(300, 3))
    b = rng.normal((3.0, 3.0, 1.0), 0.05, size=(300, 3))
    # Add large absolute coordinates to verify local centering.
    points = np.vstack([a, b]) + np.array([37_500_000.0, 6_200_000.0, 180.0])
    labels = gpu_backend._gpu_labels(points, min_cluster_size=25, min_samples=6, epsilon=0.40)
    clusters = np.unique(labels[labels >= 0])
    if len(clusters) < 2:
        raise RuntimeError(f"GPU HDBSCAN smoke test failed: clusters={clusters.tolist()}")
    print(f"Centered cuML HDBSCAN OK: {len(points)} points, {len(clusters)} clusters")


if __name__ == "__main__":
    main()
