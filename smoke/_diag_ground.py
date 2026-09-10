import numpy as np
import laspy
from pathlib import Path

path = Path(r"c:\Users\roman\Documents\Cursor\3dit1_00811cr\3dit1_00811cr1_2.las")
print("open", path.name, "size_gb", round(path.stat().st_size / 1e9, 2))
las = laspy.read(str(path))
cls = np.asarray(las.classification)
x = np.asarray(las.x, dtype=np.float64)
y = np.asarray(las.y, dtype=np.float64)
z = np.asarray(las.z, dtype=np.float64)
print("n", f"{len(cls):,}")
vals, cnts = np.unique(cls, return_counts=True)
for v, c in zip(vals, cnts):
    print(f"  class {int(v)}: {c:,}")

g = cls == 2
cell = 1.0
gx, gy, gz = x[g], y[g], z[g]
xmin, ymin = gx.min(), gy.min()
ix = np.floor((gx - xmin) / cell).astype(np.int64)
iy = np.floor((gy - ymin) / cell).astype(np.int64)
ny = int(iy.max()) + 1
lin = ix * ny + iy
order = np.argsort(lin)
lin_s = lin[order]
z_s = gz[order]
uniq, starts = np.unique(lin_s, return_index=True)
mins = np.minimum.reduceat(z_s, starts)
grid = np.full(int(uniq.max()) + 1, np.nan)
grid[uniq] = mins
h = gz - grid[lin]
print(
    "ground h above cell-min: p50=%.3f p90=%.3f p99=%.3f max=%.3f"
    % (np.percentile(h, 50), np.percentile(h, 90), np.percentile(h, 99), h.max())
)
for thr in (0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.35):
    print(f"  ground pts with h>{thr:.2f}: {(h > thr).sum():,}")

# also vs morphological opening proxy: use percentile low of neighbors - skip
# denser cell 0.5
cell2 = 0.5
ix2 = np.floor((gx - xmin) / cell2).astype(np.int64)
iy2 = np.floor((gy - ymin) / cell2).astype(np.int64)
ny2 = int(iy2.max()) + 1
lin2 = ix2 * ny2 + iy2
order2 = np.argsort(lin2)
uniq2, starts2 = np.unique(lin2[order2], return_index=True)
mins2 = np.minimum.reduceat(gz[order2], starts2)
grid2 = np.full(int(uniq2.max()) + 1, np.nan)
grid2[uniq2] = mins2
h2 = gz - grid2[lin2]
print(
    "ground h cell0.5: p50=%.3f p90=%.3f p99=%.3f"
    % (np.percentile(h2, 50), np.percentile(h2, 90), np.percentile(h2, 99))
)
for thr in (0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25):
    print(f"  ground pts h0.5>{thr:.2f}: {(h2 > thr).sum():,}")
