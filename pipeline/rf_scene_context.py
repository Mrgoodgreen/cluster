"""RF features defined once for a whole geometry-classified scene.

This implements the existing monolithic feature definitions. Prediction batches
only bound temporary matrices; they do not redefine ground, voxel origins,
intensity normalization or the reference building sample.
"""
from __future__ import annotations

import gc
import pickle
import time
from pathlib import Path

import laspy
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree

from classify_tls import log
from classify_tls_all_in_one import write_las_atomic
from classify_tls_tiling import available_ram_bytes
from etalon_rf_refiner import apply_refiner


class SceneFeatures:
    def __init__(self, x, y, z, cls, intensity, *, block=1_000_000):
        if not len(x):
            raise ValueError('Cannot prepare RF context for an empty scene')
        self.x,self.y,self.z,self.cls,self.intensity=x,y,z,cls,intensity
        self.n=len(x); self.block=block
        self.origin=(float(x.min()),float(y.min()),float(z.min()))
        available=available_ram_bytes()
        peak=self.n*48+512*1024**2  # sort keys/order/copies and small working blocks
        if available is not None and peak>available*.75:
            raise MemoryError('RF_SCENE_CONTEXT_RAM: insufficient RAM for global voxel aggregation; '
                              'increase container RAM. No window-dependent fallback was applied.')
        log('[RF-SCENE] Preparing one ground grid and intensity scale')
        nx=int(np.floor(x.max()-self.origin[0]))+1
        ny=int(np.floor(y.max()-self.origin[1]))+1
        if available is not None and nx*ny*24>available*.20:
            raise MemoryError('RF_SCENE_GRID_RAM: scene extent too large for a dense 1m ground grid')
        grid=np.full(nx*ny,np.inf,dtype=np.float64)
        n_ground=int(np.count_nonzero(cls==2))
        self.ny=ny
        if n_ground<100:
            grid.fill(float(np.median(z)))
        else:
            for start,end in self.blocks():
                mask=cls[start:end]==2
                ix=np.floor(x[start:end][mask]-self.origin[0]).astype(np.int64)
                iy=np.floor(y[start:end][mask]-self.origin[1]).astype(np.int64)
                np.minimum.at(grid,ix*ny+iy,z[start:end][mask])
            grid[~np.isfinite(grid)]=float(np.median(z[cls==2]))
        self.ground=minimum_filter(grid.reshape(nx,ny),size=3,mode='nearest').ravel()
        del grid
        self.intensity_scale=self._intensity_scale()
        log('[RF-SCENE] Preparing the shared building reference')
        tall=np.empty(self.n,dtype=bool)
        for start,end in self.blocks():
            tall[start:end]=(cls[start:end]==6)&((z[start:end]-self.ground_at(x[start:end],y[start:end]))>=3.5)
        tidx=np.flatnonzero(tall)
        del tall
        if len(tidx)>250_000:
            tidx=np.random.default_rng(0).choice(tidx,250_000,replace=False)
        self.buildings=cKDTree(np.column_stack([x[tidx],y[tidx]])) if len(tidx) else None
        del tidx
        log('[RF-SCENE] Aggregating global 0.35m voxel features')
        keys=np.empty(self.n,dtype=np.int64)
        for start,end in self.blocks():
            keys[start:end]=self.keys_at(x[start:end],y[start:end],z[start:end])
        order=np.argsort(keys,kind='mergesort')
        sorted_keys=keys[order]
        del keys
        self.keys,starts,self.counts=np.unique(sorted_keys,return_index=True,return_counts=True)
        del sorted_keys
        zs=z[order]
        self.span=np.maximum.reduceat(zs,starts)-np.minimum.reduceat(zs,starts)
        del zs
        hs=np.empty(self.n,dtype=np.float64)
        for start,end in self.blocks():
            ids=order[start:end]
            hs[start:end]=z[ids]-self.ground_at(x[ids],y[ids])
        self.mean_height=np.add.reduceat(hs,starts)/self.counts
        del hs,order,starts
        gc.collect()
        log(f'[RF-SCENE] Context ready: {len(self.keys):,} voxel groups; intensity p95={self.intensity_scale:g}')

    def blocks(self):
        for start in range(0,self.n,self.block):
            yield start,min(start+self.block,self.n)

    def ground_at(self,x,y):
        ix=np.floor(x-self.origin[0]).astype(np.int64)
        iy=np.floor(y-self.origin[1]).astype(np.int64)
        return self.ground[ix*self.ny+iy]

    def keys_at(self,x,y,z):
        ix=np.floor((x-self.origin[0])/.35).astype(np.int64)
        iy=np.floor((y-self.origin[1])/.35).astype(np.int64)
        iz=np.floor((z-self.origin[2])/.35).astype(np.int64)
        # Keep the trained feature definition, including its existing hash scheme.
        return ix*73856093 ^ iy*19349663 ^ iz*83492791

    def _intensity_scale(self):
        if self.intensity is None:
            return 1.0
        hist=np.zeros(65536,dtype=np.int64)
        for start,end in self.blocks():
            values=self.intensity[start:end]
            if np.any(values<0) or np.any(values>65535) or np.any(values!=np.floor(values)):
                raise ValueError('RF-SCENE expects original LAS uint16 intensity values')
            hist+=np.bincount(values.astype(np.int64),minlength=65536)
        hist[0]=0
        n=int(hist.sum())
        if not n:return 1.0
        rank=(n-1)*.95; lo=int(np.floor(rank)); hi=int(np.ceil(rank))
        cumulative=hist.cumsum()
        low=np.searchsorted(cumulative,lo+1)
        high=np.searchsorted(cumulative,hi+1)
        fraction=rank-lo
        # Match NumPy's stable linear interpolation on either side of 0.5.
        value=(low+(high-low)*fraction if fraction<.5 else high-(high-low)*(1-fraction))
        return max(float(value),1e-6)

    def features(self,ids):
        x,y,z=self.x[ids],self.y[ids],self.z[ids]
        h=z-self.ground_at(x,y)
        group=np.searchsorted(self.keys,self.keys_at(x,y,z))
        distance=(self.buildings.query(np.column_stack([x,y]),k=1,workers=-1)[0]
                  if self.buildings is not None else np.full(len(x),50.0))
        intensity=(np.clip(self.intensity[ids]/self.intensity_scale,0,2.0)
                   if self.intensity is not None else np.zeros(len(x)))
        return np.column_stack([h,np.log1p(self.counts[group].astype(np.float64)),
            self.span[group],self.mean_height[group],intensity,np.clip(distance,0,20),
            self.cls[ids].astype(np.float64),(h>=.30).astype(float),
            (h>=2.20).astype(float),(h>=4.00).astype(float)])


def run_rf_scene(input_path: Path, output_path: Path, model_path: Path, *, max_batch_points=0):
    started=time.perf_counter()
    if max_batch_points<0:raise ValueError('max_batch_points must be non-negative')
    with laspy.open(str(input_path)) as reader:
        parent=reader.header.point_count*(reader.header.point_format.size+36)
    available=available_ram_bytes()
    if available is not None and parent>available*.8:
        raise MemoryError('RF_PARENT_EXCEEDS_RAM: cannot safely load the scene')
    las=laspy.read(str(input_path))
    if not len(las.points):
        write_las_atomic(las,output_path);return
    x,y,z=(np.asarray(las[d],dtype=np.float64) for d in ('x','y','z'))
    cls=np.asarray(las.classification,dtype=np.uint8).copy()
    intensity=np.asarray(las.intensity,dtype=np.float64)
    loaded=time.perf_counter()
    log(f'[RF-SCENE] Loaded {len(cls):,} points in {loaded-started:.1f}s')
    context=SceneFeatures(x,y,z,cls,intensity)
    prepared=time.perf_counter()
    with model_path.open('rb') as stream:
        blob=pickle.load(stream)
    model_loaded=time.perf_counter()
    result=cls.copy()
    # Spatial context is fixed. Shrinking only prediction batches cannot change it.
    batch=min(max_batch_points or 1_000_000,1_000_000)
    if batch<1:raise ValueError('max_batch_points must be non-negative')
    start=0;changed=0
    while start<len(cls):
        available=available_ram_bytes()
        if available is not None:
            batch=min(batch,max(1,int(available*.35/400)))
        end=min(start+batch,len(cls))
        failed=False
        try:
            features=context.features(slice(start,end))
            prediction,n=apply_refiner(cls[start:end],features,blob['model'],
                                      conf_min=float(blob.get('conf_min',.60)))
        except MemoryError:
            failed=True
        if failed:
            if 'features' in locals():del features
            gc.collect()
            if batch<=1000:raise MemoryError('RF_PREDICTION_RAM: minimum prediction batch failed')
            batch=max(1000,batch//2)
            log(f'[RF-SCENE] MemoryError: retry prediction batch={batch:,} with unchanged context')
            continue
        result[start:end]=prediction;changed+=n
        del features,prediction
        start=end
        log(f'[RF-SCENE] predicted {start:,}/{len(cls):,}')
    las.classification=result
    predicted=time.perf_counter()
    write_las_atomic(las,output_path)
    written=time.perf_counter()
    log(f'[RF-SCENE] Seconds: load={loaded-started:.1f}; context={prepared-loaded:.1f}; '
        f'model={model_loaded-prepared:.1f}; predict={predicted-model_loaded:.1f}; write={written-predicted:.1f}')
    log(f'[RF-SCENE] Done in {(written-started)/60:.1f} min; changed={changed:,}')
