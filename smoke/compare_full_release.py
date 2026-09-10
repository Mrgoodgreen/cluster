"""Compare full candidate/baseline LAS against GT, with a conservative release gate.

Gate is specified before the run: accuracy, ground precision/recall, useful
ground/building retention, car/noise removal and outside-training-crop accuracy.
Any regression prevents automatic release. Histograms alone cannot pass it.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import laspy
import numpy as np

LABELS={2:'ground',3:'grass',4:'medium_vegetation',5:'trees',6:'buildings',7:'noise',91:'vehicle'}


def metrics(m):
    total=int(m.sum());rows=m.sum(1);columns=m.sum(0)
    per_class={}
    for c,name in LABELS.items():
        precision=float(m[c,c]/columns[c]) if columns[c] else 0.0
        recall=float(m[c,c]/rows[c]) if rows[c] else None
        f1=(2*precision*recall/(precision+recall) if recall is not None and precision+recall else 0.0)
        per_class[name]=dict(precision=precision,recall=recall,f1=f1,gt_points=int(rows[c]),predicted_points=int(columns[c]))
    return dict(points=total,accuracy=float(m.trace()/total) if total else 0.0,
                classes=per_class,
                ground_to_removal=int(m[2,7]+m[2,91]),
                buildings_to_removal=int(m[6,7]+m[6,91]),
                vehicles_left_in_useful=int(rows[91]-m[91,7]-m[91,91]),
                noise_left_in_useful=int(rows[7]-m[7,7]-m[7,91]),
                vehicle_noise_in_ground=int(m[91,2]+m[7,2]))


def release_gate(old,new,out_old,out_new):
    checks={}
    checks['full_accuracy']=new['accuracy']>=old['accuracy']
    checks['outside_training_crop_accuracy']=out_new['accuracy']>=out_old['accuracy']
    for field in ('precision','recall'):
        a=old['classes']['ground'][field];b=new['classes']['ground'][field]
        checks['ground_'+field]=b>=a if a is not None and b is not None else a==b
    for field in ('ground_to_removal','buildings_to_removal','vehicles_left_in_useful',
                  'noise_left_in_useful','vehicle_noise_in_ground'):
        checks[field]=new[field]<=old[field]
    return dict(automatic_release_allowed=all(checks.values()),checks=checks,
                failed=[k for k,v in checks.items() if not v])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt',type=Path,required=True);p.add_argument('--old',type=Path,required=True)
    p.add_argument('--new',type=Path,required=True);p.add_argument('--training-crop-meta',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    meta=json.loads(args.training_crop_meta.read_text())
    matrices=[np.zeros((256,256),dtype=np.int64) for _ in range(4)]
    transitions=np.zeros((256,256),dtype=np.int64)
    seen=0;changed=0
    with laspy.open(args.gt) as gt,laspy.open(args.old) as old,laspy.open(args.new) as new:
        expected=gt.header.point_count
        for reader in (old,new):
            if reader.header.point_count!=expected:raise ValueError('Point count mismatch')
            if not np.array_equal(reader.header.scales,gt.header.scales) or not np.array_equal(reader.header.offsets,gt.header.offsets):
                raise ValueError('Coordinate encoding differs')
        for g in gt.chunk_iterator(500_000):
            a=old.read_points(len(g));b=new.read_points(len(g))
            if len(a)!=len(g) or len(b)!=len(g):raise ValueError('Truncated point data')
            for dimension in ('X','Y','Z'):
                if not np.array_equal(g[dimension],a[dimension]):raise ValueError('Baseline coordinate/order mismatch')
            if g.array.dtype!=b.array.dtype:raise ValueError('Candidate point record format changed')
            for field in g.array.dtype.names:
                if field=='classification':continue
                if np.ascontiguousarray(g.array[field]).tobytes()!=np.ascontiguousarray(b.array[field]).tobytes():
                    raise ValueError(f'Candidate changed original {field} values')
            truth=np.asarray(g.classification,dtype=np.int64)
            before=np.asarray(a.classification,dtype=np.int64);after=np.asarray(b.classification,dtype=np.int64)
            outside=(np.abs(np.asarray(g.x)-meta['cx'])>meta['half'])|(np.abs(np.asarray(g.y)-meta['cy'])>meta['half'])
            for matrix,pred,mask in ((matrices[0],before,slice(None)),(matrices[1],after,slice(None)),
                                      (matrices[2],before,outside),(matrices[3],after,outside)):
                matrix+=np.bincount(truth[mask]*256+pred[mask],minlength=65536).reshape(256,256)
            transitions+=np.bincount(before*256+after,minlength=65536).reshape(256,256)
            changed+=int(np.count_nonzero(before!=after));seen+=len(g)
            if seen%5_000_000==0:print(f'[COMPARE] {seen:,}/{expected:,}',flush=True)
        if seen!=expected:raise ValueError('Source is truncated')
    reports=[metrics(m) for m in matrices]
    report=dict(source=str(args.gt),baseline=str(args.old),candidate=str(args.new),
                integrity='all source point attributes except classification preserved; baseline XYZ matched',
                old=reports[0],new=reports[1],outside_training_crop_old=reports[2],outside_training_crop_new=reports[3],
                changed_points=changed,changed_fraction=changed/max(1,seen),
                gate=release_gate(*reports),
                confusion_matrices=[{str(i):{str(j):int(m[i,j]) for j in np.flatnonzero(m[i])}
                                     for i in np.flatnonzero(m.sum(1))} for m in matrices],
                transition={str(i):{str(j):int(transitions[i,j]) for j in np.flatnonzero(transitions[i])}
                            for i in np.flatnonzero(transitions.sum(1))},
                note='Outside crop is a held-out area of the same scan, not an independent new scene.')
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('old','new','gate','changed_fraction')},indent=2),flush=True)


if __name__=='__main__':main()
