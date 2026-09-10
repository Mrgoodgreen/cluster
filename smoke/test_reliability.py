"""Regression checks for memory planning, RF splitting and LAS completion."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import laspy
import numpy as np

CLUSTER=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(CLUSTER/'pipeline'),str(CLUSTER/'worker'),str(CLUSTER/'smoke'),str(CLUSTER/'manager')]
sys.path.insert(0,str(CLUSTER/'pipeline'/'GPU_0_0_1'))
import classify_tls_tiling as tiling
import classify_tls_rf_tiling as rf
import classify_tls_all_in_one as pipeline
import etalon_rf_refiner as refiner
import train_etalon_refiner as trainer
import gpu_backend
from las_result import InvalidLasResult, validate_result
from worker_agent import WorkerAgent
from app import services
from app.models import Base, Subtask
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session


class MemoryTests(unittest.TestCase):
    def memory(self, files):
        def read(path,*a,**kw):
            if str(path) in files:return files[str(path)]
            raise FileNotFoundError(path)
        with patch.object(Path,'read_text',read):
            return tiling.available_ram_bytes()

    def test_v2_ancestor_limit(self):
        self.assertEqual(self.memory({
            '/proc/meminfo':'MemAvailable: 100000 kB',
            '/proc/self/cgroup':'0::/jobs/worker',
            '/proc/self/mountinfo':'1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw',
            '/sys/fs/cgroup/jobs/worker/memory.max':'max',
            '/sys/fs/cgroup/jobs/memory.max':'6000',
            '/sys/fs/cgroup/jobs/memory.current':'4500',
            '/sys/fs/cgroup/memory.max':'max',
        }),1500)

    def test_v1_limit_and_exhausted_zero(self):
        files={'/proc/meminfo':'MemAvailable: 100000 kB',
            '/proc/self/cgroup':'5:memory:/docker/id',
            '/proc/self/mountinfo':'1 0 0:1 /docker/id /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory',
            '/sys/fs/cgroup/memory/memory.limit_in_bytes':'1000',
            '/sys/fs/cgroup/memory/memory.usage_in_bytes':'1200'}
        self.assertEqual(self.memory(files),0)

    def test_host_without_cgroup(self):
        self.assertEqual(self.memory({'/proc/meminfo':'MemAvailable: 12345 kB'}),12345*1024)

    def test_rf_budget_does_not_use_user_cap_over_ram(self):
        with patch.object(rf,'available_ram_bytes',return_value=2*1024**3):
            self.assertLess(rf.rf_point_budget(50_000_000,100_000_000),5_000_000)
        with patch.object(rf,'available_ram_bytes',return_value=0):
            with self.assertRaisesRegex(MemoryError,'RF_RAM_BUDGET'):
                rf.rf_point_budget(5_000_000,1)


class RFTests(unittest.TestCase):
    def test_gpu_error_releases_traceback_before_smaller_retry(self):
        refs=[]
        def labels(pts,*a):
            if not refs:
                allocation=np.ones(1024)
                refs.append(weakref.ref(allocation))
                raise MemoryError('out of memory')
            self.assertIsNone(refs[0]())
            return np.zeros(len(pts),dtype=np.int32)
        x=np.arange(30.);y=np.ones(30);z=np.zeros(30)
        with patch.object(gpu_backend,'_gpu_labels',new=labels),patch.object(gpu_backend,'_free_gpu'):
            list(gpu_backend._rect_clusters(np.arange(30),x,y,z,(0,30,0,15),min_cluster_size=5,min_samples=3,epsilon=.1))
        self.assertIsNone(refs[0]())

    def test_features_ground_comes_from_prediction(self):
        xyz=np.arange(4,dtype=float); pred=np.array([2,6,7,91],dtype=np.uint8)
        with patch.object(refiner,'dtm_from_ground',return_value=xyz) as dtm,patch.object(refiner,'build_features',return_value='features'):
            self.assertEqual(refiner.prediction_features(xyz,xyz,xyz,pred,None),'features')
            np.testing.assert_array_equal(dtm.call_args.args[3],[True,False,False,False])

    def _window(self, cap, fail_once=False):
        x,y=np.meshgrid(np.arange(31.),np.arange(31.));x=x.ravel();y=y.ravel()
        z=np.zeros_like(x);cls=np.full(len(x),2,dtype=np.uint8)
        calls=[]
        def classify(tx,ty,tz,tc,intensity,model):
            calls.append(len(tx))
            if fail_once and len(calls)==1:raise MemoryError('injected')
            return tc.copy(),0
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp)/'model';model.write_bytes(b'fake')
            with patch.object(rf,'refine_classification',side_effect=classify),patch.object(rf,'rf_point_budget',return_value=cap):
                chunks=list(rf.refine_window_adaptive(x,y,z,cls,None,tiling.TileRect(0,31,0,31),0,30,30,cap,model))
        indices=np.concatenate([i for i,c in chunks])
        np.testing.assert_array_equal(np.sort(indices),np.arange(len(x)))
        return calls

    def test_actual_cap_and_exact_core_coverage(self):
        self.assertLessEqual(max(self._window(100)),100)

    def test_memoryerror_retries_smaller_windows(self):
        calls=self._window(2000,True)
        self.assertGreater(len(calls),1)
        self.assertLess(max(calls[1:]),calls[0])

    def test_irreducible_context_fails_instead_of_shrinking_overlap(self):
        a=np.zeros(500);c=np.full(500,2,dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp)/'model';model.write_bytes(b'fake')
            with patch.object(rf,'rf_point_budget',return_value=100):
                with self.assertRaisesRegex(MemoryError,'RF_CONTEXT_EXCEEDS_BUDGET'):
                    list(rf.refine_window_adaptive(a,a,a,c,None,tiling.TileRect(0,1,0,1),8,0,0,100,model))


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.src=Path(self.tmp.name)/'in.las'; self.dst=Path(self.tmp.name)/'out.las'
        las=laspy.LasData(laspy.LasHeader(point_format=7,version='1.4'))
        las.x=np.arange(200.)/10;las.y=np.arange(200.)/20;las.z=np.zeros(200)
        las.classification=np.full(200,1,dtype=np.uint8);las.write(self.src)
        las.classification=np.full(200,2,dtype=np.uint8);las.write(self.dst)

    def test_valid_legacy_and_receipt_cache(self):
        self.assertTrue(validate_result(self.src,self.dst))
        receipt=self.dst.with_name(self.dst.name+'.tls-complete.json')
        self.assertEqual(json.loads(receipt.read_text())['provenance']['source'],'legacy_verified')
        with patch('las_result.laspy.open',side_effect=AssertionError('cached output was reread')):
            self.assertTrue(validate_result(self.src,self.dst))

    def test_truncated_file_not_skipped_or_removed(self):
        self.dst.write_bytes(self.dst.read_bytes()[:-100])
        before=self.dst.read_bytes()
        with self.assertRaises(InvalidLasResult):validate_result(self.src,self.dst)
        self.assertEqual(self.dst.read_bytes(),before)

    def test_changed_coordinates_invalidates_receipt(self):
        validate_result(self.src,self.dst)
        las=laspy.read(self.dst);las.X[50]+=1;las.write(self.dst)
        with self.assertRaisesRegex(InvalidLasResult,'coordinates'):
            validate_result(self.src,self.dst)

    def test_source_change_invalidates_receipt(self):
        validate_result(self.src,self.dst)
        las=laspy.read(self.src);las.Y[100]+=1;las.write(self.src)
        with self.assertRaises(InvalidLasResult):validate_result(self.src,self.dst)

    def test_missing_result_is_only_false_case(self):
        self.dst.unlink();self.assertFalse(validate_result(self.src,self.dst))

    def test_worker_cancel_during_verification(self):
        w=WorkerAgent('http://unused','test','test',sys.executable,Path('fake.py'))
        w._current_input=self.src;w._validation_subtask_id=1
        with patch.object(w,'heartbeat',return_value=True):
            with self.assertRaises(InterruptedError):w._check_output_exists_with_retry(self.dst)
        self.assertEqual(w._cancel_reason,'manager')
        self.assertFalse(self.dst.with_name(self.dst.name+'.tls-complete.json').exists())

    def test_training_rejects_different_coordinate_order(self):
        las=laspy.read(self.dst);las.X=np.asarray(las.X)[::-1];las.write(self.dst)
        with self.assertRaisesRegex(ValueError,'order'):
            trainer.load_pair(self.src,self.dst)

    def test_manager_delegates_existing_output_validation_and_keeps_fifo(self):
        engine=create_engine('sqlite:///:memory:')
        self.addCleanup(engine.dispose)
        Base.metadata.create_all(engine)
        root=Path(self.tmp.name)
        (root/'input1').mkdir();(root/'input2').mkdir();(root/'output1').mkdir()
        (root/'input1'/'a.las').write_bytes(self.src.read_bytes())
        (root/'input2'/'b.las').write_bytes(self.src.read_bytes())
        (root/'output1'/'a.las').write_bytes(b'nonempty but broken')
        with patch.dict(services.settings.storage_roots,{'test':root},clear=True),Session(engine) as db:
            first=services.create_task(db,'test/input1','test/output1')
            second=services.create_task(db,'test/input2','test/output2')
            rows=db.scalars(select(Subtask).order_by(Subtask.id)).all()
            self.assertEqual([r.status for r in rows],['pending','pending'])
            a=services.claim_subtask(db,'worker1','host1')
            b=services.claim_subtask(db,'worker2','host2')
            self.assertEqual(a.task_id,first.id)
            self.assertEqual(b.task_id,second.id)
            self.assertEqual(a.status,'processing')
            services.create_task(db,'test/input1','test/output1')
            self.assertIsNone(services.claim_subtask(db,'worker3','host3'),
                              'two workers must not write to the same output concurrently')

    def test_mono_memory_is_released_before_tile_fallback(self):
        refs=[]
        def failed(*a,**kw):
            allocated=np.ones(1024)
            refs.append(weakref.ref(allocated))
            raise MemoryError('injected mono failure')
        def tiled(*a,**kw):
            self.assertIsNone(refs[0](), 'failed mono arrays retained by traceback')
        # Avoid unittest.mock retaining the exception/traceback in call results.
        with patch.object(pipeline,'_run_pipeline_monolithic',new=failed),patch.object(tiling,'run_pipeline_tiled',new=tiled),patch.object(tiling,'should_use_tiles',return_value=(False,'test')):
            pipeline.run_pipeline_all_in_one(self.src,self.dst,auto_tile=True,hdbscan_backend='cpu')

    def test_parent_preflight_refuses_read_when_container_memory_exhausted(self):
        with patch.object(tiling,'available_ram_bytes',return_value=1),patch.object(tiling.laspy,'read',side_effect=AssertionError('read called before budget check')):
            with self.assertRaisesRegex(MemoryError,'TILE_PARENT_EXCEEDS_RAM'):
                tiling.run_pipeline_tiled(self.src,self.dst,run_arrays=lambda *a,**kw:None)


if __name__=='__main__':
    unittest.main(verbosity=2)
