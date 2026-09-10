"""Global RF context must reproduce monolithic features for any prediction batches."""
import sys
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import laspy
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
from etalon_rf_refiner import prediction_features,apply_refiner
import rf_scene_context as scene


class ContextTests(unittest.TestCase):
    def sample(self,n=2400,ground=True):
        rng=np.random.default_rng(5)
        x=rng.uniform(2040,2055,n);y=rng.uniform(3300,3312,n)
        z=rng.uniform(40,47,n)
        cls=rng.choice([2,3,5,6,7,91],n).astype(np.uint8)
        if not ground:cls[cls==2]=6
        intensity=rng.integers(0,65536,n).astype(float)
        return x,y,z,cls,intensity

    def check_case(self,data):
        expected=prediction_features(*data)
        with patch.object(scene,'available_ram_bytes',return_value=None):
            context=scene.SceneFeatures(*data,block=173)
        actual=context.features(slice(None))
        np.testing.assert_array_equal(actual,expected)
        for size in (17,73,901):
            chunked=np.concatenate([context.features(slice(i,i+size)) for i in range(0,len(data[0]),size)])
            np.testing.assert_array_equal(chunked,actual)
        shuffled=np.random.default_rng(8).permutation(len(data[0]))
        np.testing.assert_array_equal(context.features(shuffled),actual[shuffled])
        target=np.where(expected[:,0]>.7,91,6)
        model=RandomForestClassifier(n_estimators=9,max_depth=5,random_state=1,n_jobs=1).fit(expected,target)
        full=apply_refiner(data[3],actual,model)[0]
        batches=np.concatenate([apply_refiner(data[3][i:i+79],context.features(slice(i,i+79)),model)[0]
                                for i in range(0,len(full),79)])
        np.testing.assert_array_equal(full,batches)

    def test_matches_monolithic_and_batch_invariant(self):self.check_case(self.sample())
    def test_ground_fallback_matches_monolithic(self):self.check_case(self.sample(500,False))
    def test_no_intensity_matches(self):
        data=list(self.sample());data[-1]=None;self.check_case(data)
    def test_global_sampling_of_buildings_matches(self):
        n=250_300
        data=list(self.sample(n));data[3][:]=6
        # Supply enough ground and many high building points to exercise sampling.
        data[3][:150]=2;data[2][:150]=35;data[2][150:]=45
        expected=prediction_features(*data)
        with patch.object(scene,'available_ram_bytes',return_value=None):
            context=scene.SceneFeatures(*data,block=4096)
        np.testing.assert_array_equal(context.features(slice(None)),expected)
    def test_preflight_limits_context_allocations(self):
        with patch.object(scene,'available_ram_bytes',return_value=100):
            with self.assertRaisesRegex(MemoryError,'RF_SCENE_CONTEXT_RAM'):
                scene.SceneFeatures(*self.sample())

    def test_full_las_write_and_prediction_oom_retry(self):
        # Exercise real LAS serialization and retry after a prediction allocation
        # fails. The context and final point order/attributes must stay unchanged.
        x,y,z,cls,intensity=self.sample(2800)
        header=laspy.LasHeader(point_format=7,version='1.4')
        header.scales=np.array([.001,.001,.001])
        las=laspy.LasData(header)
        las.x=x;las.y=y;las.z=z;las.classification=cls
        las.intensity=intensity.astype(np.uint16)
        las.red=np.arange(len(x),dtype=np.uint16)
        las.gps_time=np.arange(len(x),dtype=float)+123456.75
        xyz=[np.asarray(las[d],dtype=float) for d in ('x','y','z')]
        features=prediction_features(*xyz,cls,intensity)
        target=np.where(features[:,0]>.7,91,6)
        model=RandomForestClassifier(n_estimators=9,max_depth=5,random_state=1,n_jobs=1).fit(features,target)
        expected=apply_refiner(cls,features,model)[0]
        real_apply=scene.apply_refiner
        attempts=[]
        def allocation_failure_once(*args,**kwargs):
            attempts.append(len(args[0]))
            if len(attempts)==1:raise MemoryError('injected prediction allocation failure')
            return real_apply(*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);source=folder/'input.las';output=folder/'output.las';weights=folder/'model.pkl'
            las.write(source)
            with weights.open('wb') as stream:pickle.dump({'model':model,'conf_min':.60},stream)
            with patch.object(scene,'available_ram_bytes',return_value=None),patch.object(scene,'apply_refiner',side_effect=allocation_failure_once):
                scene.run_rf_scene(source,output,weights,max_batch_points=2000)
            result=laspy.read(output)
            np.testing.assert_array_equal(result.classification,expected)
            for field in las.points.array.dtype.names:
                if field!='classification':
                    np.testing.assert_array_equal(result.points.array[field],las.points.array[field])
        self.assertEqual(attempts,[2000,1000,1000,800])


if __name__=='__main__':unittest.main(verbosity=2)
