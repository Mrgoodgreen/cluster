import sys
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import laspy
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from compare_full_release import metrics,release_gate,main


class GateTests(unittest.TestCase):
    def baseline(self):
        m=np.zeros((256,256),dtype=np.int64)
        for c in (2,3,5,6,7,91):m[c,c]=100
        m[2,7]=10;m[6,91]=10;m[91,2]=10;m[7,6]=10
        return m

    def test_identical_can_release(self):
        a=metrics(self.baseline())
        self.assertTrue(release_gate(a,a,a,a)['automatic_release_allowed'])

    def test_improvement_can_release(self):
        old=self.baseline();new=old.copy()
        for a,b in ((2,7),(6,91),(91,2),(7,6)):
            new[a,a]+=new[a,b];new[a,b]=0
        self.assertTrue(release_gate(metrics(old),metrics(new),metrics(old),metrics(new))['automatic_release_allowed'])

    def test_accuracy_gain_cannot_hide_facade_loss(self):
        old=self.baseline();new=old.copy()
        new[91,91]+=10;new[91,2]=0
        new[6,6]-=1;new[6,7]+=1
        result=release_gate(metrics(old),metrics(new),metrics(old),metrics(new))
        self.assertFalse(result['automatic_release_allowed'])
        self.assertIn('buildings_to_removal',result['failed'])

    def test_seen_crop_gain_cannot_hide_outside_regression(self):
        a=metrics(self.baseline());b={**a,'accuracy':a['accuracy']-.001}
        self.assertFalse(release_gate(a,a,a,b)['automatic_release_allowed'])

    def run_comparison(self,corrupt_attribute=False):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            gt=laspy.LasData(laspy.LasHeader(point_format=7,version='1.4'))
            gt.x=np.arange(12,dtype=float);gt.y=np.zeros(12);gt.z=np.zeros(12)
            gt.classification=np.tile([2,3,5,6,7,91],2).astype(np.uint8)
            gt.intensity=np.arange(12,dtype=np.uint16)
            gt.write(folder/'gt.las');gt.write(folder/'new.las')
            gt.classification[0]=7;gt.classification[3]=91
            gt.write(folder/'old.las')
            if corrupt_attribute:
                new=laspy.read(folder/'new.las');new.intensity[4]=123
                new.write(folder/'new.las')
            (folder/'crop.json').write_text(json.dumps({'cx':0,'cy':0,'half':.5}))
            argv=['compare','--gt',str(folder/'gt.las'),'--old',str(folder/'old.las'),
                  '--new',str(folder/'new.las'),'--training-crop-meta',str(folder/'crop.json'),
                  '--output',str(folder/'report.json')]
            with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):main()
            return json.loads((folder/'report.json').read_text())

    def test_streaming_las_comparison_and_outside_crop(self):
        report=self.run_comparison()
        self.assertTrue(report['gate']['automatic_release_allowed'])
        self.assertEqual(report['changed_points'],2)
        self.assertEqual(report['new']['accuracy'],1.0)
        self.assertEqual(report['outside_training_crop_new']['points'],11)

    def test_modified_point_attribute_cannot_release(self):
        with self.assertRaisesRegex(ValueError,'intensity'):
            self.run_comparison(corrupt_attribute=True)


if __name__=='__main__':unittest.main(verbosity=2)
