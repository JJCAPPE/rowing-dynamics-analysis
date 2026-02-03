
import unittest
from unittest.mock import MagicMock
import sys
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import types

# Mock Sports2D package before importing our module
sports2d_pkg = types.ModuleType("Sports2D")
sports2d_mod = types.ModuleType("Sports2D.Sports2D")
sports2d_pkg.Sports2D = sports2d_mod
sys.modules["Sports2D"] = sports2d_pkg
sys.modules["Sports2D.Sports2D"] = sports2d_mod
from Sports2D import Sports2D

# Add repo root to path so we can import rowing_pose
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rowing_pose.pose2d_sports2d import infer_pose2d_sports2d, Pose2DResult

class TestSports2DIntegration(unittest.TestCase):
    def setUp(self):
        Sports2D.process = MagicMock()
        self.tmp_dir = tempfile.mkdtemp()
        self.video_path = Path(self.tmp_dir) / "test_video.mp4"
        self.video_path.touch()
        self.out_npz = Path(self.tmp_dir) / "pose2d.npz"
        
        # Create a dummy stabilization file
        self.stab_npz = Path(self.tmp_dir) / "stabilization.npz"
        T = 10
        A = np.zeros((T, 2, 3), dtype=np.float32)
        # Identity transforms
        for i in range(T):
            A[i] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        
        np.savez(self.stab_npz, A=A, fps=30.0, width=1920, height=1080)
        
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        
    def test_infer_pose2d_sports2d(self):
        # Mock Sports2D.process side effects (creating TRC file)
        def side_effect(config):
            base = config.get("base", {})
            out_dir = Path(base.get("result_dir"))
            video_input = base.get("video_input")
            if isinstance(video_input, list):
                video_input = video_input[0] if video_input else ""
            vid_name = Path(str(video_input)).stem
            # Create the expected folder structure
            res_dir = out_dir # Sports2D might create a subdir or output directly depending on version
            # The wrapper looks specifically for *.trc recursively
            
            trc_file = res_dir / f"{vid_name}_trc.trc"
            
            # Create dummy TRC content
            # Header
            header = ["PathFileType\t4\t(X/Y/Z)\ttest.trc", 
                      "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames", 
                      "30.00\t30.00\t10\t17\tm\t30.00\t1\t10", 
                      "Frame#\tTime\tNose_X\tNose_Y\tL_Shoulder_X\tL_Shoulder_Y\tUnused_X", 
                      "1\t0.00\t100.0\t100.0\t200.0\t200.0\t0.0",
                      "2\t0.03\t101.0\t101.0\t201.0\t201.0\t0.0",
                      # ... simulate 10 frames or just enough
                      ]
            # Fill remaining frames
            for i in range(3, 11):
                 header.append(f"{i}\t{i*0.03}\t{100+i}.0\t{100+i}.0\t{200+i}.0\t{200+i}.0\t0.0")

            with open(trc_file, 'w') as f:
                f.write("\n".join(header))
                
        Sports2D.process.side_effect = side_effect
        
        result = infer_pose2d_sports2d(
            video_path=self.video_path,
            stabilization_npz=self.stab_npz,
            out_npz=self.out_npz,
            model_name="rtmpose",
            mode="balanced"
        )
        
        # Verify Sports2D called
        Sports2D.process.assert_called_once()
        
        # Verify result
        self.assertIsInstance(result, Pose2DResult)
        self.assertEqual(result.J2d_px.shape, (10, 17, 3))
        self.assertAlmostEqual(result.fps, 30.0)
        
        # Verified Nose (index 0)
        # Frame 0: 100, 100. Stabilization is identity. Result should be 100, 100.
        self.assertAlmostEqual(result.J2d_px[0, 0, 0], 100.0)
        self.assertAlmostEqual(result.J2d_px[0, 0, 1], 100.0)
        
        # Verify Left Shoulder (index 5)
        # Frame 0: 200, 200
        self.assertAlmostEqual(result.J2d_px[0, 5, 0], 200.0)
        
        # Verify output file
        self.assertTrue(self.out_npz.exists())

if __name__ == "__main__":
    unittest.main()
