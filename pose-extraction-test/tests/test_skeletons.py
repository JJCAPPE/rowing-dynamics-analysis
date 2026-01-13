import numpy as np

from rowing_pose.skeletons import H36M17_JOINT_NAMES, coco17_to_h36m17


def test_coco17_to_h36m17_shapes_and_values() -> None:
    T = 1
    X = np.zeros((T, 17, 3), dtype=np.float32)

    # COCO indices:
    # 0 nose, 5 lsho, 6 rsho, 11 lhip, 12 rhip, 13 lknee, 14 rknee, 15 lank, 16 rank
    X[0, 0] = (20.0, 20.0, 0.4)  # nose
    X[0, 5] = (10.0, 10.0, 0.5)  # lsho
    X[0, 6] = (30.0, 10.0, 0.9)  # rsho
    X[0, 11] = (10.0, 0.0, 0.8)  # lhip
    X[0, 12] = (30.0, 0.0, 0.6)  # rhip
    X[0, 13] = (10.0, -10.0, 0.7)  # lknee
    X[0, 14] = (30.0, -10.0, 0.3)  # rknee
    X[0, 15] = (10.0, -20.0, 0.2)  # lank
    X[0, 16] = (30.0, -20.0, 0.1)  # rank

    out = coco17_to_h36m17(X)
    assert out.shape == (T, 17, 3)
    assert len(H36M17_JOINT_NAMES) == 17

    pelvis = out[0, 0]
    thorax = out[0, 8]
    spine = out[0, 7]
    neck = out[0, 9]
    head = out[0, 10]

    # pelvis = avg(hips)
    assert np.allclose(pelvis[:2], (20.0, 0.0))
    assert np.isclose(pelvis[2], 0.7)  # (0.8+0.6)/2

    # thorax = avg(shoulders)
    assert np.allclose(thorax[:2], (20.0, 10.0))
    assert np.isclose(thorax[2], 0.7)  # (0.5+0.9)/2

    # spine = avg(pelvis, thorax)
    assert np.allclose(spine[:2], (20.0, 5.0))
    assert np.isclose(spine[2], 0.7)  # (0.7+0.7)/2

    # neck = avg(thorax, nose)
    assert np.allclose(neck[:2], (20.0, 15.0))
    assert np.isclose(neck[2], 0.55)  # (0.7+0.4)/2

    # head = nose
    assert np.allclose(head, X[0, 0])

