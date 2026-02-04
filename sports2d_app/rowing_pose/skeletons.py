from __future__ import annotations

from typing import Tuple

import numpy as np


COCO17_JOINT_NAMES: Tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


# Human3.6M 17-joint order commonly used by MotionBERT / VideoPose3D-style pipelines.
H36M17_JOINT_NAMES: Tuple[str, ...] = (
    "pelvis",  # 0
    "right_hip",  # 1
    "right_knee",  # 2
    "right_ankle",  # 3
    "left_hip",  # 4
    "left_knee",  # 5
    "left_ankle",  # 6
    "spine",  # 7
    "thorax",  # 8
    "neck",  # 9
    "head",  # 10
    "left_shoulder",  # 11
    "left_elbow",  # 12
    "left_wrist",  # 13
    "right_shoulder",  # 14
    "right_elbow",  # 15
    "right_wrist",  # 16
)


def coco17_to_h36m17(X: np.ndarray) -> np.ndarray:
    """Map COCO-17 keypoints to H36M-17 format expected by MotionBERT.

    Input/Output shape:
    - input:  (T, 17, 3)  (x, y, conf)
    - output: (T, 17, 3)  (x, y, conf)

    Missing H36M joints (pelvis/spine/thorax/neck) are constructed deterministically:
    - pelvis: average(left_hip, right_hip)
    - thorax: average(left_shoulder, right_shoulder)
    - spine: average(pelvis, thorax)
    - neck: average(thorax, nose)
    - head: nose
    """

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[1] != 17 or X.shape[2] != 3:
        raise ValueError(f"Expected shape (T,17,3). Got {X.shape}")

    T = X.shape[0]
    out = np.zeros((T, 17, 3), dtype=np.float32)

    nose = X[:, 0]
    lsho = X[:, 5]
    rsho = X[:, 6]
    lelb = X[:, 7]
    relb = X[:, 8]
    lwri = X[:, 9]
    rwri = X[:, 10]
    lhip = X[:, 11]
    rhip = X[:, 12]
    lkne = X[:, 13]
    rkne = X[:, 14]
    lank = X[:, 15]
    rank = X[:, 16]

    pelvis = 0.5 * (lhip + rhip)
    thorax = 0.5 * (lsho + rsho)
    spine = 0.5 * (pelvis + thorax)
    neck = 0.5 * (thorax + nose)
    head = nose

    out[:, 0] = pelvis
    out[:, 1] = rhip
    out[:, 2] = rkne
    out[:, 3] = rank
    out[:, 4] = lhip
    out[:, 5] = lkne
    out[:, 6] = lank
    out[:, 7] = spine
    out[:, 8] = thorax
    out[:, 9] = neck
    out[:, 10] = head
    out[:, 11] = lsho
    out[:, 12] = lelb
    out[:, 13] = lwri
    out[:, 14] = rsho
    out[:, 15] = relb
    out[:, 16] = rwri

    return out

