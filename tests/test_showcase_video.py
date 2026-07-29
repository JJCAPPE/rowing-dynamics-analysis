from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.render_showcase_video import (
    _phase_state,
    _prepare_force_curves,
    _project_pose,
)


def test_phase_state_expands_each_half_cycle_to_full_progress() -> None:
    assert _phase_state(3, 0.25, True) == ("DRIVE", 0.5)
    assert _phase_state(3, 0.75, False) == ("RECOVERY", 0.5)
    assert _phase_state(-1, float("nan"), False) == ("READY", 0.0)


def test_prepare_force_curves_preserves_rp3_metrics() -> None:
    segments = pd.DataFrame(
        {
            "video_stroke_idx": [4, 4, 4],
            "rp3_stroke_number": [12, 12, 12],
            "s_force": [0.0, 0.5, 1.0],
            "force_raw": [0.0, 500.0, 0.0],
            "rp3_drive_s": [0.8, 0.8, 0.8],
            "rp3_cycle_s": [2.4, 2.4, 2.4],
        }
    )

    curve = _prepare_force_curves(segments)[4]

    assert curve.rp3_stroke_number == 12
    assert curve.peak_force == 500.0
    assert curve.drive_time_s == 0.8
    assert curve.stroke_rate == 25.0
    np.testing.assert_allclose(curve.s, [0.0, 0.5, 1.0])


def test_project_pose_fits_requested_panel() -> None:
    pose = np.zeros((17, 3), dtype=np.float32)
    pose[:, 0] = np.linspace(-1.0, 1.0, 17)
    pose[:, 1] = np.linspace(-0.5, 0.5, 17)

    points = _project_pose(
        pose,
        rect=(100, 200, 400, 300),
        bounds=(-2.0, 0.0, 0.0, 1.0),
    )

    assert np.all(points[:, 0] >= 100)
    assert np.all(points[:, 0] <= 500)
    assert np.all(points[:, 1] >= 200)
    assert np.all(points[:, 1] <= 500)
