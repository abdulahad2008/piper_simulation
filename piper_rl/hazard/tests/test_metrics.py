import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from piper_rl.hazard.geometry import closest_points_segments, segment_capsule_distance  # noqa: E402
from piper_rl.hazard.metrics import BladeFrame, Capsule, ExposureAccumulator, blade_hazard  # noqa: E402

A = np.array


def test_parallel_segments_distance():
    _, _, cp, cq = closest_points_segments(A([0, 0, 0.]), A([1, 0, 0.]), A([0, 1, 0.]), A([1, 1, 0.]))
    assert np.isclose(np.linalg.norm(cp - cq), 1.0)


def test_point_segment_degenerate():
    d, *_ = segment_capsule_distance(A([0, 0, 0.]), A([0, 0, 0.]), A([0, 0, 1.]), A([0, 0, 3.]), 0.25)
    assert np.isclose(d, 0.75)


def test_edge_facing_human_scores_high():
    # edge along x at z=0, cutting direction +y, human capsule 10 cm away in +y
    blade = BladeFrame(A([0, 0, 0.]), A([0.2, 0, 0.]), A([0, 1, 0.]), vel=A([0, 0.3, 0.]))
    human = [Capsule(A([0.1, 0.15, -0.1]), A([0.1, 0.15, 0.1]), 0.05, name="forearm")]
    s = blade_hazard(blade, human, d0=0.15, v0=0.25)
    assert np.isclose(s.edge_distance, 0.10)
    assert np.isclose(s.alignment, 1.0)
    assert np.isclose(s.closing_speed, 0.3)
    expected = 1.0 * (1 + 0.3 / 0.25) * (1 - 0.10 / 0.15)
    assert np.isclose(s.hazard, expected)


def test_spine_facing_human_scores_zero():
    # same geometry, cutting direction flipped to -y: spine faces the person
    blade = BladeFrame(A([0, 0, 0.]), A([0.2, 0, 0.]), A([0, -1, 0.]), vel=A([0, 0.3, 0.]))
    human = [Capsule(A([0.1, 0.15, -0.1]), A([0.1, 0.15, 0.1]), 0.05)]
    s = blade_hazard(blade, human)
    assert s.alignment == 0.0 and s.hazard == 0.0 and s.closing_speed == 0.0


def test_far_human_scores_zero_even_if_facing():
    blade = BladeFrame(A([0, 0, 0.]), A([0.2, 0, 0.]), A([0, 1, 0.]), vel=A([0, 1.0, 0.]))
    human = [Capsule(A([0.1, 0.5, -0.1]), A([0.1, 0.5, 0.1]), 0.05)]
    s = blade_hazard(blade, human, d0=0.15)
    assert s.alignment == 1.0 and s.hazard == 0.0


def test_receding_blade_has_no_closing_speed():
    blade = BladeFrame(A([0, 0, 0.]), A([0.2, 0, 0.]), A([0, 1, 0.]), vel=A([0, -0.3, 0.]))
    human = [Capsule(A([0.1, 0.15, -0.1]), A([0.1, 0.15, 0.1]), 0.05, vel=A([0, 0, 0.]))]
    s = blade_hazard(blade, human)
    assert s.closing_speed == 0.0
    assert np.isclose(s.hazard, 1.0 * (1 - 0.10 / 0.15))


def test_human_moving_into_blade_counts_as_closing():
    blade = BladeFrame(A([0, 0, 0.]), A([0.2, 0, 0.]), A([0, 1, 0.]), vel=A([0, 0, 0.]))
    human = [Capsule(A([0.1, 0.15, -0.1]), A([0.1, 0.15, 0.1]), 0.05, vel=A([0, -0.2, 0.]))]
    s = blade_hazard(blade, human)
    assert np.isclose(s.closing_speed, 0.2)


def test_nearest_of_several_parts_is_used():
    blade = BladeFrame(A([0, 0, 0.]), A([0.2, 0, 0.]), A([0, 1, 0.]))
    humans = [Capsule(A([0.1, 0.5, -0.1]), A([0.1, 0.5, 0.1]), 0.05, name="far"),
              Capsule(A([0.1, 0.12, -0.1]), A([0.1, 0.12, 0.1]), 0.05, name="near")]
    s = blade_hazard(blade, humans)
    assert s.nearest == "near" and np.isclose(s.edge_distance, 0.07)


def test_penetration_is_negative_and_flagged_as_cut():
    blade = BladeFrame(A([0, 0, 0.]), A([0.2, 0, 0.]), A([0, 1, 0.]))
    humans = [Capsule(A([0.1, 0.02, -0.1]), A([0.1, 0.02, 0.1]), 0.05)]
    acc = ExposureAccumulator(dt=0.05)
    acc.update(blade_hazard(blade, humans))
    out = acc.summary()
    assert out["cut"] and out["min_edge_distance"] < 0


def test_exposure_integrates_hazard():
    blade = BladeFrame(A([0, 0, 0.]), A([0.2, 0, 0.]), A([0, 1, 0.]))
    human = [Capsule(A([0.1, 0.15, -0.1]), A([0.1, 0.15, 0.1]), 0.05)]
    acc = ExposureAccumulator(dt=0.05)
    for _ in range(10):
        acc.update(blade_hazard(blade, human))
    assert np.isclose(acc.summary()["exposure"], 10 * 0.05 * (1 - 0.10 / 0.15))


@pytest.mark.skipif(pytest.importorskip("mujoco", reason="mujoco not installed") is None, reason="")
def test_mujoco_bridge_demo_scene():
    import mujoco
    from piper_rl.hazard.mujoco_bridge import read_blade, read_human_capsules

    path = os.path.join(os.path.dirname(__file__), "..", "assets", "demo_scene.xml")
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    blade = read_blade(model, data)
    caps = read_human_capsules(model, data)
    assert len(caps) == 2 and {c.name for c in caps} == {"human_forearm", "human_hand"}
    assert np.allclose(blade.cut_dir, [0, 0, 1])           # identity orientation: +Z is the edge
    assert np.isclose(np.linalg.norm(blade.tip - blade.heel), 0.2)
    s = blade_hazard(blade, caps)
    assert s.edge_distance > 0.15 and s.hazard == 0.0     # parked human, no hazard
