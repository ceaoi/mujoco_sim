from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mujoco_sim.scripts.base.base import (
    BASE_HUD_UPDATE_HZ,
    MujocoDeploy,
    _base_state_from_free_joint,
    _find_default_base_free_joint,
)


TWO_FREE_JOINT_MODEL = """
<mujoco model="base_hud_test">
    <worldbody>
        <body name="base_link">
            <freejoint name="base_freejoint"/>
            <geom type="sphere" size="0.1" mass="1"/>
        </body>
        <body name="projectile_ball">
            <freejoint name="projectile_ball_freejoint"/>
            <geom type="sphere" size="0.05" mass="0.1"/>
        </body>
    </worldbody>
</mujoco>
"""


class _FakeViewer:
    def __init__(self):
        self.texts = None

    def set_texts(self, texts):
        self.texts = texts


def _make_model_and_data():
    model = mujoco.MjModel.from_xml_string(TWO_FREE_JOINT_MODEL)
    return model, mujoco.MjData(model)


def test_default_base_uses_first_free_joint_instead_of_projectile():
    model, _ = _make_model_and_data()

    joint_id, body_id, qpos_adr, dof_adr = _find_default_base_free_joint(model)

    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) == "base_freejoint"
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) == "base_link"
    assert qpos_adr == 0
    assert dof_adr == 0


def test_default_base_requires_a_free_joint():
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")

    with pytest.raises(ValueError, match="free joint"):
        _find_default_base_free_joint(model)


def test_base_state_reports_world_pose_and_base_frame_velocities():
    model, data = _make_model_and_data()
    _, _, qpos_adr, dof_adr = _find_default_base_free_joint(model)
    yaw = np.pi / 2.0
    data.qpos[qpos_adr:qpos_adr + 7] = [
        0.0,
        0.0,
        1.25,
        np.cos(yaw / 2.0),
        0.0,
        0.0,
        np.sin(yaw / 2.0),
    ]
    data.qvel[dof_adr:dof_adr + 6] = [1.0, 0.0, 0.0, 0.1, -0.2, 0.3]

    z_world, yaw_world, angular_velocity_base, linear_velocity_base = (
        _base_state_from_free_joint(data, qpos_adr, dof_adr)
    )

    assert z_world == pytest.approx(1.25)
    assert yaw_world == pytest.approx(np.pi / 2.0)
    np.testing.assert_allclose(angular_velocity_base, [0.1, -0.2, 0.3])
    np.testing.assert_allclose(linear_velocity_base, [0.0, -1.0, 0.0], atol=1e-12)


def test_hud_uses_top_left_two_column_si_format():
    deploy = object.__new__(MujocoDeploy)
    deploy.viewer = _FakeViewer()
    deploy._base_qpos_adr = 0
    deploy._base_dof_adr = 0
    deploy.data = SimpleNamespace(
        qpos=np.array([0.0, 0.0, 0.625, 1.0, 0.0, 0.0, 0.0]),
        qvel=np.array([1.0, -2.0, 3.0, 0.1, -0.2, 0.3]),
    )

    deploy._update_base_state_hud()

    font, gridpos, labels, values = deploy.viewer.texts
    assert font == mujoco.mjtFontScale.mjFONTSCALE_150
    assert gridpos == mujoco.mjtGridPos.mjGRID_TOPLEFT
    assert labels.splitlines() == [
        "Base state",
        "z_world [m]",
        "yaw_world [rad]",
        "omega_base [rad/s]",
        "velocity_base [m/s]",
    ]
    assert values.splitlines() == [
        "",
        "+0.625",
        "+0.000",
        "[+0.100, -0.200, +0.300]",
        "[+1.000, -2.000, +3.000]",
    ]


def test_hud_20_hz_interval_for_default_simulation_timestep():
    simulation_dt = 0.0005
    interval = round(1.0 / (BASE_HUD_UPDATE_HZ * simulation_dt))

    assert BASE_HUD_UPDATE_HZ == 20.0
    assert interval == 100
