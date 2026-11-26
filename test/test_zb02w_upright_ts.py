"""Eight-action student deployment and headless complex-terrain integration."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mujoco_sim.configs import MUJOCO_SIM_ROOT, PROJECT_ROOT, Zb02wUprightConfig, Zb02wUprightTsConfig
from mujoco_sim.scripts.base.base import MujocoDeploy
from mujoco_sim.scripts.base.base_wl import MujocoDeployWl
from mujoco_sim.scripts.zb02w_flat import Zb02wFlatDeploy
from mujoco_sim.scripts.zb02w_gyrate import Zb02wGyrateDeploy
from mujoco_sim.scripts.zb02w_upright import Zb02wUprightDeploy
from mujoco_sim.scripts.zb02w_upright_ts import Zb02wUprightTsDeploy


class FakePolicy:
    def __init__(self):
        self.action = np.arange(8, dtype=np.float32)[None, :]
        self.calls = []

    def get_inputs(self):
        return [SimpleNamespace(name=name, shape=shape) for name, shape in (
            ("obs", [1, 98]), ("h_in", [3, 1, 256]), ("c_in", [3, 1, 256]),
        )]

    def get_outputs(self):
        return [SimpleNamespace(name=name, shape=shape) for name, shape in (
            ("actions", [1, 8]), ("h_out", [3, 1, 256]), ("c_out", [3, 1, 256]),
        )]

    def run(self, names, inputs):
        self.calls.append((names, {k: v.copy() for k, v in inputs.items()}))
        return self.action.copy(), inputs["h_in"] + 1, inputs["c_in"] + 2


class FakeGamepad:
    def __init__(self):
        self.rising_buttons = set()

    def is_button_rising_edge(self, name):
        return name in self.rising_buttons

    def is_axis_pressed(self, name, threshold=0.5):
        return False

    def is_button_pressed(self, name):
        return False

    def get_cmd(self):
        return np.zeros(3, dtype=np.float32)

    def poll(self):
        return True


@pytest.fixture
def deploy(monkeypatch):
    policy = FakePolicy()

    def fake_flat_init(self, config, device):
        self.config = config
        self.num_obs = config.num_obs
        self.num_actions = config.num_actions
        self.sim_dt = config.simulation_dt
        self.robot = SimpleNamespace(nu=16)
        self.data = SimpleNamespace(qpos=np.zeros(23), qvel=np.zeros(22))
        self.data.qpos[3] = 1.0
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.cmd = np.array([0.2, -0.3, 0.4], dtype=np.float32)
        self.send_plotjuggler_data = lambda *_: None
        self._init_control()
        self.gamepad = FakeGamepad()

    monkeypatch.setattr(Zb02wFlatDeploy, "__init__", fake_flat_init)
    monkeypatch.setattr(MujocoDeployWl, "_make_onnx_session", lambda *_: policy)
    return Zb02wUprightTsDeploy(Zb02wUprightTsConfig(plotjuggler_enabled=False))


def test_config_preserves_upright_and_enables_terrain_student():
    cfg = Zb02wUprightTsConfig()
    assert isinstance(cfg, Zb02wUprightConfig)
    assert cfg.policy_path == PROJECT_ROOT / "logs/rsl_rl/su_terrain_mix/1_exported/policy.onnx"
    assert cfg.terrain_xml_path == MUJOCO_SIM_ROOT / "assets/terrains/rough_stairs.xml"
    assert cfg.num_obs == 45 and cfg.num_actions == 8 and cfg.num_actions_pos == 12
    assert cfg.is_rnn and cfg.simulation_dt * cfg.control_decimation == pytest.approx(0.01)
    assert cfg.kpsPos == (300.,) * 4 + (160.,) * 4 + (300.,) * 2 + (160.,) * 2
    assert cfg.kdsPos == (3.,) * 12 and cfg.kdsVel == (0.5,) * 4
    assert issubclass(Zb02wUprightTsDeploy, Zb02wUprightDeploy)
    assert Zb02wUprightTsDeploy.update_tau is Zb02wGyrateDeploy.update_tau
    assert Zb02wUprightConfig().num_actions == 16


def test_observation_order_and_model_layout(deploy):
    deploy.data.qpos[7:] = np.linspace(-0.6, 0.9, 16)
    deploy.data.qvel[3:6] = [1., -2., 3.]
    deploy.data.qvel[6:] = np.linspace(-4., 3.5, 16)
    deploy.action[:] = np.arange(8) - 3
    deploy.update_obs()
    expected = np.concatenate((
        deploy.data.qvel[3:6] * 0.5,
        [0., 0., -1.],
        (deploy.data.qpos[7:][deploy.leg_joint_idx] - deploy.default_angles) * 0.5,
        deploy.data.qvel[6:][deploy.leg_joint_idx] * 0.03,
        deploy.data.qvel[6:][deploy.wheel_joint_idx] * 0.03,
        deploy.action, deploy.cmd,
    ))
    np.testing.assert_allclose(deploy.obs, expected, atol=1e-6)
    for mode in (0, 1, 0):
        deploy.mode = mode
        deploy.update_model_in()
        assert deploy.model_in.shape == (98,)
        np.testing.assert_array_equal(deploy.model_in[:45], deploy.obs)
        np.testing.assert_array_equal(deploy.model_in[45:96], np.zeros(51))
        np.testing.assert_array_equal(deploy.model_in[96:], np.eye(2)[mode])


def test_eight_actions_map_to_full_joint_targets_and_lstm_updates(deploy):
    raw_action = np.array([200., -200., 2., -2., 3., -3., 0.001, -0.00025], dtype=np.float32)
    deploy.policy.action[:] = raw_action
    deploy.update_model_in()
    deploy.update_action()
    np.testing.assert_array_equal(deploy.action, raw_action)
    expected = deploy.default_angles.copy()
    expected[4:8] += [50., -50., 1., -1.]
    expected[10:] += [1.5, -1.5]
    np.testing.assert_allclose(deploy.targ_dof_pos, expected)
    np.testing.assert_allclose(deploy.targ_dof_vel, [0., 0., 0.01, 0.], atol=1e-8)
    names, inputs = deploy.policy.calls[0]
    assert names == ["actions", "h_out", "c_out"]
    assert set(inputs) == {"obs", "h_in", "c_in"}
    assert inputs["obs"].shape == (1, 98)
    for value in inputs.values():
        assert value.dtype == np.float32 and value.flags.c_contiguous
    np.testing.assert_array_equal(deploy.h_in, np.ones((3, 1, 256)))
    np.testing.assert_array_equal(deploy.c_in, np.full((3, 1, 256), 2))
    deploy.update_action()
    np.testing.assert_array_equal(deploy.policy.calls[1][1]["h_in"], np.ones((3, 1, 256)))
    np.testing.assert_array_equal(deploy.c_in, np.full((3, 1, 256), 4))
    deploy.policy.action[0, 6:] = [-200., 200.]
    deploy.update_action()
    np.testing.assert_allclose(deploy.targ_dof_vel, [0., 0., -1999.99, 1999.99], atol=1e-4)


def test_mode_switch_preserves_history_and_reset_clears_control(deploy):
    deploy.update_model_in()
    deploy.update_action()
    for button, expected in (("Y", 1), ("B", 1), ("X", 0)):
        deploy.gamepad.rising_buttons = {button}
        deploy._update_mode_from_gamepad()
        assert deploy.mode == expected
        np.testing.assert_array_equal(deploy.h_in, np.ones((3, 1, 256)))
        deploy.gamepad.rising_buttons.clear()
        deploy._update_mode_from_gamepad()
        assert deploy.mode == expected
    deploy.update_tau()
    assert deploy.tau.shape == deploy._target_tau.shape == deploy._tau_response_delta.shape == (16,)
    assert np.isfinite(deploy.tau).all() and np.any(deploy.tau)
    deploy.mode = 1
    deploy._reset_control()
    assert deploy.mode == 0
    for value in (deploy.h_in, deploy.c_in, deploy.tau, deploy._target_tau,
                  deploy._tau_response_delta, deploy.model_in, deploy._scaled_action, deploy.targ_dof_vel):
        assert np.count_nonzero(value) == 0
    np.testing.assert_array_equal(deploy.targ_dof_pos, deploy.default_angles)


def test_torque_response_targets_all_sixteen_actuators(deploy):
    deploy.targ_dof_pos[:] = deploy.default_angles + 0.1
    deploy.targ_dof_vel[:] = [0., 0., 1., -2.]
    leg_torque = deploy.kpsPos * deploy.targ_dof_pos
    wheel_torque = deploy.kdsVel * deploy.targ_dof_vel
    deploy.update_tau()
    np.testing.assert_allclose(deploy._target_tau[deploy.leg_actions_to_mujoco], leg_torque)
    np.testing.assert_allclose(deploy._target_tau[deploy.wheel_actions_to_mujoco], wheel_torque)
    np.testing.assert_allclose(deploy.tau, deploy._target_tau * deploy._tau_response_alpha)
    for _ in range(round(deploy.torque_tracking_time / deploy.sim_dt) - 1):
        deploy.update_tau()
    np.testing.assert_allclose(deploy.tau, deploy._target_tau * 0.95, atol=1e-5)


def test_real_onnx_complex_terrain_headless(monkeypatch, tmp_path):
    """No viewer or real gamepad: exercise the actual model and physics loop."""
    config = Zb02wUprightTsConfig(plotjuggler_enabled=False)

    def merged_xml(self, robot_xml_path, ball_xml_path, terrain_xml_path):
        path = tmp_path / "scene.xml"
        path.write_text(
            f'<mujoco><include file="{robot_xml_path}"/>'
            f'<include file="{ball_xml_path}"/>'
            f'<include file="{terrain_xml_path}"/></mujoco>'
        )
        return str(path)

    def fake_gamepad(self):
        self.remote_enabled = False
        self.gamepad = FakeGamepad()

    monkeypatch.setattr(MujocoDeploy, "_build_merged_xml", merged_xml)
    monkeypatch.setattr(MujocoDeploy, "_init_gamepad", fake_gamepad)
    sim = Zb02wUprightTsDeploy(config)
    for name in ("stair_mid_mid_up_1", "ramp_mid_row_down", "platform_left_high_120"):
        assert mujoco.mj_name2id(sim.robot, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
    assert sim.projectile_manager.ball_body_id >= 0
    assert sim.robot.nu == 16
    assert sim.h_in.shape == sim.c_in.shape == (3, 1, 256)
    for mode in (0, 1):
        sim.reset()
        assert sim.mode == 0 and sim.data.qpos[2] == pytest.approx(0.62)
        sim.mode = mode
        for _ in range(400):
            sim.step()
            for value in (sim.data.qpos, sim.data.qvel, sim.data.ctrl, sim.action, sim.h_in, sim.c_in):
                assert np.isfinite(value).all()
        assert np.any(sim.h_in) and np.any(sim.c_in)
        assert sim.counter == 400
    sim.reset()
    assert np.count_nonzero(sim.action) == np.count_nonzero(sim.h_in) == np.count_nonzero(sim.c_in) == 0
