from types import SimpleNamespace

import numpy as np

from mujoco_sim.configs import Zb02wGyrateConfig
from mujoco_sim.scripts.zb02w_gyrate import Zb02wGyrateDeploy


class FakeGamepad:
    def __init__(self):
        self.rising_buttons = set()

    def is_button_rising_edge(self, name):
        return name in self.rising_buttons


class FakePolicy:
    def __init__(self, action, h_out):
        self.action = action
        self.h_out = h_out
        self.requested_outputs = None
        self.inputs = None

    def run(self, requested_outputs, inputs):
        self.requested_outputs = requested_outputs
        self.inputs = inputs
        return self.action, self.h_out


def test_gyrate_observation_matches_training_order_and_scaling():
    config = Zb02wGyrateConfig(plotjuggler_enabled=False)
    deploy = object.__new__(Zb02wGyrateDeploy)
    deploy.leg_joint_idx = np.asarray(config.leg_joint_idx, dtype=np.intp)
    deploy.wheel_joint_idx = np.asarray(config.wheel_joint_idx, dtype=np.intp)
    deploy.default_angles = np.asarray(config.default_angles_leg, dtype=np.float32)
    deploy.num_actions_pos = config.num_actions_pos
    deploy.num_wheels = len(config.wheel_joint_idx)
    deploy.num_actions = config.num_actions
    deploy.obs = np.zeros(config.num_obs, dtype=np.float32)
    deploy.action = np.arange(config.num_actions, dtype=np.float32) - 8.0
    deploy.cmd = np.zeros(3, dtype=np.float32)
    deploy.plotjuggler_enabled = False
    deploy.plotjuggler = None
    plotjuggler_messages = []
    deploy.send_plotjuggler_data = (
        lambda name, value: plotjuggler_messages.append((name, value))
    )

    joint_pos = np.linspace(-0.8, 0.7, config.num_actions, dtype=np.float32)
    joint_vel = np.linspace(-4.0, 3.5, config.num_actions, dtype=np.float32)
    qpos = np.zeros(7 + config.num_actions, dtype=np.float32)
    qvel = np.zeros(6 + config.num_actions, dtype=np.float32)
    qpos[3:7] = np.array(
        [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],
        dtype=np.float32,
    )
    qpos[7:] = joint_pos
    qvel[3:6] = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    qvel[6:] = joint_vel
    deploy.data = SimpleNamespace(qpos=qpos, qvel=qvel)

    deploy.update_obs()

    expected = np.concatenate(
        (
            qvel[3:6] * 0.5,
            np.array([0.0, -1.0, 0.0], dtype=np.float32),
            (joint_pos[deploy.leg_joint_idx] - deploy.default_angles) * 0.5,
            joint_vel[deploy.leg_joint_idx] * 0.03,
            joint_vel[deploy.wheel_joint_idx] * 0.03,
            deploy.action,
        )
    )

    assert deploy.obs.shape == (50,)
    assert deploy.obs.dtype == np.float32
    np.testing.assert_allclose(deploy.obs, expected, atol=1.0e-6)
    assert plotjuggler_messages[0][0] == "world_ang_vel_z"
    np.testing.assert_allclose(plotjuggler_messages[0][1], -2.0, atol=1.0e-6)


def test_mode_buttons_persistently_select_expected_mode():
    deploy = object.__new__(Zb02wGyrateDeploy)
    deploy.mode = 0
    deploy.gamepad = FakeGamepad()

    for button, expected_mode in (("Y", 1), ("B", 2), ("X", 0)):
        deploy.gamepad.rising_buttons = {button}
        deploy._update_mode_from_gamepad()
        assert deploy.mode == expected_mode

        deploy.gamepad.rising_buttons.clear()
        deploy._update_mode_from_gamepad()
        assert deploy.mode == expected_mode


def test_model_input_appends_mode_one_hot():
    deploy = object.__new__(Zb02wGyrateDeploy)
    deploy.num_obs = 50
    deploy.obs = np.arange(50, dtype=np.float32)
    deploy.model_in = np.zeros(53, dtype=np.float32)

    expected_modes = np.eye(3, dtype=np.float32)
    for mode in range(3):
        deploy.mode = mode
        deploy.update_model_in()
        assert deploy.model_in.shape == (53,)
        assert deploy.model_in.dtype == np.float32
        np.testing.assert_array_equal(deploy.model_in[:50], deploy.obs)
        np.testing.assert_array_equal(deploy.model_in[50:], expected_modes[mode])


def test_gru_policy_inference_updates_hidden_state_and_action_targets():
    deploy = object.__new__(Zb02wGyrateDeploy)
    deploy.model_in = np.arange(53, dtype=np.float32)
    deploy.h_in = np.zeros((1, 1, 256), dtype=np.float32)
    deploy.policy_input_name = "obs"
    deploy.policy_output_name = "actions"
    action = np.arange(16, dtype=np.float32)[None, :]
    h_out = np.ones((1, 1, 256), dtype=np.float32)
    deploy.policy = FakePolicy(action, h_out)
    deploy.num_actions_pos = 12
    deploy.action_scale_pos = np.float32(0.5)
    deploy.action_scale_vel = np.float32(20.0)
    deploy.default_angles = np.linspace(-0.4, 0.4, 12, dtype=np.float32)
    deploy.wheel_joint_idx = np.asarray((3, 7, 11, 15), dtype=np.intp)
    deploy.data = SimpleNamespace(qvel=np.zeros(22, dtype=np.float32))
    deploy.cmd = np.zeros(3, dtype=np.float32)
    deploy.wheel_stop_pid_enabled = False
    deploy.wheel_stop_pid_integral = np.zeros(4, dtype=np.float32)
    deploy.wheel_stop_pid_previous_error = np.zeros(4, dtype=np.float32)
    deploy.wheel_stop_pid_active = False
    deploy.plotjuggler_enabled = False
    deploy.plotjuggler = None

    deploy.update_action()

    assert deploy.policy.requested_outputs == ["actions", "h_out"]
    assert set(deploy.policy.inputs) == {"obs", "h_in"}
    assert deploy.policy.inputs["obs"].shape == (1, 53)
    assert deploy.policy.inputs["obs"].dtype == np.float32
    np.testing.assert_array_equal(deploy.policy.inputs["h_in"], np.zeros_like(h_out))
    np.testing.assert_array_equal(deploy.h_in, h_out)
    np.testing.assert_array_equal(deploy.action, action.squeeze(0))
    np.testing.assert_allclose(
        deploy.targ_dof_pos,
        action[0, :12] * deploy.action_scale_pos + deploy.default_angles,
    )
    np.testing.assert_allclose(
        deploy.targ_dof_vel,
        action[0, 12:] * deploy.action_scale_vel,
    )


def test_reset_control_restores_default_mode_and_clears_gru_state():
    deploy = object.__new__(Zb02wGyrateDeploy)
    deploy.default_angles = np.zeros(12, dtype=np.float32)
    deploy.targ_dof_pos = np.ones(12, dtype=np.float32)
    deploy.targ_dof_vel = np.ones(4, dtype=np.float32)
    deploy.wheel_stop_pid_integral = np.ones(4, dtype=np.float32)
    deploy.wheel_stop_pid_previous_error = np.ones(4, dtype=np.float32)
    deploy.wheel_stop_pid_active = True
    deploy.is_rnn = False
    deploy.mode = 2
    deploy.h_in = np.ones((1, 1, 256), dtype=np.float32)

    deploy._reset_control()

    assert deploy.mode == 0
    np.testing.assert_array_equal(deploy.h_in, np.zeros_like(deploy.h_in))
