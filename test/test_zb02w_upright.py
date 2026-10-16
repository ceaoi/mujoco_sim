from types import SimpleNamespace

import numpy as np

from mujoco_sim.configs import PROJECT_ROOT, Zb02wGyrateConfig, Zb02wUprightConfig
from mujoco_sim.scripts.zb02w_flat import Zb02wFlatDeploy
from mujoco_sim.scripts.zb02w_gyrate import Zb02wGyrateDeploy
from mujoco_sim.scripts.zb02w_upright import Zb02wUprightDeploy


class FakePolicy:
    def __init__(self, action, routing_h_out, estimator_h_out):
        self.action = action
        self.routing_h_out = routing_h_out
        self.estimator_h_out = estimator_h_out
        self.requested_outputs = None
        self.inputs = None

    def run(self, requested_outputs, inputs):
        self.requested_outputs = requested_outputs
        self.inputs = inputs
        return self.action, self.routing_h_out, self.estimator_h_out


class FakeInput:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakePolicyMetadata:
    def get_inputs(self):
        return [
            FakeInput("obs", [1, 56]),
            FakeInput("routing_h_in", [1, 1, 128]),
            FakeInput("estimator_h_in", [2, 1, 64]),
        ]


def test_upright_config_uses_exported_moe_policy_and_policy_observation_size():
    config = Zb02wUprightConfig()

    assert isinstance(config, Zb02wGyrateConfig)
    assert config.policy_path == (
        PROJECT_ROOT / "logs/rsl_rl/su_flat_student/1_exported/policy.onnx"
    )
    assert config.num_obs == 53
    assert config.num_actions == 16
    assert not config.is_rnn


def test_upright_initializes_hidden_states_from_onnx_metadata(monkeypatch):
    def fake_flat_init(deploy, config, device):
        deploy.num_obs = config.num_obs
        deploy.num_wheels = 4
        deploy.sim_dt = config.simulation_dt
        deploy.tau = np.zeros(config.num_actions, dtype=np.float32)
        deploy.policy = FakePolicyMetadata()

    monkeypatch.setattr(Zb02wFlatDeploy, "__init__", fake_flat_init)

    deploy = Zb02wUprightDeploy(
        Zb02wUprightConfig(plotjuggler_enabled=False)
    )

    assert deploy.mode == 0
    assert deploy.model_in.shape == (56,)
    assert deploy.routing_h_in.shape == (1, 1, 128)
    assert deploy.estimator_h_in.shape == (2, 1, 64)
    assert deploy.routing_h_in.dtype == np.float32
    assert deploy.estimator_h_in.dtype == np.float32


def test_upright_observation_matches_training_order_and_scaling():
    config = Zb02wUprightConfig(plotjuggler_enabled=False)
    deploy = object.__new__(Zb02wUprightDeploy)
    deploy.leg_joint_idx = np.asarray(config.leg_joint_idx, dtype=np.intp)
    deploy.wheel_joint_idx = np.asarray(config.wheel_joint_idx, dtype=np.intp)
    deploy.default_angles = np.asarray(config.default_angles_leg, dtype=np.float32)
    deploy.num_actions_pos = config.num_actions_pos
    deploy.num_wheels = len(config.wheel_joint_idx)
    deploy.num_actions = config.num_actions
    deploy.obs = np.zeros(config.num_obs, dtype=np.float32)
    deploy.action = np.arange(config.num_actions, dtype=np.float32) - 8.0
    deploy.cmd = np.array([0.5, -0.25, 0.75], dtype=np.float32)
    deploy.plotjuggler_enabled = False
    deploy.plotjuggler = None
    deploy.send_plotjuggler_data = lambda *_: None

    joint_pos = np.linspace(-0.8, 0.7, config.num_actions, dtype=np.float32)
    joint_vel = np.linspace(-4.0, 3.5, config.num_actions, dtype=np.float32)
    qpos = np.zeros(7 + config.num_actions, dtype=np.float32)
    qvel = np.zeros(6 + config.num_actions, dtype=np.float32)
    qpos[3] = 1.0
    qpos[7:] = joint_pos
    qvel[3:6] = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    qvel[6:] = joint_vel
    deploy.data = SimpleNamespace(qpos=qpos, qvel=qvel)

    deploy.update_obs()

    expected = np.concatenate(
        (
            qvel[3:6] * 0.5,
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
            (joint_pos[deploy.leg_joint_idx] - deploy.default_angles) * 0.5,
            joint_vel[deploy.leg_joint_idx] * 0.03,
            joint_vel[deploy.wheel_joint_idx] * 0.03,
            deploy.action,
            deploy.cmd,
        )
    )

    assert deploy.obs.shape == (53,)
    assert deploy.obs.dtype == np.float32
    np.testing.assert_allclose(deploy.obs, expected, atol=1.0e-6)


def test_upright_model_input_appends_mode_one_hot():
    deploy = object.__new__(Zb02wUprightDeploy)
    deploy.num_obs = 53
    deploy.obs = np.arange(53, dtype=np.float32)
    deploy.model_in = np.zeros(56, dtype=np.float32)

    expected_modes = np.eye(3, dtype=np.float32)
    for mode in range(3):
        deploy.mode = mode
        deploy.update_model_in()
        np.testing.assert_array_equal(deploy.model_in[:53], deploy.obs)
        np.testing.assert_array_equal(deploy.model_in[53:], expected_modes[mode])


def test_upright_policy_inference_updates_both_hidden_states_and_targets():
    deploy = object.__new__(Zb02wUprightDeploy)
    deploy.model_in = np.arange(56, dtype=np.float32)
    deploy.routing_h_in = np.zeros((1, 1, 256), dtype=np.float32)
    deploy.estimator_h_in = np.zeros((1, 1, 256), dtype=np.float32)
    deploy.policy_input_name = "obs"
    deploy.policy_output_name = "actions"
    action = np.arange(16, dtype=np.float32)[None, :]
    routing_h_out = np.ones((1, 1, 256), dtype=np.float32)
    estimator_h_out = np.full((1, 1, 256), 2.0, dtype=np.float32)
    deploy.policy = FakePolicy(action, routing_h_out, estimator_h_out)
    deploy.num_actions_pos = 12
    deploy.action_scale_pos = np.float32(0.5)
    deploy.action_scale_vel = np.float32(20.0)
    deploy.default_angles = np.linspace(-0.4, 0.4, 12, dtype=np.float32)
    deploy.wheel_joint_idx = np.asarray((3, 7, 11, 15), dtype=np.intp)
    deploy.data = SimpleNamespace(qvel=np.zeros(22, dtype=np.float32))
    deploy.plotjuggler_enabled = False
    deploy.plotjuggler = None

    deploy.update_action()

    assert deploy.policy.requested_outputs == [
        "actions",
        "routing_h_out",
        "estimator_h_out",
    ]
    assert set(deploy.policy.inputs) == {
        "obs",
        "routing_h_in",
        "estimator_h_in",
    }
    assert deploy.policy.inputs["obs"].shape == (1, 56)
    assert deploy.policy.inputs["obs"].dtype == np.float32
    np.testing.assert_array_equal(deploy.routing_h_in, routing_h_out)
    np.testing.assert_array_equal(deploy.estimator_h_in, estimator_h_out)
    np.testing.assert_array_equal(deploy.action, action.squeeze(0))
    np.testing.assert_allclose(
        deploy.targ_dof_pos,
        action[0, :12] * deploy.action_scale_pos + deploy.default_angles,
    )
    np.testing.assert_allclose(
        deploy.targ_dof_vel,
        action[0, 12:] * deploy.action_scale_vel,
    )


def test_upright_reset_clears_mode_and_both_hidden_states():
    deploy = object.__new__(Zb02wUprightDeploy)
    deploy.default_angles = np.zeros(12, dtype=np.float32)
    deploy.targ_dof_pos = np.ones(12, dtype=np.float32)
    deploy.targ_dof_vel = np.ones(4, dtype=np.float32)
    deploy.wheel_stop_pid_integral = np.ones(4, dtype=np.float32)
    deploy.wheel_stop_pid_previous_error = np.ones(4, dtype=np.float32)
    deploy.wheel_stop_pid_active = True
    deploy.is_rnn = False
    deploy.mode = 2
    deploy.routing_h_in = np.ones((1, 1, 256), dtype=np.float32)
    deploy.estimator_h_in = np.ones((1, 1, 256), dtype=np.float32)
    deploy.tau = np.ones(16, dtype=np.float32)
    deploy._target_tau = np.ones(16, dtype=np.float32)
    deploy._tau_response_delta = np.ones(16, dtype=np.float32)

    deploy._reset_control()

    assert deploy.mode == 0
    np.testing.assert_array_equal(
        deploy.routing_h_in,
        np.zeros_like(deploy.routing_h_in),
    )
    np.testing.assert_array_equal(
        deploy.estimator_h_in,
        np.zeros_like(deploy.estimator_h_in),
    )
    np.testing.assert_array_equal(deploy.tau, np.zeros_like(deploy.tau))
    np.testing.assert_array_equal(
        deploy._target_tau,
        np.zeros_like(deploy._target_tau),
    )
    np.testing.assert_array_equal(
        deploy._tau_response_delta,
        np.zeros_like(deploy._tau_response_delta),
    )


def test_upright_reuses_gyrate_mode_switching_and_torque_response():
    assert (
        Zb02wUprightDeploy._update_mode_from_gamepad
        is Zb02wGyrateDeploy._update_mode_from_gamepad
    )
    assert Zb02wUprightDeploy.update_tau is Zb02wGyrateDeploy.update_tau
