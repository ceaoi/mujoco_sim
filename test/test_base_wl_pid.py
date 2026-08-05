from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from mujoco_sim.scripts.base.base_wl import MujocoDeployWl, wheel_stop_pid


class _FakePolicy:
    def __init__(self, action):
        self._action = np.asarray(action, dtype=np.float32)[None, :]

    def run(self, output_names, inputs):
        return [self._action.copy()]


def _make_deploy(action, cmd, wheel_velocity, kp=1.0, ki=0.0, kd=0.0):
    deploy = object.__new__(MujocoDeployWl)
    deploy.model_in = np.zeros(1, dtype=np.float32)
    deploy.is_rnn = False
    deploy.policy = _FakePolicy(action)
    deploy.policy_input_name = "obs"
    deploy.policy_output_name = "action"
    deploy.num_actions_pos = 12
    deploy.default_angles = np.zeros(12, dtype=np.float32)
    deploy.action_scale_pos = np.float32(1.0)
    deploy.action_scale_vel = np.float32(1.0)
    deploy.wheel_action_vel_deadzone = np.float32(0.1)
    deploy.wheel_joint_idx = [3, 7, 11, 15]
    deploy.num_wheels = 4
    deploy.cmd = np.asarray(cmd, dtype=np.float32)
    deploy.ctrl_dt = 0.02
    deploy.wheel_stop_pid_enabled = True
    deploy.wheel_stop_pid_kp = np.float32(kp)
    deploy.wheel_stop_pid_ki = np.float32(ki)
    deploy.wheel_stop_pid_kd = np.float32(kd)
    deploy.wheel_stop_pid_output_limit = np.float32(5.0)
    deploy.wheel_stop_pid_integral = np.zeros(4, dtype=np.float32)
    deploy.wheel_stop_pid_previous_error = np.zeros(4, dtype=np.float32)
    deploy.wheel_stop_pid_active = False
    deploy.targ_dof_pos = deploy.default_angles.copy()
    deploy.targ_dof_vel = np.zeros(4, dtype=np.float32)

    qvel = np.zeros(22, dtype=np.float32)
    qvel[6 + np.asarray(deploy.wheel_joint_idx)] = wheel_velocity
    deploy.data = SimpleNamespace(qvel=qvel)
    return deploy


def test_wheel_stop_pid_computes_pid_terms_for_each_wheel():
    output, integral, previous_error = wheel_stop_pid(
        error=np.array([1.0, -2.0, 0.5, -0.5], dtype=np.float32),
        integral=np.zeros(4, dtype=np.float32),
        previous_error=np.zeros(4, dtype=np.float32),
        kp=1.0,
        ki=2.0,
        kd=0.5,
        dt=0.1,
        output_limit=100.0,
    )

    np.testing.assert_allclose(output, [6.2, -12.4, 3.1, -3.1], atol=1e-6)
    np.testing.assert_allclose(integral, [0.1, -0.2, 0.05, -0.05], atol=1e-6)
    np.testing.assert_allclose(previous_error, [1.0, -2.0, 0.5, -0.5])


def test_wheel_stop_pid_limits_output_and_prevents_integral_windup():
    output, integral, _ = wheel_stop_pid(
        error=np.array([10.0, -10.0, 10.0, -10.0], dtype=np.float32),
        integral=np.zeros(4, dtype=np.float32),
        previous_error=np.array([10.0, -10.0, 10.0, -10.0], dtype=np.float32),
        kp=1.0,
        ki=1.0,
        kd=0.0,
        dt=1.0,
        output_limit=5.0,
    )

    np.testing.assert_allclose(output, [5.0, -5.0, 5.0, -5.0])
    np.testing.assert_allclose(integral, np.zeros(4))

    _, unwinding_integral, _ = wheel_stop_pid(
        error=np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32),
        integral=np.array([10.0, -10.0, 10.0, -10.0], dtype=np.float32),
        previous_error=np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32),
        kp=0.0,
        ki=1.0,
        kd=0.0,
        dt=1.0,
        output_limit=5.0,
    )
    np.testing.assert_allclose(unwinding_integral, [9.0, -9.0, 9.0, -9.0])


def test_update_action_adds_pid_only_for_zero_command_without_derivative_kick():
    action = np.concatenate(
        [np.zeros(12, dtype=np.float32), np.array([1.0, -1.0, 0.2, -0.2])]
    )
    deploy = _make_deploy(
        action=action,
        cmd=[0.0, 0.0, 0.0],
        wheel_velocity=[2.0, -2.0, 1.0, -1.0],
        kp=1.0,
        kd=100.0,
    )

    deploy.update_action()

    policy_target = np.array([0.9, -0.9, 0.1, -0.1], dtype=np.float32)
    np.testing.assert_allclose(
        deploy.targ_dof_vel,
        policy_target + np.array([-2.0, 2.0, -1.0, 1.0], dtype=np.float32),
    )
    assert deploy.wheel_stop_pid_active


def test_nonzero_command_preserves_policy_target_and_clears_pid_state():
    action = np.concatenate(
        [np.zeros(12, dtype=np.float32), np.array([1.0, -1.0, 0.2, -0.2])]
    )
    deploy = _make_deploy(
        action=action,
        cmd=[0.1, 0.0, 0.0],
        wheel_velocity=[2.0, -2.0, 1.0, -1.0],
    )
    deploy.wheel_stop_pid_integral[:] = 3.0
    deploy.wheel_stop_pid_previous_error[:] = 2.0
    deploy.wheel_stop_pid_active = True

    deploy.update_action()

    np.testing.assert_allclose(deploy.targ_dof_vel, [0.9, -0.9, 0.1, -0.1])
    np.testing.assert_allclose(deploy.wheel_stop_pid_integral, np.zeros(4))
    np.testing.assert_allclose(deploy.wheel_stop_pid_previous_error, np.zeros(4))
    assert not deploy.wheel_stop_pid_active


def test_disabled_pid_preserves_policy_target_and_clears_pid_state():
    action = np.concatenate(
        [np.zeros(12, dtype=np.float32), np.array([1.0, -1.0, 0.2, -0.2])]
    )
    deploy = _make_deploy(
        action=action,
        cmd=[0.0, 0.0, 0.0],
        wheel_velocity=[2.0, -2.0, 1.0, -1.0],
    )
    deploy.wheel_stop_pid_enabled = False
    deploy.wheel_stop_pid_integral[:] = 3.0
    deploy.wheel_stop_pid_previous_error[:] = 2.0
    deploy.wheel_stop_pid_active = True

    deploy.update_action()

    np.testing.assert_allclose(deploy.targ_dof_vel, [0.9, -0.9, 0.1, -0.1])
    np.testing.assert_allclose(deploy.wheel_stop_pid_integral, np.zeros(4))
    np.testing.assert_allclose(deploy.wheel_stop_pid_previous_error, np.zeros(4))
    assert not deploy.wheel_stop_pid_active


def test_reset_control_clears_wheel_stop_pid_state():
    deploy = _make_deploy(
        action=np.zeros(16, dtype=np.float32),
        cmd=[0.0, 0.0, 0.0],
        wheel_velocity=np.zeros(4, dtype=np.float32),
    )
    deploy.is_rnn = False
    deploy.wheel_stop_pid_integral[:] = 3.0
    deploy.wheel_stop_pid_previous_error[:] = 2.0
    deploy.wheel_stop_pid_active = True

    deploy._reset_control()

    np.testing.assert_allclose(deploy.wheel_stop_pid_integral, np.zeros(4))
    np.testing.assert_allclose(deploy.wheel_stop_pid_previous_error, np.zeros(4))
    assert not deploy.wheel_stop_pid_active


def test_wheel_stop_pid_config_is_present_in_all_deployments():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    config_names = (
        "m20_flat.yaml",
        "m20_rough.yaml",
        "zb02w_flat.yaml",
        "zb02w_rough.yaml",
        "zb02w_ts.yaml",
    )

    enabled_configs = {"zb02w_ts.yaml"}
    for config_name in config_names:
        with (config_dir / config_name).open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        assert config["wheel_stop_pid_enabled"] is (config_name in enabled_configs)
        assert isinstance(config["wheel_stop_pid_kp"], float)
        assert isinstance(config["wheel_stop_pid_ki"], float)
        assert isinstance(config["wheel_stop_pid_kd"], float)
        assert isinstance(config["wheel_stop_pid_output_limit"], float)
