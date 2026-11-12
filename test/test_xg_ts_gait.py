from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from mujoco_sim import run_script
from mujoco_sim.configs import XgTsConfig, XgTsGaitConfig
from mujoco_sim.scripts.xg_ts import XgTsDeploy
from mujoco_sim.scripts.xg_ts_gait import XgTsGaitDeploy
from mujoco_sim.utils.gait_phase_generator import XgGaitPhaseGenerator


MINIMAL_MJCF = """
<mujoco>
  <worldbody>
    <body name="base_link">
      <freejoint/>
      <inertial pos="0.1 0.2 0.3" mass="1" diaginertia="1 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_gait_config_inherits_current_xg_control_parameters():
    base = XgTsConfig()
    gait = XgTsGaitConfig()
    assert replace(gait, num_obs=53).__dict__ == {
        **base.__dict__,
        "gait_period": 0.36,
        "gait_command_threshold": 0.01,
        "gait_acceleration_threshold": 5.0,
        "gait_angular_velocity_threshold": 0.5,
    }
    assert gait.num_obs == 55
    assert issubclass(XgTsGaitDeploy, XgTsDeploy)


def test_generator_preserves_training_float32_clock_with_new_triggers():
    config = XgTsGaitConfig()
    gait = XgGaitPhaseGenerator(config)
    clock = torch.zeros(1, dtype=torch.float32)
    # Include many wraparounds and both sides of every mode threshold.
    cases = [
        ((0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0), 10),
        ((2.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0), 10),
        ((0.0, 0.01, 0.0), 5.0, (0.5, 0.0, -0.5), 10),
        ((0.0, 0.010001, 0.0), 0.0, (0.0, 0.0, 0.0), 57),
        ((0.0, -0.2, 0.0), 0.0, (0.0, 0.0, 0.0), 1500),
        ((0.0, 0.0, -0.3), 0.0, (0.0, 0.0, 0.0), 21),
        ((0.0, 0.0, 0.0), 5.00001, (0.0, 0.0, 0.0), 25),
        ((0.0, 0.0, 0.0), -5.00001, (0.0, 0.0, 0.0), 25),
        ((0.0, 0.0, 0.0), 0.0, (-0.6, 0.0, 0.0), 25),
        ((0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.6), 25),
        ((0.0, 0.0, 0.0), 0.0, (0.0, 2.0, 0.0), 100),
        ((0.0, 0.0, 0.02), 0.0, (0.0, 0.0, 0.0), 5),
        ((0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0), 100),
    ]
    for command, acceleration, angular_velocity, steps in cases:
        cmd = torch.tensor([command], dtype=torch.float32)
        ay = torch.tensor([acceleration], dtype=torch.float32)
        omega = torch.tensor([angular_velocity], dtype=torch.float32)
        for _ in range(steps):
            mode = (
                (torch.norm(cmd[:, 1:3], dim=-1) > config.gait_command_threshold)
                | (ay.abs() > config.gait_acceleration_threshold)
                | (omega[:, 0].abs() > config.gait_angular_velocity_threshold)
                | (omega[:, 2].abs() > config.gait_angular_velocity_threshold)
            )
            reset = (~mode) & (clock.remainder(config.gait_period) <= float(gait.dt))
            clock = torch.where(reset, 0.0, clock + float(gait.dt))
            phase_a = clock.remainder(config.gait_period) / config.gait_period
            phase_b = (phase_a + 0.5).remainder(1.0) * (~reset).float()
            expected = torch.stack((phase_a, phase_b), dim=-1)[0].numpy() - np.float32(0.5)
            actual = gait.update(cmd[0].numpy(), acceleration, omega[0].numpy())
            np.testing.assert_array_equal(actual, expected)
            assert gait.clock == clock.item()
            assert gait.walk_mode == mode.item()
            assert actual.dtype == np.float32
    np.testing.assert_array_equal(gait.observation, [-0.5, -0.5])


@pytest.mark.parametrize("signal", ["ay", "wx", "wz"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("relative_value, expected_mode", [(0.99, False), (1.0, False), (1.01, True)])
def test_motion_trigger_thresholds_are_strict_and_symmetric(signal, sign, relative_value, expected_mode):
    config = XgTsGaitConfig()
    gait = XgGaitPhaseGenerator(config)
    ay = 0.0
    omega = np.zeros(3, dtype=np.float32)
    if signal == "ay":
        ay = sign * relative_value * config.gait_acceleration_threshold
    else:
        omega[0 if signal == "wx" else 2] = (
            sign * relative_value * config.gait_angular_velocity_threshold
        )
    gait.update(np.zeros(3), ay, omega)
    assert gait.walk_mode is expected_mode


def test_pitch_rate_alone_does_not_trigger_gait():
    gait = XgGaitPhaseGenerator(XgTsGaitConfig())
    for pitch_rate in [-10.0, 10.0]:
        gait.update(np.zeros(3), 0.0, np.array([0.0, pitch_rate, 0.0]))
        assert not gait.walk_mode
        np.testing.assert_array_equal(gait.observation, [-0.5, -0.5])


def test_generator_reset_and_reads_do_not_advance_clock():
    gait = XgGaitPhaseGenerator(XgTsGaitConfig())
    np.testing.assert_array_equal(gait.observation, [-0.5, -0.5])
    gait.update(np.array([0.0, 0.2, 0.0]), 0.0, np.zeros(3))
    before = gait.clock
    observation = gait.observation
    observation[:] = 99.0
    assert gait.clock == before
    assert (gait.observation < 0.5).all()
    gait.reset()
    assert gait.clock == gait.phase_a == gait.phase_b == 0.0
    assert not gait.walk_mode
    np.testing.assert_array_equal(gait.observation, [-0.5, -0.5])


def test_stop_on_first_tick_resets_at_inclusive_dt_boundary():
    gait = XgGaitPhaseGenerator(XgTsGaitConfig())
    gait.update(np.array([0.0, 0.2, 0.0]), 0.0, np.zeros(3))
    assert gait.clock == gait.dt
    gait.update(np.zeros(3), 0.0, np.zeros(3))
    assert gait.clock == 0.0
    np.testing.assert_array_equal(gait.observation, [-0.5, -0.5])


@pytest.mark.parametrize(
    "overrides",
    [
        {"gait_period": 0.0}, {"simulation_dt": 0.0}, {"gait_period": float("nan")},
        {"gait_command_threshold": -1.0},
        {"gait_acceleration_threshold": float("inf")},
        {"gait_acceleration_threshold": -1.0},
        {"gait_angular_velocity_threshold": float("nan")},
        {"gait_angular_velocity_threshold": -1.0},
    ],
)
def test_invalid_gait_parameters_fail_at_startup(overrides):
    with pytest.raises(ValueError):
        XgGaitPhaseGenerator(XgTsGaitConfig(**overrides))


@pytest.mark.parametrize(
    "yaw, linear_acceleration, angular_velocity, angular_acceleration",
    [
        (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        (np.pi / 2, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        (np.pi / 2, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 3.0)),
        (np.pi / 2, (0.0, 0.0, 0.0), (0.0, 0.0, 2.0), (0.0, 0.0, 0.0)),
        (0.7, (2.0, -1.0, 3.0), (0.2, 0.3, -0.7), (-0.5, 0.8, 1.2)),
    ],
)
def test_base_gait_state_includes_rotation_and_com_acceleration(
    yaw, linear_acceleration, angular_velocity, angular_acceleration
):
    deploy = object.__new__(XgTsGaitDeploy)
    deploy.robot = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    deploy.data = mujoco.MjData(deploy.robot)
    deploy._base_body_id = 1
    deploy._base_dof_adr = 0
    deploy._base_com_jacobian = np.zeros((3, deploy.robot.nv))
    deploy._base_com_jacobian_dot = np.zeros_like(deploy._base_com_jacobian)
    deploy.data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    deploy.data.qvel[3:6] = angular_velocity
    mujoco.mj_forward(deploy.robot, deploy.data)
    # Prescribe kinematic accelerations after forward dynamics so the oracle is
    # independent of this minimal body's gravity and inertial parameters.
    deploy.data.qacc[:3] = linear_acceleration
    deploy.data.qacc[3:6] = angular_acceleration
    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    offset = np.array([0.1, 0.2, 0.3])
    omega = np.array(angular_velocity)
    expected = (
        rotation.T @ np.array(linear_acceleration)
        + np.cross(angular_acceleration, offset)
        + np.cross(omega, np.cross(omega, offset))
    )
    ay, measured_omega = deploy._base_gait_state()
    assert ay == pytest.approx(expected[1], abs=1.0e-12)
    np.testing.assert_array_equal(measured_omega, omega)


def test_phase_advances_once_per_completed_interval_using_previous_command(monkeypatch):
    deploy = object.__new__(XgTsGaitDeploy)
    deploy.robot = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    deploy.data = mujoco.MjData(deploy.robot)
    deploy.counter = 0
    deploy.control_decimation = 40
    deploy.cmd = np.zeros(3, dtype=np.float32)
    deploy.gait = XgGaitPhaseGenerator(XgTsGaitConfig())
    deploy._base_gait_state = lambda: (0.0, np.zeros(3))
    policy_phases = []

    def parent_step(self):
        if self.counter % self.control_decimation == 0:
            policy_phases.append(self.gait.observation)
            self.cmd[:] = [0.0, 0.2 if self.counter < 80 else 0.0, 0.0]
        self.counter += 1

    monkeypatch.setattr(XgTsDeploy, "step", parent_step)
    for _ in range(81):
        deploy.step()
    np.testing.assert_array_equal(policy_phases[0], [-0.5, -0.5])
    assert len(policy_phases) == 3
    assert deploy.gait.clock == np.float32(0.04)
    assert deploy.gait.walk_mode
    assert deploy.cmd[1] == 0.0
    assert policy_phases[1][0] == pytest.approx(0.02 / 0.36 - 0.5)
    assert policy_phases[2][0] == pytest.approx(0.04 / 0.36 - 0.5)
    for _ in range(40):
        deploy.step()
    assert deploy.gait.clock == np.float32(0.06)
    assert not deploy.gait.walk_mode


def test_observation_appends_phases_and_does_not_advance_generator(monkeypatch):
    deploy = object.__new__(XgTsGaitDeploy)
    deploy.obs = np.zeros(55, dtype=np.float32)
    deploy.gait = XgGaitPhaseGenerator(XgTsGaitConfig())
    deploy.gait.update(np.array([0.0, 0.2, 0.0]), 0.0, np.zeros(3))
    deploy.send_plotjuggler_data = lambda *args: None
    def parent_observations(self):
        self.obs[:53] = np.arange(53)

    monkeypatch.setattr(XgTsDeploy, "update_obs", parent_observations)
    clock = deploy.gait.clock
    for _ in range(3):
        deploy.update_obs()
    np.testing.assert_array_equal(deploy.obs[:53], np.arange(53))
    np.testing.assert_array_equal(deploy.obs[53:], deploy.gait.observation)
    assert deploy.gait.clock == clock


def _policy_values(shapes):
    return [
        SimpleNamespace(name=name, shape=list(shape), type="tensor(float)")
        for name, shape in shapes.items()
    ]


@pytest.mark.parametrize("fault", ["53_dim", "state", "type", "names", "output", "config"])
def test_signature_rejects_incompatible_models(fault):
    deploy = object.__new__(XgTsGaitDeploy)
    deploy.config = XgTsGaitConfig(num_obs=53 if fault == "config" else 55)
    inputs = _policy_values({
        "policy_obs": (1, 55), "h_in": (3, 1, 256), "c_in": (3, 1, 256),
    })
    outputs = _policy_values({
        "actions": (1, 16), "h_out": (3, 1, 256), "c_out": (3, 1, 256),
    })
    if fault == "53_dim":
        inputs[0].shape = [1, 53]
    elif fault == "state":
        inputs[1].shape = [1, 1, 256]
    elif fault == "type":
        inputs[0].type = "tensor(double)"
    elif fault == "names":
        inputs[0].name = "obs"
    elif fault == "output":
        outputs[0].shape = [1, 12]
    deploy.policy = SimpleNamespace(get_inputs=lambda: inputs, get_outputs=lambda: outputs)
    with pytest.raises(ValueError):
        deploy._validate_student_policy_signature()


class ScriptedGamepad:
    command = (0.0, 0.0, 0.0)

    def poll(self):
        return True

    def get_cmd(self):
        return self.command

    def is_axis_pressed(self, *args, **kwargs):
        return False

    def is_button_pressed(self, *args, **kwargs):
        return False


def test_real_model_runs_500_control_intervals_headless(monkeypatch):
    def no_hardware_gamepad(self):
        self.gamepad = ScriptedGamepad()
        self.remote_enabled = False

    monkeypatch.setattr(XgTsGaitDeploy, "_init_gamepad", no_hardware_gamepad)
    deploy = XgTsGaitDeploy(XgTsGaitConfig(plotjuggler_enabled=False))
    assert not Path(deploy._scene_directory.name).exists()
    deploy.reset()
    mujoco.mj_forward(deploy.robot, deploy.data)
    np.testing.assert_array_equal(deploy.obs[53:], [-0.5, -0.5])
    initial_phases = None
    commands = [
        (0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0),
        (0.0, 0.0, 0.2), (0.0, 0.0, 0.0),
    ]
    for command in commands:
        deploy.gamepad.command = command
        for _ in range(100):
            for _ in range(deploy.control_decimation):
                deploy.step()
            if initial_phases is None:
                initial_phases = deploy.obs[53:].copy()
            assert deploy.model_in.shape == (55,)
            for values in (deploy.obs, deploy.action, deploy.tau, deploy.h_in, deploy.c_in,
                           deploy.data.qpos, deploy.data.qvel):
                assert np.isfinite(values).all()
    np.testing.assert_array_equal(initial_phases, [-0.5, -0.5])
    assert deploy.counter == 500 * deploy.control_decimation
    print(f"XG_TS_GAIT_HEADLESS_OK intervals=500 time={deploy.data.time:.2f}s")
    deploy.reset()
    assert deploy.counter == 0
    assert deploy.gait.clock == 0
    np.testing.assert_array_equal(deploy.obs[53:], [-0.5, -0.5])
    assert not deploy.h_in.any() and not deploy.c_in.any()


def test_run_script_discovers_xg_ts_gait(monkeypatch):
    executed = []
    monkeypatch.setattr(run_script, "configure_headless_mujoco_gl", lambda: False)
    monkeypatch.setattr(
        run_script.runpy, "run_path",
        lambda path, run_name: executed.append((Path(path), run_name)),
    )
    run_script.main(["--filename=xg_ts_gait"])
    assert executed == [(Path(run_script.WORK_DIR) / "scripts/xg_ts_gait.py", "__main__")]
