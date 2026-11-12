from __future__ import annotations

import math

import numpy as np

from mujoco_sim.configs import M20FlatConfig, XgTsGaitConfig


class GaitPhaseGenerator:
    """Single-env gait generator for MuJoCo deployment.

    Mirrors the logic of IsaacLab `GaitStateCommand` and outputs
    `[phase, clock_fl, clock_fr, clock_hl, clock_hr]`.
    """

    def __init__(self, config: M20FlatConfig):
        self.gait_freq = float(config.gait_freq)
        self.no_constraint_lin_acc_threshold = float(config.no_constraint_lin_acc_threshold)
        self.stop_hold_time_factor = float(config.stop_hold_time_factor)
        self.switch_threshold = float(config.switch_threshold)

        self.gait_offset_leftward = np.asarray(config.gait_offset_leftward, dtype=np.float32)
        self.gait_offset_rightward = np.asarray(config.gait_offset_rightward, dtype=np.float32)
        self._dt = np.asarray(
            config.control_decimation * config.simulation_dt,
            dtype=np.float32,
        )
        self.reset()

    def reset(self) -> None:
        self.gait_phase = np.float32(-1e-9)
        self.gait_clock = np.zeros(4, dtype=np.float32)
        self.gait_offsets = self.gait_offset_leftward.copy()

        self.stop_timer = np.float32(0.0)
        self.last_should_run_clock = False
        
    def _update_gait(self, cmd: np.ndarray, base_lin_acc_y: float) -> np.ndarray:
        # matches: is_active = norm(cmd[1:3]) > 0.0
        is_active = float(np.linalg.norm(cmd[1:3])) > 0.0
        # matches: is_unstable = (~is_active) & (abs(base_lin_acc_y) > threshold)
        is_unstable = (not is_active) and (abs(float(base_lin_acc_y)) > self.no_constraint_lin_acc_threshold)

        # stop-hold logic
        if is_active:
            self.stop_timer = np.float32(self.stop_hold_time_factor / self.gait_freq)
        else:
            self.stop_timer = np.float32(max(0.0, float(self.stop_timer - self._dt)))
        is_unstable = is_unstable or ((not is_active) and (self.stop_timer > 0.0))

        should_run_clock = is_active
        if should_run_clock:
            self.gait_phase = np.float32((float(self.gait_phase) + float(self._dt) * self.gait_freq) % 1.0)
        else:
            self.gait_phase = np.float32(-1e-9)

        if is_unstable:
            self.gait_phase = np.float32(-2e-9)

        # choose offsets from lateral/yaw command sign when starting a run segment
        is_leftward = (cmd[1] > 0.0) or ((cmd[1] == 0.0) and (cmd[2] > 0.0))
        new_offsets = self.gait_offset_leftward if is_leftward else self.gait_offset_rightward
        if (not self.last_should_run_clock) and should_run_clock:
            self.gait_offsets = new_offsets.copy()

        phase_per_leg = (float(self.gait_phase) + self.gait_offsets) * (1.0 if self.gait_phase >= 0.0 else 0.0)
        self.gait_clock = np.sin(2.0 * math.pi * phase_per_leg).astype(np.float32)
        if is_unstable:
            self.gait_clock[:] = self.switch_threshold

        self.last_should_run_clock = should_run_clock

        out = np.zeros(5, dtype=np.float32)
        out[0] = self.gait_phase
        out[1:] = self.gait_clock
        return out


class XgGaitPhaseGenerator:
    """Use XG's float32 gait clock with command and acceleration/rate triggers."""

    def __init__(self, config: XgTsGaitConfig):
        self.period = np.float32(config.gait_period)
        self.dt = np.float32(config.control_decimation * config.simulation_dt)
        self.command_threshold = np.float32(config.gait_command_threshold)
        self.acceleration_threshold = np.float32(config.gait_acceleration_threshold)
        self.angular_velocity_threshold = np.float32(config.gait_angular_velocity_threshold)
        if not np.isfinite(self.period) or self.period <= 0:
            raise ValueError("gait_period must be finite and positive")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("gait control period must be finite and positive")
        for name, threshold in (
            ("gait_command_threshold", self.command_threshold),
            ("gait_acceleration_threshold", self.acceleration_threshold),
            ("gait_angular_velocity_threshold", self.angular_velocity_threshold),
        ):
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.reset()

    def reset(self) -> None:
        self.clock = np.float32(0.0)
        self.phase_a = np.float32(0.0)
        self.phase_b = np.float32(0.0)
        self.walk_mode = False

    @property
    def observation(self) -> np.ndarray:
        """Read the centered phase pair without advancing the clock."""
        return np.array([self.phase_a, self.phase_b], dtype=np.float32) - np.float32(0.5)

    def update(
        self, cmd: np.ndarray, base_com_ay: float, base_ang_vel: np.ndarray
    ) -> np.ndarray:
        """Advance once after a completed control interval, using its command."""
        command = np.asarray(cmd, dtype=np.float32)
        angular_velocity = np.asarray(base_ang_vel, dtype=np.float32)
        self.walk_mode = bool(
            np.linalg.norm(command[1:3]) > self.command_threshold
            or np.abs(np.float32(base_com_ay)) > self.acceleration_threshold
            or np.abs(angular_velocity[0]) > self.angular_velocity_threshold
            or np.abs(angular_velocity[2]) > self.angular_velocity_threshold
        )
        reset = not self.walk_mode and np.remainder(self.clock, self.period) <= self.dt
        self.clock = np.float32(0.0) if reset else self.clock + self.dt
        self.phase_a = np.remainder(self.clock, self.period) / self.period
        self.phase_b = (
            np.float32(0.0)
            if reset
            else np.remainder(self.phase_a + np.float32(0.5), np.float32(1.0))
        )
        return self.observation
