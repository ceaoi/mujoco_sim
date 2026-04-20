from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml


class GaitGenerator:
    """Single-env gait generator for MuJoCo deployment.

    Mirrors the logic of IsaacLab `GaitStateCommand` and outputs
    `[phase, clock_fl, clock_fr, clock_hl, clock_hr]`.
    """

    def __init__(self, yaml_path: str):
        yaml_path = Path(yaml_path)
        with yaml_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.gait_freq = float(cfg.get("gait_freq", 2.0))
        self.no_constraint_lin_acc_threshold = float(cfg.get("no_constraint_lin_acc_threshold", 4.0))
        self.stop_hold_time_factor = float(cfg.get("stop_hold_time_factor", 1.5))
        self.switch_threshold = float(cfg["switch_threshold"])

        self.gait_offset_leftward = np.asarray(cfg.get("gait_offset_leftward", [0.0, 0.5, 0.5, 0.0]), dtype=np.float32)
        self.gait_offset_rightward = np.asarray(cfg.get("gait_offset_rightward", [0.5, 0.0, 0.0, 0.5]), dtype=np.float32)
        self._dt = np.asarray(cfg["control_decimation"] * cfg["simulation_dt"], dtype=np.float32)
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