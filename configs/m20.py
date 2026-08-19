from dataclasses import dataclass
from pathlib import Path

from .base import MUJOCO_SIM_ROOT, PROJECT_ROOT, WheelLeggedConfig


@dataclass(frozen=True, kw_only=True)
class M20FlatConfig(WheelLeggedConfig):
    policy_path: Path = PROJECT_ROOT / "logs/rsl_rl/m20_flat/1_exported/policy.onnx"
    xml_path: Path = MUJOCO_SIM_ROOT / "assets/robots/M20_mjcf/mjcf/M20.xml"

    simulation_dt: float = 0.0005
    control_decimation: int = 40

    gait_freq: float = 2.0
    no_constraint_lin_acc_threshold: float = 2.0
    stop_hold_time_factor: float = 1.5
    switch_threshold: float = -0.1
    gait_offset_leftward: tuple[float, ...] = (0.5, 0.0, 0.0, 0.5)
    gait_offset_rightward: tuple[float, ...] = (0.0, 0.5, 0.5, 0.0)

    leg_joint_idx: tuple[int, ...] = (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14)
    leg_joint_idxx: tuple[int, ...] = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14)
    wheel_joint_idx: tuple[int, ...] = (3, 7, 11, 15)
    leg_actions_to_mujoco: tuple[int, ...] = (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14)
    wheel_actions_to_mujoco: tuple[int, ...] = (3, 7, 11, 15)

    kpsPos: tuple[float, ...] = (80.0,) * 12
    kdsPos: tuple[float, ...] = (2.0,) * 12
    kpsVel: tuple[float, ...] = (0.0,) * 4
    kdsVel: tuple[float, ...] = (0.6,) * 4
    default_angles_leg: tuple[float, ...] = (
        0.0, 0.0, 0.0, 0.0,
        -0.6, -0.6, 0.6, 0.6,
        1.0, 1.0, -1.0, -1.0,
    )

    max_command_rate: tuple[float, ...] = (2.0, 2.0, 2.0)
    action_scale_pos: float = 0.5
    action_scale_vel: float = 20.0

    num_actions: int = 16
    num_actions_pos: int = 12
    num_obs: int = 53
    num_obs_hist: int = 10

    cmd_init: tuple[float, ...] = (0.0, 0.0, 0.0)
    cmd_range: tuple[float, ...] = (2.0, 1.0, 1.0)
    cmd_deadzone: tuple[float, ...] = (0.01, 0.01, 0.01)
    wheel_action_vel_deadzone: float = 0.1
    wheel_stop_pid_enabled: bool = False
    wheel_stop_pid_kp: float = 0.0
    wheel_stop_pid_ki: float = 0.0
    wheel_stop_pid_kd: float = 0.0
    wheel_stop_pid_output_limit: float = 5.0
    is_rnn: bool = True


@dataclass(frozen=True, kw_only=True)
class M20RoughConfig(M20FlatConfig):
    policy_path: Path = PROJECT_ROOT / "logs/rsl_rl/m20_rough/1_exported/policy.onnx"
    terrain_xml_path: Path | None = MUJOCO_SIM_ROOT / "assets/terrains/rough_stairs.xml"
    max_command_rate: tuple[float, ...] = (3.0, 3.0, 3.0)
    cmd_range: tuple[float, ...] = (1.0, 1.0, 1.0)
