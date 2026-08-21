import math
from dataclasses import dataclass
from pathlib import Path

from .base_cfg import MUJOCO_SIM_ROOT, PROJECT_ROOT, WheelLeggedConfig


@dataclass(frozen=True, kw_only=True)
class Zb02wFlatConfig(WheelLeggedConfig):
    policy_path: Path = PROJECT_ROOT / "logs/rsl_rl/zb02w_flat/1_exported/policy.onnx"
    xml_path: Path = PROJECT_ROOT / "source/manager_based/locomotion/assets/robots/zb02w/mjcf/mjcf/zb02w.xml"

    simulation_dt: float = 0.0005
    control_decimation: int = 40

    leg_joint_idx: tuple[int, ...] = (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14)
    leg_joint_idxx: tuple[int, ...] = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14)
    wheel_joint_idx: tuple[int, ...] = (3, 7, 11, 15)
    leg_actions_to_mujoco: tuple[int, ...] = (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14)
    wheel_actions_to_mujoco: tuple[int, ...] = (3, 7, 11, 15)

    kpsPos: tuple[float, ...] = (160.0,) * 12
    kdsPos: tuple[float, ...] = (3.0,) * 12
    kpsVel: tuple[float, ...] = (0.0,) * 4
    kdsVel: tuple[float, ...] = (0.5,) * 4
    default_angles_leg: tuple[float, ...] = (
        0.0, 0.0, 0.0, 0.0,
        0.4, 0.4, -0.4, -0.4,
        -0.8, -0.8, 0.8, 0.8,
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
class Zb02wRoughConfig(Zb02wFlatConfig):
    policy_path: Path = PROJECT_ROOT / "logs/rsl_rl/zb02w_rough/1_exported/policy.onnx"
    terrain_xml_path: Path | None = MUJOCO_SIM_ROOT / "assets/terrains/rough_stairs.xml"
    max_command_rate: tuple[float, ...] = (3.0, 3.0, 3.0)
    cmd_range: tuple[float, ...] = (1.0, 1.0, 1.0)


@dataclass(frozen=True, kw_only=True)
class Zb02wTsConfig(Zb02wRoughConfig):
    policy_path: Path = PROJECT_ROOT / "logs/rsl_rl/zb02w_student/1_exported/policy.onnx"
    max_command_rate: tuple[float, ...] = (10.0, 4.0, 4.0)
    cmd_range: tuple[float, ...] = (4.0, 1.5, 1.5)
    cmd_deadzone: tuple[float, ...] = (0.0001, 0.01, 0.01)
    wheel_action_vel_deadzone: float = 0.0
    wheel_stop_pid_enabled: bool = True
    wheel_stop_pid_kp: float = 1.0
    wheel_stop_pid_ki: float = 2.0
    wheel_stop_pid_kd: float = 0.00005
    wheel_stop_pid_output_limit: float = 20.0


@dataclass(frozen=True, kw_only=True)
class Zb02wDepthConfig(Zb02wTsConfig):
    policy_path: Path = PROJECT_ROOT / "logs/rsl_rl/zb02w_student/1_exported/policy.onnx"
    cmd_range: tuple[float, ...] = (3.0, 1.5, 1.5)

    depth_camera_name: str = "depth_camera"
    depth_camera_link: str = "base_link"
    depth_camera_pos: tuple[float, float, float] = (0.375, 0.0175, 0.10225)
    depth_camera_quat: tuple[float, float, float, float] = (
        math.cos(0.5 * 45.0 * math.pi / 180.0),
        0.0,
        math.sin(0.5 * 45.0 * math.pi / 180.0),
        0.0,
    )
    depth_camera_width: int = 64
    depth_camera_height: int = 36
    depth_camera_fovy: float = 47.83
    depth_camera_near: float = 0.05
    depth_camera_update_period: float = 1.0 / 60.0
    depth_min: float = 0.3
    depth_max: float = 3.0
    depth_camera_display: bool = True
    depth_camera_display_update_period: float = 1.0 / 10.0
    depth_camera_display_scale: int = 4
    depth_pointcloud_display: bool = True
    depth_pointcloud_stride: int = 1
    depth_pointcloud_radius: float = 0.01
