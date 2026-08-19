from dataclasses import dataclass
from pathlib import Path


MUJOCO_SIM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MUJOCO_SIM_ROOT.parent

FloatTuple = tuple[float, ...]
IntTuple = tuple[int, ...]


@dataclass(frozen=True, kw_only=True)
class MujocoSimConfig:
    """Common configuration for a MuJoCo deployment."""

    xml_path: Path
    simulation_dt: float
    control_decimation: int
    num_actions: int
    num_obs: int
    num_obs_hist: int
    cmd_range: FloatTuple
    cmd_deadzone: FloatTuple
    cmd_init: FloatTuple = (0.0, 0.0, 0.0)
    terrain_xml_path: Path | None = None
    plotjuggler_enabled: bool = False


@dataclass(frozen=True, kw_only=True)
class WheelLeggedConfig(MujocoSimConfig):
    """Shared policy and controller configuration for wheel-legged robots."""

    policy_path: Path
    leg_joint_idx: IntTuple
    leg_joint_idxx: IntTuple
    wheel_joint_idx: IntTuple
    leg_actions_to_mujoco: IntTuple
    wheel_actions_to_mujoco: IntTuple
    kpsPos: FloatTuple
    kdsPos: FloatTuple
    kpsVel: FloatTuple
    kdsVel: FloatTuple
    default_angles_leg: FloatTuple
    max_command_rate: FloatTuple
    action_scale_pos: float
    action_scale_vel: float
    num_actions_pos: int
    wheel_action_vel_deadzone: float
    wheel_stop_pid_enabled: bool = False
    wheel_stop_pid_kp: float = 0.0
    wheel_stop_pid_ki: float = 0.0
    wheel_stop_pid_kd: float = 0.0
    wheel_stop_pid_output_limit: float = 5.0
    is_rnn: bool = False
    plotjuggler_enabled: bool = True
