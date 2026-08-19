from dataclasses import dataclass
from pathlib import Path

from .base import PROJECT_ROOT, MujocoSimConfig


@dataclass(frozen=True, kw_only=True)
class Wh044xConfig(MujocoSimConfig):
    xml_path: Path = PROJECT_ROOT / "robots/wh044x/mjcf/wh044x_generated.xml"
    chassis_build_dir: Path = PROJECT_ROOT / "build"

    simulation_dt: float = 0.001
    control_decimation: int = 20

    num_actions: int = 8
    num_obs: int = 8
    num_obs_hist: int = 10

    cmd_init: tuple[float, ...] = (0.0, 0.0, 0.0)
    cmd_range: tuple[float, ...] = (1.0, 1.0, 2.0)
    cmd_deadzone: tuple[float, ...] = (0.01, 0.01, 0.01)
