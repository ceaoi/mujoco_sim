from dataclasses import dataclass
from pathlib import Path

from .base_cfg import PROJECT_ROOT


@dataclass(frozen=True, kw_only=True)
class RemoteFileSyncConfig:
    """Configuration for copying the newest matching remote file locally."""

    remote_url: str = "sftp://l40/data/chensj/wl_lab2/logs/rsl_rl/zb02w_teacher"
    file_pattern: str = "model_*.pt"
    local_dir: Path = PROJECT_ROOT / "logs/rsl_rl/zb02w_teacher/100_model"
