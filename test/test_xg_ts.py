from pathlib import Path

import mujoco
import numpy as np

from mujoco_sim import run_script
from mujoco_sim.configs import (
    MUJOCO_SIM_ROOT,
    PROJECT_ROOT,
    WheelLeggedConfig,
    XgTsConfig,
)
from mujoco_sim.scripts.xg_ts import XgTsDeploy
from mujoco_sim.scripts.zb02w_flat import Zb02wFlatDeploy


def test_xg_ts_config_matches_training_and_student_interfaces():
    config = XgTsConfig()

    assert isinstance(config, WheelLeggedConfig)
    assert config.xml_path == (
        PROJECT_ROOT / "../wl_loco/wl_loco/assets/robots/xg/mjcf/mjcf/xg.xml"
    )
    assert config.policy_path == (
        PROJECT_ROOT / "logs/rsl_rl/xg_student/1_exported/policy.onnx"
    )
    assert config.terrain_xml_path == (
        MUJOCO_SIM_ROOT / "assets/terrains/rough_stairs.xml"
    )
    assert config.simulation_dt == 0.0005
    assert config.control_decimation == 40
    assert config.leg_joint_idx == (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14)
    assert config.leg_joint_idxx == (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14)
    assert config.wheel_joint_idx == (3, 7, 11, 15)
    assert config.leg_actions_to_mujoco == config.leg_joint_idx
    assert config.wheel_actions_to_mujoco == config.wheel_joint_idx
    assert config.default_angles_leg == (
        -0.1,
        0.1,
        -0.1,
        0.1,
        0.4,
        0.4,
        -0.4,
        -0.4,
        -0.8,
        -0.8,
        0.8,
        0.8,
    )
    assert config.num_actions == 16
    assert config.num_actions_pos == 12
    assert config.num_obs == 53
    assert config.is_rnn


def test_xg_ts_config_uses_reference_ts_command_and_stop_pid_settings():
    config = XgTsConfig()

    assert config.max_command_rate == (10.0, 4.0, 4.0)
    assert config.cmd_range == (4.0, 1.5, 1.5)
    assert config.cmd_deadzone == (0.0001, 0.01, 0.01)
    assert config.wheel_action_vel_deadzone == 0.0
    assert config.wheel_stop_pid_enabled
    assert config.wheel_stop_pid_kp == 1.0
    assert config.wheel_stop_pid_ki == 2.0
    assert config.wheel_stop_pid_kd == 0.00005
    assert config.wheel_stop_pid_output_limit == 20.0


def test_xg_xml_actuator_order_matches_config_indices():
    config = XgTsConfig()
    model = mujoco.MjModel.from_xml_path(str(config.xml_path))

    assert model.nu == 16
    assert [model.actuator(i).name for i in range(model.nu)] == [
        f"{leg}_{joint}_joint_motor"
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf", "wheel")
    ]


def test_xg_ts_reset_uses_training_base_height(monkeypatch):
    monkeypatch.setattr(Zb02wFlatDeploy, "reset", lambda self: None)
    deploy = object.__new__(XgTsDeploy)
    deploy.data = type("Data", (), {"qpos": np.zeros(23)})()

    deploy.reset()

    assert deploy.data.qpos[2] == 0.5082


def test_run_script_discovers_xg_ts_entrypoint(monkeypatch):
    executed = []
    monkeypatch.setattr(run_script, "configure_headless_mujoco_gl", lambda: False)
    monkeypatch.setattr(
        run_script.runpy,
        "run_path",
        lambda path, run_name: executed.append((Path(path), run_name)),
    )

    run_script.main(["--filename=xg_ts"])

    assert executed == [
        (Path(run_script.WORK_DIR) / "scripts/xg_ts.py", "__main__"),
    ]
