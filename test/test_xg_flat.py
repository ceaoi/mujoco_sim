from pathlib import Path

import numpy as np
import onnxruntime as ort

from mujoco_sim import run_script
from mujoco_sim.configs import (
    PROJECT_ROOT,
    WheelLeggedConfig,
    XgFlatConfig,
    XgTsConfig,
)
from mujoco_sim.scripts.xg_flat import XgFlatDeploy
from mujoco_sim.scripts.zb02w_flat import Zb02wFlatDeploy


def test_xg_flat_config_matches_feed_forward_training_interface():
    config = XgFlatConfig()

    assert isinstance(config, WheelLeggedConfig)
    assert config.policy_path == (
        PROJECT_ROOT / "logs/rsl_rl/xg_teacher_flat1/1_exported/policy.onnx"
    )
    assert config.xml_path == (
        PROJECT_ROOT / "../wl_loco/wl_loco/assets/robots/xg/mjcf/mjcf/xg.xml"
    )
    assert config.terrain_xml_path is None
    assert config.num_obs == 53
    assert config.num_actions == 16
    assert config.num_actions_pos == 12
    assert not config.is_rnn


def test_xg_flat_preserves_xg_joint_and_control_configuration():
    flat = XgFlatConfig()
    student = XgTsConfig()

    assert flat.leg_joint_idx == (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14)
    assert flat.wheel_joint_idx == (3, 7, 11, 15)
    assert flat.leg_actions_to_mujoco == flat.leg_joint_idx
    assert flat.wheel_actions_to_mujoco == flat.wheel_joint_idx
    assert flat.default_angles_leg == (
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
    assert flat.kpsPos == student.kpsPos
    assert flat.kdsPos == student.kdsPos
    assert flat.kpsVel == student.kpsVel
    assert flat.kdsVel == student.kdsVel
    assert flat.action_scale_pos == student.action_scale_pos
    assert flat.action_scale_vel == student.action_scale_vel


def test_xg_flat_onnx_is_single_input_feed_forward_policy():
    config = XgFlatConfig()
    policy = ort.InferenceSession(
        str(config.policy_path),
        providers=["CPUExecutionProvider"],
    )

    assert [(value.name, value.shape) for value in policy.get_inputs()] == [
        ("obs", [1, 53])
    ]
    assert [(value.name, value.shape) for value in policy.get_outputs()] == [
        ("actions", [1, 16])
    ]

    actions = policy.run(
        ["actions"],
        {"obs": np.zeros((1, 53), dtype=np.float32)},
    )[0]
    assert actions.shape == (1, 16)
    assert actions.dtype == np.float32
    assert np.isfinite(actions).all()


def test_xg_flat_deploy_does_not_create_or_pass_recurrent_state():
    deploy = object.__new__(XgFlatDeploy)
    deploy.config = XgFlatConfig()

    deploy._init_control()

    assert not hasattr(deploy, "h_in")
    assert not hasattr(deploy, "c_in")
    inputs = deploy._build_policy_inputs(np.zeros((1, 53), dtype=np.float32))
    assert list(inputs) == ["obs"]


def test_xg_flat_runs_one_control_step_with_read_only_external_mjcf(monkeypatch):
    merged_paths = []
    build_merged_xml = XgFlatDeploy._build_merged_xml

    def track_merged_xml(self, *args, **kwargs):
        path = build_merged_xml(self, *args, **kwargs)
        merged_paths.append(Path(path))
        return path

    monkeypatch.setattr(XgFlatDeploy, "_build_merged_xml", track_merged_xml)
    monkeypatch.setattr(XgFlatDeploy, "_init_gamepad", lambda self: None)

    deploy = XgFlatDeploy(XgFlatConfig(plotjuggler_enabled=False))
    deploy.reset()
    deploy.update_obs()
    deploy.update_model_in()
    deploy.update_action()
    deploy.update_tau()

    assert len(merged_paths) == 1
    assert not merged_paths[0].exists()
    assert deploy.model_in.shape == (53,)
    assert deploy.action.shape == (16,)
    assert deploy.tau.shape == (16,)
    assert np.isfinite(deploy.action).all()
    assert np.isfinite(deploy.tau).all()


def test_xg_flat_reset_uses_training_base_height(monkeypatch):
    monkeypatch.setattr(Zb02wFlatDeploy, "reset", lambda self: None)
    deploy = object.__new__(XgFlatDeploy)
    deploy.data = type("Data", (), {"qpos": np.zeros(23)})()

    deploy.reset()

    assert deploy.data.qpos[2] == 0.5082


def test_run_script_discovers_xg_flat_entrypoint(monkeypatch):
    executed = []
    monkeypatch.setattr(run_script, "configure_headless_mujoco_gl", lambda: False)
    monkeypatch.setattr(
        run_script.runpy,
        "run_path",
        lambda path, run_name: executed.append((Path(path), run_name)),
    )

    run_script.main(["--filename=xg_flat"])

    assert executed == [
        (Path(run_script.WORK_DIR) / "scripts/xg_flat.py", "__main__"),
    ]
