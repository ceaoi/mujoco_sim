import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from mujoco_sim.configs import (
    MUJOCO_SIM_ROOT,
    PROJECT_ROOT,
    M20FlatConfig,
    M20RoughConfig,
    MujocoSimConfig,
    Wh044xConfig,
    WheelLeggedConfig,
    Zb02wFlatConfig,
    Zb02wDepthConfig,
    Zb02wGyrateConfig,
    Zb02wRoughConfig,
    Zb02wTsConfig,
)
from mujoco_sim.utils.gait_phase_generator import GaitPhaseGenerator


def test_configuration_hierarchy_uses_flat_as_each_robot_base():
    assert issubclass(WheelLeggedConfig, MujocoSimConfig)
    assert issubclass(M20RoughConfig, M20FlatConfig)
    assert issubclass(Zb02wRoughConfig, Zb02wFlatConfig)
    assert issubclass(Zb02wTsConfig, Zb02wRoughConfig)
    assert issubclass(Zb02wDepthConfig, Zb02wTsConfig)
    assert issubclass(Wh044xConfig, MujocoSimConfig)


def test_m20_rough_only_overrides_scene_and_command_parameters():
    flat = M20FlatConfig()
    rough = M20RoughConfig()

    assert flat.policy_path == PROJECT_ROOT / "logs/rsl_rl/m20_flat/1_exported/policy.onnx"
    assert rough.policy_path == PROJECT_ROOT / "logs/rsl_rl/m20_rough/1_exported/policy.onnx"
    assert flat.terrain_xml_path is None
    assert rough.terrain_xml_path == MUJOCO_SIM_ROOT / "assets/terrains/rough_stairs.xml"
    assert rough.max_command_rate == (3.0, 3.0, 3.0)
    assert rough.cmd_range == (1.0, 1.0, 1.0)
    assert rough.xml_path == flat.xml_path
    assert rough.kpsPos == flat.kpsPos
    assert rough.gait_freq == flat.gait_freq


def test_zb02w_rough_and_student_configs_inherit_expected_values():
    flat = Zb02wFlatConfig()
    rough = Zb02wRoughConfig()
    student = Zb02wTsConfig()

    assert flat.terrain_xml_path is None
    assert rough.terrain_xml_path == MUJOCO_SIM_ROOT / "assets/terrains/rough_stairs.xml"
    assert student.terrain_xml_path == rough.terrain_xml_path
    assert student.xml_path == flat.xml_path
    assert student.kpsPos == flat.kpsPos
    assert student.policy_path == PROJECT_ROOT / "logs/rsl_rl/zb02w_student/1_exported/policy.onnx"
    assert student.max_command_rate == (10.0, 4.0, 4.0)
    assert student.cmd_range == (4.0, 1.5, 1.5)
    assert student.cmd_deadzone == (0.0001, 0.01, 0.01)
    assert student.wheel_stop_pid_enabled
    assert student.wheel_stop_pid_ki == 2.0
    assert student.wheel_stop_pid_output_limit == 20.0
    assert flat.plotjuggler_enabled
    assert rough.plotjuggler_enabled
    assert student.plotjuggler_enabled


def test_zb02w_gyrate_config_uses_feedforward_policy_and_actor_observation_size():
    flat = Zb02wFlatConfig()
    gyrate = Zb02wGyrateConfig()

    assert isinstance(gyrate, Zb02wFlatConfig)
    assert gyrate.policy_path == PROJECT_ROOT / "logs/rsl_rl/gyrate/1_exported/policy.onnx"
    assert gyrate.policy_path.is_absolute()
    assert gyrate.num_obs == 50
    assert gyrate.num_actions == 16
    assert not gyrate.is_rnn
    assert gyrate.xml_path == flat.xml_path
    assert gyrate.terrain_xml_path is None


def test_zb02w_depth_config_inherits_student_policy_and_camera_defaults():
    student = Zb02wTsConfig()
    depth = Zb02wDepthConfig()

    assert depth.policy_path == student.policy_path
    assert depth.wheel_stop_pid_enabled == student.wheel_stop_pid_enabled
    assert depth.wheel_stop_pid_ki == student.wheel_stop_pid_ki
    assert depth.terrain_xml_path == student.terrain_xml_path
    assert depth.depth_camera_name == "depth_camera"
    assert depth.depth_camera_link == "base_link"
    assert depth.depth_camera_pos == (0.375, 0.0175, 0.10225)
    assert depth.depth_camera_quat == pytest.approx(
        (math.cos(math.pi / 8.0), 0.0, math.sin(math.pi / 8.0), 0.0)
    )
    assert depth.depth_camera_width == 64
    assert depth.depth_camera_height == 36
    assert depth.depth_camera_fovy == pytest.approx(47.83)
    assert depth.depth_camera_near == pytest.approx(0.05)
    assert depth.depth_camera_update_period == pytest.approx(1.0 / 60.0)
    assert depth.depth_min == pytest.approx(0.05)
    assert depth.depth_max == pytest.approx(3.0)
    assert depth.depth_camera_display
    assert depth.depth_camera_display_update_period == pytest.approx(1.0 / 10.0)
    assert depth.depth_camera_display_scale == 4
    assert depth.depth_pointcloud_display
    assert depth.depth_pointcloud_stride == 1
    assert depth.depth_pointcloud_radius == pytest.approx(0.01)


def test_configs_use_absolute_paths_immutable_sequences_and_frozen_instances():
    config = M20FlatConfig()

    assert isinstance(config.xml_path, Path)
    assert isinstance(config.policy_path, Path)
    assert config.xml_path.is_absolute()
    assert config.policy_path.is_absolute()
    assert isinstance(config.cmd_range, tuple)
    assert isinstance(config.leg_joint_idx, tuple)

    with pytest.raises(FrozenInstanceError):
        config.simulation_dt = 0.001  # type: ignore[misc]

    state = np.arange(16)
    joint_idx = np.asarray(config.leg_joint_idx, dtype=np.intp)
    np.testing.assert_array_equal(state[joint_idx], state[list(config.leg_joint_idx)])


def test_wh044x_config_keeps_chassis_specific_paths_and_dimensions():
    config = Wh044xConfig()

    assert config.xml_path == PROJECT_ROOT / "robots/wh044x/mjcf/wh044x_generated.xml"
    assert config.chassis_build_dir == PROJECT_ROOT / "build"
    assert config.simulation_dt == 0.001
    assert config.control_decimation == 20
    assert config.num_actions == 8
    assert not config.plotjuggler_enabled


def test_gait_phase_generator_reads_m20_config_object():
    config = M20FlatConfig()
    gait = GaitPhaseGenerator(config)

    assert gait.gait_freq == config.gait_freq
    assert gait.no_constraint_lin_acc_threshold == config.no_constraint_lin_acc_threshold
    assert gait.stop_hold_time_factor == config.stop_hold_time_factor
    assert gait.switch_threshold == config.switch_threshold
    assert float(gait._dt) == pytest.approx(
        config.control_decimation * config.simulation_dt
    )
    np.testing.assert_allclose(gait.gait_offset_leftward, config.gait_offset_leftward)
    np.testing.assert_allclose(gait.gait_offset_rightward, config.gait_offset_rightward)
