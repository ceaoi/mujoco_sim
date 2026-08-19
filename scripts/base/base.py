import os
import time
import warnings
from dataclasses import replace
from importlib import import_module
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from pathlib import Path

from mujoco_sim.configs import MujocoSimConfig
from mujoco_sim.utils.gamepad_pygame import Gamepad
from mujoco_sim.utils.projectile import ProjectileManager


BASE_HUD_UPDATE_HZ = 20.0


def _find_default_base_free_joint(model):
    free_joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if free_joint_ids.size == 0:
        raise ValueError("The model must contain a free joint for the default base HUD.")

    joint_id = int(free_joint_ids[0])
    return (
        joint_id,
        int(model.jnt_bodyid[joint_id]),
        int(model.jnt_qposadr[joint_id]),
        int(model.jnt_dofadr[joint_id]),
    )


def _base_state_from_free_joint(data, qpos_adr, dof_adr):
    base_quat = data.qpos[qpos_adr + 3:qpos_adr + 7].copy()
    w, x, y, z = base_quat
    yaw_world = float(np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    ))

    linear_velocity_world = data.qvel[dof_adr:dof_adr + 3].copy()
    rotation_base_to_world = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation_base_to_world, base_quat)
    linear_velocity_base = (
        rotation_base_to_world.reshape(3, 3).T @ linear_velocity_world
    )
    angular_velocity_base = data.qvel[dof_adr + 3:dof_adr + 6].copy()

    return (
        float(data.qpos[qpos_adr + 2]),
        yaw_world,
        angular_velocity_base,
        linear_velocity_base,
    )


class MujocoDeploy:

    mujoco_workspace_dir = str(Path(__file__).resolve().parents[2])

    def __init__(self, config: MujocoSimConfig, device="cpu"):
        self.config = config
        self.device = device
        self._init_plotjuggler()

        self.control_decimation = config.control_decimation
        self.sim_dt = config.simulation_dt

        self.num_obs = int(config.num_obs)
        self.num_actions = int(config.num_actions)
        self.num_obs_hist = int(config.num_obs_hist)
        self.obs_hist_dim = self.num_obs * self.num_obs_hist

        self.cmd_range = np.array(config.cmd_range, dtype=np.float32)
        self.cmd_deadzone = np.array(config.cmd_deadzone, dtype=np.float32)
        self.cmd = np.array(config.cmd_init, dtype=np.float32)

        ball_xml_path = f"{self.mujoco_workspace_dir}/assets/ball/ball.xml"

        merged_xml_path = self._build_merged_xml(
            config.xml_path,
            ball_xml_path,
            config.terrain_xml_path,
        )
        self.robot = mujoco.MjModel.from_xml_path(merged_xml_path)
        self.data = mujoco.MjData(self.robot)
        (
            self._base_joint_id,
            self._base_body_id,
            self._base_qpos_adr,
            self._base_dof_adr,
        ) = _find_default_base_free_joint(self.robot)
        self.robot.opt.timestep = self.sim_dt
        self.ctrl_dt = self.sim_dt * self.control_decimation
        self._base_hud_update_interval = max(
            1,
            round(1.0 / (BASE_HUD_UPDATE_HZ * self.robot.opt.timestep)),
        )

        self.gamepad = Gamepad(joystick_index=0)
        self.gamepad.connect()

        self.counter = 0
        self.viewer = None

        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.obs_hist = np.zeros(self.obs_hist_dim, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.tau = np.zeros(self.num_actions, dtype=np.float32)

        self.follow_camera = True
        self.prev_l2_pressed = False
        self.prev_r2_pressed = False

        self.projectile_manager = ProjectileManager(self.robot, self.data)
        self.prev_a_pressed = False

        self._init_control()

    # ---- optional telemetry ----

    def _init_plotjuggler(self) -> None:
        self.plotjuggler_enabled = bool(self.config.plotjuggler_enabled)
        self.plotjuggler = None
        if not self.plotjuggler_enabled:
            return

        try:
            data_vis = import_module("data_vis")
            self.plotjuggler = data_vis.PlotJugglerUDP("127.0.0.1", 5005)
        except Exception as exc:
            self._disable_plotjuggler("initialization failed", exc)

    def _disable_plotjuggler(self, reason: str, exc: Exception) -> None:
        self.plotjuggler_enabled = False
        self.plotjuggler = None
        if self.config.plotjuggler_enabled:
            self.config = replace(self.config, plotjuggler_enabled=False)
        warnings.warn(
            f"PlotJuggler {reason}; telemetry disabled: {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    def send_plotjuggler_data(self, name: str, value: Any) -> None:
        if not self.plotjuggler_enabled or self.plotjuggler is None:
            return

        try:
            self.plotjuggler.send_data(name, value)
        except Exception as exc:
            self._disable_plotjuggler("send failed", exc)

    # ---- hooks (subclass overrides) ----

    def _init_control(self):
        pass

    def _reset_control(self):
        pass

    # ---- abstract (subclass must implement) ----

    def update_obs(self):
        pass

    def update_model_in(self):
        pass

    def update_action(self):
        pass

    def update_tau(self):
        pass

    # ---- common simulation loop ----

    def reset(self):
        mujoco.mj_resetData(self.robot, self.data)
        self.counter = 0
        self.obs[:] = 0.0
        self.obs_hist[:] = 0.0
        self.action[:] = 0.0
        self.tau[:] = 0.0
        self.cmd = np.array(self.config.cmd_init, dtype=np.float32)

        self.prev_l2_pressed = False
        self.prev_r2_pressed = False
        self.prev_a_pressed = False
        self.projectile_manager.reset()

        self._reset_control()

    def run(self, duration=1e3):
        self.reset()
        start = time.time()

        with mujoco.viewer.launch_passive(self.robot, self.data) as viewer:
            self.viewer = viewer
            next_tick = time.perf_counter()
            while viewer.is_running() and (time.time() - start < duration):
                next_tick += self.robot.opt.timestep
                self.step()

                if self.counter % 10 == 0:
                    with viewer.lock():
                        if self.follow_camera:
                            self.set_camera_follow()
                    if self.counter % self._base_hud_update_interval == 0:
                        self._update_base_state_hud()
                    viewer.sync()

                remain = next_tick - time.perf_counter()
                if remain > 0:
                    if remain > 5e-4:
                        time.sleep(remain)
                else:
                    next_tick = time.perf_counter()

    def step(self):
        if self.counter % self.control_decimation == 0:
            ok = self.gamepad.poll()
            if not ok:
                print("Warning: [deploy_mujoco] Gamepad not connected or failed to read.")
            self.handle_gamepad_events()

            self.update_cmd()
            self.update_obs()
            self.obs_hist[:-self.num_obs] = self.obs_hist[self.num_obs:]
            self.obs_hist[-self.num_obs:] = self.obs
            self.update_model_in()
            self.update_action()

        self.update_tau()
        self.data.ctrl[:] = self.tau

        mujoco.mj_step(self.robot, self.data)
        self.counter += 1

    def update_cmd(self):
        self.cmd = np.asarray(self.gamepad.get_cmd(), dtype=np.float32) * self.cmd_range
        self.cmd[0] = 0.0 if abs(self.cmd[0]) < self.cmd_deadzone[0] else self.cmd[0]
        self.cmd[1] = 0.0 if abs(self.cmd[1]) < self.cmd_deadzone[1] else self.cmd[1]
        self.cmd[2] = 0.0 if abs(self.cmd[2]) < self.cmd_deadzone[2] else self.cmd[2]

    # ---- viewer ----

    def _update_base_state_hud(self):
        if self.viewer is None:
            return

        z_world, yaw_world, angular_velocity_base, linear_velocity_base = (
            _base_state_from_free_joint(
                self.data,
                self._base_qpos_adr,
                self._base_dof_adr,
            )
        )
        labels = "\n".join((
            "Base state",
            "z_world [m]",
            "yaw_world [rad]",
            "omega_base [rad/s]",
            "velocity_base [m/s]",
        ))
        values = "\n".join((
            "",
            f"{z_world:+.3f}",
            f"{yaw_world:+.3f}",
            "[" + ", ".join(f"{value:+.3f}" for value in angular_velocity_base) + "]",
            "[" + ", ".join(f"{value:+.3f}" for value in linear_velocity_base) + "]",
        ))
        self.viewer.set_texts((
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            labels,
            values,
        ))

    def set_camera_follow(self):
        if self.viewer is None:
            return

        base_pos = self.data.qpos[self._base_qpos_adr:self._base_qpos_adr + 3].copy()
        camera_offset = np.array([-2.0, 0.0, 1.0], dtype=np.float32)

        self.viewer.cam.lookat[:] = base_pos
        self.viewer.cam.distance = float(np.linalg.norm(camera_offset))
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -20

    def handle_gamepad_events(self):
        l2_pressed = self.gamepad.is_axis_pressed("L2", threshold=0.5)
        r2_pressed = self.gamepad.is_axis_pressed("R2", threshold=0.5)

        if (not self.prev_r2_pressed) and r2_pressed:
            self.follow_camera = not self.follow_camera
            mode = "FOLLOW" if self.follow_camera else "FIXED"
            print(f"[deploy_mujoco] Camera mode -> {mode}")

        if (not self.prev_l2_pressed) and l2_pressed:
            print("[deploy_mujoco] Reset Mujoco")
            self.reset()

        self.prev_r2_pressed = r2_pressed
        self.prev_l2_pressed = l2_pressed

        a_pressed = self.gamepad.is_button_pressed("A")

        if (not self.prev_a_pressed) and a_pressed:
            self.projectile_manager.spawn_ball_towards_robot(speed=6.0)

        self.prev_a_pressed = a_pressed

    def _build_merged_xml(self, robot_xml_path, ball_xml_path, terrain_xml_path=None):
        robot_xml_path = Path(robot_xml_path).resolve()
        ball_xml_path = Path(ball_xml_path).resolve()

        if not robot_xml_path.exists():
            raise FileNotFoundError(f"robot xml not found: {robot_xml_path}")
        if not ball_xml_path.exists():
            raise FileNotFoundError(f"ball xml not found: {ball_xml_path}")

        out_dir = robot_xml_path.parent
        merged_xml_path = out_dir / "tmp_merged.xml"

        robot_rel = os.path.relpath(robot_xml_path, start=out_dir)
        ball_rel = os.path.relpath(ball_xml_path, start=out_dir)

        includes = f"""<include file="{robot_rel}"/>
        <include file="{ball_rel}"/>"""

        if terrain_xml_path is not None:
            terrain_xml_path = Path(terrain_xml_path).resolve()
            if not terrain_xml_path.exists():
                raise FileNotFoundError(f"terrain xml not found: {terrain_xml_path}")
            terrain_rel = os.path.relpath(terrain_xml_path, start=out_dir)
            includes += f"""
        <include file="{terrain_rel}"/>"""

        merged_text = f"""<mujoco model="merged_scene">
        {includes}
    </mujoco>
    """

        merged_xml_path.write_text(merged_text, encoding="utf-8")

        print(f"[deploy_mujoco] Temporary merged xml created: {merged_xml_path}")
        return str(merged_xml_path)
