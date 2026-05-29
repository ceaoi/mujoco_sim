import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from pathlib import Path

from .utils.gamepad_pygame import Gamepad
from .utils.projectile import ProjectileManager


class MujocoDeploy:

    mujoco_workspace_dir = os.path.dirname(__file__)

    def __init__(self, yaml_filename, device="cpu"):
        with open(f"{self.mujoco_workspace_dir}/configs/{yaml_filename}", "r") as f:
            config: dict = yaml.load(f, Loader=yaml.FullLoader)  # type: ignore
            xml_path = config["xml_path"].replace("{mujoco_workspace_dir}", self.mujoco_workspace_dir)

        self.config = config
        self.device = device

        self.control_decimation = config["control_decimation"]
        self.sim_dt = config["simulation_dt"]

        self.num_obs = int(config["num_obs"])
        self.num_actions = int(config["num_actions"])
        self.num_obs_hist = int(config["num_obs_hist"])
        self.obs_hist_dim = self.num_obs * self.num_obs_hist

        self.cmd_range = np.array(config["cmd_range"], dtype=np.float32)
        self.cmd_deadzone = np.array(config["cmd_deadzone"], dtype=np.float32)
        self.cmd = np.array(config.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32)

        ball_xml_path = f"{self.mujoco_workspace_dir}/robots/ball/ball.xml"
        merged_xml_path = self._build_merged_xml(xml_path, ball_xml_path)
        self.robot = mujoco.MjModel.from_xml_path(merged_xml_path)
        self.data = mujoco.MjData(self.robot)
        self.robot.opt.timestep = self.sim_dt
        self.ctrl_dt = self.sim_dt * self.control_decimation

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
        self.cmd = np.array(self.config.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32)

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
                    if self.follow_camera:
                        self.set_camera_follow()
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

    # ---- camera ----

    def set_camera_follow(self):
        if self.viewer is None:
            return

        base_pos = self.data.qpos[0:3].copy()
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

    def _build_merged_xml(self, robot_xml_path, ball_xml_path):
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

        merged_text = f"""<mujoco model="merged_scene">
            <include file="{robot_rel}"/>
            <include file="{ball_rel}"/>
        </mujoco>
        """

        merged_xml_path.write_text(merged_text, encoding="utf-8")

        print(f"[deploy_mujoco] Temporary merged xml created: {merged_xml_path}")
        return str(merged_xml_path)
