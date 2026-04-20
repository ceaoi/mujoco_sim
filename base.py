import os
import time

import mujoco
import mujoco.viewer
import numpy as np

import yaml
import onnxruntime as ort

from pathlib import Path

from .utils.gamepad_pygame import Gamepad
from .utils.deploy_func import quat_rotate_inverse, pd_ctrl
from .utils.projectile import ProjectileManager

from data_vis import PlotJugglerUDP
plotjuggler = PlotJugglerUDP("localhost", 5005)

class MujocoDeploy:

    mujoco_workspace_dir = os.path.dirname(__file__)

    def __init__(self, yaml_filename, device="cpu"):
        with open(f"{self.mujoco_workspace_dir}/configs/{yaml_filename}", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            policy_path = config["policy_path"].replace("{mujoco_workspace_dir}", self.mujoco_workspace_dir)
            xml_path = config["xml_path"].replace("{mujoco_workspace_dir}", self.mujoco_workspace_dir)

        # 保存配置
        self.config = config
        self.device = device

        # control / joint mapping
        self.control_decimation = config["control_decimation"]
        self.leg_joint_idx = config["leg_joint_idx"]
        self.wheel_joint_idx = config["wheel_joint_idx"]
        self.leg_joint_idx_to_mujoco = config["leg_joint_idx_to_mujoco"]
        self.wheel_joint_idx_to_mujoco = config["wheel_joint_idx_to_mujoco"]

        # gains
        self.kpsPos = np.array(config["kpsPos"], dtype=np.float32)
        self.kdsPos = np.array(config["kdsPos"], dtype=np.float32)
        self.kpsVel = np.array(config["kpsVel"], dtype=np.float32)
        self.kdsVel = np.array(config["kdsVel"], dtype=np.float32)

        # scales / constants
        self.default_angles = np.array(config["default_angles_leg"], dtype=np.float32)
        self.action_scale_pos = np.float32(config["action_scale_pos"])
        self.action_scale_vel = np.float32(config["action_scale_vel"])

            # self.ang_vel_scale = np.float32(config["ang_vel_scale"])
            # self.dof_pos_scale = np.float32(config["dof_pos_scale"])
            # self.dof_vel_leg_scale = np.float32(config["dof_vel_leg_scale"])
            # self.dof_vel_wheel_scale = np.float32(config["dof_vel_wheel_scale"])
        self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        self.cmd_range = np.array(config["cmd_range"], dtype=np.float32)
        self.cmd_deadzone = np.array(config["cmd_deadzone"], dtype=np.float32)
        self.wheel_action_vel_deadzone = np.float32(config["wheel_action_vel_deadzone"])

        # dims
        self.num_obs = int(config["num_obs"])
        self.num_actions = int(config["num_actions"])
        self.num_actions_pos = int(config["num_actions_pos"])
        self.num_obs_hist = int(config["num_obs_hist"])
        self.obs_hist_dim = self.num_obs * self.num_obs_hist
        self.num_wheels = len(self.wheel_joint_idx)

        # init cmd
        self.cmd = np.array(config.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32)

        # 初始化 Mujoco 模型和数据
        ball_xml_path = f"{self.mujoco_workspace_dir}/robots/ball/ball.xml"
        merged_xml_path = self._build_merged_xml(xml_path, ball_xml_path)
        self.robot = mujoco.MjModel.from_xml_path(merged_xml_path)
        # self.robot = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.robot)
        self.robot.opt.timestep = config["simulation_dt"]

        # 加载 ONNX Policy
        self.is_rnn = bool(config.get("is_rnn", False))
        self.policy = self._make_onnx_session(policy_path)
        print(f"[deploy_mujoco] Loaded ONNX policy: {policy_path}")

        self.policy_input_names = [x.name for x in self.policy.get_inputs()]
        self.policy_output_names = [x.name for x in self.policy.get_outputs()]
        print(f"[deploy_mujoco] policy inputs: {self.policy_input_names}")
        print(f"[deploy_mujoco] policy outputs: {self.policy_output_names}")
        self.policy_input_name = self.policy.get_inputs()[0].name
        self.policy_output_name = self.policy.get_outputs()[0].name

        if self.is_rnn:
            # 尽量从 onnx 的输入shape里读 hidden 尺寸
            h_input = self.policy.get_inputs()[1]
            h_shape = h_input.shape   # 一般是 [num_layers, 1, hidden_size]
            self.rnn_num_layers = int(h_shape[0])
            self.rnn_hidden_size = int(h_shape[2])
            self.h_in = np.zeros((self.rnn_num_layers, 1, self.rnn_hidden_size), dtype=np.float32)
            self.c_in = np.zeros((self.rnn_num_layers, 1, self.rnn_hidden_size), dtype=np.float32)

        # 遥控
        self.gamepad = Gamepad(joystick_index=0)
        self.gamepad.connect()

        # runtime buffers
        self.counter = 0
        self.viewer = None

        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.obs_hist = np.zeros(self.obs_hist_dim, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)

        self.targ_dof_pos = self.default_angles.copy()
        self.targ_dof_vel = np.zeros(self.num_wheels, dtype=np.float32)
        self.tau = np.zeros(self.num_actions, dtype=np.float32)

        # camera / viewer states
        self.follow_camera = True
        self.prev_l2_pressed = False
        self.prev_r2_pressed = False

        self.projectile_manager = ProjectileManager(self.robot, self.data)
        self.prev_a_pressed = False

    def reset(self):
        mujoco.mj_resetData(self.robot, self.data)
        self.counter = 0
        self.obs[:] = 0.0
        self.obs_hist[:] = 0.0
        self.action[:] = 0.0
        self.targ_dof_pos = self.default_angles.copy()
        self.targ_dof_vel[:] = 0.0
        self.tau[:] = 0.0
        self.cmd = np.array(self.config.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32)

        # self.follow_camera = True
        self.prev_l2_pressed = False
        self.prev_r2_pressed = False
        self.prev_a_pressed = False
        self.projectile_manager.reset()

        if self.is_rnn:
            self.h_in[:] = 0.0
            self.c_in[:] = 0.0

    def run(self, duration = 1e3):
        self.reset()
        start = time.time()

        with mujoco.viewer.launch_passive(self.robot, self.data) as viewer:
            self.viewer = viewer
            next_tick = time.perf_counter()
            while viewer.is_running() and (time.time() - start < duration):
                next_tick += self.robot.opt.timestep
                self.step()

                # if self.counter % 10 == 0: # 每 10 步更新一次视觉
                if self.follow_camera:
                    self.set_camera_follow()
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                remain = next_tick - time.perf_counter()
                if remain > 0:
                    if remain > 5e-4:
                        time.sleep(remain)
                else:
                    next_tick = time.perf_counter()

    def step(self):
        # 每 control_decimation 次更新一次策略
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
            # plotjuggler.send_array("obs", self.obs)
            # plotjuggler.send_array("action", self.action)

        self.update_tau()
        self.data.ctrl[:] = self.tau

        mujoco.mj_step(self.robot, self.data)
        self.counter += 1

    def update_cmd(self):
        self.cmd = np.asarray(self.gamepad.get_cmd(), dtype=np.float32) * self.cmd_range
        self.cmd[0] = 0.0 if abs(self.cmd[0]) < self.cmd_deadzone[0] else self.cmd[0]
        self.cmd[1] = 0.0 if abs(self.cmd[1]) < self.cmd_deadzone[1] else self.cmd[1]
        self.cmd[2] = 0.0 if abs(self.cmd[2]) < self.cmd_deadzone[2] else self.cmd[2]

    def update_obs(self):
        raise NotImplementedError("update_obs must be implemented by subclass")
    
    def update_model_in(self):
        raise NotImplementedError("modelupdate_model_in must be implemented by subclass, let 'self.model_in = self.obs or self.obs_hist'")

    def update_action(self):
        model_in = self.model_in
        inp = np.ascontiguousarray(model_in, dtype=np.float32)
            # ONNX 一般要求 [batch, dim]
        if inp.ndim == 1:
            inp = inp[None, :]
        if self.is_rnn:
            action, h_out, c_out = self.policy.run(
                [self.policy_output_name, "h_out", "c_out"],
                {
                    self.policy_input_name: inp,
                    "h_in": self.h_in,
                    "c_in": self.c_in,
                }
            )
            self.action = np.asarray(action, dtype=np.float32).squeeze()
            self.h_in = np.asarray(h_out, dtype=np.float32)
            self.c_in = np.asarray(c_out, dtype=np.float32)
        else:
            action = self.policy.run([self.policy_output_name], {self.policy_input_name: inp})[0]
            self.action = np.asarray(action, dtype=np.float32).squeeze()

        self.targ_dof_pos = (
            self.action[:self.num_actions_pos] * self.action_scale_pos + self.default_angles
        )
        self.targ_dof_vel = self.action[self.num_actions_pos:] * self.action_scale_vel
        self.targ_dof_vel = self.targ_dof_vel * (np.abs(self.targ_dof_vel) >= self.wheel_action_vel_deadzone)  # 死区

    def update_tau(self):
        tau_pos = pd_ctrl(
            self.targ_dof_pos - self.data.qpos[7:][self.leg_joint_idx],
            -self.data.qvel[6:][self.leg_joint_idx],
            self.kpsPos,
            self.kdsPos,
        )
        tau_vel = pd_ctrl(
            np.zeros(len(self.wheel_joint_idx), dtype=np.float32),
            self.targ_dof_vel - self.data.qvel[6:][self.wheel_joint_idx],
            self.kpsVel,
            self.kdsVel,
        )
        self.tau[self.leg_joint_idx_to_mujoco] = np.clip(tau_pos, -76.4, 76.4)
        self.tau[self.wheel_joint_idx_to_mujoco] = np.clip(tau_vel, -21.6, 21.6)

    def _make_onnx_session(self, onnx_path: str) -> ort.InferenceSession:
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
    
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

        # R2 上升沿：切换相机模式
        if (not self.prev_r2_pressed) and r2_pressed:
            self.follow_camera = not self.follow_camera
            mode = "FOLLOW" if self.follow_camera else "FIXED"
            print(f"[deploy_mujoco] Camera mode -> {mode}")

        # L2 上升沿：reset mujoco
        if (not self.prev_l2_pressed) and l2_pressed:
            print("[deploy_mujoco] Reset Mujoco")
            self.reset()

        self.prev_r2_pressed = r2_pressed
        self.prev_l2_pressed = l2_pressed

        # A 上升沿：发球撞击robot
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

        # 临时文件放在 ball.xml 所在目录
        out_dir = robot_xml_path.parent
        merged_xml_path = out_dir / f"tmp_merged.xml"

        # include 路径建议写成“相对 merged_xml_path 的相对路径”
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