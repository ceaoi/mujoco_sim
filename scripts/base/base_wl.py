import numpy as np
import onnxruntime as ort

from .base import MujocoDeploy
from mujoco_sim.utils.deploy_func import pd_ctrl

from data_vis import PlotJugglerUDP
plotjuggler = PlotJugglerUDP("localhost", 5005)


class MujocoDeployWl(MujocoDeploy):

    def _init_control(self):
        config = self.config

        self.leg_joint_idx = config["leg_joint_idx"]
        self.leg_joint_idxx = config["leg_joint_idxx"]
        self.wheel_joint_idx = config["wheel_joint_idx"]
        self.leg_actions_to_mujoco = config["leg_actions_to_mujoco"]
        self.wheel_actions_to_mujoco = config["wheel_actions_to_mujoco"]

        self.kpsPos = np.array(config["kpsPos"], dtype=np.float32)
        self.kdsPos = np.array(config["kdsPos"], dtype=np.float32)
        self.kpsVel = np.array(config["kpsVel"], dtype=np.float32)
        self.kdsVel = np.array(config["kdsVel"], dtype=np.float32)

        self.default_angles = np.array(config["default_angles_leg"], dtype=np.float32)
        self.action_scale_pos = np.float32(config["action_scale_pos"])
        self.action_scale_vel = np.float32(config["action_scale_vel"])
        self.wheel_action_vel_deadzone = np.float32(config["wheel_action_vel_deadzone"])

        self.num_actions_pos = int(config["num_actions_pos"])
        self.num_wheels = len(self.wheel_joint_idx)

        self.targ_dof_pos = self.default_angles.copy()
        self.targ_dof_vel = np.zeros(self.num_wheels, dtype=np.float32)

        policy_path = config["policy_path"].replace("{mujoco_workspace_dir}", self.mujoco_workspace_dir)
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
            h_input = self.policy.get_inputs()[1]
            h_shape = h_input.shape
            self.rnn_num_layers = int(h_shape[0])
            self.rnn_hidden_size = int(h_shape[2])
            self.h_in = np.zeros((self.rnn_num_layers, 1, self.rnn_hidden_size), dtype=np.float32)
            self.c_in = np.zeros((self.rnn_num_layers, 1, self.rnn_hidden_size), dtype=np.float32)

    def _reset_control(self):
        self.targ_dof_pos = self.default_angles.copy()
        self.targ_dof_vel[:] = 0.0

        if self.is_rnn:
            self.h_in[:] = 0.0
            self.c_in[:] = 0.0

    def update_action(self):
        model_in = self.model_in
        inp = np.ascontiguousarray(model_in, dtype=np.float32)
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
        self.targ_dof_vel = self.targ_dof_vel * (np.abs(self.targ_dof_vel) >= self.wheel_action_vel_deadzone)

    def update_tau(self):
        self.tau[self.leg_actions_to_mujoco] = pd_ctrl(
            self.targ_dof_pos - self.data.qpos[7:][self.leg_joint_idx],
            -self.data.qvel[6:][self.leg_joint_idx],
            self.kpsPos,
            self.kdsPos,
        )
        self.tau[self.wheel_actions_to_mujoco] = pd_ctrl(
            np.zeros(len(self.wheel_joint_idx), dtype=np.float32),
            self.targ_dof_vel - self.data.qvel[6:][self.wheel_joint_idx],
            self.kpsVel,
            self.kdsVel,
        )

    def _make_onnx_session(self, onnx_path: str) -> ort.InferenceSession:
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
