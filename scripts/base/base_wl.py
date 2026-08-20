import numpy as np
import onnxruntime as ort

from mujoco_sim.configs import WheelLeggedConfig
from .base import MujocoDeploy
from mujoco_sim.utils.deploy_func import pd_ctrl


def wheel_stop_pid(error, integral, previous_error, kp, ki, kd, dt, output_limit):
    """Compute a vectorized wheel-speed PID correction with anti-windup."""
    error = np.asarray(error, dtype=np.float32)
    integral = np.asarray(integral, dtype=np.float32)
    previous_error = np.asarray(previous_error, dtype=np.float32)

    derivative = (error - previous_error) / np.float32(dt)
    candidate_integral = integral + error * np.float32(dt)
    unsaturated_output = kp * error + ki * candidate_integral + kd * derivative

    drives_further_into_saturation = np.logical_or(
        np.logical_and(unsaturated_output > output_limit, error > 0.0),
        np.logical_and(unsaturated_output < -output_limit, error < 0.0),
    )
    new_integral = np.where(
        drives_further_into_saturation,
        integral,
        candidate_integral,
    )

    output = kp * error + ki * new_integral + kd * derivative
    output = np.clip(output, -output_limit, output_limit).astype(np.float32, copy=False)
    return output, new_integral.astype(np.float32, copy=False), error.copy()


class MujocoDeployWl(MujocoDeploy):

    def __init__(self, config: WheelLeggedConfig, device="cpu"):
        super().__init__(config, device)

    def _init_control(self):
        config = self.config

        self.leg_joint_idx = np.asarray(config.leg_joint_idx, dtype=np.intp)
        self.leg_joint_idxx = np.asarray(config.leg_joint_idxx, dtype=np.intp)
        self.wheel_joint_idx = np.asarray(config.wheel_joint_idx, dtype=np.intp)
        self.leg_actions_to_mujoco = np.asarray(
            config.leg_actions_to_mujoco,
            dtype=np.intp,
        )
        self.wheel_actions_to_mujoco = np.asarray(
            config.wheel_actions_to_mujoco,
            dtype=np.intp,
        )

        self.kpsPos = np.array(config.kpsPos, dtype=np.float32)
        self.kdsPos = np.array(config.kdsPos, dtype=np.float32)
        self.kpsVel = np.array(config.kpsVel, dtype=np.float32)
        self.kdsVel = np.array(config.kdsVel, dtype=np.float32)

        self.default_angles = np.array(config.default_angles_leg, dtype=np.float32)
        self.action_scale_pos = np.float32(config.action_scale_pos)
        self.action_scale_vel = np.float32(config.action_scale_vel)
        self.wheel_action_vel_deadzone = np.float32(config.wheel_action_vel_deadzone)
        self.wheel_stop_pid_enabled = bool(config.wheel_stop_pid_enabled)
        self.wheel_stop_pid_kp = np.float32(config.wheel_stop_pid_kp)
        self.wheel_stop_pid_ki = np.float32(config.wheel_stop_pid_ki)
        self.wheel_stop_pid_kd = np.float32(config.wheel_stop_pid_kd)
        self.wheel_stop_pid_output_limit = np.float32(config.wheel_stop_pid_output_limit)

        if self.wheel_stop_pid_output_limit <= 0.0:
            raise ValueError("wheel_stop_pid_output_limit must be greater than zero")

        self.num_actions_pos = int(config.num_actions_pos)
        self.num_wheels = len(self.wheel_joint_idx)

        self.targ_dof_pos = self.default_angles.copy()
        self.targ_dof_vel = np.zeros(self.num_wheels, dtype=np.float32)
        self.wheel_stop_pid_integral = np.zeros(self.num_wheels, dtype=np.float32)
        self.wheel_stop_pid_previous_error = np.zeros(self.num_wheels, dtype=np.float32)
        self.wheel_stop_pid_active = False

        policy_path = config.policy_path
        self.is_rnn = bool(config.is_rnn)
        self.policy = self._make_onnx_session(str(policy_path))
        print(f"[deploy_mujoco] Loaded ONNX policy: {policy_path}")

        self.policy_input_names = [x.name for x in self.policy.get_inputs()]
        self.policy_output_names = [x.name for x in self.policy.get_outputs()]
        print(f"[deploy_mujoco] policy inputs: {self.policy_input_names}")
        print(f"[deploy_mujoco] policy outputs: {self.policy_output_names}")
        self.policy_input_name = self.policy.get_inputs()[0].name
        self.policy_output_name = self.policy.get_outputs()[0].name

        if self.is_rnn:
            policy_inputs = {item.name: item for item in self.policy.get_inputs()}
            missing_state_inputs = [
                name for name in ("h_in", "c_in") if name not in policy_inputs
            ]
            if missing_state_inputs:
                raise ValueError(
                    "RNN policy is missing required recurrent inputs: "
                    f"{missing_state_inputs}; available inputs: {self.policy_input_names}"
                )

            h_shape = self._concrete_recurrent_state_shape(
                policy_inputs["h_in"].shape,
                "h_in",
            )
            c_shape = self._concrete_recurrent_state_shape(
                policy_inputs["c_in"].shape,
                "c_in",
            )
            if c_shape != h_shape:
                raise ValueError(
                    "RNN policy h_in and c_in shapes must match, got "
                    f"{h_shape} and {c_shape}"
                )

            self.rnn_num_layers = h_shape[0]
            self.rnn_hidden_size = h_shape[2]
            self.h_in = np.zeros(h_shape, dtype=np.float32)
            self.c_in = np.zeros(c_shape, dtype=np.float32)

    @staticmethod
    def _concrete_recurrent_state_shape(shape, input_name):
        if len(shape) != 3 or any(
            isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or dim <= 0
            for dim in shape
        ):
            raise ValueError(
                f"RNN policy input {input_name!r} must have a fixed, positive 3D "
                f"shape, got {shape}"
            )
        return tuple(int(dim) for dim in shape)

    def _reset_control(self):
        self.targ_dof_pos = self.default_angles.copy()
        self.targ_dof_vel[:] = 0.0
        self.wheel_stop_pid_integral[:] = 0.0
        self.wheel_stop_pid_previous_error[:] = 0.0
        self.wheel_stop_pid_active = False

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
                self._build_policy_inputs(inp),
            )
            self.action = np.asarray(action, dtype=np.float32).squeeze()
            self.h_in = np.asarray(h_out, dtype=np.float32)
            self.c_in = np.asarray(c_out, dtype=np.float32)
        else:
            action = self.policy.run(
                [self.policy_output_name],
                self._build_policy_inputs(inp),
            )[0]
            self.action = np.asarray(action, dtype=np.float32).squeeze()

        self.send_plotjuggler_data("actions", self.action)
        self.targ_dof_pos = (
            self.action[:self.num_actions_pos] * self.action_scale_pos + self.default_angles
        )
        self.targ_dof_vel = self.action[self.num_actions_pos:] * self.action_scale_vel
        # self.targ_dof_vel = np.sign(self.targ_dof_vel) * np.maximum(
        #     np.abs(self.targ_dof_vel) - self.wheel_action_vel_deadzone,
        #     0.0,
        # )

        dqj_wheel = self.data.qvel[6:][self.wheel_joint_idx]
        self.send_plotjuggler_data("wheel_vel", dqj_wheel)
        if self.wheel_stop_pid_enabled and np.linalg.norm(self.cmd) < 1e-3 and np.abs(np.mean(dqj_wheel)) < 2.0:
            wheel_velocity = self.data.qvel[6:][self.wheel_joint_idx]
            wheel_velocity_error = -np.asarray(wheel_velocity, dtype=np.float32)
            if not self.wheel_stop_pid_active:
                self.wheel_stop_pid_previous_error[:] = wheel_velocity_error

            pid_output, self.wheel_stop_pid_integral, self.wheel_stop_pid_previous_error = (
                wheel_stop_pid(
                    wheel_velocity_error,
                    self.wheel_stop_pid_integral,
                    self.wheel_stop_pid_previous_error,
                    self.wheel_stop_pid_kp,
                    self.wheel_stop_pid_ki,
                    self.wheel_stop_pid_kd,
                    self.ctrl_dt,
                    self.wheel_stop_pid_output_limit,
                )
            )
            self.targ_dof_vel += pid_output
            self.wheel_stop_pid_active = True
            # print(f"pid_output: {pid_output}")
        else:
            self.wheel_stop_pid_integral[:] = 0.0
            self.wheel_stop_pid_previous_error[:] = 0.0
            self.wheel_stop_pid_active = False

    def _build_policy_inputs(self, policy_obs):
        inputs = {
            self.policy_input_name: np.ascontiguousarray(
                policy_obs,
                dtype=np.float32,
            )
        }
        if self.is_rnn:
            inputs["h_in"] = np.ascontiguousarray(self.h_in, dtype=np.float32)
            inputs["c_in"] = np.ascontiguousarray(self.c_in, dtype=np.float32)
        return inputs

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
