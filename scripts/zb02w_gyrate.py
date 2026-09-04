import numpy as np

from mujoco_sim.configs import Zb02wGyrateConfig
from mujoco_sim.scripts.base.base_wl import wheel_stop_pid
from mujoco_sim.scripts.zb02w_flat import Zb02wFlatDeploy
from mujoco_sim.utils.deploy_func import quat_rotate_inverse


class Zb02wGyrateDeploy(Zb02wFlatDeploy):

    num_modes = 3

    def __init__(self, config: Zb02wGyrateConfig, device="cpu"):
        super().__init__(config, device)

        # Policy 观测保持训练端顺序：50 维 policy 后拼接 3 维 mode one-hot。
        self.mode = 0
        self.model_in = np.zeros(self.num_obs + self.num_modes, dtype=np.float32)

        # 当前导出的 actor encoder 是 GRU，仅使用 h_in/h_out，不包含 LSTM 的
        # c_in/c_out。因此 hidden state 由本部署类独立维护，不改动通用 LSTM 基类。
        policy_inputs = {item.name: item for item in self.policy.get_inputs()}
        h_shape = self._concrete_recurrent_state_shape(
            policy_inputs["h_in"].shape,
            "h_in",
        )
        self.h_in = np.zeros(h_shape, dtype=np.float32)

    def _reset_control(self):
        super()._reset_control()
        self.mode = 0
        self.h_in[:] = 0.0

    def handle_gamepad_events(self):
        super().handle_gamepad_events()
        self._update_mode_from_gamepad()

    def _update_mode_from_gamepad(self):
        """根据模式按键上升沿切换 mode，并保持到下一次模式按键触发。"""
        if self.gamepad.is_button_rising_edge("X"):
            mode = 0
        elif self.gamepad.is_button_rising_edge("Y"):
            mode = 1
        elif self.gamepad.is_button_rising_edge("B"):
            mode = 2
        else:
            return

        self.mode = mode
        print(f"[deploy_mujoco] Policy mode -> {self.mode}")

    @staticmethod
    def compute_world_ang_vel_z(base_ang_vel, projected_gravity):
        world_z_base = -projected_gravity
        return float(np.dot(base_ang_vel, world_z_base))

    def update_obs(self):
        base_quat = self.data.qpos[3:7].copy()
        joint_pos = self.data.qpos[7:]
        joint_vel = self.data.qvel[6:]
        base_ang_vel = self.data.qvel[3:6]

        projected_gravity = quat_rotate_inverse(
            base_quat,
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        leg_joint_pos_rel = joint_pos[self.leg_joint_idx] - self.default_angles

        offset = 0
        self.obs[offset:offset + 3] = base_ang_vel * 0.5
        offset += 3

        self.obs[offset:offset + 3] = projected_gravity
        offset += 3

        self.obs[offset:offset + self.num_actions_pos] = leg_joint_pos_rel * 0.5
        offset += self.num_actions_pos

        self.obs[offset:offset + self.num_actions_pos] = (
            joint_vel[self.leg_joint_idx] * 0.03
        )
        offset += self.num_actions_pos

        self.obs[offset:offset + self.num_wheels] = (
            joint_vel[self.wheel_joint_idx] * 0.03
        )
        offset += self.num_wheels

        self.obs[offset:offset + self.num_actions] = self.action

        world_ang_vel_z = self.compute_world_ang_vel_z(
            base_ang_vel,
            projected_gravity,
        )
        self.send_plotjuggler_data("world_ang_vel_z", world_ang_vel_z)

    def update_model_in(self):
        """在 50 维 policy 观测末尾拼接当前 mode 的 one-hot。"""
        self.model_in[:self.num_obs] = self.obs
        self.model_in[self.num_obs:] = 0.0
        self.model_in[self.num_obs + self.mode] = 1.0

    def update_action(self):
        """使用 GRU ONNX policy 推理，并更新下一控制步的 hidden state。"""
        inp = np.ascontiguousarray(self.model_in, dtype=np.float32)
        if inp.ndim == 1:
            inp = inp[None, :]

        action, h_out = self.policy.run(
            [self.policy_output_name, "h_out"],
            {
                self.policy_input_name: inp,
                "h_in": np.ascontiguousarray(self.h_in, dtype=np.float32),
            },
        )
        self.action = np.asarray(action, dtype=np.float32).squeeze()
        self.h_in = np.asarray(h_out, dtype=np.float32)

        self.send_plotjuggler_data("actions", self.action)
        self.targ_dof_pos = (
            self.action[:self.num_actions_pos] * self.action_scale_pos
            + self.default_angles
        )
        self.targ_dof_vel = (
            self.action[self.num_actions_pos:] * self.action_scale_vel
        )

        dqj_wheel = self.data.qvel[6:][self.wheel_joint_idx]
        self.send_plotjuggler_data("wheel_vel", dqj_wheel)


if __name__ == "__main__":
    deploy = Zb02wGyrateDeploy(Zb02wGyrateConfig())
    deploy.run()
