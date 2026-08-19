from mujoco_sim.configs import Zb02wFlatConfig
from mujoco_sim.scripts.base.base_wl import MujocoDeployWl
from mujoco_sim.utils.gait_phase_generator import GaitPhaseGenerator
from mujoco_sim.utils.deploy_func import quat_rotate_inverse, pd_ctrl
import numpy as np

class Zb02wFlatDeploy(MujocoDeployWl):

    def __init__(self, config: Zb02wFlatConfig, device="cpu"):
        super().__init__(config, device)
        self.max_delta_cmd = np.array(config.max_command_rate, dtype=np.float32) * self.ctrl_dt

    def reset(self):
        super().reset()
        self.data.qpos[2] = 0.62

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
        self.send_plotjuggler_data("targ_pos", self.targ_dof_pos)
        self.send_plotjuggler_data("targ_vel", self.targ_dof_vel)
        self.send_plotjuggler_data("tau", self.tau)

    def update_model_in(self):
        self.model_in = self.obs

    def update_cmd(self):
        cmd = np.asarray(self.gamepad.get_cmd(), dtype=np.float32) * self.cmd_range
        self.cmd += np.clip(cmd - self.cmd, -self.max_delta_cmd, self.max_delta_cmd)
        self.cmd[0] = 0.0 if abs(self.cmd[0]) < self.cmd_deadzone[0] else self.cmd[0]
        self.cmd[1] = 0.0 if abs(self.cmd[1]) < self.cmd_deadzone[1] else self.cmd[1]
        self.cmd[2] = 0.0 if abs(self.cmd[2]) < self.cmd_deadzone[2] else self.cmd[2]

    def update_obs(self):
        base_quat = self.data.qpos[3:7].copy()  # MuJoCo freejoint quat: [w, x, y, z]
        qj = self.data.qpos[7:][self.leg_joint_idx]  # 按照 joint_idx 重新排序
        dqj = self.data.qvel[6:]  # 按照 joint_idx 重新排序
        omega = self.data.qvel[3:6].copy() # free joint 的线速度在 global frame，rotational velocity 在 local body frame

        # 训练端 projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_orientation = quat_rotate_inverse(base_quat, gravity_world)
        qj_rel = (qj - self.default_angles)
        dqj_leg = dqj[self.leg_joint_idx]
        dqj_wheel = dqj[self.wheel_joint_idx]

        offset = 0

        # encoder obs term 1: base_ang_vel (scaled)
        self.obs[offset:offset + 3] = omega * 0.5
        offset += 3

        self.obs[offset:offset + 3] = gravity_orientation
        offset += 3

        self.obs[offset:offset + self.num_actions_pos] = qj_rel * 0.5
        offset += self.num_actions_pos

        self.obs[offset:offset + self.num_actions_pos] = dqj_leg * 0.03
        offset += self.num_actions_pos

        self.obs[offset:offset + self.num_wheels] = dqj_wheel * 0.03
        offset += self.num_wheels

        self.obs[offset:offset + self.num_actions] = self.action
        offset += self.num_actions

        # encoder obs term 7: velocity_commands (raw generated command)
        self.obs[offset:offset + 3] = self.cmd
        offset += 3
        self.send_plotjuggler_data("cmd", self.cmd)
        self.send_plotjuggler_data("obs", self.obs)

if __name__ == "__main__":
    deploy = Zb02wFlatDeploy(Zb02wFlatConfig())
    deploy.run()
