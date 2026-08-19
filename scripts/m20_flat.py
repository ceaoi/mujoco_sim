from mujoco_sim.configs import M20FlatConfig
from mujoco_sim.scripts.base.base_wl import MujocoDeployWl
from mujoco_sim.utils.gait_phase_generator import GaitPhaseGenerator
from mujoco_sim.utils.deploy_func import quat_rotate_inverse
import numpy as np
from data_vis import PlotJugglerUDP
pj = PlotJugglerUDP()

class M20FlatDeploy(MujocoDeployWl):

    def __init__(self, config: M20FlatConfig, device="cpu"):
        super().__init__(config, device)
        self.gait = GaitPhaseGenerator(config)
        self.max_delta_cmd = np.array(config.max_command_rate, dtype=np.float32) * self.ctrl_dt

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
        # base_lin_acc_world = self.data.qacc[0:3].copy()
        # base_lin_acc_body = quat_rotate_inverse(
        # base_quat, base_lin_acc_world)
        # imu_lin_acc_xy = base_lin_acc_body[:2]

        # 训练端 projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_orientation = quat_rotate_inverse(base_quat, gravity_world)
        qj_rel = (qj - self.default_angles)
        dqj_leg = dqj[self.leg_joint_idx]
        dqj_wheel = dqj[self.wheel_joint_idx]

        offset = 0
        # encoder obs term 1: imu_lin_acc_xy (scaled)
        # self.obs[offset:offset + 2] = imu_lin_acc_xy * 0.04
        # offset += 2

        # encoder obs term 2: base_ang_vel (scaled)
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
        pj.send_array("cmd", self.cmd)
        pj.send_array("obs", self.obs)

        # gait_state = self.gait._update_gait(self.cmd, float(base_lin_acc_body[1]))
        # self.obs[offset:offset + 5] = gait_state

if __name__ == "__main__":
    deploy = M20FlatDeploy(M20FlatConfig())
    deploy.run()
