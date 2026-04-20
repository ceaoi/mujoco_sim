from mujoco_sim.base import MujocoDeploy
from mujoco_sim.utils.gait_generator import GaitGenerator
from mujoco_sim.utils.deploy_func import quat_rotate_inverse
import numpy as np

class M20FlatDeploy(MujocoDeploy):

    def __init__(self, yaml_filename, device="cpu"):
        super().__init__(yaml_filename, device)
        self.gait = GaitGenerator(f"{self.mujoco_workspace_dir}/configs/{yaml_filename}")

    def update_obs(self):
        base_quat = self.data.qpos[3:7].copy()  # MuJoCo freejoint quat: [w, x, y, z]
        qj = self.data.qpos[7:][self.leg_joint_idx]  # 按照 joint_idx 重新排序
        dqj = self.data.qvel[6:]  # 按照 joint_idx 重新排序
        omega_world = self.data.qvel[3:6].copy()
        base_lin_acc_world = self.data.qacc[0:3].copy()
        # 训练端使用 body-frame 角速度：quat_rotate_inverse(base_quat, world_omega)
        omega_body = quat_rotate_inverse(base_quat, omega_world)
        base_lin_acc_body = quat_rotate_inverse(base_quat, base_lin_acc_world)
        imu_lin_acc_xy = base_lin_acc_body[:2]

        # 训练端 projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_orientation = quat_rotate_inverse(base_quat, gravity_world)
        qj_rel = (qj - self.default_angles)
        dqj_leg = dqj[self.leg_joint_idx]
        dqj_wheel = dqj[self.wheel_joint_idx]

            # test observations
            # plotjuggler.send_array("omega_world", omega_world)
            # plotjuggler.send_array("omega_body", omega_body)
            # plotjuggler.send_array("gravity_orientation", gravity_orientation)

        offset = 0
        # encoder obs term 1: imu_lin_acc_xy (scaled)
        self.obs[offset:offset + 2] = imu_lin_acc_xy * 0.04
        offset += 2

        # encoder obs term 2: base_ang_vel (scaled)
        self.obs[offset:offset + 3] = omega_body * 0.5
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

        gait_state = self.gait._update_gait(self.cmd, float(base_lin_acc_body[1]))
        self.obs[offset:offset + 5] = gait_state

if __name__ == "__main__":
    deploy = M20FlatDeploy("m20_flat.yaml")
    deploy.run()