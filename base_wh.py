import numpy as np

from .base import MujocoDeploy


class MujocoDeployWh(MujocoDeploy):

    def _init_control(self):
        self.turn_joint_idx = [0, 2, 4, 6]
        self.wheel_joint_idx = [1, 3, 5, 7]

        self.default_angles = np.zeros(self.num_actions, dtype=np.float32)
        self.targ_dof_pos = self.default_angles.copy()

        self.kp_turn = .0
        self.kd_turn = 0.
        self.kp_wheel = 0.0
        self.kd_wheel = 0.0

    def _reset_control(self):
        self.targ_dof_pos = self.default_angles.copy()

    def update_action(self):
        self.action[:] = 0.0

    def update_tau(self):
        pos_err = self.targ_dof_pos - self.data.qpos[7:7 + self.num_actions]
        vel_err = -self.data.qvel[6:6 + self.num_actions]

        tau = np.zeros(self.num_actions, dtype=np.float32)
        tau[self.turn_joint_idx] = (
            self.kp_turn * pos_err[self.turn_joint_idx]
            + self.kd_turn * vel_err[self.turn_joint_idx]
        )
        tau[self.wheel_joint_idx] = (
            self.kp_wheel * pos_err[self.wheel_joint_idx]
            + self.kd_wheel * vel_err[self.wheel_joint_idx]
        )
        self.tau[:] = tau
