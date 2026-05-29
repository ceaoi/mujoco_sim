import numpy as np

from .base import MujocoDeploy


class MujocoDeployWh(MujocoDeploy):

    def _init_control(self):
        self.turn_joint_idx = [0, 2, 4, 6]
        self.wheel_joint_idx = [1, 3, 5, 7]

    #     self.default_angles = np.zeros(self.num_actions, dtype=np.float32)
    #     self.targ_dof_pos = self.default_angles.copy()

    # def _reset_control(self):
    #     self.targ_dof_pos = self.default_angles.copy()

    def update_action(self):
        self.action[:] = 0.0
        if self.counter % 5000 < 2000:
            self.action[1] = 1.57
            self.action[3] = 1.57
            self.action[5] = 1.57
            self.action[7] = 1.57

    def update_tau(self):
        self.tau[:] = self.action # pos + vel, not tau
