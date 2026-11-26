"""Deploy the terrain-mix student LSTM with eight actions on complex terrain."""

import numpy as np

from mujoco_sim.configs import Zb02wUprightTsConfig
from mujoco_sim.scripts.zb02w_flat import Zb02wFlatDeploy
from mujoco_sim.scripts.zb02w_upright import Zb02wUprightDeploy


class Zb02wUprightTsDeploy(Zb02wUprightDeploy):
    """Two-mode StudentMoE actor, retaining the upright torque response."""

    num_modes = 2
    mode_offset = 96

    def __init__(self, config: Zb02wUprightTsConfig, device="cpu"):
        # The common LSTM loader owns h/c; the upright constructor loads two GRUs.
        Zb02wFlatDeploy.__init__(self, config, device)
        self._init_tau_response()
        self.mode = 0
        self.model_in = np.zeros(self.mode_offset + self.num_modes, dtype=np.float32)

    def _init_control(self):
        super()._init_control()
        # Eight policy outputs drive targets for all sixteen physical actuators.
        self.tau = np.zeros(self.robot.nu, dtype=np.float32)
        self._scaled_action = np.zeros(self.num_actions, dtype=np.float32)

    def _reset_control(self):
        Zb02wFlatDeploy._reset_control(self)
        self.tau[:] = 0.0
        self._target_tau[:] = 0.0
        self._tau_response_delta[:] = 0.0
        self._scaled_action[:] = 0.0
        self.model_in[:] = 0.0
        self.mode = 0

    def _update_mode_from_gamepad(self):
        if self.gamepad.is_button_rising_edge("X"):
            self.mode = 0
        elif self.gamepad.is_button_rising_edge("Y"):
            self.mode = 1
        else:
            return
        print(f"[deploy_mujoco] Policy mode -> {self.mode}")

    def update_obs(self):
        """Build 45 values, including the previous eight raw policy actions."""
        super().update_obs()

    def update_model_in(self):
        # StudentMoE's ONNX graph only consumes policy and mode.  The unused
        # foot-height [45:51] and privileged [51:96] slices remain zero.
        self.model_in[:self.num_obs] = self.obs
        self.model_in[self.mode_offset:] = 0.0
        self.model_in[self.mode_offset + self.mode] = 1.0

    def update_action(self):
        action, h_out, c_out = self.policy.run(
            [self.policy_output_name, "h_out", "c_out"],
            self._build_policy_inputs(self.model_in[None, :]),
        )
        self.action[:] = action[0]
        self.h_in = np.asarray(h_out, dtype=np.float32)
        self.c_in = np.asarray(c_out, dtype=np.float32)

        np.clip(self.action, -100.0, 100.0, out=self._scaled_action)
        self._scaled_action[:6] *= self.action_scale_pos
        self._scaled_action[6:] *= self.action_scale_vel
        self.targ_dof_pos[:] = self.default_angles
        self.targ_dof_pos[4:8] += self._scaled_action[:4]
        self.targ_dof_pos[10:12] += self._scaled_action[4:6]
        self.targ_dof_vel[:2] = 0.0
        rear_velocity = self._scaled_action[6:]
        self.targ_dof_vel[2:] = np.sign(rear_velocity) * np.maximum(
            np.abs(rear_velocity) - self.wheel_action_vel_deadzone, 0.0
        )

        self.send_plotjuggler_data("actions", self.action)
        self.send_plotjuggler_data("wheel_vel", self.data.qvel[6:][self.wheel_joint_idx])


if __name__ == "__main__":
    deploy = Zb02wUprightTsDeploy(Zb02wUprightTsConfig())
    deploy.run()
