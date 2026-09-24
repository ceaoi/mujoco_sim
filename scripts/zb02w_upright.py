import numpy as np

from mujoco_sim.configs import Zb02wUprightConfig
from mujoco_sim.scripts.zb02w_flat import Zb02wFlatDeploy
from mujoco_sim.scripts.zb02w_gyrate import Zb02wGyrateDeploy


class Zb02wUprightDeploy(Zb02wGyrateDeploy):
    """Deploy the upright residual MoE policy with both GRU states."""

    def __init__(self, config: Zb02wUprightConfig, device="cpu"):
        super().__init__(config, device)

        policy_inputs = {item.name: item for item in self.policy.get_inputs()}
        mode0_h_shape = self._concrete_recurrent_state_shape(
            policy_inputs["mode0_h_in"].shape,
            "mode0_h_in",
        )
        self.mode0_h_in = np.zeros(mode0_h_shape, dtype=np.float32)

    def _reset_control(self):
        super()._reset_control()
        self.mode0_h_in[:] = 0.0

    def update_obs(self):
        """Build the 53-dimensional policy observation used during training."""
        Zb02wFlatDeploy.update_obs(self)

    def update_model_in(self):
        """Append the current three-mode one-hot vector to the policy observation."""
        self.model_in[:self.num_obs] = self.obs
        self.model_in[self.num_obs:] = 0.0
        self.model_in[self.num_obs + self.mode] = 1.0

    def update_action(self):
        """Run the dual-GRU ONNX policy and retain both next hidden states."""
        inp = np.ascontiguousarray(self.model_in, dtype=np.float32)
        if inp.ndim == 1:
            inp = inp[None, :]

        action, h_out, mode0_h_out = self.policy.run(
            [self.policy_output_name, "h_out", "mode0_h_out"],
            {
                self.policy_input_name: inp,
                "h_in": np.ascontiguousarray(self.h_in, dtype=np.float32),
                "mode0_h_in": np.ascontiguousarray(
                    self.mode0_h_in,
                    dtype=np.float32,
                ),
            },
        )
        self.action = np.asarray(action, dtype=np.float32).squeeze()
        self.h_in = np.asarray(h_out, dtype=np.float32)
        self.mode0_h_in = np.asarray(mode0_h_out, dtype=np.float32)

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
    deploy = Zb02wUprightDeploy(Zb02wUprightConfig())
    deploy.run()
