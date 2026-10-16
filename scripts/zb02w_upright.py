import numpy as np

from mujoco_sim.configs import Zb02wUprightConfig
from mujoco_sim.scripts.zb02w_flat import Zb02wFlatDeploy
from mujoco_sim.scripts.zb02w_gyrate import Zb02wGyrateDeploy


class Zb02wUprightDeploy(Zb02wGyrateDeploy):
    """Deploy the upright student MoE policy with both GRU states."""

    def __init__(self, config: Zb02wUprightConfig, device="cpu"):
        # Skip Zb02wGyrateDeploy.__init__: the gyrate policy exposes a single
        # h_in state, while StudentMoE exports explicit routing and estimator
        # GRU states.
        Zb02wFlatDeploy.__init__(self, config, device)
        self._init_tau_response()

        self.mode = 0
        self.model_in = np.zeros(
            self.num_obs + self.num_modes,
            dtype=np.float32,
        )

        policy_inputs = {item.name: item for item in self.policy.get_inputs()}
        routing_h_shape = self._concrete_recurrent_state_shape(
            policy_inputs["routing_h_in"].shape,
            "routing_h_in",
        )
        estimator_h_shape = self._concrete_recurrent_state_shape(
            policy_inputs["estimator_h_in"].shape,
            "estimator_h_in",
        )
        self.routing_h_in = np.zeros(routing_h_shape, dtype=np.float32)
        self.estimator_h_in = np.zeros(estimator_h_shape, dtype=np.float32)

    def _reset_control(self):
        Zb02wFlatDeploy._reset_control(self)
        self.tau[:] = 0.0
        self._target_tau[:] = 0.0
        self._tau_response_delta[:] = 0.0
        self.mode = 0
        self.routing_h_in[:] = 0.0
        self.estimator_h_in[:] = 0.0

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

        action, routing_h_out, estimator_h_out = self.policy.run(
            [
                self.policy_output_name,
                "routing_h_out",
                "estimator_h_out",
            ],
            {
                self.policy_input_name: inp,
                "routing_h_in": np.ascontiguousarray(
                    self.routing_h_in,
                    dtype=np.float32,
                ),
                "estimator_h_in": np.ascontiguousarray(
                    self.estimator_h_in,
                    dtype=np.float32,
                ),
            },
        )
        self.action = np.asarray(action, dtype=np.float32).squeeze()
        self.routing_h_in = np.asarray(routing_h_out, dtype=np.float32)
        self.estimator_h_in = np.asarray(estimator_h_out, dtype=np.float32)

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
