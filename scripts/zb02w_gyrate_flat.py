import numpy as np

from mujoco_sim.configs import Zb02wGyrateFlatConfig
from mujoco_sim.scripts.zb02w_gyrate import Zb02wGyrateDeploy
from mujoco_sim.scripts.zb02w_flat import Zb02wFlatDeploy


class Zb02wGyrateFlatDeploy(Zb02wGyrateDeploy):
    """Deploy the two-mode gyrate policy with a feed-forward MLP actor."""

    num_modes = 2

    def __init__(self, config: Zb02wGyrateFlatConfig, device="cpu"):
        # Skip Zb02wGyrateDeploy.__init__: that deployment owns the hidden state
        # required by its GRU policy. The flat policy is a plain MLP.
        Zb02wFlatDeploy.__init__(self, config, device)

        self.mode = 0
        self.model_in = np.zeros(
            self.num_obs + self.num_modes,
            dtype=np.float32,
        )

    def _reset_control(self):
        Zb02wFlatDeploy._reset_control(self)
        self.mode = 0

    def _update_mode_from_gamepad(self):
        """Select one of the two modes and keep it until the next button edge."""
        if self.gamepad.is_button_rising_edge("X"):
            mode = 0
        elif self.gamepad.is_button_rising_edge("Y"):
            mode = 1
        else:
            return

        self.mode = mode
        print(f"[deploy_mujoco] Policy mode -> {self.mode}")

    def update_action(self):
        """Run the MLP policy without recurrent state inputs or outputs."""
        Zb02wFlatDeploy.update_action(self)


if __name__ == "__main__":
    deploy = Zb02wGyrateFlatDeploy(Zb02wGyrateFlatConfig())
    deploy.run()
