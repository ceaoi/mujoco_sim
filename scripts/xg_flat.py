from mujoco_sim.configs import XgFlatConfig
from mujoco_sim.scripts.zb02w_flat import Zb02wFlatDeploy


class XgFlatDeploy(Zb02wFlatDeploy):
    """Deploy the feed-forward XG policy on flat terrain."""

    def reset(self):
        super().reset()
        self.data.qpos[2] = 0.5082


if __name__ == "__main__":
    deploy = XgFlatDeploy(XgFlatConfig())
    deploy.run()
