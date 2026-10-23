from mujoco_sim.configs import XgTsConfig
from mujoco_sim.scripts.zb02w_flat import Zb02wFlatDeploy


class XgTsDeploy(Zb02wFlatDeploy):
    """Deploy the recurrent XG student with the shared wheel-legged controller."""

    def reset(self):
        super().reset()
        self.data.qpos[2] = 0.5082


if __name__ == "__main__":
    deploy = XgTsDeploy(XgTsConfig())
    deploy.run()
