from mujoco_sim.configs import Zb02wTsConfig
from mujoco_sim.scripts.zb02w_flat import Zb02wFlatDeploy

class Zb02wRoughDeploy(Zb02wFlatDeploy):
    pass

if __name__ == "__main__":
    deploy = Zb02wRoughDeploy(Zb02wTsConfig())
    deploy.run()
