from mujoco_sim.scripts.zb02w_flat import Zb02wFlatDeploy

yaml_filename = "zb02w_ts.yaml"

class Zb02wRoughDeploy(Zb02wFlatDeploy):
    pass

if __name__ == "__main__":
    deploy = Zb02wRoughDeploy(yaml_filename)
    deploy.run()