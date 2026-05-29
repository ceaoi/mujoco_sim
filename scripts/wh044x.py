from mujoco_sim.base_wh import MujocoDeployWh
import numpy as np


class Wh044xDeploy(MujocoDeployWh):

    _init_z = 0.2

    def update_model_in(self):
        self.model_in = self.obs

    def update_obs(self):
        self.obs[:] = 0.0

    def reset(self):
        super().reset()
        self.data.qpos[2] = self._init_z

    def set_camera_follow(self):
        if self.viewer is None:
            return
        base_pos = self.data.qpos[0:3].copy()
        camera_offset = np.array([-2.0, 1.0, 4.0], dtype=np.float32)
        self.viewer.cam.lookat[:] = base_pos
        self.viewer.cam.distance = float(np.linalg.norm(camera_offset))
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -20


if __name__ == "__main__":
    deploy = Wh044xDeploy("wh044x.yaml")
    deploy.run()
