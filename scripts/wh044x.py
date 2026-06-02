import os
import sys

import numpy as np
import yaml

from mujoco_sim.scripts.base.base_wh import MujocoDeployWh

yaml_filename = "wh044x.yaml"

def _load_chassis_build_dir(yaml_filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mujoco_workspace_dir = os.path.dirname(script_dir)
    config_path = os.path.join(mujoco_workspace_dir, "configs", yaml_filename)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader) or {}

    build_dir = config["chassis_build_dir"]
    return (
        build_dir.replace("{mujoco_workspace_dir}", mujoco_workspace_dir)
    )


_build_dir = _load_chassis_build_dir(yaml_filename)
if _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)

import chassis as cpp_chassis # type: ignore


class Wh044xDeploy(MujocoDeployWh):

    _init_z = 0.2

    def _init_control(self):
        self.turn_joint_idx = [0, 2, 4, 6]
        self.wheel_joint_idx = [1, 3, 5, 7]

        self.wheel_radius = 0.065  # m, from MJCF

        self.chassis = cpp_chassis.Chassis()
        self.chassis.set_rate_limit(dt = self.ctrl_dt, vel_rate = 1.0e0, steering_rate = 10.0e0)
        self.prev_r1_pressed = False

    def update_action(self):
        if (self.gamepad.get_button("R1") & (not self.prev_r1_pressed)):
            self.chassis.mode = (self.chassis.mode + 1) % 3
            print(f"Switched to mode {self.chassis.mode}")
        self.prev_r1_pressed = self.gamepad.get_button("R1")
        vx = float(self.cmd[0])
        vy = float(self.cmd[1])        
        omega = float(self.cmd[2])
        self.chassis.step(vx, vy, omega)
        self.action[self.turn_joint_idx] = np.array(self.chassis.steering_angle)
        self.action[self.wheel_joint_idx] = np.array(self.chassis.vel) / self.wheel_radius

    def update_tau(self):
        self.tau[:] = self.action # pos + vel, not tau

    def reset(self):
        super().reset()
        self.data.qpos[2] = self._init_z

    def set_camera_follow(self):
        if self.viewer is None:
            return
        base_pos = self.data.qpos[0:3].copy()
        camera_offset = np.array([-0.5, 0.5, 3.0], dtype=np.float32)
        self.viewer.cam.lookat[:] = base_pos
        self.viewer.cam.lookat[2] += 0.5
        self.viewer.cam.distance = float(np.linalg.norm(camera_offset))
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -20


if __name__ == "__main__":
    deploy = Wh044xDeploy("wh044x.yaml")
    deploy.run()
