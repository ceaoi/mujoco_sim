import numpy as np
import mujoco


class ProjectileManager:
    def __init__(
        self,
        robot,
        data,
        ball_body_name="projectile_ball",
        ball_joint_name="projectile_ball_freejoint",
        park_position=(100.0, 100.0, 100.0),
    ):
        self.robot = robot
        self.data = data

        self.ball_body_name = ball_body_name
        self.ball_joint_name = ball_joint_name
        self.park_position = np.array(park_position, dtype=np.float64)

        self.ball_body_id = mujoco.mj_name2id(
            self.robot, mujoco.mjtObj.mjOBJ_BODY, self.ball_body_name
        )
        self.ball_joint_id = mujoco.mj_name2id(
            self.robot, mujoco.mjtObj.mjOBJ_JOINT, self.ball_joint_name
        )

        if self.ball_body_id == -1:
            raise ValueError(
                f"[ProjectileManager] body '{self.ball_body_name}' not found in xml."
            )
        if self.ball_joint_id == -1:
            raise ValueError(
                f"[ProjectileManager] joint '{self.ball_joint_name}' not found in xml."
            )

        joint_type = self.robot.jnt_type[self.ball_joint_id]
        if joint_type != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(
                f"[ProjectileManager] joint '{self.ball_joint_name}' must be a freejoint."
            )

        self.ball_qpos_adr = self.robot.jnt_qposadr[self.ball_joint_id]
        self.ball_qvel_adr = self.robot.jnt_dofadr[self.ball_joint_id]

        self.reset()

    def reset(self):
        self.park_ball()

    def park_ball(self):
        qpos_adr = self.ball_qpos_adr
        qvel_adr = self.ball_qvel_adr

        # freejoint qpos = [x, y, z, qw, qx, qy, qz]
        self.data.qpos[qpos_adr:qpos_adr + 3] = self.park_position
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )

        # freejoint qvel = [vx, vy, vz, wx, wy, wz]
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    def spawn_ball_towards_robot(
        self,
        speed=1.0,
        radius_range=(1.0, 1.0),
        height_range=(0.5, 0.5),
        target_height=0.0,
        random_yaw_range=(0.0, 2.0 * np.pi),
        angular_vel=None,
    ):
        """
        从机器人周围随机角度、空中位置生成球，并以固定速度撞向机器人。

        Args:
            speed: 球的线速度大小（m/s）
            radius_range: 生成半径范围（相对机器人 base）
            height_range: 生成高度范围
            target_height: 目标撞击高度（相对机器人 base）
            random_yaw_range: 随机角度范围，默认全方向 [0, 2pi]
            angular_vel: 可选角速度，None 则设为 0
        """
        base_pos = self.data.qpos[0:3].copy()

        radius = np.random.uniform(radius_range[0], radius_range[1])
        theta = np.random.uniform(random_yaw_range[0], random_yaw_range[1])
        height = np.random.uniform(height_range[0], height_range[1])

        spawn_pos = base_pos + np.array(
            [
                radius * np.cos(theta),
                radius * np.sin(theta),
                height,
            ],
            dtype=np.float64,
        )

        target_pos = base_pos + np.array([0.0, 0.0, target_height], dtype=np.float64)

        direction = target_pos - spawn_pos
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            print("[ProjectileManager] Skip spawn: direction norm too small.")
            return

        direction = direction / norm
        linear_vel = direction * float(speed)

        if angular_vel is None:
            angular_vel = np.zeros(3, dtype=np.float64)
        else:
            angular_vel = np.asarray(angular_vel, dtype=np.float64)

        qpos_adr = self.ball_qpos_adr
        qvel_adr = self.ball_qvel_adr

        self.data.qpos[qpos_adr:qpos_adr + 3] = spawn_pos
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )

        self.data.qvel[qvel_adr:qvel_adr + 3] = linear_vel
        self.data.qvel[qvel_adr + 3:qvel_adr + 6] = angular_vel

        print(
            "[ProjectileManager] Spawn ball: "
            f"pos={spawn_pos}, vel={linear_vel}, target={target_pos}"
        )

    def spawn_ball_from_sector(
        self,
        speed=1.0,
        sector="front",
        radius_range=(1.5, 2.5),
        height_range=(0.6, 1.2),
        target_height=0.35,
        angular_vel=None,
    ):
        """
        从指定扇区发球。扇区基于机器人世界坐标的水平面定义：
            front: [-45°, 45°]
            left : [45°, 135°]
            back : [135°, 225°]
            right: [225°, 315°]
        """
        sector_ranges = {
            "front": (-np.pi / 4, np.pi / 4),
            "left": (np.pi / 4, 3 * np.pi / 4),
            "back": (3 * np.pi / 4, 5 * np.pi / 4),
            "right": (5 * np.pi / 4, 7 * np.pi / 4),
        }

        if sector not in sector_ranges:
            raise ValueError(
                f"[ProjectileManager] Unknown sector '{sector}', "
                f"choose from {list(sector_ranges.keys())}"
            )

        self.spawn_ball_towards_robot(
            speed=speed,
            radius_range=radius_range,
            height_range=height_range,
            target_height=target_height,
            random_yaw_range=sector_ranges[sector],
            angular_vel=angular_vel,
        )