import math
import warnings

import cv2
import mujoco
import numpy as np

from mujoco_sim.configs import Zb02wDepthConfig
from mujoco_sim.scripts.zb02w_ts import Zb02wRoughDeploy


DEPTH_WINDOW_NAME = "ZB02W Depth"
# MuJoCo/OpenGL cameras look along local -Z with +Y up.  The training camera
# convention uses +X forward with +Z up, so this fixed rotation aligns the
# camera frames before applying the configured training-side rotation.
CAMERA_FRAME_ALIGNMENT_QUAT = np.array(
    [0.5, 0.5, -0.5, -0.5],
    dtype=np.float64,
)
POINTCLOUD_RGBA = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
POINTCLOUD_ROTATION = np.eye(3, dtype=np.float64).reshape(-1)


class Zb02wDepthDeploy(Zb02wRoughDeploy):

    def __init__(self, config: Zb02wDepthConfig, device="cpu"):
        super().__init__(config, device)
        self._depth_renderer = self._create_depth_renderer()
        self._depth_renderer.enable_depth_rendering()

        shape = (config.depth_camera_height, config.depth_camera_width)
        self.depth_image_metric = np.full(shape, config.depth_max, dtype=np.float32)
        self.depth_image = np.zeros((1, 1, *shape), dtype=np.float32)
        self.depth_points_world = np.empty((0, 3), dtype=np.float32)
        self._depth_valid_mask = np.zeros(shape, dtype=bool)
        self._next_depth_update_time = 0.0
        self.depth_camera_display_enabled = bool(config.depth_camera_display)
        self.depth_pointcloud_display_enabled = bool(
            config.depth_pointcloud_display
        )
        self._depth_window_created = False
        self._depth_camera_id = mujoco.mj_name2id(
            self.robot,
            mujoco.mjtObj.mjOBJ_CAMERA,
            config.depth_camera_name,
        )
        if self._depth_camera_id < 0:
            raise RuntimeError(
                f"Compiled depth camera not found: {config.depth_camera_name!r}"
            )

    def _load_model(self, merged_xml_path: str) -> mujoco.MjModel:
        config = self.config
        self._validate_depth_camera_config(config)

        spec = mujoco.MjSpec.from_file(merged_xml_path)
        target_body = spec.body(config.depth_camera_link)
        if target_body is None:
            raise ValueError(
                f"Depth camera target link not found: {config.depth_camera_link!r}"
            )
        if spec.camera(config.depth_camera_name) is not None:
            raise ValueError(
                f"Depth camera name already exists: {config.depth_camera_name!r}"
            )

        configured_quat = np.asarray(config.depth_camera_quat, dtype=np.float64)
        configured_quat /= np.linalg.norm(configured_quat)
        mujoco_quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(
            mujoco_quat,
            configured_quat,
            CAMERA_FRAME_ALIGNMENT_QUAT,
        )
        target_body.add_camera(
            name=config.depth_camera_name,
            pos=config.depth_camera_pos,
            quat=mujoco_quat,
            fovy=config.depth_camera_fovy,
            resolution=(config.depth_camera_width, config.depth_camera_height),
        )
        model = spec.compile()
        if not math.isfinite(model.stat.extent) or model.stat.extent <= 0.0:
            raise ValueError("compiled model extent must be finite and positive")
        # MuJoCo stores znear relative to model.stat.extent.  Use an absolute
        # camera-space distance so distant parked objects cannot move the near
        # clipping plane into the configured depth range.
        model.vis.map.znear = config.depth_camera_near / model.stat.extent
        return model

    @staticmethod
    def _validate_depth_camera_config(config: Zb02wDepthConfig) -> None:
        pos = np.asarray(config.depth_camera_pos, dtype=np.float64)
        quat = np.asarray(config.depth_camera_quat, dtype=np.float64)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            raise ValueError("depth_camera_pos must contain three finite values")
        if quat.shape != (4,) or not np.all(np.isfinite(quat)):
            raise ValueError("depth_camera_quat must be a finite wxyz quaternion")
        if np.linalg.norm(quat) <= np.finfo(np.float64).eps:
            raise ValueError("depth_camera_quat must have non-zero norm")
        if config.depth_camera_width <= 0 or config.depth_camera_height <= 0:
            raise ValueError("depth camera width and height must be positive")
        if not 0.0 < config.depth_camera_fovy < 180.0:
            raise ValueError("depth_camera_fovy must be between 0 and 180 degrees")
        if (
            not math.isfinite(config.depth_camera_near)
            or not 0.0 < config.depth_camera_near <= config.depth_min
        ):
            raise ValueError(
                "depth_camera_near must satisfy "
                "0 < depth_camera_near <= depth_min"
            )
        if (
            not math.isfinite(config.depth_camera_update_period)
            or config.depth_camera_update_period <= 0.0
        ):
            raise ValueError("depth_camera_update_period must be positive")
        if (
            isinstance(config.depth_camera_display_scale, bool)
            or not isinstance(config.depth_camera_display_scale, int)
            or config.depth_camera_display_scale <= 0
        ):
            raise ValueError(
                "depth_camera_display_scale must be a positive integer"
            )
        if (
            not math.isfinite(config.depth_min)
            or not math.isfinite(config.depth_max)
            or not 0.0 <= config.depth_min < config.depth_max
        ):
            raise ValueError("depth range must satisfy 0 <= depth_min < depth_max")
        if (
            isinstance(config.depth_pointcloud_stride, bool)
            or not isinstance(config.depth_pointcloud_stride, int)
            or config.depth_pointcloud_stride <= 0
        ):
            raise ValueError("depth_pointcloud_stride must be a positive integer")
        if (
            not math.isfinite(config.depth_pointcloud_radius)
            or config.depth_pointcloud_radius <= 0.0
        ):
            raise ValueError("depth_pointcloud_radius must be positive")

    def _create_depth_renderer(self) -> mujoco.Renderer:
        return mujoco.Renderer(
            self.robot,
            height=self.config.depth_camera_height,
            width=self.config.depth_camera_width,
        )

    def _process_depth_image(
        self,
        raw_depth: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        expected_shape = (
            self.config.depth_camera_height,
            self.config.depth_camera_width,
        )
        raw_depth = np.asarray(raw_depth, dtype=np.float32)
        if raw_depth.shape != expected_shape:
            raise ValueError(
                f"Expected depth image shape {expected_shape}, got {raw_depth.shape}"
            )

        valid_mask = (
            np.isfinite(raw_depth)
            & (raw_depth > 0.0)
            & (raw_depth < self.config.depth_max)
        )
        metric = np.nan_to_num(
            raw_depth,
            nan=self.config.depth_max,
            posinf=self.config.depth_max,
            neginf=self.config.depth_min,
        )
        metric = np.clip(
            metric,
            self.config.depth_min,
            self.config.depth_max,
        ).astype(np.float32, copy=False)
        proximity = (
            (self.config.depth_max - metric)
            / (self.config.depth_max - self.config.depth_min)
        ).astype(np.float32, copy=False)
        return metric, proximity[None, None, ...], valid_mask

    def _depth_to_world_points(
        self,
        metric_depth: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        if valid_mask.shape != metric_depth.shape:
            raise ValueError(
                "valid_mask and metric_depth must have the same shape"
            )
        height, width = metric_depth.shape
        stride = self.config.depth_pointcloud_stride
        v, u = np.mgrid[0:height:stride, 0:width:stride]
        depth = metric_depth[::stride, ::stride]
        sampled_valid = valid_mask[::stride, ::stride]

        if not np.any(sampled_valid):
            return np.empty((0, 3), dtype=np.float32)

        u = u[sampled_valid]
        v = v[sampled_valid]
        depth = depth[sampled_valid]

        fovy = math.radians(float(self.robot.cam_fovy[self._depth_camera_id]))
        focal_length = height / (2.0 * math.tan(fovy / 2.0))
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0

        x_camera = (u - center_x) * depth / focal_length
        y_camera = -(v - center_y) * depth / focal_length
        z_camera = -depth
        points_camera = np.stack(
            (x_camera, y_camera, z_camera),
            axis=1,
        )

        rotation_world_from_camera = self.data.cam_xmat[
            self._depth_camera_id
        ].reshape(3, 3)
        translation_world_from_camera = self.data.cam_xpos[
            self._depth_camera_id
        ]
        return (
            points_camera @ rotation_world_from_camera.T
            + translation_world_from_camera
        ).astype(np.float32, copy=False)

    def _draw_depth_pointcloud(self) -> None:
        viewer = getattr(self, "viewer", None)
        if not self.depth_pointcloud_display_enabled or viewer is None:
            return

        scene = viewer.user_scn
        if scene is None:
            return

        point_size = np.array(
            [self.config.depth_pointcloud_radius, 0.0, 0.0],
            dtype=np.float64,
        )
        with viewer.lock():
            scene.ngeom = 0
            point_count = min(len(self.depth_points_world), scene.maxgeom)
            for index in range(point_count):
                mujoco.mjv_initGeom(
                    scene.geoms[index],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=point_size,
                    pos=self.depth_points_world[index],
                    mat=POINTCLOUD_ROTATION,
                    rgba=POINTCLOUD_RGBA,
                )
            scene.ngeom = point_count

    def _clear_depth_pointcloud(self) -> None:
        self.depth_points_world = np.empty((0, 3), dtype=np.float32)
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return

        scene = viewer.user_scn
        if scene is None:
            return
        try:
            with viewer.lock():
                scene.ngeom = 0
        except RuntimeError:
            # The passive viewer can already be closed when run() unwinds.
            pass

    def _update_depth_camera(self) -> None:
        self._depth_renderer.update_scene(
            self.data,
            camera=self.config.depth_camera_name,
        )
        raw_depth = self._depth_renderer.render()
        (
            self.depth_image_metric,
            self.depth_image,
            self._depth_valid_mask,
        ) = self._process_depth_image(raw_depth)
        self.depth_points_world = self._depth_to_world_points(
            self.depth_image_metric,
            self._depth_valid_mask,
        )
        self._draw_depth_pointcloud()

        if self.depth_camera_display_enabled:
            display = np.rint(self.depth_image[0, 0] * 255.0).astype(np.uint8)
            display = cv2.resize(
                display,
                dsize=None,
                fx=self.config.depth_camera_display_scale,
                fy=self.config.depth_camera_display_scale,
                interpolation=cv2.INTER_NEAREST,
            )
            try:
                cv2.imshow(DEPTH_WINDOW_NAME, display)
                cv2.waitKey(1)
                self._depth_window_created = True
            except cv2.error as exc:
                self.depth_camera_display_enabled = False
                warnings.warn(
                    f"OpenCV depth display disabled: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _update_depth_camera_if_due(self) -> None:
        now = float(self.data.time)
        if now + np.finfo(np.float64).eps < self._next_depth_update_time:
            return

        self._update_depth_camera()
        elapsed_periods = max(
            1,
            math.floor(
                (now - self._next_depth_update_time)
                / self.config.depth_camera_update_period
            )
            + 1,
        )
        self._next_depth_update_time += (
            elapsed_periods * self.config.depth_camera_update_period
        )

    def step(self):
        self._update_depth_camera_if_due()
        super().step()

    def reset(self):
        super().reset()
        if hasattr(self, "depth_image_metric"):
            self.depth_image_metric.fill(self.config.depth_max)
            self.depth_image.fill(0.0)
            self._depth_valid_mask.fill(False)
            self._clear_depth_pointcloud()
            self._next_depth_update_time = 0.0

    def close_depth_camera(self) -> None:
        if hasattr(self, "depth_points_world"):
            self._clear_depth_pointcloud()
        if self._depth_renderer is not None:
            self._depth_renderer.close()
            self._depth_renderer = None
        if self._depth_window_created:
            try:
                cv2.destroyWindow(DEPTH_WINDOW_NAME)
            except cv2.error:
                pass
            self._depth_window_created = False

    def run(self, duration=1e3):
        try:
            super().run(duration)
        finally:
            self.close_depth_camera()


if __name__ == "__main__":
    deploy = Zb02wDepthDeploy(Zb02wDepthConfig())
    deploy.run()
