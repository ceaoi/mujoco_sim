from types import SimpleNamespace

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pytest

from mujoco_sim.configs import Zb02wDepthConfig
from mujoco_sim.scripts.zb02w_depth import (
    CAMERA_FRAME_ALIGNMENT_QUAT,
    DEPTH_WINDOW_NAME,
    Zb02wDepthDeploy,
)
from mujoco_sim.scripts.zb02w_ts import Zb02wRoughDeploy


MINIMAL_MJCF = """
<mujoco>
  <worldbody>
    <body name="base_link">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _bare_deploy(**config_overrides):
    deploy = object.__new__(Zb02wDepthDeploy)
    config_kwargs = {
        "plotjuggler_enabled": False,
        "depth_camera_display": False,
        **config_overrides,
    }
    deploy.config = Zb02wDepthConfig(**config_kwargs)
    deploy._depth_figure = None
    deploy._depth_axes = None
    deploy._depth_image_artist = None
    deploy._next_depth_display_update_time = 0.0
    return deploy


def test_load_model_attaches_camera_to_requested_body(tmp_path):
    xml_path = tmp_path / "robot.xml"
    xml_path.write_text(MINIMAL_MJCF)
    deploy = _bare_deploy()

    model = deploy._load_model(str(xml_path))

    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        deploy.config.depth_camera_name,
    )
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        deploy.config.depth_camera_link,
    )
    assert camera_id >= 0
    assert model.cam_bodyid[camera_id] == body_id
    np.testing.assert_allclose(
        model.cam_pos[camera_id],
        deploy.config.depth_camera_pos,
    )
    configured_quat = np.asarray(deploy.config.depth_camera_quat)
    configured_quat /= np.linalg.norm(configured_quat)
    expected_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(
        expected_quat,
        configured_quat,
        CAMERA_FRAME_ALIGNMENT_QUAT,
    )
    np.testing.assert_allclose(model.cam_quat[camera_id], expected_quat, atol=1e-7)
    assert model.cam_fovy[camera_id] == pytest.approx(
        deploy.config.depth_camera_fovy
    )
    np.testing.assert_array_equal(
        model.cam_resolution[camera_id],
        (deploy.config.depth_camera_width, deploy.config.depth_camera_height),
    )
    assert model.vis.map.znear * model.stat.extent == pytest.approx(
        deploy.config.depth_camera_near
    )

    camera_rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(camera_rotation, model.cam_quat[camera_id])
    look_direction = camera_rotation.reshape(3, 3) @ np.array([0.0, 0.0, -1.0])
    np.testing.assert_allclose(
        look_direction,
        (np.sqrt(0.5), 0.0, -np.sqrt(0.5)),
        atol=1e-7,
    )


def test_zero_training_rotation_aligns_camera_forward_with_link_x(tmp_path):
    xml_path = tmp_path / "aligned_robot.xml"
    xml_path.write_text(MINIMAL_MJCF)
    deploy = _bare_deploy(depth_camera_quat=(1.0, 0.0, 0.0, 0.0))

    model = deploy._load_model(str(xml_path))

    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        deploy.config.depth_camera_name,
    )
    camera_rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(camera_rotation, model.cam_quat[camera_id])
    camera_rotation = camera_rotation.reshape(3, 3)
    np.testing.assert_allclose(
        camera_rotation @ np.array([0.0, 0.0, -1.0]),
        (1.0, 0.0, 0.0),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        camera_rotation @ np.array([0.0, 1.0, 0.0]),
        (0.0, 0.0, 1.0),
        atol=1e-7,
    )


def test_load_model_rejects_missing_link_and_invalid_quaternion(tmp_path):
    xml_path = tmp_path / "robot.xml"
    xml_path.write_text(MINIMAL_MJCF)

    with pytest.raises(ValueError, match="target link not found"):
        _bare_deploy(depth_camera_link="missing_link")._load_model(str(xml_path))

    with pytest.raises(ValueError, match="non-zero norm"):
        _bare_deploy(depth_camera_quat=(0.0, 0.0, 0.0, 0.0))._load_model(
            str(xml_path)
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"depth_camera_width": 0}, "width and height"),
        ({"depth_camera_fovy": 180.0}, "between 0 and 180"),
        ({"depth_camera_near": 0.0}, "depth_camera_near"),
        ({"depth_camera_near": float("nan")}, "depth_camera_near"),
        ({"depth_camera_near": 3.0}, "depth_camera_near"),
        (
            {"depth_camera_near": 0.1, "depth_min": 0.05},
            "depth_camera_near",
        ),
        ({"depth_camera_update_period": 0.0}, "update_period"),
        ({"depth_camera_update_period": float("nan")}, "update_period"),
        (
            {"depth_camera_display_update_period": 0.0},
            "display_update_period",
        ),
        (
            {"depth_camera_display_update_period": float("nan")},
            "display_update_period",
        ),
        ({"depth_camera_display_scale": 0}, "display_scale"),
        ({"depth_camera_display_scale": True}, "display_scale"),
        ({"depth_min": 3.0, "depth_max": 3.0}, "depth range"),
        ({"depth_max": float("inf")}, "depth range"),
        ({"depth_pointcloud_stride": 0}, "stride"),
        ({"depth_pointcloud_stride": True}, "stride"),
        ({"depth_pointcloud_radius": 0.0}, "radius"),
        ({"depth_pointcloud_radius": float("nan")}, "radius"),
    ],
)
def test_depth_camera_config_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        Zb02wDepthDeploy._validate_depth_camera_config(
            Zb02wDepthConfig(**overrides)
        )


def test_depth_processing_clips_normalizes_and_preserves_shapes():
    deploy = _bare_deploy(depth_camera_width=3, depth_camera_height=2)
    raw_depth = np.array(
        [[0.1, 0.3, 1.65], [3.0, np.inf, np.nan]],
        dtype=np.float32,
    )

    metric, normalized, valid_mask = deploy._process_depth_image(raw_depth)

    assert metric.dtype == np.float32
    assert metric.shape == (2, 3)
    np.testing.assert_allclose(
        metric,
        [[0.1, 0.3, 1.65], [3.0, 3.0, 3.0]],
    )
    assert normalized.dtype == np.float32
    assert normalized.shape == (1, 1, 2, 3)
    expected_normalized = (
        deploy.config.depth_max - metric
    ) / (deploy.config.depth_max - deploy.config.depth_min)
    np.testing.assert_allclose(normalized[0, 0], expected_normalized, atol=1e-6)
    np.testing.assert_array_equal(
        valid_mask,
        [[True, True, True], [False, False, False]],
    )

    with pytest.raises(ValueError, match="Expected depth image shape"):
        deploy._process_depth_image(np.zeros((3, 2), dtype=np.float32))


class _FakeRenderer:

    def __init__(self, images):
        self.images = iter(images)
        self.update_calls = []
        self.render_calls = 0
        self.closed = False

    def update_scene(self, data, camera):
        self.update_calls.append((data, camera))

    def render(self):
        self.render_calls += 1
        return next(self.images)

    def close(self):
        self.closed = True


class _FakeImageArtist:

    def __init__(self):
        self.data_updates = []

    def set_data(self, data):
        self.data_updates.append(np.array(data, copy=True))


class _FakeFigureManager:

    def __init__(self):
        self.window_titles = []

    def set_window_title(self, title):
        self.window_titles.append(title)


class _FakeFigureCanvas:
    required_interactive_framework = "test"

    def __init__(self):
        self.manager = _FakeFigureManager()
        self.draw_idle_calls = 0
        self.flush_events_calls = 0
        self.flush_events_exception = None

    def draw_idle(self):
        self.draw_idle_calls += 1

    def flush_events(self):
        self.flush_events_calls += 1
        if self.flush_events_exception is not None:
            raise self.flush_events_exception


class _FakeFigure:

    def __init__(self):
        self.canvas = _FakeFigureCanvas()
        self.size_calls = []
        self.adjust_calls = []

    def get_dpi(self):
        return 100.0

    def set_size_inches(self, width, height, *, forward):
        self.size_calls.append((width, height, forward))

    def subplots_adjust(self, **kwargs):
        self.adjust_calls.append(kwargs)


class _FakeAxes:

    def __init__(self):
        self.axis_off_calls = 0
        self.imshow_calls = []
        self.image_artist = _FakeImageArtist()

    def set_axis_off(self):
        self.axis_off_calls += 1

    def imshow(self, data, **kwargs):
        self.imshow_calls.append((np.array(data, copy=True), kwargs))
        return self.image_artist


class _CountingLock:

    def __init__(self, viewer):
        self.viewer = viewer

    def __enter__(self):
        self.viewer.lock_entries += 1

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeViewer:

    def __init__(self, scene):
        self.user_scn = scene
        self.lock_entries = 0

    def lock(self):
        return _CountingLock(self)


def _projection_deploy(
    tmp_path,
    *,
    world_camera_quat=(1.0, 0.0, 0.0, 0.0),
):
    xml_path = tmp_path / "projection_robot.xml"
    xml_path.write_text(MINIMAL_MJCF)
    deploy = _bare_deploy(
        depth_camera_width=3,
        depth_camera_height=3,
        depth_camera_fovy=90.0,
        depth_camera_pos=(1.0, 2.0, 3.0),
        depth_camera_quat=(1.0, 0.0, 0.0, 0.0),
        depth_pointcloud_stride=1,
    )
    deploy.robot = deploy._load_model(str(xml_path))
    deploy.data = mujoco.MjData(deploy.robot)
    mujoco.mj_forward(deploy.robot, deploy.data)
    deploy._depth_camera_id = mujoco.mj_name2id(
        deploy.robot,
        mujoco.mjtObj.mjOBJ_CAMERA,
        deploy.config.depth_camera_name,
    )
    world_camera_rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(
        world_camera_rotation,
        np.asarray(world_camera_quat, dtype=np.float64),
    )
    deploy.data.cam_xpos[deploy._depth_camera_id] = (1.0, 2.0, 3.0)
    deploy.data.cam_xmat[deploy._depth_camera_id] = world_camera_rotation
    return deploy


def test_depth_to_world_points_uses_camera_axes_and_adds_translation(tmp_path):
    deploy = _projection_deploy(tmp_path)
    metric = np.full((3, 3), 2.0, dtype=np.float32)
    valid = np.zeros((3, 3), dtype=bool)
    valid[1, 1] = True
    valid[0, 0] = True

    points = deploy._depth_to_world_points(metric, valid)

    np.testing.assert_allclose(points[1], (1.0, 2.0, 1.0), atol=1e-6)
    np.testing.assert_allclose(
        points[0],
        (-1.0 / 3.0, 10.0 / 3.0, 1.0),
        atol=1e-6,
    )


def test_depth_to_world_points_applies_camera_rotation(tmp_path):
    half_sqrt_two = np.sqrt(0.5)
    deploy = _projection_deploy(
        tmp_path,
        world_camera_quat=(half_sqrt_two, 0.0, 0.0, half_sqrt_two),
    )
    metric = np.ones((3, 3), dtype=np.float32)
    valid = np.zeros((3, 3), dtype=bool)
    valid[1, 2] = True

    points = deploy._depth_to_world_points(metric, valid)

    np.testing.assert_allclose(
        points,
        [[1.0, 2.0 + 2.0 / 3.0, 2.0]],
        atol=1e-6,
    )


def test_depth_to_world_points_honors_stride_and_empty_mask(tmp_path):
    deploy = _projection_deploy(tmp_path)
    deploy.config = Zb02wDepthConfig(
        plotjuggler_enabled=False,
        depth_camera_display=False,
        depth_camera_width=3,
        depth_camera_height=3,
        depth_camera_fovy=90.0,
        depth_camera_pos=(1.0, 2.0, 3.0),
        depth_camera_quat=(1.0, 0.0, 0.0, 0.0),
        depth_pointcloud_stride=2,
    )
    metric = np.ones((3, 3), dtype=np.float32)
    valid = np.ones((3, 3), dtype=bool)

    points = deploy._depth_to_world_points(metric, valid)

    assert points.shape == (4, 3)
    empty = deploy._depth_to_world_points(metric, np.zeros_like(valid))
    assert empty.shape == (0, 3)
    assert empty.dtype == np.float32


def test_draw_depth_pointcloud_populates_user_scene_and_caps_count(tmp_path):
    deploy = _projection_deploy(tmp_path)
    scene = mujoco.MjvScene(deploy.robot, maxgeom=2)
    viewer = _FakeViewer(scene)
    deploy.viewer = viewer
    deploy.depth_pointcloud_display_enabled = True
    deploy.depth_points_world = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )

    deploy._draw_depth_pointcloud()

    assert viewer.lock_entries == 1
    assert scene.ngeom == 2
    np.testing.assert_allclose(scene.geoms[0].pos, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(
        scene.geoms[0].size,
        (deploy.config.depth_pointcloud_radius,) * 3,
    )
    np.testing.assert_allclose(scene.geoms[0].rgba, (1.0, 0.0, 0.0, 1.0))


def test_depth_update_uses_simulation_time_and_displays_proximity(monkeypatch):
    deploy = _bare_deploy(
        depth_camera_width=2,
        depth_camera_height=1,
        depth_camera_display=True,
    )
    renderer = _FakeRenderer(
        [
            np.array([[0.3, 3.0]], dtype=np.float32),
            np.array([[0.9, 3.0]], dtype=np.float32),
            np.array([[1.65, 3.0]], dtype=np.float32),
        ]
    )
    deploy._depth_renderer = renderer
    deploy.data = SimpleNamespace(time=0.0)
    deploy._next_depth_update_time = 0.0
    deploy.depth_camera_display_enabled = True
    deploy.depth_pointcloud_display_enabled = True
    deploy.viewer = None
    figure = _FakeFigure()
    axes = _FakeAxes()
    show_calls = []
    monkeypatch.setattr(
        deploy,
        "_depth_to_world_points",
        lambda metric, valid: np.empty((0, 3), dtype=np.float32),
    )
    monkeypatch.setattr(plt, "subplots", lambda: (figure, axes))
    monkeypatch.setattr(plt, "show", lambda **kwargs: show_calls.append(kwargs))

    deploy._update_depth_camera_if_due()
    deploy.data.time = deploy.config.depth_camera_update_period / 2.0
    deploy._update_depth_camera_if_due()
    deploy.data.time = deploy.config.depth_camera_update_period
    deploy._update_depth_camera_if_due()
    deploy.data.time = deploy.config.depth_camera_display_update_period
    deploy._update_depth_camera_if_due()

    assert renderer.render_calls == 3
    assert [call[1] for call in renderer.update_calls] == [
        deploy.config.depth_camera_name,
        deploy.config.depth_camera_name,
        deploy.config.depth_camera_name,
    ]
    expected_first = (
        (deploy.config.depth_max - 0.3)
        / (deploy.config.depth_max - deploy.config.depth_min)
    )
    expected_second = (
        (deploy.config.depth_max - 1.65)
        / (deploy.config.depth_max - deploy.config.depth_min)
    )
    assert figure.canvas.manager.window_titles == [DEPTH_WINDOW_NAME]
    assert figure.size_calls == [(0.08, 0.04, True)]
    assert figure.adjust_calls == [
        {"left": 0.0, "right": 1.0, "bottom": 0.0, "top": 1.0}
    ]
    assert axes.axis_off_calls == 1
    assert len(axes.imshow_calls) == 1
    first_image, image_options = axes.imshow_calls[0]
    np.testing.assert_allclose(
        first_image,
        [[expected_first, 0.0]],
        atol=1e-6,
    )
    assert image_options == {
        "cmap": "turbo",
        "vmin": 0.0,
        "vmax": 1.0,
        "interpolation": "nearest",
        "origin": "upper",
        "aspect": "auto",
    }
    assert show_calls == [{"block": False}]
    assert len(axes.image_artist.data_updates) == 1
    np.testing.assert_allclose(
        axes.image_artist.data_updates[0],
        [[expected_second, 0.0]],
        atol=1e-6,
    )
    assert figure.canvas.draw_idle_calls == 2
    assert figure.canvas.flush_events_calls == 2
    assert deploy._next_depth_update_time > deploy.data.time
    assert deploy._next_depth_display_update_time == pytest.approx(
        2.0 * deploy.config.depth_camera_display_update_period
    )


def test_matplotlib_failure_does_not_disable_pointcloud(monkeypatch):
    deploy = _bare_deploy(
        depth_camera_width=1,
        depth_camera_height=1,
        depth_camera_display=True,
    )
    deploy._depth_renderer = _FakeRenderer(
        [np.array([[1.0]], dtype=np.float32)]
    )
    deploy.data = SimpleNamespace(time=0.0)
    deploy.depth_camera_display_enabled = True
    deploy.depth_pointcloud_display_enabled = True
    deploy.viewer = None
    figure = _FakeFigure()
    figure.canvas.flush_events_exception = RuntimeError(
        "GUI backend is not available"
    )
    axes = _FakeAxes()
    expected_points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    monkeypatch.setattr(
        deploy,
        "_depth_to_world_points",
        lambda metric, valid: expected_points,
    )
    drawn = []
    monkeypatch.setattr(
        deploy,
        "_draw_depth_pointcloud",
        lambda: drawn.append(deploy.depth_points_world.copy()),
    )
    monkeypatch.setattr(plt, "subplots", lambda: (figure, axes))
    monkeypatch.setattr(plt, "show", lambda **kwargs: None)
    monkeypatch.setattr(plt, "close", lambda target: None)

    with pytest.warns(RuntimeWarning, match="Matplotlib depth display disabled"):
        deploy._update_depth_camera()

    assert not deploy.depth_camera_display_enabled
    assert deploy.depth_pointcloud_display_enabled
    np.testing.assert_array_equal(deploy.depth_points_world, expected_points)
    np.testing.assert_array_equal(drawn, [expected_points])


def test_noninteractive_matplotlib_backend_disables_display(monkeypatch):
    deploy = _bare_deploy(depth_camera_width=1, depth_camera_height=1)
    deploy.depth_image = np.zeros((1, 1, 1, 1), dtype=np.float32)
    deploy.depth_camera_display_enabled = True
    figure = _FakeFigure()
    figure.canvas.required_interactive_framework = None
    axes = _FakeAxes()
    closed = []
    monkeypatch.setattr(plt, "subplots", lambda: (figure, axes))
    monkeypatch.setattr(plt, "get_backend", lambda: "agg")
    monkeypatch.setattr(plt, "close", closed.append)

    with pytest.warns(RuntimeWarning, match="backend 'agg' is not interactive"):
        deploy._display_depth_image()

    assert not deploy.depth_camera_display_enabled
    assert closed == [figure]
    assert deploy._depth_figure is None


def test_reset_clears_depth_buffers_and_restarts_schedule(monkeypatch):
    deploy = _bare_deploy(depth_camera_width=2, depth_camera_height=1)
    deploy.depth_image_metric = np.zeros((1, 2), dtype=np.float32)
    deploy.depth_image = np.ones((1, 1, 1, 2), dtype=np.float32)
    deploy._depth_valid_mask = np.ones((1, 2), dtype=bool)
    deploy.depth_points_world = np.ones((2, 3), dtype=np.float32)
    scene = SimpleNamespace(ngeom=2)
    viewer = _FakeViewer(scene)
    deploy.viewer = viewer
    deploy._next_depth_update_time = 10.0
    deploy._next_depth_display_update_time = 10.0
    monkeypatch.setattr(Zb02wRoughDeploy, "reset", lambda self: None)

    deploy.reset()

    np.testing.assert_array_equal(
        deploy.depth_image_metric,
        np.full((1, 2), deploy.config.depth_max, dtype=np.float32),
    )
    np.testing.assert_array_equal(deploy.depth_image, np.zeros((1, 1, 1, 2)))
    np.testing.assert_array_equal(deploy._depth_valid_mask, [[False, False]])
    assert deploy.depth_points_world.shape == (0, 3)
    assert scene.ngeom == 0
    assert viewer.lock_entries == 1
    assert deploy._next_depth_update_time == 0.0
    assert deploy._next_depth_display_update_time == 0.0


def test_close_depth_camera_releases_renderer_and_window(monkeypatch):
    deploy = _bare_deploy(depth_camera_width=1, depth_camera_height=1)
    renderer = _FakeRenderer([])
    deploy._depth_renderer = renderer
    deploy.depth_points_world = np.ones((1, 3), dtype=np.float32)
    deploy.viewer = None
    figure = object()
    deploy._depth_figure = figure
    deploy._depth_axes = object()
    deploy._depth_image_artist = object()
    closed = []
    monkeypatch.setattr(plt, "close", closed.append)

    deploy.close_depth_camera()
    deploy.close_depth_camera()

    assert renderer.closed
    assert deploy._depth_renderer is None
    assert deploy.depth_points_world.shape == (0, 3)
    assert closed == [figure]
    assert deploy._depth_figure is None
    assert deploy._depth_axes is None
    assert deploy._depth_image_artist is None
