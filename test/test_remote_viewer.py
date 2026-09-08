from types import SimpleNamespace

import pytest

from mujoco_sim.scripts.base.base import MujocoDeploy


class _FakeServer:
    def __init__(self, port=8080):
        self.port = port
        self.stop_calls = 0

    def get_port(self):
        return self.port

    def stop(self):
        self.stop_calls += 1


class _FakeScene:
    def __init__(self):
        self.gui_calls = 0
        self.update_calls = []
        self.camera_tracking_enabled = True

    def create_visualization_gui(self):
        self.gui_calls += 1

    def update_from_mjdata(self, data):
        self.update_calls.append(data)


def test_run_selects_native_or_remote_viewer():
    deploy = object.__new__(MujocoDeploy)
    calls = []
    deploy.reset = lambda: calls.append("reset")
    deploy._run_native_viewer = lambda duration: calls.append(("native", duration))
    deploy._run_remote_viewer = lambda duration: calls.append(("remote", duration))

    deploy.remote_enabled = False
    deploy.run(duration=2.0)
    assert calls == ["reset", ("native", 2.0)]

    calls.clear()
    deploy.remote_enabled = True
    deploy.run(duration=3.0)
    assert calls == ["reset", ("remote", 3.0)]


def test_create_remote_viewer_uses_configured_endpoint(monkeypatch):
    server = _FakeServer(port=8080)
    scene = _FakeScene()
    created = []
    viser_module = SimpleNamespace(
        ViserServer=lambda host, port: created.append((host, port)) or server
    )
    mjviser_module = SimpleNamespace(
        ViserMujocoScene=lambda actual_server, robot, num_envs: (
            created.append((actual_server, robot, num_envs)) or scene
        )
    )
    monkeypatch.setattr(
        "mujoco_sim.scripts.base.base.import_module",
        lambda name: {"viser": viser_module, "mjviser": mjviser_module}[name],
    )

    deploy = object.__new__(MujocoDeploy)
    deploy.config = SimpleNamespace(remote_host="127.0.0.1", remote_viewer_port=8080)
    deploy.robot = object()

    actual_server, actual_scene = deploy._create_remote_viewer()

    assert actual_server is server
    assert actual_scene is scene
    assert created == [
        ("127.0.0.1", 8080),
        (server, deploy.robot, 1),
    ]
    assert scene.gui_calls == 1
    assert server.stop_calls == 0


def test_create_remote_viewer_rejects_viser_port_fallback(monkeypatch):
    server = _FakeServer(port=8081)
    viser_module = SimpleNamespace(ViserServer=lambda host, port: server)
    mjviser_module = SimpleNamespace(ViserMujocoScene=lambda *args, **kwargs: _FakeScene())
    monkeypatch.setattr(
        "mujoco_sim.scripts.base.base.import_module",
        lambda name: {"viser": viser_module, "mjviser": mjviser_module}[name],
    )

    deploy = object.__new__(MujocoDeploy)
    deploy.config = SimpleNamespace(remote_host="127.0.0.1", remote_viewer_port=8080)
    deploy.robot = object()

    with pytest.raises(RuntimeError, match="selected 8081.*does not match the SSH tunnel"):
        deploy._create_remote_viewer()
    assert server.stop_calls == 1


def test_remote_viewer_publishes_at_configured_rate_and_cleans_up(monkeypatch):
    server = _FakeServer()
    scene = _FakeScene()
    clock = SimpleNamespace(now=0.0)
    step_calls = []

    monkeypatch.setattr(
        "mujoco_sim.scripts.base.base.mujoco.mj_forward",
        lambda robot, data: None,
    )
    monkeypatch.setattr(
        "mujoco_sim.scripts.base.base.time.perf_counter",
        lambda: clock.now,
    )

    deploy = object.__new__(MujocoDeploy)
    deploy.config = SimpleNamespace(remote_viewer_update_hz=10.0)
    deploy.robot = SimpleNamespace(opt=SimpleNamespace(timestep=0.05))
    deploy.data = object()
    deploy.mjviser_scene = None
    deploy._create_remote_viewer = lambda: (server, scene)
    deploy.step = lambda: step_calls.append(clock.now)

    def pace(next_tick):
        clock.now += 0.05
        return next_tick

    deploy._pace_simulation = pace
    deploy._run_remote_viewer(duration=0.35)

    assert len(step_calls) == 7
    assert scene.update_calls == [deploy.data] * 4
    assert deploy.mjviser_scene is None
    assert server.stop_calls == 1


def test_remote_viewer_skips_expired_render_periods(monkeypatch):
    server = _FakeServer()
    scene = _FakeScene()
    clock = SimpleNamespace(now=0.0)

    monkeypatch.setattr(
        "mujoco_sim.scripts.base.base.mujoco.mj_forward",
        lambda robot, data: None,
    )
    monkeypatch.setattr(
        "mujoco_sim.scripts.base.base.time.perf_counter",
        lambda: clock.now,
    )

    deploy = object.__new__(MujocoDeploy)
    deploy.config = SimpleNamespace(remote_viewer_update_hz=10.0)
    deploy.robot = SimpleNamespace(opt=SimpleNamespace(timestep=0.01))
    deploy.data = object()
    deploy.mjviser_scene = None
    deploy._create_remote_viewer = lambda: (server, scene)
    deploy.step = lambda: None

    def pace(next_tick):
        clock.now += 0.25
        return next_tick

    deploy._pace_simulation = pace
    deploy._run_remote_viewer(duration=0.5)

    assert scene.update_calls == [deploy.data] * 2
    assert server.stop_calls == 1


def test_remote_viewer_stops_server_on_keyboard_interrupt(monkeypatch):
    server = _FakeServer()
    scene = _FakeScene()
    monkeypatch.setattr(
        "mujoco_sim.scripts.base.base.mujoco.mj_forward",
        lambda robot, data: None,
    )

    deploy = object.__new__(MujocoDeploy)
    deploy.config = SimpleNamespace(remote_viewer_update_hz=30.0)
    deploy.robot = SimpleNamespace(opt=SimpleNamespace(timestep=0.01))
    deploy.data = object()
    deploy.mjviser_scene = None
    deploy._create_remote_viewer = lambda: (server, scene)
    deploy.step = lambda: (_ for _ in ()).throw(KeyboardInterrupt())

    deploy._run_remote_viewer(duration=1.0)

    assert deploy.mjviser_scene is None
    assert server.stop_calls == 1


def test_remote_viewer_stops_server_and_propagates_step_failure(monkeypatch):
    server = _FakeServer()
    scene = _FakeScene()
    monkeypatch.setattr(
        "mujoco_sim.scripts.base.base.mujoco.mj_forward",
        lambda robot, data: None,
    )

    deploy = object.__new__(MujocoDeploy)
    deploy.config = SimpleNamespace(remote_viewer_update_hz=30.0)
    deploy.robot = SimpleNamespace(opt=SimpleNamespace(timestep=0.01))
    deploy.data = object()
    deploy.mjviser_scene = None
    deploy._create_remote_viewer = lambda: (server, scene)
    deploy.step = lambda: (_ for _ in ()).throw(RuntimeError("step failed"))

    with pytest.raises(RuntimeError, match="step failed"):
        deploy._run_remote_viewer(duration=1.0)

    assert deploy.mjviser_scene is None
    assert server.stop_calls == 1


@pytest.mark.parametrize("update_hz", (0.0, -1.0, float("inf"), float("nan")))
def test_remote_viewer_rejects_invalid_update_rate(update_hz):
    deploy = object.__new__(MujocoDeploy)
    deploy.config = SimpleNamespace(remote_viewer_update_hz=update_hz)

    with pytest.raises(ValueError, match="remote_viewer_update_hz"):
        deploy._run_remote_viewer(duration=1.0)


class _CameraGamepad:
    def is_axis_pressed(self, name, threshold):
        return name == "R2"

    def is_button_pressed(self, name):
        return False


def _camera_deploy(scene):
    deploy = object.__new__(MujocoDeploy)
    deploy.gamepad = _CameraGamepad()
    deploy.prev_l2_pressed = False
    deploy.prev_r2_pressed = False
    deploy.prev_a_pressed = False
    deploy.follow_camera = True
    deploy.mjviser_scene = scene
    deploy.projectile_manager = SimpleNamespace(spawn_ball_towards_robot=lambda speed: None)
    return deploy


def test_r2_toggles_native_camera_follow():
    deploy = _camera_deploy(scene=None)

    deploy.handle_gamepad_events()

    assert not deploy.follow_camera


def test_r2_toggles_mjviser_camera_tracking_from_web_state():
    scene = _FakeScene()
    scene.camera_tracking_enabled = False
    deploy = _camera_deploy(scene)

    deploy.handle_gamepad_events()

    assert deploy.follow_camera
    assert scene.camera_tracking_enabled
