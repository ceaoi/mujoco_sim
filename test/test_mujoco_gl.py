import os

from mujoco_sim import run_script
from mujoco_sim.utils.mujoco_gl import configure_headless_mujoco_gl


def test_headless_process_uses_egl_by_default():
    environ = {}

    assert configure_headless_mujoco_gl(environ)
    assert environ["MUJOCO_GL"] == "egl"


def test_display_process_keeps_default_backend_selection():
    environ = {"DISPLAY": ":0"}

    assert not configure_headless_mujoco_gl(environ)
    assert "MUJOCO_GL" not in environ


def test_explicit_mujoco_backend_is_preserved_without_display():
    environ = {"MUJOCO_GL": "osmesa"}

    assert not configure_headless_mujoco_gl(environ)
    assert environ["MUJOCO_GL"] == "osmesa"


def test_run_script_selects_egl_before_loading_target(monkeypatch):
    loaded = []
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setattr(run_script.os.path, "isfile", lambda path: True)

    def fake_run_path(path, run_name):
        loaded.append((path, run_name, os.environ.get("MUJOCO_GL")))

    monkeypatch.setattr(run_script.runpy, "run_path", fake_run_path)

    run_script.main(["--filename=zb02w_depth"])

    assert loaded == [
        (
            os.path.join(run_script.WORK_DIR, "scripts", "zb02w_depth.py"),
            "__main__",
            "egl",
        )
    ]
