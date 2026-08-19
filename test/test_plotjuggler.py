from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from mujoco_sim.configs import M20FlatConfig, Wh044xConfig
from mujoco_sim.scripts.base.base import MujocoDeploy


def _make_deploy(config):
    deploy = object.__new__(MujocoDeploy)
    deploy.config = config
    return deploy


class _FakeSender:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.messages = []

    def send_data(self, name, value):
        self.messages.append((name, value))


def test_disabled_plotjuggler_does_not_import_data_vis(monkeypatch):
    deploy = _make_deploy(M20FlatConfig(plotjuggler_enabled=False))

    def fail_if_imported(_):
        raise AssertionError("data_vis must not be imported when telemetry is disabled")

    monkeypatch.setattr("mujoco_sim.scripts.base.base.import_module", fail_if_imported)
    deploy._init_plotjuggler()
    deploy.send_plotjuggler_data("obs", np.ones(3))

    assert not deploy.plotjuggler_enabled
    assert deploy.plotjuggler is None


def test_enabled_plotjuggler_creates_one_sender_and_sends_data(monkeypatch):
    deploy = _make_deploy(M20FlatConfig())
    module = SimpleNamespace(PlotJugglerUDP=_FakeSender)
    monkeypatch.setattr(
        "mujoco_sim.scripts.base.base.import_module",
        lambda name: module,
    )

    deploy._init_plotjuggler()
    values = np.array([1.0, 2.0], dtype=np.float32)
    deploy.send_plotjuggler_data("actions", values)

    assert deploy.plotjuggler_enabled
    assert deploy.plotjuggler.host == "127.0.0.1"
    assert deploy.plotjuggler.port == 5005
    assert deploy.plotjuggler.messages == [("actions", values)]


def test_missing_data_vis_warns_and_overrides_runtime_config(monkeypatch):
    deploy = _make_deploy(M20FlatConfig())

    def raise_missing(_):
        raise ModuleNotFoundError("No module named 'data_vis'")

    monkeypatch.setattr("mujoco_sim.scripts.base.base.import_module", raise_missing)

    with pytest.warns(RuntimeWarning, match="initialization failed.*ModuleNotFoundError"):
        deploy._init_plotjuggler()

    assert not deploy.plotjuggler_enabled
    assert deploy.plotjuggler is None
    assert not deploy.config.plotjuggler_enabled


def test_sender_construction_failure_warns_and_disables_telemetry(monkeypatch):
    deploy = _make_deploy(M20FlatConfig())

    def fail_to_construct(*_):
        raise OSError("socket unavailable")

    module = SimpleNamespace(PlotJugglerUDP=fail_to_construct)
    monkeypatch.setattr(
        "mujoco_sim.scripts.base.base.import_module",
        lambda name: module,
    )

    with pytest.warns(RuntimeWarning, match="initialization failed.*socket unavailable"):
        deploy._init_plotjuggler()

    assert not deploy.plotjuggler_enabled
    assert deploy.plotjuggler is None
    assert not deploy.config.plotjuggler_enabled


def test_send_failure_warns_once_and_disables_telemetry():
    class FailingSender:
        def send_data(self, name, value):
            raise OSError("network unavailable")

    deploy = _make_deploy(M20FlatConfig())
    deploy.plotjuggler_enabled = True
    deploy.plotjuggler = FailingSender()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deploy.send_plotjuggler_data("obs", np.ones(3))
        deploy.send_plotjuggler_data("obs", np.ones(3))

    assert len(caught) == 1
    assert "send failed" in str(caught[0].message)
    assert not deploy.plotjuggler_enabled
    assert deploy.plotjuggler is None
    assert not deploy.config.plotjuggler_enabled


def test_non_wheel_legged_config_disables_plotjuggler_by_default():
    assert not Wh044xConfig().plotjuggler_enabled
