import pytest

from mujoco_sim import remote_client


class _FakeResponse:
    def __init__(self, status):
        self.status = status
        self.closed = False

    def close(self):
        self.closed = True


def test_remote_client_argument_defaults():
    args = remote_client.build_argument_parser().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.gamepad_port == 5555
    assert args.viewer_port == 8080
    assert args.index == 0
    assert args.hz == 50.0
    assert not args.no_browser


def test_remote_client_does_not_accept_filename():
    with pytest.raises(SystemExit):
        remote_client.main(["--filename=zb02w_rough"])


def test_viewer_readiness_uses_http_status(monkeypatch):
    response = _FakeResponse(status=200)
    calls = []
    monkeypatch.setattr(
        remote_client,
        "urlopen",
        lambda url, timeout: calls.append((url, timeout)) or response,
    )

    assert remote_client.is_viewer_ready("http://127.0.0.1:8080")
    assert calls == [("http://127.0.0.1:8080", remote_client.VIEWER_PROBE_TIMEOUT)]
    assert response.closed


def test_viewer_readiness_returns_false_on_connection_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(remote_client, "urlopen", fail)

    assert not remote_client.is_viewer_ready("http://127.0.0.1:8080")


def test_remote_client_opens_browser_once_and_cleans_up(monkeypatch):
    events = []

    class FakeGamepad:
        def __init__(self, joystick_index, deadzone):
            events.append(("gamepad", joystick_index, deadzone))
            self.joystick = object()
            self.poll_count = 0

        def connect(self):
            raise AssertionError("already connected")

        def poll(self):
            self.poll_count += 1
            if self.poll_count > 2:
                raise KeyboardInterrupt
            return True

        def get_state(self):
            return {"state": "snapshot"}

        def close(self):
            events.append("gamepad_closed")

    class FakeSender:
        def __init__(self, host, port, retry_interval):
            events.append(("sender", host, port, retry_interval))

        def send_state(self, state):
            events.append(("sent", state))
            return True

        def close(self):
            events.append("sender_closed")

    probes = []
    browser_urls = []
    monkeypatch.setattr(remote_client, "Gamepad", FakeGamepad)
    monkeypatch.setattr(remote_client, "RemoteGamepadSender", FakeSender)
    monkeypatch.setattr(
        remote_client,
        "is_viewer_ready",
        lambda url: probes.append(url) or True,
    )
    monkeypatch.setattr(
        remote_client.webbrowser,
        "open",
        lambda url: browser_urls.append(url) or True,
    )
    monkeypatch.setattr(remote_client.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(remote_client.time, "sleep", lambda duration: None)

    result = remote_client.main([])

    assert result == 0
    assert probes == ["http://127.0.0.1:8080"]
    assert browser_urls == ["http://127.0.0.1:8080"]
    assert ("sent", {"state": "snapshot"}) in events
    assert events[-2:] == ["sender_closed", "gamepad_closed"]


def test_remote_client_no_browser_skips_viewer_probe(monkeypatch):
    events = []

    class FakeGamepad:
        def __init__(self, joystick_index, deadzone):
            self.joystick = object()

        def poll(self):
            raise KeyboardInterrupt

        def close(self):
            events.append("gamepad_closed")

    class FakeSender:
        def __init__(self, host, port, retry_interval):
            pass

        def close(self):
            events.append("sender_closed")

    monkeypatch.setattr(remote_client, "Gamepad", FakeGamepad)
    monkeypatch.setattr(remote_client, "RemoteGamepadSender", FakeSender)
    monkeypatch.setattr(
        remote_client,
        "is_viewer_ready",
        lambda url: (_ for _ in ()).throw(AssertionError("unexpected probe")),
    )

    assert remote_client.main(["--no-browser"]) == 0
    assert events == ["sender_closed", "gamepad_closed"]


def test_browser_failure_prints_manual_url_and_does_not_raise(monkeypatch, capsys):
    monkeypatch.setattr(remote_client.webbrowser, "open", lambda url: False)

    remote_client._open_browser_once("http://127.0.0.1:8080")

    assert "open http://127.0.0.1:8080 manually" in capsys.readouterr().out
