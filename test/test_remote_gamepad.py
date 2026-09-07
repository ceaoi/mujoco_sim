import json
import socket
import time
from types import SimpleNamespace

import pytest

from mujoco_sim.scripts.base.base import MujocoDeploy
from mujoco_sim.utils.gamepad_pygame import Gamepad
from mujoco_sim.utils.remote_gamepad import (
    PROTOCOL_VERSION,
    RemoteGamepadSender,
    VirtualGamepad,
    decode_gamepad_message,
    encode_gamepad_message,
)


def _state(**overrides):
    state = Gamepad._make_default_state()
    state.update(overrides)
    return state


def _poll_until(gamepad, predicate, timeout=0.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert gamepad.poll()
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("condition was not reached before timeout")


def test_protocol_round_trip_preserves_complete_state():
    expected = _state(
        leftJS_x=-0.25,
        leftJS_y=0.5,
        rightJS_x=0.75,
        rightJS_y=-1.0,
        L2=0.2,
        R2=1.0,
        A=True,
        Y=True,
        R1=True,
    )

    sequence, actual = decode_gamepad_message(
        encode_gamepad_message(expected, sequence=17).rstrip(b"\n")
    )

    assert sequence == 17
    assert actual == expected


@pytest.mark.parametrize(
    "payload",
    (
        {"version": PROTOCOL_VERSION + 1, "sequence": 0, "state": _state()},
        {"version": PROTOCOL_VERSION, "sequence": -1, "state": _state()},
        {
            "version": PROTOCOL_VERSION,
            "sequence": 0,
            "state": _state(leftJS_x=True),
        },
        {
            "version": PROTOCOL_VERSION,
            "sequence": 0,
            "state": _state(leftJS_x=float("nan")),
        },
        {
            "version": PROTOCOL_VERSION,
            "sequence": 0,
            "state": _state(A=1),
        },
    ),
)
def test_protocol_rejects_invalid_messages(payload):
    frame = json.dumps(payload).encode("utf-8")
    with pytest.raises(ValueError):
        decode_gamepad_message(frame)


def test_virtual_gamepad_handles_fragmented_and_coalesced_latest_frames():
    virtual = VirtualGamepad(port=0)
    assert virtual.connect()
    client = socket.create_connection((virtual.host, virtual.port), timeout=0.5)
    try:
        first = encode_gamepad_message(_state(leftJS_x=0.25), sequence=0)
        second = encode_gamepad_message(
            _state(leftJS_x=0.75, A=True),
            sequence=1,
        )

        split_at = len(first) // 2
        client.sendall(first[:split_at])
        assert virtual.poll()
        assert virtual.get_axis("leftJS_x") == 0.0

        client.sendall(first[split_at:] + second)
        _poll_until(
            virtual,
            lambda: virtual.get_axis("leftJS_x") == pytest.approx(0.75),
        )
        assert virtual.is_button_pressed("A")
        assert virtual.is_button_rising_edge("A")

        assert virtual.poll()
        assert not virtual.is_button_rising_edge("A")
    finally:
        client.close()
        virtual.close()


def test_virtual_gamepad_holds_last_state_and_accepts_reconnection():
    virtual = VirtualGamepad(port=0)
    assert virtual.connect()
    first_client = socket.create_connection((virtual.host, virtual.port), timeout=0.5)
    try:
        first_client.sendall(
            encode_gamepad_message(_state(leftJS_y=0.6, X=True), sequence=4)
        )
        _poll_until(virtual, lambda: virtual.get_axis("leftJS_y") == 0.6)
        expected = virtual.get_state()

        first_client.close()
        _poll_until(virtual, lambda: not virtual.connected)
        assert virtual.get_state() == expected

        second_client = socket.create_connection(
            (virtual.host, virtual.port),
            timeout=0.5,
        )
        try:
            second_client.sendall(
                encode_gamepad_message(_state(rightJS_x=-0.4), sequence=0)
            )
            _poll_until(
                virtual,
                lambda: virtual.get_axis("rightJS_x") == -0.4,
            )
            assert virtual.get_axis("leftJS_y") == 0.0
            assert not virtual.is_button_pressed("X")
        finally:
            second_client.close()
    finally:
        virtual.close()


def test_invalid_frame_does_not_change_virtual_gamepad_state():
    virtual = VirtualGamepad(port=0)
    assert virtual.connect()
    client = socket.create_connection((virtual.host, virtual.port), timeout=0.5)
    try:
        client.sendall(encode_gamepad_message(_state(R2=0.8), sequence=0))
        _poll_until(virtual, lambda: virtual.get_axis("R2") == 0.8)
        expected = virtual.get_state()

        client.sendall(b'{"version":99}\n')
        for _ in range(5):
            assert virtual.poll()
        assert virtual.get_state() == expected
    finally:
        client.close()
        virtual.close()


def test_remote_sender_delivers_state_to_virtual_gamepad():
    virtual = VirtualGamepad(port=0)
    assert virtual.connect()
    sender = RemoteGamepadSender(virtual.host, virtual.port, retry_interval=0.0)
    try:
        assert sender.send_state(_state(B=True, leftJS_y=-0.5))
        _poll_until(virtual, lambda: virtual.is_button_pressed("B"))
        assert virtual.get_cmd()[0] == -0.5
    finally:
        sender.close()
        virtual.close()


def test_base_selects_local_gamepad_by_default(monkeypatch):
    created = []

    class FakeGamepad:
        def __init__(self, joystick_index):
            created.append(("local", joystick_index))

        def connect(self):
            return False

    monkeypatch.setattr("mujoco_sim.scripts.base.base.Gamepad", FakeGamepad)
    deploy = object.__new__(MujocoDeploy)
    deploy.config = SimpleNamespace(remote_gamepad_enabled=False)

    deploy._init_gamepad()

    assert not deploy.remote_gamepad_enabled
    assert created == [("local", 0)]
    assert isinstance(deploy.gamepad, FakeGamepad)


def test_base_selects_virtual_gamepad_when_enabled(monkeypatch):
    created = []

    class FakeVirtualGamepad:
        def __init__(self, host, port):
            created.append((host, port))

        def connect(self):
            return True

    monkeypatch.setattr(
        "mujoco_sim.scripts.base.base.VirtualGamepad",
        FakeVirtualGamepad,
    )
    deploy = object.__new__(MujocoDeploy)
    deploy.config = SimpleNamespace(
        remote_gamepad_enabled=True,
        remote_gamepad_host="127.0.0.1",
        remote_gamepad_port=5555,
    )

    deploy._init_gamepad()

    assert deploy.remote_gamepad_enabled
    assert created == [("127.0.0.1", 5555)]
    assert isinstance(deploy.gamepad, FakeVirtualGamepad)


def test_base_reports_virtual_gamepad_bind_failure():
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    occupied_port = blocker.getsockname()[1]
    try:
        deploy = object.__new__(MujocoDeploy)
        deploy.config = SimpleNamespace(
            remote_gamepad_enabled=True,
            remote_gamepad_host="127.0.0.1",
            remote_gamepad_port=occupied_port,
        )

        with pytest.raises(
            RuntimeError,
            match=rf"127\.0\.0\.1:{occupied_port}.*Address already in use",
        ):
            deploy._init_gamepad()
    finally:
        blocker.close()
