"""TCP transport for forwarding a pygame gamepad through an SSH tunnel."""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

from mujoco_sim.utils.gamepad_pygame import Gamepad


PROTOCOL_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555
DEFAULT_SEND_HZ = 50.0
DEFAULT_RETRY_INTERVAL = 1.0
MAX_FRAME_BYTES = 8192
RECV_CHUNK_BYTES = 4096

AXIS_KEYS = ("leftJS_x", "leftJS_y", "rightJS_x", "rightJS_y")
TRIGGER_KEYS = ("L2", "R2")
BUTTON_KEYS = ("A", "B", "X", "Y", "L1", "R1")
STATE_KEYS = AXIS_KEYS + TRIGGER_KEYS + BUTTON_KEYS


def _validate_number(value: Any, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"gamepad state {name!r} must be a number")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"gamepad state {name!r} must be finite")
    if result < lower or result > upper:
        raise ValueError(
            f"gamepad state {name!r} must be in [{lower}, {upper}], got {result}"
        )
    return result


def validate_gamepad_state(state: Any) -> dict[str, float | bool]:
    """Validate and normalize one complete gamepad state snapshot."""
    if not isinstance(state, Mapping):
        raise ValueError("gamepad state must be an object")

    actual_keys = set(state)
    expected_keys = set(STATE_KEYS)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"gamepad state keys do not match; missing={missing}, extra={extra}"
        )

    normalized: dict[str, float | bool] = {}
    for name in AXIS_KEYS:
        normalized[name] = _validate_number(state[name], name, -1.0, 1.0)
    for name in TRIGGER_KEYS:
        normalized[name] = _validate_number(state[name], name, 0.0, 1.0)
    for name in BUTTON_KEYS:
        value = state[name]
        if not isinstance(value, bool):
            raise ValueError(f"gamepad state {name!r} must be a boolean")
        normalized[name] = value
    return normalized


def encode_gamepad_message(state: Any, sequence: int) -> bytes:
    """Encode one state snapshot as a newline-delimited JSON protocol frame."""
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("gamepad message sequence must be a non-negative integer")

    payload = {
        "version": PROTOCOL_VERSION,
        "sequence": sequence,
        "state": validate_gamepad_state(state),
    }
    return (
        json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def decode_gamepad_message(frame: bytes) -> tuple[int, dict[str, float | bool]]:
    """Decode and validate one JSON frame without its trailing newline."""
    try:
        payload = json.loads(frame.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid gamepad JSON frame: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("gamepad message must be an object")
    if set(payload) != {"version", "sequence", "state"}:
        raise ValueError(
            "gamepad message keys must be exactly version, sequence, and state"
        )
    if payload["version"] != PROTOCOL_VERSION:
        raise ValueError(
            f"unsupported gamepad protocol version: {payload['version']!r}"
        )

    sequence = payload["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("gamepad message sequence must be a non-negative integer")
    return sequence, validate_gamepad_state(payload["state"])


class RemoteGamepadSender:
    """TCP client used on the local machine to forward physical gamepad state."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        retry_interval: float = DEFAULT_RETRY_INTERVAL,
    ):
        if not host:
            raise ValueError("remote gamepad host must not be empty")
        if port < 1 or port > 65535:
            raise ValueError("remote gamepad port must be in [1, 65535]")
        if retry_interval < 0.0:
            raise ValueError("retry_interval must be non-negative")

        self.host = host
        self.port = int(port)
        self.retry_interval = float(retry_interval)
        self.sequence = 0
        self.connected = False
        self._socket: socket.socket | None = None
        self._next_connect_time = 0.0

    def connect(self) -> bool:
        if self._socket is not None:
            return True

        now = time.monotonic()
        if now < self._next_connect_time:
            return False

        try:
            client = socket.create_connection((self.host, self.port), timeout=1.0)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client.settimeout(None)
        except OSError as exc:
            self._next_connect_time = now + self.retry_interval
            print(
                f"[remote_gamepad_sender] Connection failed to "
                f"{self.host}:{self.port}: {exc}"
            )
            return False

        self._socket = client
        self.connected = True
        self._next_connect_time = 0.0
        print(f"[remote_gamepad_sender] Connected to {self.host}:{self.port}")
        return True

    def send_state(self, state: Any) -> bool:
        if not self.connect():
            return False

        frame = encode_gamepad_message(state, self.sequence)
        try:
            assert self._socket is not None
            self._socket.sendall(frame)
        except OSError as exc:
            print(f"[remote_gamepad_sender] Send failed: {exc}")
            self.close()
            self._next_connect_time = time.monotonic() + self.retry_interval
            return False

        self.sequence += 1
        return True

    def close(self) -> None:
        active_socket = getattr(self, "_socket", None)
        if active_socket is not None:
            with suppress(OSError):
                active_socket.close()
        self._socket = None
        self.connected = False

    def __del__(self):
        self.close()


class VirtualGamepad(Gamepad):
    """Non-blocking TCP server exposing remote state through the Gamepad API."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        max_frame_bytes: int = MAX_FRAME_BYTES,
    ):
        if not host:
            raise ValueError("virtual gamepad host must not be empty")
        if port < 0 or port > 65535:
            raise ValueError("virtual gamepad port must be in [0, 65535]")
        if max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be positive")

        super().__init__()
        self.host = host
        self.port = int(port)
        self.max_frame_bytes = int(max_frame_bytes)
        self.server_socket: socket.socket | None = None
        self.client_socket: socket.socket | None = None
        self.client_address: tuple[str, int] | None = None
        self._receive_buffer = bytearray()
        self._last_sequence = -1

    def connect(self) -> bool:
        self.close()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            server.setblocking(False)
        except OSError:
            server.close()
            raise

        self.server_socket = server
        self.port = int(server.getsockname()[1])
        print(f"[virtual_gamepad] Listening on {self.host}:{self.port}")
        return True

    def close(self) -> None:
        self._close_client()
        server_socket = getattr(self, "server_socket", None)
        if server_socket is not None:
            with suppress(OSError):
                server_socket.close()
        self.server_socket = None

    def poll(self) -> bool:
        self.prev_state = dict(self.state)
        if self.server_socket is None:
            return False

        self._accept_client()
        latest_state = self._receive_latest_state()
        if latest_state is not None:
            self.state = latest_state
        return True

    def _accept_client(self) -> None:
        if self.server_socket is None or self.client_socket is not None:
            return

        try:
            client, address = self.server_socket.accept()
        except BlockingIOError:
            return
        except OSError as exc:
            raise RuntimeError(f"virtual gamepad accept failed: {exc}") from exc

        client.setblocking(False)
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.client_socket = client
        self.client_address = (str(address[0]), int(address[1]))
        self.connected = True
        self._receive_buffer.clear()
        self._last_sequence = -1
        print(f"[virtual_gamepad] Client connected: {address[0]}:{address[1]}")

    def _receive_latest_state(self) -> dict[str, float | bool] | None:
        if self.client_socket is None:
            return None

        latest_state = None
        while self.client_socket is not None:
            try:
                chunk = self.client_socket.recv(RECV_CHUNK_BYTES)
            except BlockingIOError:
                break
            except OSError as exc:
                print(f"[virtual_gamepad] Receive failed: {exc}")
                self._close_client()
                break

            if not chunk:
                print("[virtual_gamepad] Client disconnected; holding last state")
                self._close_client()
                break

            self._receive_buffer.extend(chunk)
            latest_state = self._consume_complete_frames(latest_state)
            if len(self._receive_buffer) > self.max_frame_bytes:
                print("[virtual_gamepad] Oversized frame; disconnecting client")
                self._close_client()
                break

        return latest_state

    def _consume_complete_frames(
        self,
        latest_state: dict[str, float | bool] | None,
    ) -> dict[str, float | bool] | None:
        while True:
            newline_index = self._receive_buffer.find(b"\n")
            if newline_index < 0:
                return latest_state

            frame = bytes(self._receive_buffer[:newline_index])
            del self._receive_buffer[:newline_index + 1]
            if not frame:
                continue
            if len(frame) > self.max_frame_bytes:
                print("[virtual_gamepad] Ignoring oversized complete frame")
                continue

            try:
                sequence, state = decode_gamepad_message(frame)
            except ValueError as exc:
                print(f"[virtual_gamepad] Ignoring invalid frame: {exc}")
                continue
            if sequence <= self._last_sequence:
                print(
                    f"[virtual_gamepad] Ignoring stale sequence {sequence}; "
                    f"latest is {self._last_sequence}"
                )
                continue

            self._last_sequence = sequence
            latest_state = state

    def _close_client(self) -> None:
        client_socket = getattr(self, "client_socket", None)
        if client_socket is not None:
            with suppress(OSError):
                client_socket.close()
        self.client_socket = None
        self.client_address = None
        self.connected = False
        receive_buffer = getattr(self, "_receive_buffer", None)
        if receive_buffer is not None:
            receive_buffer.clear()
        self._last_sequence = -1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forward a local pygame gamepad to a remote MuJoCo simulation."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--index", type=int, default=0, help="pygame joystick index")
    parser.add_argument("--hz", type=float, default=DEFAULT_SEND_HZ)
    parser.add_argument("--deadzone", type=float, default=0.05)
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=DEFAULT_RETRY_INTERVAL,
    )
    args = parser.parse_args()

    if args.hz <= 0.0:
        parser.error("--hz must be greater than zero")
    if args.retry_interval < 0.0:
        parser.error("--retry-interval must be non-negative")

    gamepad = Gamepad(joystick_index=args.index, deadzone=args.deadzone)
    sender = RemoteGamepadSender(
        host=args.host,
        port=args.port,
        retry_interval=args.retry_interval,
    )
    period = 1.0 / args.hz

    print("[remote_gamepad_sender] Starting. Press Ctrl+C to quit.")
    try:
        while True:
            if gamepad.joystick is None and not gamepad.connect():
                time.sleep(args.retry_interval)
                continue
            if not gamepad.poll():
                print("[remote_gamepad_sender] Gamepad disconnected; retrying")
                time.sleep(args.retry_interval)
                continue

            sender.send_state(gamepad.get_state())
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[remote_gamepad_sender] Stopped by user.")
    finally:
        sender.close()
        gamepad.close()


if __name__ == "__main__":
    main()
