"""Local launcher for remote gamepad forwarding and the mjviser browser."""

from __future__ import annotations

import argparse
import math
import sys
import time
import webbrowser
from contextlib import closing
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mujoco_sim.utils.gamepad_pygame import Gamepad
from mujoco_sim.utils.remote_gamepad import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_RETRY_INTERVAL,
    DEFAULT_SEND_HZ,
    RemoteGamepadSender,
)


DEFAULT_VIEWER_PORT = 8080
VIEWER_PROBE_TIMEOUT = 0.2


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forward a local gamepad and open the remote mjviser page."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--gamepad-port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--viewer-port", type=int, default=DEFAULT_VIEWER_PORT)
    parser.add_argument("--index", type=int, default=0, help="pygame joystick index")
    parser.add_argument("--hz", type=float, default=DEFAULT_SEND_HZ)
    parser.add_argument("--deadzone", type=float, default=0.05)
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=DEFAULT_RETRY_INTERVAL,
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="send gamepad state without opening a browser",
    )
    return parser


def _validate_arguments(parser: argparse.ArgumentParser, args) -> None:
    if not args.host:
        parser.error("--host must not be empty")
    for name in ("gamepad_port", "viewer_port"):
        port = getattr(args, name)
        if port < 1 or port > 65535:
            parser.error(f"--{name.replace('_', '-')} must be in [1, 65535]")
    if args.index < 0:
        parser.error("--index must be non-negative")
    if not math.isfinite(args.hz) or args.hz <= 0.0:
        parser.error("--hz must be finite and greater than zero")
    if not math.isfinite(args.retry_interval) or args.retry_interval < 0.0:
        parser.error("--retry-interval must be finite and non-negative")


def viewer_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def is_viewer_ready(url: str, timeout: float = VIEWER_PROBE_TIMEOUT) -> bool:
    try:
        with closing(urlopen(url, timeout=timeout)) as response:
            return 200 <= response.status < 400
    except (OSError, TimeoutError, URLError):
        return False


def _open_browser_once(url: str) -> None:
    try:
        opened = webbrowser.open(url)
    except (OSError, webbrowser.Error) as exc:
        print(f"[remote_client] Could not open browser: {exc}; open {url} manually")
        return
    if not opened:
        print(f"[remote_client] Browser did not open; open {url} manually")


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)

    url = viewer_url(args.host, args.viewer_port)
    gamepad = Gamepad(joystick_index=args.index, deadzone=args.deadzone)
    sender = RemoteGamepadSender(
        host=args.host,
        port=args.gamepad_port,
        retry_interval=args.retry_interval,
    )
    send_period = 1.0 / args.hz
    browser_handled = args.no_browser
    next_viewer_probe = 0.0

    print(
        f"[remote_client] Forwarding gamepad to {args.host}:{args.gamepad_port}; "
        f"viewer URL is {url}. Press Ctrl+C to quit."
    )
    try:
        while True:
            now = time.monotonic()
            if not browser_handled and now >= next_viewer_probe:
                if is_viewer_ready(url):
                    _open_browser_once(url)
                    browser_handled = True
                next_viewer_probe = now + args.retry_interval

            if gamepad.joystick is None and not gamepad.connect():
                time.sleep(args.retry_interval)
                continue
            if not gamepad.poll():
                print("[remote_client] Gamepad disconnected; retrying")
                time.sleep(args.retry_interval)
                continue

            sender.send_state(gamepad.get_state())
            time.sleep(send_period)
    except KeyboardInterrupt:
        print("\n[remote_client] Stopped by user.")
    finally:
        sender.close()
        gamepad.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
