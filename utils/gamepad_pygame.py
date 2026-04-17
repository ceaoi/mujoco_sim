import time
from typing import Dict, Union, Optional

import pygame


class Gamepad:
    """
    Gamepad wrapper based on pygame.

    Default mapping assumes an Xbox-like controller on Linux.
    Axis mapping follows the same semantic names as the previous jsdev version:

    state keys:
        leftJS_x, leftJS_y, rightJS_x, rightJS_y, L2, R2
        A, B, X, Y, L1, R1

    Notes:
    - Different controllers/platforms may expose different axis/button indices.
    - You can override indices in __init__ if needed.
    """

    def __init__(
        self,
        joystick_index: int = 0,
        deadzone: float = 0.05,
        axis_map: Optional[Dict[str, int]] = None,
        button_map: Optional[Dict[str, int]] = None,
    ):
        self.joystick_index = joystick_index
        self.deadzone = deadzone

        self.connected = False
        self.joystick = None

        self.state = self._make_default_state()
        self.prev_state = dict(self.state)

        # Default pygame.joystick mapping for many Xbox-like controllers on Linux.
        # You may need to adjust these on your machine.
        self.axis_map = axis_map or {
            "leftJS_x": 0,
            "leftJS_y": 1,
            "L2": 2,
            "rightJS_x": 3,
            "rightJS_y": 4,
            "R2": 5,
        }

        self.button_map = button_map or {
            "A": 0,
            "B": 1,
            "X": 2,
            "Y": 3,
            "L1": 4,
            "R1": 5,
        }

        self._pygame_inited = False

    @staticmethod
    def _make_default_state() -> Dict[str, Union[float, bool]]:
        return {
            "leftJS_x": 0.0,
            "leftJS_y": 0.0,
            "rightJS_x": 0.0,
            "rightJS_y": 0.0,
            "L2": 0.0,
            "R2": 0.0,
            "A": False,
            "B": False,
            "X": False,
            "Y": False,
            "L1": False,
            "R1": False,
        }

    def _ensure_pygame(self):
        if not self._pygame_inited:
            pygame.init()
            pygame.joystick.init()
            self._pygame_inited = True

    def connect(self) -> bool:
        self.close()
        self._ensure_pygame()

        count = pygame.joystick.get_count()
        if count <= self.joystick_index:
            self.connected = False
            print(
                f"[pygame_gamepad] No joystick at index {self.joystick_index}. "
                f"Detected count: {count}"
            )
            return False

        try:
            js = pygame.joystick.Joystick(self.joystick_index)
            js.init()
            self.joystick = js
            self.connected = True
            print(
                f"[pygame_gamepad] Connected: index={self.joystick_index}, "
                f"name={js.get_name()}"
            )
            return True
        except pygame.error as e:
            self.joystick = None
            self.connected = False
            print(f"[pygame_gamepad] Failed to connect joystick: {e}")
            return False

    def close(self):
        if self.joystick is not None:
            try:
                self.joystick.quit()
            except Exception:
                pass
        self.joystick = None
        self.connected = False

    def reset_state(self):
        self.state = self._make_default_state()
        self.prev_state = dict(self.state)

    def reconnect(self) -> bool:
        return self.connect()

    def poll(self) -> bool:
        if self.joystick is None:
            self.connected = False
            return False

        self.prev_state = dict(self.state)

        try:
            pygame.event.pump()
        except pygame.error as e:
            print(f"[pygame_gamepad] event pump failed: {e}")
            self.close()
            return False

        try:
            self._update_axes()
            self._update_buttons()
        except pygame.error as e:
            print(f"[pygame_gamepad] read failed: {e}")
            self.close()
            return False

        self.connected = True
        return True

    def _safe_get_axis(self, idx: int) -> float:
        if self.joystick is None:
            return 0.0
        if idx < 0 or idx >= self.joystick.get_numaxes():
            return 0.0
        v = float(self.joystick.get_axis(idx))
        return 0.0 if abs(v) < self.deadzone else v

    def _safe_get_button(self, idx: int) -> bool:
        if self.joystick is None:
            return False
        if idx < 0 or idx >= self.joystick.get_numbuttons():
            return False
        return bool(self.joystick.get_button(idx))

    @staticmethod
    def _normalize_trigger(v: float) -> float:
        # Many pygame backends expose triggers in [-1, 1].
        # Map to [0, 1]. If your controller already gives [0, 1], this still stays usable.
        return max(0.0, min(1.0, 0.5 * (v + 1.0)))

    def _update_axes(self):
        lx = self._safe_get_axis(self.axis_map["leftJS_x"])
        ly = self._safe_get_axis(self.axis_map["leftJS_y"])
        l2 = self._safe_get_axis(self.axis_map["L2"])
        rx = self._safe_get_axis(self.axis_map["rightJS_x"])
        ry = self._safe_get_axis(self.axis_map["rightJS_y"])
        r2 = self._safe_get_axis(self.axis_map["R2"])

        # Keep sign convention aligned with your previous jsdev version
        self.state["leftJS_x"] = -lx
        self.state["leftJS_y"] = -ly
        self.state["rightJS_x"] = -rx
        self.state["rightJS_y"] = -ry
        self.state["L2"] = self._normalize_trigger(l2)
        self.state["R2"] = self._normalize_trigger(r2)

    def _update_buttons(self):
        for name, idx in self.button_map.items():
            self.state[name] = self._safe_get_button(idx)

    def is_button_pressed(self, name: str) -> bool:
        return bool(self.state.get(name, False))

    def is_button_rising_edge(self, name: str) -> bool:
        prev_v = bool(self.prev_state.get(name, False))
        curr_v = bool(self.state.get(name, False))
        return (not prev_v) and curr_v

    def is_axis_pressed(self, name: str, threshold: float = 0.5) -> bool:
        return float(self.state.get(name, 0.0)) > threshold

    def is_axis_rising_edge(self, name: str, threshold: float = 0.5) -> bool:
        prev_on = float(self.prev_state.get(name, 0.0)) > threshold
        curr_on = float(self.state.get(name, 0.0)) > threshold
        return (not prev_on) and curr_on

    def get_axis(self, name: str) -> float:
        return float(self.state.get(name, 0.0))

    def get_button(self, name: str) -> bool:
        return bool(self.state.get(name, False))

    def get_cmd(self):
        """
        Returns normalized [vx_cmd, vy_cmd, yaw_cmd].
        MujocoDeploy can multiply this by cmd_range.
        """
        return [
            self.get_axis("leftJS_y"),
            self.get_axis("leftJS_x"),
            self.get_axis("rightJS_x"),
        ]

    def get_state(self) -> Dict[str, Union[float, bool]]:
        return dict(self.state)

    def print_mapping_info(self):
        if self.joystick is None:
            print("[pygame_gamepad] No joystick connected.")
            return
        print(f"[pygame_gamepad] name={self.joystick.get_name()}")
        print(f"[pygame_gamepad] num_axes={self.joystick.get_numaxes()}")
        print(f"[pygame_gamepad] num_buttons={self.joystick.get_numbuttons()}")
        print(f"[pygame_gamepad] axis_map={self.axis_map}")
        print(f"[pygame_gamepad] button_map={self.button_map}")

    def __del__(self):
        self.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="pygame joystick index")
    parser.add_argument("--hz", type=float, default=20.0, help="print rate")
    parser.add_argument("--deadzone", type=float, default=0.05)
    args = parser.parse_args()

    gamepad = Gamepad(
        joystick_index=args.index,
        deadzone=args.deadzone,
    )

    if not gamepad.connect():
        raise SystemExit(1)

    gamepad.print_mapping_info()
    print("[pygame_gamepad] Start polling. Press Ctrl+C to quit.")

    period = 1.0 / max(args.hz, 1e-6)

    try:
        while True:
            ok = gamepad.poll()
            if not ok:
                print("[pygame_gamepad] Disconnected.")
                break

            s = gamepad.get_state()
            cmd = gamepad.get_cmd()

            line = (
                f"\r"
                f"leftJS=({s['leftJS_x']:+.3f}, {s['leftJS_y']:+.3f})  "
                f"rightJS=({s['rightJS_x']:+.3f}, {s['rightJS_y']:+.3f})  "
                f"L2={s['L2']:+.3f}  R2={s['R2']:+.3f}  "
                f"A={int(s['A'])} B={int(s['B'])} X={int(s['X'])} Y={int(s['Y'])} "
                f"L1={int(s['L1'])} R1={int(s['R1'])}  "
                f"cmd=({cmd[0]:+.3f}, {cmd[1]:+.3f}, {cmd[2]:+.3f})"
            )
            print(line, end="", flush=True)
            time.sleep(period)

    except KeyboardInterrupt:
        print("\n[pygame_gamepad] Stopped by user.")
    finally:
        gamepad.close()
        try:
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass