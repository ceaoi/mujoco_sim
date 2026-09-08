"""Select a MuJoCo OpenGL backend before MuJoCo is imported."""

from __future__ import annotations

import os
from collections.abc import MutableMapping


def configure_headless_mujoco_gl(
    environ: MutableMapping[str, str] | None = None,
) -> bool:
    """Select EGL for a headless process unless the backend is explicit.

    This function must run before importing :mod:`mujoco`.  It returns whether
    it changed the environment, which lets command-line entry points report the
    automatic selection without coupling this helper to logging.
    """

    if environ is None:
        environ = os.environ

    if "MUJOCO_GL" in environ or environ.get("DISPLAY"):
        return False

    environ["MUJOCO_GL"] = "egl"
    return True
